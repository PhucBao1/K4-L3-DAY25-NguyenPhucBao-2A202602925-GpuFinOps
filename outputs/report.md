# NimbusAI — Báo cáo tối ưu chi phí GPU

**Kỳ báo cáo:** hàng tháng  
**Chi phí baseline:** $27,133  
**Chi phí sau tối ưu:** $14,626  
**Tiết kiệm dự kiến (Projected savings):** $12,507  (**46%**)

## Tiết kiệm theo đòn bẩy

| Đòn bẩy | Tiết kiệm (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,212 |
| Purchasing (spot/reserved) | $10,040 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Tính bền vững (Sustainability)

- Năng lượng mỗi truy vấn: 0.24 Wh
- Carbon mỗi truy vấn: 0.091 gCO2e
- Vùng tốt nhất (Best region — sạch nhất, giá điện trung bình): europe-north1
- Vùng điện rẻ nhất (Cheapest power): us-east-wa

## Vì sao GPU-Util "nói dối"

| GPU | Loại | GPU-Util | MFU | MBU | Chi phí on-demand / tháng |
|---|---|---|---|---|---|
| gpu-h100-4 | H100 | 98% | 0.19 | 0.21 | $1,800 |
| gpu-a10g-1 | A10G | 97% | 0.27 | 0.30 | $720 |

GPU-Util của `nvidia-smi` là tỷ lệ thời gian có **ít nhất một kernel đang chạy**, không phải tỷ lệ FLOPs được dùng. Kernel nhỏ phải chờ dữ liệu từ HBM (memory stall), chi phí khởi chạy kernel / đồng bộ CPU-GPU, hoặc batch quá nhỏ đều khiến GPU hiện "bận" 98% trong khi Tensor Core gần như ngồi không. Với MFU ~0.2, mỗi FLOP hữu ích đắt gấp ~2 lần so với khi chạy ở MFU 0.4 (mức tốt cho training): trả tiền H100 nhưng chỉ nhận hiệu năng của GPU rẻ hơn. Cách xử lý: profile (Nsight / DCGM `SM_ACTIVE`, `PIPE_TENSOR_ACTIVE`), tăng batch size / fuse kernel, hoặc right-size xuống GPU rẻ hơn (đòn bẩy *Right-size util-lies*).

## Khuyến nghị theo thứ tự ROI

| # | Đòn bẩy | Tiết kiệm / tháng | Công sức / rủi ro |
|---|---|---|---|
| 1 | Purchasing (spot/reserved) | $10,040 | đổi hợp đồng mua, không sửa code; cần checkpoint cho spot và xác nhận duty cycle trước khi cam kết reserved |
| 2 | Inference (cascade/cache/batch) | $1,212 | sửa gateway/routing; cần eval chất lượng của model nhỏ |
| 3 | Right-size util-lies | $655 | đổi loại GPU sau khi profile kernel; rủi ro thấp |
| 4 | Kill idle GPUs | $600 | auto-shutdown / scale-to-zero; dễ nhất, làm ngay |

Inference tính theo đơn vị sản lượng: **$6.488 → $1.126 / 1M-token** (-82.6%). Số tiền tuyệt đối còn nhỏ vì traffic hiện tại nhỏ, nhưng khoản tiết kiệm tăng tỷ lệ thuận với số token, nên làm song song với Purchasing.

## Phần mở rộng 3 — Kinh tế học của cache

**Giả định và nguồn:**
- Giá cache theo bảng giá prompt caching của Anthropic: cache read = 0.1× giá input; cache write TTL 5 phút = 1.25× (phụ phí 25%), TTL 1 giờ = 2× (phụ phí 100%). Mức 0.1× trùng với `cache_discount=0.10` sẵn có của lab. Đây là giá riêng của Anthropic: OpenAI không thu phí ghi cache, Gemini tính phí lưu trữ cache theo giờ.
- Dữ liệu không có mã prefix nên không đếm được số lần đọc lại thật. Ước lượng: một prefix ghi 1 lần, đọc lại k lần thì hit-rate h = k/(k+1), suy ra **k = h/(1−h)**. Đây là ước lượng tự đặt, không phải công thức chuẩn.
- **Cách A:** h = cached / toàn bộ input (tính cả câu hỏi của user, vốn không cache được) → ước lượng thấp, thận trọng.
- **Cách B:** h = cached / phần input có thể cache (prefix tĩnh: 70% input với team assistant/rag, 10% với team khác, theo `data/generate.py`) → sát với mức tái sử dụng thật hơn.
- Break-even = phụ phí ghi / (1 − 0.10): **0.278 lần đọc** (5 phút), **1.111 lần đọc** (1 giờ).

| Tier | Hit (A) | Đọc/ghi (A) | Hit (B) | Đọc/ghi (B) | Có lợi? 5 phút (A / B) | Có lợi? 1 giờ (A / B) |
|---|---|---|---|---|---|---|
| small | 31.9% | 0.47 | 72.6% | 2.64 | có / có | không / có |
| large | 31.9% | 0.47 | 72.1% | 2.59 | có / có | không / có |

- Break-even giống nhau cho model nhỏ và lớn vì phụ phí ghi tỷ lệ với giá input: yếu tố quyết định là **prefix được đọc lại bao nhiêu lần**, không phải model đắt hay rẻ.
- **Cache 5 phút:** có lợi theo cả hai cách ước lượng → kết luận chắc chắn. M2 dùng cổng thận trọng (cách A, 5 phút).
- **Cache 1 giờ:** kết luận **phụ thuộc cách ước lượng** (A: lỗ, B: có lợi). Chỉ nên bật khi đo được số lần đọc thật (`usage.cache_read_input_tokens` / `cache_creation_input_tokens` trên gateway) vượt ~1.1 lần mỗi lần ghi, ví dụ khi traffic thưa hơn 1 lần / 5 phút khiến cache 5 phút hết hạn trước khi được đọc.

## Phần mở rộng 4 — Ngân sách reasoning

Reasoning chiếm **8.4% traffic** nhưng **16.5% chi phí $** và **94.0% năng lượng** ($42 và 894 kWh mỗi tháng). Mỗi query reasoning tốn ~80 lần năng lượng vì sinh chuỗi suy luận dài (~6 lần output token) ở pha decode, vốn bị giới hạn bởi băng thông bộ nhớ: mỗi token phải đọc lại toàn bộ weights và KV-cache.

Mô phỏng trần: giữ reasoning cho request phức tạp nhất, hạ cấp request đơn giản nhất trước. Dữ liệu không có nhãn độ phức tạp nên dùng `input_tokens` làm proxy (prompt dài ≈ tác vụ nhiều bước); khi có router chấm điểm độ phức tạp thì thay bằng điểm đó.

| Trần reasoning (% traffic) | Ngưỡng X: chỉ reasoning khi input ≥ | Request bị hạ cấp | Tiết kiệm $ / tháng | Tiết kiệm kWh / tháng | % năng lượng inference |
|---|---|---|---|---|---|
| 10% | 408 token | 0 | $0.00 | 0 | -0% |
| 5% | 2,034 token | 81 | $13.50 | 299 | -31% |
| 0% | tắt reasoning | 201 | $30.90 | 888 | -93% |

**Quy tắc routing đề xuất:** chỉ bật reasoning cho request eval có **input ≥ 2,034 token** (tương đương trần 5% traffic) hoặc khi câu trả lời đầu không qua bước kiểm tra; còn lại dùng chế độ thường. Trần 10% không tiết kiệm gì vì traffic reasoning hiện chỉ 8.4%. Trần 5% tiết kiệm $13.50 / tháng nhưng giảm 31% năng lượng inference: lợi ích chính của ngân sách reasoning là **năng lượng và carbon**, không phải tiền.

## Tính bền vững theo vùng

Năng lượng inference ~950 kWh / tháng (từ M2).

| Vùng | $/kWh | gCO2/kWh | Tiền điện / tháng | kgCO2e / tháng |
|---|---|---|---|---|
| us-east-1 | 0.120 | 380 | $114.03 | 361.1 |
| us-west-2 | 0.070 | 120 | $66.52 | 114.0 |
| europe-north1 | 0.090 | 30 | $85.52 | 28.5 |
| europe-central2 | 0.180 | 660 | $171.05 | 627.2 |
| us-east-wa | 0.055 | 90 | $52.26 | 85.5 |

- **europe-north1** là lựa chọn tốt nhất khi cân cả carbon lẫn chi phí: sạch nhất (thủy điện Na Uy, 30 gCO2/kWh) và giá điện ở mức trung bình (0.09 $/kWh, rẻ hơn us-east-1). Nếu chỉ xét giá điện thì us-east-wa rẻ nhất; us-west-2 / us-east-wa là lựa chọn cân bằng ở Mỹ (rẻ hơn us-east-1 và ít carbon hơn >65%).
- Carbon và tiền đi cùng chiều: rời us-east-1 / europe-central2 giảm cả hai. Tiền điện nhỏ so với tiền thuê GPU, nên lý do chọn vùng chủ yếu là carbon cộng với giá GPU tại vùng đó.
- Đánh đổi: europe-north1 xa user Mỹ (+~100 ms RTT) → dùng cho training / batch chịu được gián đoạn; inference real-time giữ ở vùng gần user nhưng sạch (us-west-2).


_Số liệu là snapshot tháng 6/2026; cần cập nhật lại giá trước khi áp dụng._