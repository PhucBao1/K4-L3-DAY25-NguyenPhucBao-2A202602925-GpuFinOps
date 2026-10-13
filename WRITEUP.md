# Lab 25 — GPU FinOps: Bài viết ngắn

**Học viên:** Nguyễn Phúc Bảo · 2A202602925
Số liệu lấy từ `python missions/run_all.py` (seed = 25); chi tiết trong `outputs/report.md`.

## 1. Baseline vs. Optimized

| | Baseline | Optimized | Tiết kiệm |
|---|---|---|---|
| Tổng chi phí GPU / tháng | $27,133 | $14,626 | **$12,507 (46%)** |
| Inference $/1M-token | $6.488 | $1.126 | **-82.6%** |
| Purchasing (8 job) / tháng | $25,667 | $15,627 | -39.1% |

Baseline giả định mọi request inference chạy model lớn, không có cache, không dùng batch, và mọi job đều thuê GPU on-demand.

## 2. Đóng góp của từng đòn bẩy

| # | Lever | $/tháng | Tỷ trọng |
|---|---|---|---|
| 1 | Purchasing (spot/reserved) | $10,040 | 80% |
| 2 | Inference (cascade/cache/batch) | $1,212 | 10% |
| 3 | Right-size util-lies | $655 | 5% |
| 4 | Kill idle GPUs | $600 | 5% |

**Purchasing đóng góp nhiều nhất** vì phần lớn hóa đơn là giờ GPU của các job train. Job `job-train-llm` dùng 8×H100 chạy 20 giờ/ngày, riêng job này đã tốn $12,000/tháng. Job này chịu được gián đoạn nên chuyển sang spot kèm checkpoint vẫn tiết kiệm 37% sau khi đã trừ chi phí chạy lại khi bị ngắt. Ba job inference chạy 24/7 có duty cycle 100%, cao hơn break-even 55%, nên mua reserved 3 năm sẽ giảm 40–44%.
Inference là lever lớn nhất **tính theo đơn vị** ($/1M-token giảm 5.8 lần), nhưng hiện chỉ đứng thứ hai về số tiền tuyệt đối vì lượng token mỗi ngày còn nhỏ. Khi traffic tăng, khoản tiết kiệm này tăng theo tỷ lệ thuận với số token.

## 3. GPU-Util "nói dối"

`gpu-h100-4` báo **GPU-Util 98% nhưng MFU chỉ 0.19**. `gpu-a10g-1` báo 97% nhưng MFU chỉ 0.27. `nvidia-smi` chỉ đo xem có kernel nào đang chạy hay không, chứ không đo GPU dùng được bao nhiêu phần sức tính toán. Vì vậy GPU vẫn hiện "bận" khi đang chờ bộ nhớ (memory stall), khi tốn thời gian khởi chạy kernel hoặc khi batch quá nhỏ.
**Tác động tài chính:** chiếc H100 tốn $1,800/tháng nhưng chỉ cho khoảng 1/5 sức tính toán tối đa. Mỗi FLOP hữu ích vì thế đắt gấp khoảng 2 lần so với một job đạt MFU 0.4. Right-size hai GPU này xuống loại GPU thấp hơn một bậc tiết kiệm **$655/tháng**. Trước khi đổi GPU nên profile bằng DCGM (`PIPE_TENSOR_ACTIVE`), vì tăng batch hoặc gộp kernel (fuse) có thể sửa tận gốc mà không cần đổi GPU.

## 4. Các phần mở rộng đã làm

### Ext 3 — `cache_is_worth_it()` (`finops/pricing.py`, M2)
- Thêm hai hàm `cache_break_even_reads()` và `cache_is_worth_it()`. M2 chỉ tính tiền tiết kiệm từ cache cho tier nào qua được điều kiện này.
- **Giả định về giá** (theo bảng giá prompt caching của Anthropic): cache read = 0.1× giá input; cache write TTL 5 phút = 1.25×, TTL 1 giờ = 2×. Từ đó break-even là **0.278 lần đọc mỗi lần ghi** (TTL 5 phút) và **1.111 lần** (TTL 1 giờ). Đây là giá riêng của Anthropic: OpenAI không thu phí ghi, Gemini tính phí lưu trữ theo giờ.
- **Giả định về số lần đọc:** dữ liệu không có mã prefix, nên mình ước lượng bằng công thức tự đặt k = h/(1−h) (prefix ghi 1 lần, đọc k lần thì hit-rate h = k/(k+1)). Có hai cách tính h:
  - Cách A, trên toàn bộ input: h = 31.9%, tương đương **0.47 lần đọc**. Cách này thận trọng vì tính cả câu hỏi của user, vốn không cache được.
  - Cách B, chỉ trên phần prefix có thể cache: h ≈ 72%, tương đương **khoảng 2.6 lần đọc**.

| | TTL 5 phút | TTL 1 giờ |
|---|---|---|
| Cách A (0.47 lần đọc) | có lợi | **lỗ** |
| Cách B (khoảng 2.6 lần đọc) | có lợi | có lợi |

- **Insight:**
  - Break-even giống hệt nhau cho model nhỏ và model lớn, vì phí ghi tỷ lệ thuận với giá input. Điều quyết định là **prefix được đọc lại bao nhiêu lần**, không phải model đắt hay rẻ.
  - Với TTL 5 phút, cache có lợi theo cả hai cách ước lượng nên kết luận chắc chắn; M2 dùng cổng thận trọng (cách A, TTL 5 phút).
  - Với TTL 1 giờ, kết luận phụ thuộc vào cách ước lượng. Chỉ nên bật khi gateway đo được số lần đọc thật lớn hơn khoảng 1.1 lần mỗi lần ghi.

### Ext 4 — Ngân sách Reasoning (M2, M5)
- Tách riêng chi phí `$` và năng lượng `Wh` của các request `is_reasoning=1`. Reasoning chỉ chiếm **8.4% traffic** nhưng tốn **16.5% tiền** và **94% năng lượng** ($42 và 894 kWh mỗi tháng).
- Lý do: reasoning sinh ra lượng output token gấp khoảng 6 lần. Phần sinh token (decode) bị giới hạn bởi băng thông bộ nhớ, mỗi token phải đọc lại toàn bộ weights và KV-cache, nên tốn khoảng 80 lần năng lượng so với query thường.
- Mô phỏng các mức trần: giữ reasoning cho request phức tạp nhất, hạ cấp request đơn giản nhất trước. Dữ liệu không có nhãn độ phức tạp nên dùng `input_tokens` làm proxy.

| Trần | Ngưỡng X (input ≥) | Hạ cấp | Tiết kiệm $/tháng | Tiết kiệm kWh/tháng |
|---|---|---|---|---|
| 10% | 408 token | 0 | $0.00 | 0 |
| 5% | 2,034 token | 81 | $13.50 | 299 (-31%) |
| 0% | tắt reasoning | 201 | $30.90 | 888 |

- **Quy tắc routing đề xuất:** chỉ bật reasoning cho request eval có **input ≥ 2,034 token** (tương đương trần 5%) hoặc khi câu trả lời đầu không qua bước kiểm tra.
- **Insight:** trần 10% (theo đề bài) không có tác dụng vì traffic reasoning đã dưới 10%. Ở trần 5%, tiền tiết kiệm nhỏ ($13.50/tháng) nhưng năng lượng inference giảm 31%: lợi ích chính của ngân sách reasoning là **năng lượng và carbon**, không phải tiền.

Kiểm chứng: `tests/test_extensions.py` có 3 test, cùng 15 test gốc cho kết quả 18/18 pass; `verify.py` vẫn đạt 11/11.

## 5. Ba hành động đầu tiên cho NimbusAI

1. **Chuyển job train sang spot và mua reserved cho inference chạy 24/7** (khoảng -$10k/tháng). Lever này lớn nhất và không cần sửa code. Điều kiện: phải bật checkpoint định kỳ cho job train và xác nhận duty cycle ≥ 55% trước khi cam kết reserved 3 năm.
2. **Tắt GPU idle tự động và profile hai GPU "util-lie"** (khoảng -$1.3k/tháng). Làm được trong một tuần. Đổi dashboard sang theo dõi MFU/MBU thay cho GPU-Util để lỗi này không lặp lại.
3. **Đưa inference qua một gateway có cascade, cache, batch và trần reasoning 5%** ($/1M-token giảm 82.6%, năng lượng inference giảm 31%). Chạy training và batch có thể gián đoạn ở `europe-north1` (30 gCO2/kWh); giữ inference real-time ở `us-west-2` để gần người dùng mà vẫn sạch (ít carbon hơn 68% so với `us-east-1`).
