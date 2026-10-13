"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}
LEVER_INFER = "Inference (cascade/cache/batch)"
LEVER_BUY = "Purchasing (spot/reserved)"
LEVER_RIGHTSIZE = "Right-size util-lies"
LEVER_IDLE = "Kill idle GPUs"
# effort / risk note per lever, used to rank recommendations by ROI
LEVER_NOTES = {
    LEVER_BUY: "đổi hợp đồng mua, không sửa code; cần checkpoint cho spot và xác nhận duty cycle trước khi cam kết reserved",
    LEVER_INFER: "sửa gateway/routing; cần eval chất lượng của model nhỏ",
    LEVER_RIGHTSIZE: "đổi loại GPU sau khi profile kernel; rủi ro thấp",
    LEVER_IDLE: "auto-shutdown / scale-to-zero; dễ nhất, làm ngay",
}


def _yn(b: bool) -> str:
    return "có" if b else "không"


def build_analysis(r1: dict, r2: dict, levers: dict, cat: dict) -> list[str]:
    """Markdown analysis sections built from live mission numbers (keeps report consistent)."""
    L = ["## Vì sao GPU-Util \"nói dối\"", "",
         "| GPU | Loại | GPU-Util | MFU | MBU | Chi phí on-demand / tháng |", "|---|---|---|---|---|---|"]
    for g in r1["lies"]:
        monthly = num(cat[g["gpu_type"]]["on_demand_hr"]) * 24 * DAYS
        L.append(f"| {g['gpu_id']} | {g['gpu_type']} | {g['gpu_util_pct']:.0f}% | {g['mfu']:.2f} | {g['mbu']:.2f} | ${monthly:,.0f} |")
    L += ["",
          "GPU-Util của `nvidia-smi` là tỷ lệ thời gian có **ít nhất một kernel đang chạy**, không phải tỷ lệ FLOPs được dùng. "
          "Kernel nhỏ phải chờ dữ liệu từ HBM (memory stall), chi phí khởi chạy kernel / đồng bộ CPU-GPU, hoặc batch quá nhỏ "
          "đều khiến GPU hiện \"bận\" 98% trong khi Tensor Core gần như ngồi không. Với MFU ~0.2, mỗi FLOP hữu ích đắt gấp ~2 lần "
          "so với khi chạy ở MFU 0.4 (mức tốt cho training): trả tiền H100 nhưng chỉ nhận hiệu năng của GPU rẻ hơn. "
          "Cách xử lý: profile (Nsight / DCGM `SM_ACTIVE`, `PIPE_TENSOR_ACTIVE`), tăng batch size / fuse kernel, "
          "hoặc right-size xuống GPU rẻ hơn (đòn bẩy *Right-size util-lies*).", ""]

    L += ["## Khuyến nghị theo thứ tự ROI", "", "| # | Đòn bẩy | Tiết kiệm / tháng | Công sức / rủi ro |", "|---|---|---|---|"]
    for i, (name, amt) in enumerate(sorted(levers.items(), key=lambda kv: -kv[1]), 1):
        L.append(f"| {i} | {name} | ${amt:,.0f} | {LEVER_NOTES.get(name, '')} |")
    L += ["", f"Inference tính theo đơn vị sản lượng: **${r2['baseline_per_m']:.3f} → ${r2['optimized_per_m']:.3f} / 1M-token** "
          f"(-{r2['savings_pct']:.1f}%). Số tiền tuyệt đối còn nhỏ vì traffic hiện tại nhỏ, nhưng khoản tiết kiệm tăng tỷ lệ thuận "
          "với số token, nên làm song song với Purchasing.", ""]

    c0 = next(iter(r2["cache"].values()))
    L += ["## Phần mở rộng 3 — Kinh tế học của cache", "",
          "**Giả định và nguồn:**",
          "- Giá cache theo bảng giá prompt caching của Anthropic: cache read = 0.1× giá input; cache write TTL 5 phút = 1.25× "
          "(phụ phí 25%), TTL 1 giờ = 2× (phụ phí 100%). Mức 0.1× trùng với `cache_discount=0.10` sẵn có của lab. "
          "Đây là giá riêng của Anthropic: OpenAI không thu phí ghi cache, Gemini tính phí lưu trữ cache theo giờ.",
          "- Dữ liệu không có mã prefix nên không đếm được số lần đọc lại thật. Ước lượng: một prefix ghi 1 lần, đọc lại k lần "
          "thì hit-rate h = k/(k+1), suy ra **k = h/(1−h)**. Đây là ước lượng tự đặt, không phải công thức chuẩn.",
          "- **Cách A:** h = cached / toàn bộ input (tính cả câu hỏi của user, vốn không cache được) → ước lượng thấp, thận trọng.",
          "- **Cách B:** h = cached / phần input có thể cache (prefix tĩnh: 70% input với team assistant/rag, 10% với team khác, "
          "theo `data/generate.py`) → sát với mức tái sử dụng thật hơn.",
          f"- Break-even = phụ phí ghi / (1 − 0.10): **{c0['break_even_5m']:.3f} lần đọc** (5 phút), "
          f"**{c0['break_even_1h']:.3f} lần đọc** (1 giờ).", "",
          "| Tier | Hit (A) | Đọc/ghi (A) | Hit (B) | Đọc/ghi (B) | Có lợi? 5 phút (A / B) | Có lợi? 1 giờ (A / B) |",
          "|---|---|---|---|---|---|---|"]
    for t, c in r2["cache"].items():
        L.append(f"| {t} | {c['hit_rate']:.1%} | {c['avg_reads']:.2f} | {c['hit_rate_cacheable']:.1%} | {c['avg_reads_cacheable']:.2f} | "
                 f"{_yn(c['worth_5m_a'])} / {_yn(c['worth_5m_b'])} | {_yn(c['worth_1h_a'])} / {_yn(c['worth_1h_b'])} |")
    L += ["", "- Break-even giống nhau cho model nhỏ và lớn vì phụ phí ghi tỷ lệ với giá input: yếu tố quyết định là "
          "**prefix được đọc lại bao nhiêu lần**, không phải model đắt hay rẻ.",
          "- **Cache 5 phút:** có lợi theo cả hai cách ước lượng → kết luận chắc chắn. M2 dùng cổng thận trọng (cách A, 5 phút).",
          "- **Cache 1 giờ:** kết luận **phụ thuộc cách ước lượng** (A: lỗ, B: có lợi). Chỉ nên bật khi đo được số lần đọc thật "
          "(`usage.cache_read_input_tokens` / `cache_creation_input_tokens` trên gateway) vượt ~1.1 lần mỗi lần ghi, "
          "ví dụ khi traffic thưa hơn 1 lần / 5 phút khiến cache 5 phút hết hạn trước khi được đọc.", ""]

    rs = r2["reasoning"]
    L += ["## Phần mở rộng 4 — Ngân sách reasoning", "",
          f"Reasoning chiếm **{rs['traffic_pct']}% traffic** nhưng **{rs['cost_pct']}% chi phí $** và **{rs['wh_pct']}% năng lượng** "
          f"(${rs['usd_daily'] * DAYS:,.0f} và {rs['wh_daily'] * DAYS / 1000:,.0f} kWh mỗi tháng). Mỗi query reasoning tốn ~80 lần năng lượng vì sinh "
          "chuỗi suy luận dài (~6 lần output token) ở pha decode, vốn bị giới hạn bởi băng thông bộ nhớ: mỗi token phải đọc lại "
          "toàn bộ weights và KV-cache.", "",
          "Mô phỏng trần: giữ reasoning cho request phức tạp nhất, hạ cấp request đơn giản nhất trước. Dữ liệu không có nhãn "
          "độ phức tạp nên dùng `input_tokens` làm proxy (prompt dài ≈ tác vụ nhiều bước); khi có router chấm điểm độ phức "
          "tạp thì thay bằng điểm đó.", "",
          "| Trần reasoning (% traffic) | Ngưỡng X: chỉ reasoning khi input ≥ | Request bị hạ cấp | Tiết kiệm $ / tháng | "
          "Tiết kiệm kWh / tháng | % năng lượng inference |", "|---|---|---|---|---|---|"]
    for c in rs["caps"]:
        x = f"{c['min_input_tokens']:,} token" if c["min_input_tokens"] else "tắt reasoning"
        L.append(f"| {c['cap_pct']}% | {x} | {c['downgraded']} | ${c['usd_saved_daily'] * DAYS:,.2f} | "
                 f"{c['wh_saved_daily'] * DAYS / 1000:,.0f} | -{c['wh_saved_daily'] / rs['total_wh_daily']:.0%} |")
    c5 = next(c for c in rs["caps"] if c["cap_pct"] == 5)
    L += ["", f"**Quy tắc routing đề xuất:** chỉ bật reasoning cho request eval có **input ≥ {c5['min_input_tokens']:,} token** "
          "(tương đương trần 5% traffic) hoặc khi câu trả lời đầu không qua bước kiểm tra; còn lại dùng chế độ thường. "
          f"Trần 10% không tiết kiệm gì vì traffic reasoning hiện chỉ {rs['traffic_pct']}%. Trần 5% tiết kiệm "
          f"${c5['usd_saved_daily'] * DAYS:,.2f} / tháng nhưng giảm {c5['wh_saved_daily'] / rs['total_wh_daily']:.0%} năng lượng "
          "inference: lợi ích chính của ngân sách reasoning là **năng lượng và carbon**, không phải tiền.", ""]
    return L


def region_table(total_wh_daily: float) -> list[str]:
    kwh = total_wh_daily * DAYS / 1000.0
    wh = kwh * 1000
    L = ["## Tính bền vững theo vùng", "",
         f"Năng lượng inference ~{kwh:,.0f} kWh / tháng (từ M2).", "",
         "| Vùng | $/kWh | gCO2/kWh | Tiền điện / tháng | kgCO2e / tháng |", "|---|---|---|---|---|"]
    for reg in sustainability.REGION_CARBON:
        L.append(f"| {reg} | {sustainability.REGION_PRICE_KWH[reg]:.3f} | {sustainability.REGION_CARBON[reg]} | "
                 f"${sustainability.energy_cost_usd(wh, reg):,.2f} | {sustainability.carbon_g(wh, reg) / 1000:,.1f} |")
    L += ["", "- **europe-north1** là lựa chọn tốt nhất khi cân cả carbon lẫn chi phí: sạch nhất (thủy điện Na Uy, 30 gCO2/kWh) "
          "và giá điện ở mức trung bình (0.09 $/kWh, rẻ hơn us-east-1). Nếu chỉ xét giá điện thì us-east-wa rẻ nhất; us-west-2 / us-east-wa là lựa chọn cân bằng "
          "ở Mỹ (rẻ hơn us-east-1 và ít carbon hơn >65%).",
          "- Carbon và tiền đi cùng chiều: rời us-east-1 / europe-central2 giảm cả hai. Tiền điện nhỏ so với tiền thuê GPU, "
          "nên lý do chọn vùng chủ yếu là carbon cộng với giá GPU tại vùng đó.",
          "- Đánh đổi: europe-north1 xa user Mỹ (+~100 ms RTT) → dùng cho training / batch chịu được gián đoạn; "
          "inference real-time giữ ở vùng gần user nhưng sạch (us-west-2).", ""]
    return L


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        LEVER_INFER: round(infer_savings),
        LEVER_BUY: round(purchasing_savings),
        LEVER_RIGHTSIZE: round(rightsize_savings),
        LEVER_IDLE: round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
        "cheapest_region": min(sustainability.REGION_PRICE_KWH, key=sustainability.REGION_PRICE_KWH.get),
    }

    analysis = build_analysis(r1, r2, levers, cat) + region_table(r2["reasoning"]["total_wh_daily"])
    md = report.build_report(baseline, optimized, levers, sustainability=sust,
                             period="hàng tháng", analysis=analysis)
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"))

    if verbose:
        if hasattr(_sys.stdout, "reconfigure"):
            _sys.stdout.reconfigure(encoding="utf-8")  # Windows console (cp1252) can't print Vietnamese
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
