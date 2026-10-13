"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 analysis: list[str] | None = None) -> str:
    """Return a markdown cost-optimization report. `analysis` = extra markdown lines appended."""
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — Báo cáo tối ưu chi phí GPU",
        "",
        f"**Kỳ báo cáo:** {period}  ",
        f"**Chi phí baseline:** ${baseline_usd:,.0f}  ",
        f"**Chi phí sau tối ưu:** ${optimized_usd:,.0f}  ",
        f"**Tiết kiệm dự kiến (Projected savings):** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
        "## Tiết kiệm theo đòn bẩy",
        "",
        "| Đòn bẩy | Tiết kiệm (USD) |",
        "|---|---|",
    ]
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} |")
    if sustainability:
        lines += [
            "",
            "## Tính bền vững (Sustainability)",
            "",
            f"- Năng lượng mỗi truy vấn: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon mỗi truy vấn: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Vùng tốt nhất (Best region — sạch nhất, giá điện trung bình): {sustainability.get('best_region', 'n/a')}",
        ]
        if sustainability.get("cheapest_region"):
            lines.append(f"- Vùng điện rẻ nhất (Cheapest power): {sustainability['cheapest_region']}")
    if analysis:
        lines += [""] + analysis
    lines += ["", "_Số liệu là snapshot tháng 6/2026; cần cập nhật lại giá trước khi áp dụng._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str) -> str:
    """Write a simple savings bar chart PNG. Returns the path. No-op if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    names = list(levers.keys())
    vals = [levers[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, vals, color="#2e548a")
    ax.set_ylabel("Tiết kiệm (USD / tháng)")
    ax.set_title("Tiết kiệm chi phí GPU theo đòn bẩy FinOps")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
