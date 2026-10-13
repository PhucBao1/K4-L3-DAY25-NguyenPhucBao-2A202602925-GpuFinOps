"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}
# Cache-write premium over base input price, by TTL. Source: Anthropic prompt-caching pricing
# (5-minute write = 1.25x input, 1-hour write = 2x, read = 0.1x). Provider-specific: OpenAI
# charges no write premium, Gemini bills cache storage per hour instead.
CACHE_WRITE_PREMIUM = {"5m": 0.25, "1h": 1.0}
# Share of input that is a cacheable static prefix, per team (mirrors data/generate.py:107).
CACHEABLE_FRAC = {"assistant": 0.7, "rag": 0.7}
CACHEABLE_FRAC_DEFAULT = 0.1
REASONING_OUTPUT_TAX = 6     # reasoning inflates output ~6x (data/generate.py)
REASONING_CAPS = (0.10, 0.05, 0.0)  # what-if: max share of traffic allowed to use reasoning


def _reads_per_write(hit: float) -> float:
    # ponytail: no prefix-id in the data, so reads/write ~ h/(1-h) (a prefix written once and
    # read k times gives h = k/(k+1)); replace with real per-prefix read counts from the gateway.
    return hit / (1 - hit) if hit < 1 else float("inf")


def cache_economics(rows: list[dict]) -> dict:
    """Per route tier: two hit-rate estimates, reads/write, break-even per TTL, worth-it gates.

    est A: hit = cached / ALL input  (includes uncacheable user text -> conservative, low)
    est B: hit = cached / CACHEABLE input (static-prefix share only -> closer to real reuse)
    The M2 gate uses the conservative pair (est A, 5-minute TTL).
    """
    out = {}
    for tier, (pin, _) in MODEL_PRICES.items():
        tr = [r for r in rows if r["route_tier"] == tier]
        inp = sum(num(r["input_tokens"]) for r in tr)
        cached = sum(num(r["cached_input_tokens"]) for r in tr)
        cacheable = sum(num(r["input_tokens"]) * CACHEABLE_FRAC.get(r["team"], CACHEABLE_FRAC_DEFAULT) for r in tr)
        hit_a = cached / inp if inp else 0.0
        hit_b = cached / cacheable if cacheable else 0.0
        c = {"hit_rate": round(hit_a, 3), "avg_reads": round(_reads_per_write(hit_a), 2),
             "hit_rate_cacheable": round(hit_b, 3), "avg_reads_cacheable": round(_reads_per_write(hit_b), 2)}
        for ttl, prem in CACHE_WRITE_PREMIUM.items():
            write_cost = pin * prem
            c[f"break_even_{ttl}"] = round(pricing.cache_break_even_reads(write_cost, pin), 3)
            c[f"worth_{ttl}_a"] = pricing.cache_is_worth_it(c["avg_reads"], write_cost, pin)
            c[f"worth_{ttl}_b"] = pricing.cache_is_worth_it(c["avg_reads_cacheable"], write_cost, pin)
        c["worth_it"] = c["worth_5m_a"]
        out[tier] = c
    return out


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    cache = cache_economics(rows)
    base_cost = opt_cost = 0.0
    total_tokens = 0
    reason = {0: {"n": 0, "usd": 0.0, "wh": 0.0}, 1: {"n": 0, "usd": 0.0, "wh": 0.0}}
    downgrade = []  # (input_tokens, cost, cost_if_plain, wh, wh_if_plain) per reasoning request
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = int(num(r["is_reasoning"]))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching (only where it pays), batch API
        tier = r["route_tier"]
        pin, pout = MODEL_PRICES[tier]
        cached_in = cached if cache[tier]["worth_it"] else 0
        cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached_in, batch=is_batch)
        opt_cost += cost
        # reasoning budget: $ and Wh split
        wh = sustainability.wh_per_query(inp + out, is_reasoning=bool(is_reasoning))
        b = reason[is_reasoning]
        b["n"] += 1
        b["usd"] += cost
        b["wh"] += wh
        if is_reasoning:
            plain_out = out // REASONING_OUTPUT_TAX
            downgrade.append((inp, cost,
                              pricing.request_cost(inp, plain_out, pin, pout, cached_in=cached_in, batch=is_batch),
                              wh, sustainability.wh_per_query(inp + plain_out)))

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # cap reasoning at a share of traffic. Routing rule: keep reasoning only for the most complex
    # requests, downgrade the simplest first.
    # ponytail: no complexity label in the data, so input_tokens is the complexity proxy; swap in a
    # router's complexity score when one exists.
    ranked = sorted(downgrade)  # simplest (fewest input tokens) first
    caps = []
    for cap in REASONING_CAPS:
        excess = max(0, reason[1]["n"] - int(cap * len(rows)))
        cut, kept = ranked[:excess], ranked[excess:]
        caps.append({"cap_pct": round(cap * 100), "downgraded": excess,
                     "min_input_tokens": kept[0][0] if kept else None,  # threshold X of the routing rule
                     "usd_saved_daily": round(sum(c - cp for _, c, cp, _, _ in cut), 2),
                     "wh_saved_daily": round(sum(w - wp for _, _, _, w, wp in cut), 1)})
    total_wh = reason[0]["wh"] + reason[1]["wh"]
    reasoning = {
        "traffic_pct": round(reason[1]["n"] / len(rows) * 100, 1),
        "cost_pct": round(reason[1]["usd"] / opt_cost * 100, 1) if opt_cost else 0.0,
        "wh_pct": round(reason[1]["wh"] / total_wh * 100, 1) if total_wh else 0.0,
        "usd_daily": round(reason[1]["usd"], 2), "wh_daily": round(reason[1]["wh"], 1),
        "total_wh_daily": round(total_wh, 1),
        "caps": caps,
    }

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print("\n-- Cache economics (read at 10%; write +25% for 5m TTL, +100% for 1h TTL) --")
        print("est A = hit over all input, est B = hit over cacheable prefix only")
        print(f"{'tier':7}{'hitA':>7}{'readsA':>8}{'hitB':>7}{'readsB':>8}{'BE 5m':>7}{'BE 1h':>7}  5m A/B       1h A/B")
        for t, c in cache.items():
            print(f"{t:7}{c['hit_rate']:>7.1%}{c['avg_reads']:>8.2f}{c['hit_rate_cacheable']:>7.1%}"
                  f"{c['avg_reads_cacheable']:>8.2f}{c['break_even_5m']:>7.3f}{c['break_even_1h']:>7.3f}"
                  f"  {c['worth_5m_a']!s:>5}/{c['worth_5m_b']!s:<6} {c['worth_1h_a']!s:>5}/{c['worth_1h_b']!s:<5}")
        rs = reasoning
        print("\n-- Reasoning budget --")
        print(f"reasoning = {rs['traffic_pct']}% of traffic, {rs['cost_pct']}% of $, {rs['wh_pct']}% of Wh")
        print(f"reasoning: ${rs['usd_daily']:,.2f}/day, {rs['wh_daily']:,.0f} Wh/day (of {rs['total_wh_daily']:,.0f} Wh)")
        for c in rs["caps"]:
            print(f"cap at {c['cap_pct']:>3}%: reasoning only if input >= {c['min_input_tokens'] or '-'} tok, "
                  f"downgrade {c['downgraded']:>4} -> save ${c['usd_saved_daily']:,.2f}/day, "
                  f"{c['wh_saved_daily']:,.0f} Wh/day")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache": cache, "reasoning": reasoning,
    }


if __name__ == "__main__":
    run()
