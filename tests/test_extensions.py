import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing
from missions import m2_inference_levers


def test_cache_break_even_and_gate():
    be = pricing.cache_break_even_reads(0.25, 1.0, 0.10)
    assert abs(be - 0.25 / 0.9) < 1e-9
    assert pricing.cache_is_worth_it(0.3, 0.25, 1.0)
    assert not pricing.cache_is_worth_it(0.2, 0.25, 1.0)
    assert abs(pricing.cache_break_even_reads(1.0, 1.0, 0.10) - 1 / 0.9) < 1e-9  # 1h TTL: 2x write
    # break-even is tier-independent when the write premium is proportional to price
    assert abs(pricing.cache_break_even_reads(0.05, 0.20) - pricing.cache_break_even_reads(0.75, 3.00)) < 1e-9


def test_reasoning_budget():
    rs = m2_inference_levers.run(verbose=False)["reasoning"]
    assert rs["cost_pct"] > rs["traffic_pct"] and rs["wh_pct"] > rs["cost_pct"]
    saved = [c["usd_saved_daily"] for c in rs["caps"]]
    assert all(s >= 0 for s in saved) and saved == sorted(saved)  # tighter cap saves more
    thresholds = [c["min_input_tokens"] for c in rs["caps"] if c["min_input_tokens"]]
    assert thresholds == sorted(thresholds)  # tighter cap -> higher complexity bar X


def test_cache_scenarios_in_m2():
    cache = m2_inference_levers.run(verbose=False)["cache"]
    for c in cache.values():
        assert c["avg_reads_cacheable"] > c["avg_reads"]  # est B (cacheable prefix) >= est A (all input)
        assert c["break_even_1h"] > c["break_even_5m"]
        assert c["worth_it"] == c["worth_5m_a"]  # M2 gate uses the conservative estimate
