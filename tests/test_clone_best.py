"""Clone-the-best-funds strategy (src/backtest/clone_best.py): point-in-time selection, slots, simulation, audit."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.backtest import clone_best as cb, engine

D = date.fromisoformat


@pytest.fixture
def two_quarter_window(monkeypatch):
    monkeypatch.setattr(cb, "TRAIL_Q", 2)
    monkeypatch.setattr(cb, "N_FUNDS", 1)


def _data():
    """Funds A and B, books at 2025Q1 and 2025Q2 (returns over Q2 and Q3), selection at 2025Q4's end
    uses quarters k = Q2, Q3. A's Q3 return (+50%) is only known in 2026 (a late 13F-implied price)."""
    books = pl.DataFrame({"fund_id": ["A", "A", "B", "B"], "quarter": ["2025Q1", "2025Q2"] * 2,
                          "known_at": [D("2025-05-15"), D("2025-08-14")] * 2,
                          "cusip": ["a1", "a2", "b1", "b2"], "weight_long": [1.0] * 4})
    books = books.with_columns(pl.col("quarter").map_elements(cb.q_idx, return_dtype=pl.Int64).alias("qi"))
    rets = pl.DataFrame({"cusip": ["a1", "a2", "b1", "b2"], "quarter": ["2025Q1", "2025Q2"] * 2,
                         "ret_next_q": [0.0, 0.5, 0.05, 0.05],
                         "label_known_at": [D("2025-06-30"), D("2026-02-14"), D("2025-06-30"), D("2025-09-30")]})
    return books, rets


def test_trailing_return_uses_only_known_data(two_quarter_window):
    books, rets = _data()
    t = cb.trailing_returns(books, rets, "2025Q4", D("2025-12-31"))
    by = dict(zip(t["fund_id"], t["trailing"]))
    assert by["B"] == pytest.approx(1.05 ** 2 - 1)
    # A's Q3 return isn't known yet: its Q2 book has no known return (weight 0 known) -> fails the coverage rule
    a = t.filter(pl.col("fund_id") == "A").row(0, named=True)
    assert a["min_known"] == 0.0
    assert cb.select(books, rets, "2025Q4")["fund_id"].to_list() == ["B"]


def test_late_book_is_ignored(two_quarter_window):
    books, rets = _data()
    late = books.with_columns(pl.when(pl.col("fund_id") == "B").then(D("2026-03-01")).otherwise(pl.col("known_at")).alias("known_at"))
    assert cb.select(late, rets, "2025Q4").is_empty()   # neither fund has 2 known books with known returns


def test_audit_poison_test_catches_a_leak(two_quarter_window, monkeypatch):
    books, rets = _data()
    leaky = {"2025Q4": pl.DataFrame({"fund_id": ["A"]})}     # what a ranking that saw A's +50% would choose
    res = {"selections": leaky, "books": books, "rets": rets, "switches": [], "filings": pl.DataFrame(),
           "rd": type("R", (), {"delist": pl.DataFrame({"cusip": [], "delisted_on": []})})()}
    aud = cb.audit(res)
    assert aud.filter(pl.col("check").str.starts_with("selection ignores"))["violations"][0] == 1


def test_slots_keep_continuing_funds():
    sel = {"2025Q1": pl.DataFrame({"fund_id": [f"f{i}" for i in range(10)]}),
           "2025Q2": pl.DataFrame({"fund_id": ["f3", "new1", "f0", "new2"] + [f"f{i}" for i in (1, 2, 4, 5, 6, 7)]})}
    slots = cb.assign_slots(sel)
    for f in ("f0", "f3", "f7"):
        assert slots["2025Q2"][f] == slots["2025Q1"][f]
    assert {slots["2025Q2"]["new1"], slots["2025Q2"]["new2"]} == {8, 9}   # the slots f8 and f9 freed
    assert sorted(slots["2025Q2"].values()) == list(range(10))


def _rd(days):
    prices = pl.DataFrame({"symbol": ["X"] * len(days), "date": days, "r": [0.01] * len(days)})
    counts = prices.group_by("symbol", engine.quarter_of(pl.col("date")).alias("Q")).agg(pl.len().alias("n_days"))
    q_days = pl.DataFrame({"date": days}).group_by(engine.quarter_of(pl.col("date")).alias("Q")).agg(pl.len().alias("n_trading"))
    filled = pl.DataFrame(schema={"cusip": pl.Utf8, "R": pl.Float64, "ret_source": pl.Utf8, "delist_date": pl.Date, "Q": pl.Utf8})
    return engine.ReturnData(days, prices, counts, q_days, filled, pl.DataFrame({"cusip": ["x"], "symbol": ["X"]}),
                             pl.DataFrame(schema={"cusip": pl.Utf8, "delisted_on": pl.Date}))


def test_simulate_cash_at_risk_free_then_switch_with_cost():
    days = [D("2025-06-02") + timedelta(days=i) for i in range(10) if (D("2025-06-02") + timedelta(days=i)).weekday() < 5]
    rd = _rd(days)
    rf = {d: 0.001 for d in days}
    # only slot 0 invests: in X from days[2]; the other 9 slots stay in cash
    sw = [cb.Switch(0, "f", "2025Q1", days[1], days[2], {"x": 1.0})]
    d, tr = cb.simulate(sw, rd, rf, days[0] - timedelta(days=1), days[-1], cost_bps=10)
    r = d["r"].to_numpy()
    assert r[0] == pytest.approx(0.001)                                   # day 1: all cash at the risk-free rate
    # day 3: every slot earns the risk-free rate, then slot 0 (a tenth of the book) pays 10 bp at the close
    assert r[2] == pytest.approx(0.001 - 0.1 * 1.001 * 0.001, rel=1e-9)
    # day 4: slot 0 (value 0.1 * 1.001 * 0.999 relative to the others' 0.1 * 1.001) earns X's +1%
    s0, other = 0.999, 1.0
    assert r[3] == pytest.approx((s0 * 1.01 + 9 * other * 1.001) / (s0 + 9 * other) - 1, rel=1e-9)
    assert tr["traded"][0] == pytest.approx(1.0)
