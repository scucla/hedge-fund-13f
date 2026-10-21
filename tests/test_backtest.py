"""Milestone 7 (Task D): calendar, return stitching, simulation, selection rules."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.backtest import engine, strategies as st
from src.backtest.run import max_drawdown

D = date.fromisoformat


def weekdays(start: str, end: str) -> list[date]:
    d, out = D(start), []
    while d <= D(end):
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# --- calendar ------------------------------------------------------------------------------

def test_calendar_trades_the_day_after_the_deadline_and_rolls_weekends():
    cal = engine.calendar(weekdays("2021-07-01", "2021-12-31"), ["2021Q2", "2021Q3"])
    q2 = cal.row(0, named=True)
    # 2021-06-30 + 45 days = Saturday 2021-08-14: filings due Monday the 16th, trade Tuesday the 17th
    assert q2["cutoff"] == D("2021-08-16") and q2["entry"] == D("2021-08-17")
    q3 = cal.row(1, named=True)
    assert q3["cutoff"] == D("2021-11-15") and q3["entry"] == D("2021-11-16")
    assert q2["exit"] == q3["entry"]
    assert (cal["entry"] > cal["cutoff"]).all()


def test_visible_uses_only_filings_public_by_the_cutoff():
    books = pl.DataFrame({"fund_id": ["a", "b"], "quarter": ["2021Q2"] * 2,
                          "known_at": [D("2021-08-16"), D("2021-08-17")], "issuer": ["x", "y"]})
    assert st.visible(books, "2021Q2", D("2021-08-16"))["fund_id"].to_list() == ["a"]


# --- returns -------------------------------------------------------------------------------

def _rd(days, prices=None, filled=None, symbols=None, delist=None) -> engine.ReturnData:
    prices = prices if prices is not None else pl.DataFrame(schema={"symbol": pl.Utf8, "date": pl.Date, "r": pl.Float64})
    counts = prices.group_by("symbol", engine.quarter_of(pl.col("date")).alias("Q")).agg(pl.len().alias("n_days"))
    q_days = pl.DataFrame({"date": days}).group_by(engine.quarter_of(pl.col("date")).alias("Q")).agg(pl.len().alias("n_trading"))
    filled = filled if filled is not None else pl.DataFrame(
        schema={"cusip": pl.Utf8, "R": pl.Float64, "ret_source": pl.Utf8, "delist_date": pl.Date, "Q": pl.Utf8})
    symbols = symbols if symbols is not None else pl.DataFrame(schema={"cusip": pl.Utf8, "symbol": pl.Utf8})
    delist = delist if delist is not None else pl.DataFrame(schema={"cusip": pl.Utf8, "delisted_on": pl.Date})
    return engine.ReturnData(days, prices, counts, q_days, filled, symbols, delist)


def test_filled_return_is_spread_over_the_quarter_and_stops_at_delisting():
    days = weekdays("2022-04-01", "2022-06-30")
    filled = pl.DataFrame({"cusip": ["A", "B"], "R": [0.21, 0.10], "ret_source": ["13f_implied", "delist_cash"],
                           "delist_date": [None, D("2022-05-13")], "Q": ["2022Q2"] * 2})
    rd = _rd(days, filled=filled)
    mat, src = engine.window_returns(rd, ["A", "B"], D("2022-03-31"), days[-1])
    assert np.prod(1 + mat[:, 0]) == pytest.approx(1.21)
    assert np.prod(1 + mat[:, 1]) == pytest.approx(1.10)
    after = np.array(days) > D("2022-05-13")
    assert (mat[after, 1] == 0).all() and (mat[~after, 1] > 0).all()
    assert set(src["source"]) == {"filled"}


def test_missing_return_is_zero_or_the_sold_out_sensitivity():
    days = weekdays("2022-04-01", "2022-06-30")
    filled = pl.DataFrame({"cusip": ["S", "N"], "R": [None, None], "ret_source": ["missing:sold_out", "missing:no_cik"],
                           "delist_date": [None, None], "Q": ["2022Q2"] * 2}, schema_overrides={"R": pl.Float64, "delist_date": pl.Date})
    rd = _rd(days, filled=filled)
    mat, src = engine.window_returns(rd, ["S", "N"], D("2022-03-31"), days[-1])
    assert (mat == 0).all() and set(src["source"]) == {"missing"}
    mat, _ = engine.window_returns(rd, ["S", "N"], D("2022-03-31"), days[-1], missing_sold_out=-0.3)
    assert np.prod(1 + mat[:, 0]) == pytest.approx(0.7) and (mat[:, 1] == 0).all()


def test_quarter_after_a_delisting_is_cash_not_missing():
    days = weekdays("2022-04-01", "2022-09-30")
    filled = pl.DataFrame({"cusip": ["A"], "R": [0.10], "ret_source": ["delist_cash"],
                           "delist_date": [D("2022-05-13")], "Q": ["2022Q2"]})
    rd = _rd(days, filled=filled, delist=pl.DataFrame({"cusip": ["A"], "delisted_on": [D("2022-05-13")]}))
    mat, src = engine.window_returns(rd, ["A"], D("2022-03-31"), days[-1], missing_sold_out=-0.3)
    assert np.prod(1 + mat[:, 0]) == pytest.approx(1.10)   # nothing in Q3, not even the sensitivity
    assert src["source"].to_list() == ["filled"]


def test_daily_prices_win_over_filled_returns():
    days = weekdays("2022-04-01", "2022-06-30")
    prices = pl.DataFrame({"symbol": ["P"] * len(days), "date": days, "r": [0.001] * len(days)})
    filled = pl.DataFrame({"cusip": ["A"], "R": [0.5], "ret_source": ["yahoo"], "delist_date": [None], "Q": ["2022Q2"]},
                          schema_overrides={"delist_date": pl.Date})
    rd = _rd(days, prices, filled, pl.DataFrame({"cusip": ["A"], "symbol": ["P"]}))
    mat, src = engine.window_returns(rd, ["A"], D("2022-03-31"), days[-1])
    assert np.allclose(mat[:, 0], 0.001) and src["source"].to_list() == ["daily"]


# --- simulation ----------------------------------------------------------------------------

def test_simulate_drift_costs_and_turnover():
    days = weekdays("2022-04-01", "2022-04-08")   # 6 trading days
    # A doubles on the first day held (the day after entry), B flat
    prices = pl.DataFrame({"symbol": ["A"] * 6 + ["B"] * 6, "date": days * 2,
                           "r": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 6})
    rd = _rd(days, prices, symbols=pl.DataFrame({"cusip": ["A", "B"], "symbol": ["A", "B"]}))
    rd.counts = rd.counts.with_columns(pl.lit(100).alias("n_days"))   # treat the week as fully priced
    windows = [{"quarter": "q1", "entry": days[0], "exit": days[3]}, {"quarter": "q2", "entry": days[3], "exit": days[5]}]
    weights = {"q1": {"A": 0.5, "B": 0.5}, "q2": {"B": 1.0}}
    daily, reb = engine.simulate(windows, weights, rd, cost_bps=10)
    # q1: bought from cash (traded 1.0); A's day-1 doubling is on days[1]
    assert reb["traded"].to_list()[0] == pytest.approx(1.0)
    assert daily["r_gross"][0] == pytest.approx(0.5)
    assert daily["r"][0] == pytest.approx(1.5 * (1 - 0.001) - 1)
    # drifted to A 2/3, B 1/3; moving to all-B trades |0-2/3| + |1-1/3| = 4/3
    assert reb["traded"].to_list()[1] == pytest.approx(4 / 3)
    assert len(daily) == 5


def test_max_drawdown():
    assert max_drawdown(np.array([0.1, -0.5, 0.2, 0.0])) == pytest.approx(-0.5)


# --- selection -----------------------------------------------------------------------------

def test_consensus_ignores_funds_with_too_many_positions():
    rows = []
    for f, n_pos, names in (("a", 30, ["x", "y"]), ("b", 30, ["x"]), ("big", 5000, ["y", "z"]), ("big2", 5000, ["z"])):
        rows += [{"fund_id": f, "issuer": i, "weight_long": 0.1, "n_positions": n_pos} for i in names]
    ranked = st.consensus(pl.DataFrame(rows))
    assert ranked[0] == "x" and "z" not in ranked


def test_to_weights_equal_and_capped():
    rep = {str(i): f"c{i}" for i in range(30)}
    w = st.to_weights([str(i) for i in range(30)], rep)
    assert len(w) == st.N_NAMES and sum(w.values()) == pytest.approx(1.0)


def test_to_weights_skips_names_delisted_before_the_trade():
    w = st.to_weights(["a", "b", "c"], {"a": "A", "b": "B", "c": "C"}, gone={"A"})
    assert set(w) == {"B", "C"}


def test_c2_buys_counts_funds_in_top_lists():
    lists = pl.DataFrame({"fund_id": ["a", "a", "b", "b"], "cusip": ["X", "Y", "Y", "ETF"], "rank": [1, 2, 1, 2]})
    ranked = st.c2_buys(lists, {"X": "x", "Y": "y"})   # ETF has no stock issuer: dropped
    assert ranked == ["y", "x"]


# --- real data -----------------------------------------------------------------------------

@pytest.mark.data
def test_signal_tables_keys(sec_db):
    from tests.conftest import table_exists
    if not table_exists(sec_db, "backtest_c1_scores"):
        pytest.skip("run src.backtest.signals")
    dup, null_cusip, bad_p = sec_db.execute("""
        SELECT count(*) - count(DISTINCT (fund_id, quarter, cusip)), count(*) FILTER (WHERE cusip IS NULL),
               count(*) FILTER (WHERE abs(p_exit + p_trim + p_hold + p_add - 1) > 1e-6)
        FROM backtest_c1_scores""").fetchone()
    assert dup == 0 and null_cusip == 0 and bad_p == 0
    dup, null_cusip, bad_rank = sec_db.execute("""
        SELECT count(*) - count(DISTINCT (method, fund_id, quarter, cusip)), count(*) FILTER (WHERE cusip IS NULL),
               (SELECT count(*) FROM (SELECT method, fund_id, quarter, max(rank) AS m, count(*) AS n
                                      FROM backtest_c2_lists GROUP BY ALL) WHERE m <> n)
        FROM backtest_c2_lists""").fetchone()
    assert dup == 0 and null_cusip == 0 and bad_rank == 0


@pytest.mark.data
def test_rebalance_outputs():
    from pathlib import Path
    path = Path("data/processed/backtest_rebalances.parquet")
    if not path.exists():
        pytest.skip("run src.backtest.run")
    reb = pl.read_parquet(path).filter(pl.col("n_names") > 0)
    shares = reb["w_daily"] + reb["w_filled"] + reb["w_missing"]
    assert (shares - 1).abs().max() < 1e-9
    assert reb["traded"].max() <= 2 + 1e-9
    assert reb.group_by("strategy").agg(pl.col("entry").is_sorted()).get_column("entry").all()
