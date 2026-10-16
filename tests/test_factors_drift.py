"""Milestone 4: returns-based factor regressions (13F clone) and style drift."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from src.models import factor_regressions as fr
from src.models import style_drift as sd
from tests.conftest import table_exists


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


# --- 13F clone returns, by hand -------------------------------------------------------

def test_holding_months_are_the_next_quarter():
    assert fr.holding_months("2025Q4") == [date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1)]
    assert fr.holding_months("2026Q1") == [date(2026, 4, 1), date(2026, 5, 1), date(2026, 6, 1)]


def test_clone_is_buy_and_hold_and_drops_stocks_once_unpriced():
    """Book 50/50 in A and B for 2026Q1 (held Apr-Jun). A: +10%, 0%, -10%.
    B: 0% in April, then no price (delisted). C is 0 weight-bearing but unpriced
    from the start, so coverage starts below 1."""
    book = pl.DataFrame({"fund_id": ["f"] * 3, "cohort": ["seed"] * 3, "quarter": ["2026Q1"] * 3,
                         "symbol": ["A", "B", "C"], "w": [0.4, 0.4, 0.2]})
    rets = pl.DataFrame({"symbol": ["A", "A", "A", "B"],
                         "month": [date(2026, 4, 1), date(2026, 5, 1), date(2026, 6, 1), date(2026, 4, 1)],
                         "ret": [0.10, 0.0, -0.10, 0.0]})
    out = fr.clone_returns(book, rets).sort("month")
    # April: A and B equal value -> (0.1 + 0) / 2; May and June: A alone
    assert out["ret"].to_list() == pytest.approx([0.05, 0.0, -0.10])
    assert out["coverage"].to_list() == pytest.approx([0.8, 0.4, 0.4])
    assert out["n_stocks"].to_list() == [2, 1, 1]


def test_clone_weights_drift_with_prices():
    """A doubles in month 1, so in month 2 it is 2/3 of the book."""
    book = pl.DataFrame({"fund_id": ["f"] * 2, "cohort": ["seed"] * 2, "quarter": ["2025Q4"] * 2,
                         "symbol": ["A", "B"], "w": [0.5, 0.5]})
    rets = pl.DataFrame({"symbol": ["A", "B", "A", "B"],
                         "month": [date(2026, 1, 1)] * 2 + [date(2026, 2, 1)] * 2,
                         "ret": [1.0, 0.0, 0.3, 0.0]})
    out = fr.clone_returns(book, rets).sort("month")
    assert out["ret"][1] == pytest.approx(0.3 * 2 / 3)


def test_newey_west_ols_recovers_known_loadings():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.04, size=(120, 6))
    true = np.array([1.1, 0.4, -0.3, 0.2, 0.0, 0.1])
    y = 0.002 + X @ true + rng.normal(0, 0.002, 120)
    beta, t, r2 = fr.ols_newey_west(y, X)
    assert beta[1:] == pytest.approx(true, abs=0.03)
    assert beta[0] == pytest.approx(0.002, abs=0.001)
    assert abs(t[1]) > 50 and r2 > 0.98


# --- style drift rule, by hand --------------------------------------------------------

def _fund(fid, styles, p):
    """One fund, one quarter per entry of `styles`, top class probability p (rest spread evenly)."""
    classes = sd.STYLE_ORDER
    rows = []
    for i, s in enumerate(styles):
        probs = {f"p_{c}": (p if c == s else (1 - p) / (len(classes) - 1)) for c in classes}
        rows.append({"fund_id": fid, "fund_name": fid, "cohort": "seed", "label": styles[0],
                     "quarter": f"{2018 + i // 4}Q{i % 4 + 1}", "predicted": s, **probs})
    return rows


def test_change_needs_two_confident_windows_and_enough_quarters():
    classes = sorted(sd.STYLE_ORDER)
    pred = pl.DataFrame(
        _fund("clear", ["value_concentrated"] * 10 + ["growth_tech"] * 10, 0.9)
        + _fund("unsure", ["value_concentrated"] * 10 + ["growth_tech"] * 10, 0.4)
        + _fund("short", ["value_concentrated"] * 6 + ["growth_tech"] * 6, 0.9)
        + _fund("flicker", ["quant", "multi_manager"] * 10, 0.9))
    f = {r["fund_id"]: r for r in sd.summarize(pred, classes).iter_rows(named=True)}
    assert f["clear"]["changed"] and (f["clear"]["early_style"], f["clear"]["late_style"]) == \
        ("value_concentrated", "growth_tech")
    assert f["clear"]["switches"] == 1 and f["clear"]["stability"] == pytest.approx(0.5)
    assert f["unsure"]["style_differs"] and not f["unsure"]["changed"]   # below CONFIDENCE
    assert not f["short"]["style_differs"]                               # < MIN_QUARTERS
    assert not f["flicker"]["changed"] and f["flicker"]["switches"] == 19


# --- real tables ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def built(sec_db):
    for t in ("fund_factor_loadings", "fund_clone_returns_monthly", "style_drift", "style_drift_funds"):
        if not table_exists(sec_db, t):
            pytest.skip(f"{t} not built")
    return sec_db


@pytest.mark.data
def test_clone_returns_keys_and_ranges(built):
    assert scalar(built, "SELECT count(*) - count(DISTINCT (fund_id, quarter, month)) FROM fund_clone_returns_monthly") == 0
    assert scalar(built, """SELECT count(*) FROM fund_clone_returns_monthly
                            WHERE coverage <= 0 OR coverage > 1 + 1e-9 OR ret IS NULL OR isnan(ret) OR ret <= -1""") == 0
    # each month lies in the quarter after the book's quarter (point in time: the book is known by then)
    assert scalar(built, """SELECT count(*) FROM fund_clone_returns_monthly WHERE
        year(month) * 4 + (month(month) - 1) // 3 <> year(make_date(substr(quarter,1,4)::INT, 1, 1)) * 4
                                                    + substr(quarter,6,1)::INT""") == 0


@pytest.mark.data
def test_factor_loadings_one_row_per_fund(built):
    assert scalar(built, "SELECT count(*) - count(DISTINCT fund_id) FROM fund_factor_loadings") == 0
    assert scalar(built, "SELECT min(n_months) FROM fund_factor_loadings") >= fr.MIN_MONTHS
    cols = " OR ".join(f"b_{f} IS NULL OR isnan(b_{f}) OR isnan(t_{f})" for f in fr.FACTORS)
    assert scalar(built, f"SELECT count(*) FROM fund_factor_loadings WHERE {cols}") == 0
    # long-only equity books: market beta positive and not absurd (biotech books go as low as ~0.1,
    # since SMB carries much of their co-movement)
    assert scalar(built, "SELECT count(*) FROM fund_factor_loadings WHERE b_Mkt_RF NOT BETWEEN 0 AND 2.5") == 0


@pytest.mark.data
def test_style_drift_covers_every_labelled_fund_quarter(built):
    assert scalar(built, "SELECT count(*) - count(DISTINCT (fund_id, quarter)) FROM style_drift") == 0
    probs = " + ".join(f"p_{s}" for s in sd.STYLE_ORDER)
    assert scalar(built, f"SELECT count(*) FROM style_drift WHERE abs({probs} - 1) > 1e-6") == 0
    assert scalar(built, """SELECT count(*) FROM style_predictions p ANTI JOIN style_drift d USING (fund_id, quarter)""") == 0
    assert scalar(built, "SELECT count(*) - count(DISTINCT fund_id) FROM style_drift_funds") == 0
    assert scalar(built, f"""SELECT count(*) FROM style_drift_funds WHERE changed AND (
        early_style = late_style OR quarters < {sd.MIN_QUARTERS} OR least(p_early, p_late) < {sd.CONFIDENCE})""") == 0
