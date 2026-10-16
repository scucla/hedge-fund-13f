"""Milestone 4: fund_fingerprint (point-in-time) and style_predictions."""

from datetime import date

import duckdb
import polars as pl
import pytest

from src.features import fingerprint
from src.features.fund_quarter import FF12
from tests.conftest import table_exists


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


# --- point-in-time rule, by hand --------------------------------------------------------

def test_characteristics_known_after_the_filing_are_not_used():
    """Two stocks, half the book each. B's fundamentals were published after
    the fund filed, so B must count as missing: the size tilt is A's alone and
    coverage is 0.5. Sector weights don't depend on characteristics being known
    (SIC is static) except through the same join, so B falls into sector_none."""
    filed = date(2026, 8, 14)
    con = duckdb.connect()
    con.register("pos", pl.DataFrame({
        "fund_id": ["f", "f"], "quarter": ["2026Q2"] * 2, "known_at": [filed] * 2,
        "symbol": ["A", "B"], "cusip": ["A0", "B0"], "weight_long": [0.5, 0.5],
        "age_q": [3, 20], "is_option": [False, False]}))
    con.register("sq", pl.DataFrame({
        "cusip": ["A0", "B0"], "quarter": ["2026Q2"] * 2, "known_at": [date(2026, 6, 30)] * 2,
        "fundamentals_known_at": [date(2026, 5, 1), date(2026, 9, 1)]}))   # B: after the filing
    con.register("z", pl.DataFrame({
        "symbol": ["A", "B"], "quarter": ["2026Q2"] * 2, "ff12_code": ["BusEq", "Hlth"],
        **{f"z_{n}": [1.0, -3.0] for n in fingerprint.TILTS}}))
    con.register("fq", pl.DataFrame({
        "fund_id": ["f"], "quarter": ["2026Q2"], "fund_name": ["F"], "known_at": [filed],
        "n_positions": [2], "top10_weight": [1.0], "hhi": [0.5], "active_share": [0.9],
        "turnover": [0.1], "gross_turnover": [0.2]}))
    for name, src in [("positions", "pos"), ("stock_quarter", "sq"), ("stock_zscores", "z"), ("fund_quarter", "fq")]:
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {src}")
    fingerprint.build(con)
    r = con.execute("SELECT * FROM fund_fingerprint").pl().row(0, named=True)
    assert r["tilt_size"] == pytest.approx(1.0)            # B's -3 is not known yet
    assert r["cov_size"] == pytest.approx(0.5)
    assert r["sector_BusEq"] == pytest.approx(0.5) and r["sector_none"] == pytest.approx(0.5)
    assert r["holding_period_q"] == pytest.approx(0.5 * 3 + 0.5 * fingerprint.AGE_CAP)   # age 20 capped
    assert r["long_held_share"] == pytest.approx(0.5)
    assert r["max_input_known_at"] <= filed


# --- the real table ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def fp(sec_db):
    if not table_exists(sec_db, "fund_fingerprint"):
        pytest.skip("fund_fingerprint not built; run src.features.fingerprint")
    return sec_db


@pytest.mark.data
def test_fingerprint_unique_keys_and_no_nulls(fp):
    assert scalar(fp, "SELECT count(*) - count(DISTINCT (fund_id, quarter)) FROM fund_fingerprint") == 0
    assert scalar(fp, "SELECT count(*) FROM fund_fingerprint WHERE fund_id IS NULL OR known_at IS NULL "
                      "OR hhi IS NULL OR top10_weight IS NULL OR holding_period_q IS NULL") == 0


@pytest.mark.data
def test_fingerprint_uses_nothing_after_the_filing_date(fp):
    assert scalar(fp, "SELECT count(*) FROM fund_fingerprint WHERE max_input_known_at > known_at") == 0


@pytest.mark.data
def test_fingerprint_comes_from_the_original_filings(fp):
    # each fund-quarter's known_at is the latest *original* filing among its included CIKs
    assert scalar(fp, """
        SELECT count(*) FROM fund_fingerprint f
        JOIN (SELECT d.fund_id, d.quarter, max(s.known_at) AS k
              FROM fund_book_decisions d JOIN seed_positions_pit s
                ON s.fund_name = d.fund_name AND s.quarter = d.quarter AND s.cik = d.cik
              WHERE d.included GROUP BY ALL) p USING (fund_id, quarter)
        WHERE f.known_at <> p.k""") == 0


@pytest.mark.data
def test_fingerprint_sector_weights_sum_to_one(fp):
    total = " + ".join(f"sector_{s}" for s in FF12) + " + sector_none"
    assert scalar(fp, f"SELECT count(*) FROM fund_fingerprint WHERE abs({total} - 1) > 1e-6") == 0


@pytest.mark.data
def test_fingerprint_ranges(fp):
    assert scalar(fp, """
        SELECT count(*) FROM fund_fingerprint
        WHERE hhi NOT BETWEEN 0 AND 1 OR top10_weight NOT BETWEEN 0 AND 1 + 1e-9
           OR active_share NOT BETWEEN 0 AND 1 OR long_held_share NOT BETWEEN 0 AND 1 + 1e-9
           OR holding_period_q NOT BETWEEN 1 AND 12 OR n_positions < 1
           OR cov_size NOT BETWEEN 0 AND 1 + 1e-9 OR abs(tilt_size) > 10""") == 0


# --- style_predictions ------------------------------------------------------------------

@pytest.fixture(scope="module")
def preds(sec_db):
    if not table_exists(sec_db, "style_predictions"):
        pytest.skip("style_predictions not built; run src.models.style_classifier")
    return sec_db


@pytest.mark.data
def test_predictions_unique_and_labelled(preds):
    assert scalar(preds, "SELECT count(*) - count(DISTINCT (fund_id, quarter)) FROM style_predictions") == 0
    assert scalar(preds, "SELECT count(*) FROM style_predictions WHERE style_guess IS NULL OR predicted IS NULL") == 0


@pytest.mark.data
def test_predictions_split_by_time_only(preds):
    from src.models.style_classifier import TEST_FROM, TRAIN
    assert scalar(preds, f"""
        SELECT count(*) FROM style_predictions
        WHERE (split = 'train' AND quarter NOT BETWEEN '{TRAIN[0]}' AND '{TRAIN[1]}')
           OR (split = 'test' AND quarter < '{TEST_FROM}')
           OR split NOT IN ('train', 'test')""") == 0


@pytest.mark.data
def test_prediction_probabilities_sum_to_one(preds):
    cols = [c for c in preds.execute("SELECT * FROM style_predictions LIMIT 0").pl().columns if c.startswith("p_")]
    assert len(cols) >= 2
    assert scalar(preds, f"SELECT count(*) FROM style_predictions WHERE abs({' + '.join(cols)} - 1) > 1e-6") == 0
