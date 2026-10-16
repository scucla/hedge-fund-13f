"""Milestone 4: reviewed labels, holdout funds, the 13D heuristic and the
seed-only rules that keep the holdout out of seed features."""

import gzip
from datetime import date

import duckdb
import polars as pl
import pytest

from src.features import activist_13d
from src.features.books import fund_id
from tests.conftest import table_exists

STYLES = {"activist", "discretionary_other", "event_driven", "growth_tech", "healthcare",
          "multi_manager", "quant", "value_concentrated"}
SEED = pl.read_csv("data/reference/seed_funds.csv", schema_overrides={"cik": pl.Utf8})
SEED_MAP = pl.read_csv("data/reference/seed_cik_map.csv", schema_overrides={"cik": pl.Utf8})
V2 = pl.read_csv("data/reference/style_labels_v2.csv")
HOLDOUT = pl.read_csv("data/reference/holdout_funds.csv", schema_overrides={"cik": pl.Utf8})


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


# --- reference files ----------------------------------------------------------------

def test_v2_labels_cover_seed_funds_with_sources():
    seed_style = dict(SEED.select("fund_name", "style_guess").unique().iter_rows())
    assert V2["fund_name"].is_unique().all() and len(V2) == 13
    for r in V2.iter_rows(named=True):
        assert r["fund_name"] in seed_style, r["fund_name"]
        assert r["style_v1"] == seed_style[r["fund_name"]], r["fund_name"]   # v1 column = seed_funds.csv
        assert r["style_v2"] in STYLES and r["confidence"] in {"high", "medium", "low"}
        assert r["reason"] and r["source"].startswith("http")


def test_holdout_funds_are_disjoint_from_the_seed_list():
    assert len(HOLDOUT) >= 15 and HOLDOUT["cik"].is_unique().all() and HOLDOUT["fund_name"].is_unique().all()
    assert not set(HOLDOUT["cik"].str.zfill(10)) & set(SEED_MAP["cik"].str.zfill(10))
    assert not {fund_id(n) for n in HOLDOUT["fund_name"]} & {fund_id(n) for n in SEED["fund_name"]}
    assert set(HOLDOUT["style"]) == STYLES                       # every style represented
    assert HOLDOUT["source"].str.starts_with("http").all() and HOLDOUT["reason"].is_not_null().all()


# --- 13D heuristic, by hand ---------------------------------------------------------

def test_parse_13d_rows(tmp_path):
    path = tmp_path / "form.gz"
    with gzip.open(path, "wt", encoding="latin-1") as f:
        f.write("Form Type   Company Name      CIK   Date Filed  File Name\n")
        f.write("SC 13D/A        ACME ACTIVIST LP          1234  2024-03-01  edgar/data/1234/a.txt\n")
        f.write("SCHEDULE 13D    ACME ACTIVIST LP          1234  2025-02-01  edgar/data/1234/b.txt\n")
        f.write("SC 13G          PASSIVE FUND              999   2024-03-01  edgar/data/999/c.txt\n")
        f.write("13F-HR          ACME ACTIVIST LP          1234  2024-02-14  edgar/data/1234/d.txt\n")
    rows = activist_13d.parse_13d(path)
    assert [(r[0], r[2], r[3]) for r in rows] == [("SC 13D/A", "0000001234", "2024-03-01"),
                                                    ("SCHEDULE 13D", "0000001234", "2025-02-01")]


def test_13d_window_is_point_in_time_and_two_years():
    """Fund filed its 13F on 2024-08-14. Counted: 13Ds on or before that date
    and within 2 years. Not counted: one filed the next day, one 3 years old."""
    con = duckdb.connect()
    con.register("fp", pl.DataFrame({"fund_id": ["f"], "quarter": ["2024Q2"], "known_at": [date(2024, 8, 14)]}))
    con.register("ciks", pl.DataFrame({"fund_id": ["f"], "cik": ["0000000001"]}))
    con.register("d", pl.DataFrame({
        "cik": ["0000000001"] * 4, "form": ["SC 13D", "SC 13D/A", "SC 13D", "SC 13D"],
        "date_filed": [date(2024, 8, 14), date(2023, 1, 5), date(2024, 8, 15), date(2021, 6, 1)],
        "file_name": ["a", "b", "c", "d"], "is_original": [True, False, True, True]}))
    con.execute("CREATE TABLE fund_fingerprint AS SELECT * FROM fp")
    con.execute("CREATE TABLE fund_ciks AS SELECT * FROM ciks")
    con.execute("CREATE TABLE filings_13d AS SELECT * FROM d")
    con.execute(activist_13d.ACTIVIST_SQL)
    assert con.execute("SELECT n_13d_2y, n_13d_orig_2y, activist_13d FROM activist_13d").fetchone() == (2, 1, True)


# --- real tables ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def m4(sec_db):
    for t in ("activist_13d", "filings_13d", "fund_fingerprint"):
        if not table_exists(sec_db, t):
            pytest.skip(f"{t} not built")
    return sec_db


@pytest.mark.data
def test_activist_13d_one_row_per_fingerprint_row(m4):
    assert scalar(m4, "SELECT count(*) - count(DISTINCT (fund_id, quarter)) FROM activist_13d") == 0
    assert scalar(m4, "SELECT count(*) FROM fund_fingerprint ANTI JOIN activist_13d USING (fund_id, quarter)") == 0
    assert scalar(m4, "SELECT count(*) FROM activist_13d WHERE n_13d_orig_2y > n_13d_2y "
                      "OR activist_13d <> (n_13d_2y > 0)") == 0
    assert scalar(m4, "SELECT count(*) FROM filings_13d WHERE cik IS NULL OR date_filed IS NULL") == 0


@pytest.mark.data
def test_holdout_cohort_matches_the_reference_file(m4):
    ids = {fund_id(n) for n in HOLDOUT["fund_name"]}
    got = {r[0] for r in m4.execute("SELECT DISTINCT fund_id FROM fund_fingerprint WHERE cohort = 'holdout'").fetchall()}
    assert got == ids
    assert scalar(m4, "SELECT count(*) FROM fund_fingerprint WHERE cohort NOT IN ('seed', 'holdout')") == 0


@pytest.mark.data
def test_crowding_counts_seed_funds_only(m4):
    assert scalar(m4, """
        WITH n AS (SELECT quarter, security_key, count(DISTINCT fund_id) AS n FROM positions
                   WHERE NOT is_option AND cohort = 'seed' GROUP BY ALL)
        SELECT count(*) FROM positions p LEFT JOIN n USING (quarter, security_key)
        WHERE NOT p.is_option AND p.crowding <> coalesce(n.n, 0) - (p.cohort = 'seed')::INT""") == 0


@pytest.mark.data
def test_seed_cross_sections_exclude_holdout_only_securities(m4):
    # every row in the z-score cross-section and the benchmark is held by a seed fund
    assert scalar(m4, """
        SELECT count(*) FROM stock_zscores z WHERE z.in_seed AND NOT EXISTS (
            SELECT 1 FROM positions p WHERE p.cohort = 'seed' AND p.symbol = z.symbol AND p.quarter = z.quarter
            UNION ALL SELECT 1 FROM seed_positions s JOIN security_master m USING (cusip)
                      WHERE s.cohort = 'seed' AND m.symbol = z.symbol AND s.quarter = z.quarter)""") == 0
