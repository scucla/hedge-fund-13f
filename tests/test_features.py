"""Milestone 3 Phase A: fund books, positions and fund_quarter."""

from datetime import date

import duckdb
import polars as pl
import pytest

from src.features import fund_quarter
from src.features.books import decide
from src.features.positions import snap_ratio
from tests.conftest import table_exists


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


# --- units --------------------------------------------------------------------------

def test_duplicate_book_keeps_primary_and_separate_books_are_added():
    totals = pl.DataFrame({
        "fund_name": ["F"] * 3, "quarter": ["2026Q2"] * 3, "cik": ["1", "2", "3"],
        "role": ["predecessor", "primary", "affiliate"], "value_usd": [100.0, 90.0, 10.0],
    })
    overlaps = pl.DataFrame({"fund_name": ["F", "F"], "quarter": ["2026Q2"] * 2,
                             "cik_a": ["1", "2"], "cik_b": ["2", "3"], "overlap": [0.97, 0.10]})
    d = {r["cik"]: r for r in decide(totals, overlaps, set()).iter_rows(named=True)}
    assert d["2"]["included"] and d["3"]["included"]
    assert not d["1"]["included"] and "duplicate of 2" in d["1"]["reason"]


def test_exclusion_list():
    totals = pl.DataFrame({"fund_name": ["F", "F"], "quarter": ["2026Q2"] * 2, "cik": ["1", "2"],
                           "role": ["primary", "affiliate"], "value_usd": [100.0, 5.0]})
    empty = pl.DataFrame(schema={"fund_name": pl.Utf8, "quarter": pl.Utf8, "cik_a": pl.Utf8,
                                 "cik_b": pl.Utf8, "overlap": pl.Float64})
    d = {r["cik"]: r for r in decide(totals, empty, {("F", "2")}).iter_rows(named=True)}
    assert d["1"]["included"] and not d["2"]["included"]


@pytest.mark.parametrize("ratio, expected", [(10.0, 10.0), (25.063, 25.0), (1.502, 1.5), (0.2005, 0.2),
                                             (1.28, None), (6.515, None), (1.0, None)])  # 1.0 is not a split
def test_snap_ratio(ratio, expected):
    assert snap_ratio(ratio) == (pytest.approx(expected) if expected else None)


def test_turnover_and_active_share_by_hand():
    """One fund: Q1 holds A (100 sh, $1000) and B ($1000). Q2 holds A (150 sh, $1500)
    and C ($500); B was sold, flat return. Buys = 50 x $10 + $500 = $1000;
    sells = $1000; average book = $2000 -> turnover 0.5. Benchmark (by market
    cap) A 60%, B 20%, C 20%; fund A 75%, C 25% -> active share
    0.5 x (0.15 + 0.20 + 0.05) = 0.20."""
    con = duckdb.connect()

    def pos(q, cusip, sym, shares, value, total):
        return {"fund_id": "f", "fund_name": "F", "quarter": q, "cusip": cusip, "put_call": "", "symbol": sym,
                "security_key": sym, "security_class": "common", "issuer_cik": sym, "shares": shares,
                "principal": 0.0, "value_usd": value, "weight_long": value / total, "is_option": False,
                "split_ratio": 1.0, "known_at": date(2026, 8, 14) if q == "2026Q2" else date(2026, 5, 15)}

    positions = pl.DataFrame([
        pos("2026Q1", "A", "A", 100.0, 1000.0, 2000.0), pos("2026Q1", "B", "B", 50.0, 1000.0, 2000.0),
        pos("2026Q2", "A", "A", 150.0, 1500.0, 2000.0), pos("2026Q2", "C", "C", 25.0, 500.0, 2000.0),
    ])
    sq = pl.DataFrame([
        {"symbol": s, "quarter": q, "security_class": "common", "issuer_cik": s, "me": me, "log_me": 1.0 * me,
         "bm": 0.5, "mom_12_1": 0.1 * i, "gp_at": 0.2, "asset_growth": 0.1, "beta": 1.0, "ff12_code": "BusEq"}
        for q in ("2026Q1", "2026Q2") for i, (s, me) in enumerate([("A", 300.0), ("B", 100.0), ("C", 100.0)])
    ])
    rq = pl.DataFrame({"symbol": ["A", "B", "C"], "quarter": ["2026Q2"] * 3, "ret_q": [0.0, 0.0, 0.0]})
    for name, df in {"positions": positions, "stock_quarter": sq, "returns_quarterly": rq}.items():
        con.register(f"{name}_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_df")
    fund_quarter.build(con)
    row = con.execute("SELECT turnover, active_share, top10_weight, hhi FROM fund_quarter WHERE quarter = '2026Q2'").fetchone()
    assert row[0] == pytest.approx(0.5)
    assert row[1] == pytest.approx(0.2)
    assert row[2] == pytest.approx(1.0)
    assert row[3] == pytest.approx(0.75**2 + 0.25**2)
    assert scalar(con, "SELECT turnover FROM fund_quarter WHERE quarter = '2026Q1'") is None
    # gross: (1000 + 1000) / 2 / 2000
    assert scalar(con, "SELECT gross_turnover FROM fund_quarter WHERE quarter = '2026Q2'") == pytest.approx(0.5)


def test_gross_turnover_sees_one_sided_selling():
    """Q1: A $1000, B $1000. Q2: A unchanged ($1000), B sold. Buys 0, sells $1000,
    average book $1500 -> turnover 0 (min), gross turnover 1000 / 2 / 1500 = 1/3."""
    con = duckdb.connect()
    rows = []
    for q, held in [("2026Q1", ["A", "B"]), ("2026Q2", ["A"])]:
        for s in held:
            rows.append({"fund_id": "f", "fund_name": "F", "quarter": q, "cusip": s, "put_call": "", "symbol": s,
                         "security_key": s, "security_class": "common", "issuer_cik": s, "shares": 100.0,
                         "principal": 0.0, "value_usd": 1000.0, "weight_long": 1 / len(held), "is_option": False,
                         "split_ratio": 1.0, "known_at": date(2026, 8, 14)})
    sq = pl.DataFrame([{"symbol": s, "quarter": q, "security_class": "common", "issuer_cik": s, "me": 100.0,
                        "log_me": 1.0, "bm": 0.5, "mom_12_1": 0.1, "gp_at": 0.2, "asset_growth": 0.1, "beta": 1.0,
                        "ff12_code": "BusEq"} for q in ("2026Q1", "2026Q2") for s in ("A", "B")])
    rq = pl.DataFrame({"symbol": ["A", "B"], "quarter": ["2026Q2"] * 2, "ret_q": [0.0, 0.0]})
    for name, df in {"positions": pl.DataFrame(rows), "stock_quarter": sq, "returns_quarterly": rq}.items():
        con.register(f"{name}_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_df")
    fund_quarter.build(con)
    turnover, gross = con.execute(
        "SELECT turnover, gross_turnover FROM fund_quarter WHERE quarter = '2026Q2'").fetchone()
    assert turnover == pytest.approx(0.0)
    assert gross == pytest.approx(1 / 3)


# --- real tables --------------------------------------------------------------------

@pytest.fixture(scope="module")
def feat(sec_db):
    if not table_exists(sec_db, "fund_quarter"):
        pytest.skip("feature tables not built")
    return sec_db


@pytest.mark.data
def test_weights_sum_to_one(feat):
    assert scalar(feat, """SELECT count(*) FROM (SELECT fund_id, quarter, sum(weight) w FROM positions GROUP BY ALL)
                           WHERE abs(w - 1) > 1e-9""") == 0
    assert scalar(feat, """SELECT count(*) FROM (SELECT fund_id, quarter, sum(weight_long) w FROM positions
                                                 WHERE NOT is_option GROUP BY ALL)
                           WHERE abs(w - 1) > 1e-9""") == 0


@pytest.mark.data
def test_unique_keys(feat):
    assert scalar(feat, """SELECT count(*) FROM (SELECT fund_id, quarter, cusip, put_call FROM positions
                                                 GROUP BY ALL HAVING count(*) > 1)""") == 0
    assert scalar(feat, "SELECT count(*) - count(DISTINCT (fund_id, quarter)) FROM fund_quarter") == 0


@pytest.mark.data
def test_turnover_and_active_share_in_unit_interval(feat):
    assert scalar(feat, "SELECT count(*) FROM fund_quarter WHERE turnover NOT BETWEEN 0 AND 1") == 0
    assert scalar(feat, "SELECT count(*) FROM fund_quarter WHERE gross_turnover NOT BETWEEN 0 AND 1") == 0
    assert scalar(feat, "SELECT count(*) FROM fund_quarter WHERE gross_turnover < turnover") == 0  # mean >= min
    assert scalar(feat, "SELECT count(*) FROM fund_quarter WHERE active_share NOT BETWEEN 0 AND 1") == 0
    assert scalar(feat, "SELECT count(*) FROM fund_quarter WHERE quarter > '2025Q3' AND turnover IS NULL "
                        "AND fund_id IN (SELECT fund_id FROM fund_quarter WHERE quarter = '2025Q3')") <= 5


@pytest.mark.data
def test_stock_features_known_by_quarter_end(feat):
    # characteristics behind the tilts are as of the quarter end...
    assert scalar(feat, """
        SELECT count(*) FROM stock_quarter
        WHERE known_at <> last_day(make_date(left(quarter, 4)::INT, right(quarter, 1)::INT * 3, 1))
           OR fundamentals_known_at > known_at OR price_date > known_at""") == 0


@pytest.mark.data
def test_holdings_features_known_at_filing(feat):
    # ...and holdings-based features become known at the filing date, which is after the quarter end
    assert scalar(feat, """
        SELECT count(*) FROM fund_quarter
        WHERE known_at <= last_day(make_date(left(quarter, 4)::INT, right(quarter, 1)::INT * 3, 1))""") == 0
    assert scalar(feat, """
        SELECT count(*) FROM positions p JOIN fund_quarter f USING (fund_id, quarter)
        WHERE p.known_at <> f.known_at""") == 0


@pytest.mark.data
def test_position_fields(feat):
    assert scalar(feat, "SELECT count(*) FROM positions WHERE age_q < 1 OR crowding < 0 OR rank < 1") == 0
    assert scalar(feat, "SELECT count(*) FROM positions WHERE top10 <> (rank <= 10)") == 0
    assert scalar(feat, "SELECT count(*) FROM positions WHERE is_new AND share_change IS NOT NULL") == 0


@pytest.mark.data
def test_netflix_split_adjusted(feat):
    # Tiger Global: 201,900 shares before the 10-for-1 split, 2,439,000 after: +20.8%, not +1,108%
    chg = scalar(feat, """SELECT share_change FROM positions WHERE fund_id = 'tiger_global_management'
                          AND symbol = 'NFLX' AND put_call = '' AND quarter = '2025Q4'""")
    assert chg == pytest.approx(2_439_000 / 2_019_000 - 1, rel=1e-6)


@pytest.mark.data
def test_every_seed_fund_has_one_book(feat):
    assert scalar(feat, "SELECT count(DISTINCT fund_id) FROM fund_quarter") >= 75
    assert scalar(feat, """SELECT count(*) FROM fund_book_decisions
                           WHERE fund_name = 'Two Sigma Investments' AND cik = '0001450144' AND included""") == 0


def test_bond_principal_unit_change_is_not_a_trade():
    """A convertible reported as $1M principal one quarter and 1,000M the next (a
    filer changing units) at the same $1000 value: no buy, no sell."""
    con = duckdb.connect()
    rows = [{"fund_id": "f", "fund_name": "F", "quarter": q, "cusip": "BOND", "put_call": "", "symbol": "BOND",
             "security_key": "BOND", "security_class": "debt", "issuer_cik": None, "shares": 0.0,
             "principal": prn, "value_usd": 1000.0, "weight_long": 1.0, "is_option": False, "split_ratio": 1.0,
             "known_at": date(2026, 8, 14)}
            for q, prn in [("2026Q1", 1e6), ("2026Q2", 1e9)]]
    sq = pl.DataFrame([{"symbol": "X", "quarter": "2026Q2", "security_class": "common", "issuer_cik": "X", "me": 1.0,
                        "log_me": 1.0, "bm": 0.5, "mom_12_1": 0.1, "gp_at": 0.2, "asset_growth": 0.1, "beta": 1.0,
                        "ff12_code": "BusEq"}])
    rq = pl.DataFrame({"symbol": ["BOND"], "quarter": ["2026Q2"], "ret_q": [0.0]})
    positions = pl.DataFrame(rows, schema_overrides={"issuer_cik": pl.Utf8})  # all-NULL column needs a type
    for name, df in {"positions": positions, "stock_quarter": sq, "returns_quarterly": rq}.items():
        con.register(f"{name}_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_df")
    fund_quarter.build(con)
    assert con.execute("SELECT turnover, gross_turnover FROM fund_quarter WHERE quarter = '2026Q2'").fetchone() == (0.0, 0.0)
