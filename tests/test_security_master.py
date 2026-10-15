"""Universe and security master: parsing units plus checks on the built tables."""

import pytest

from src.securities.cusip_map import best_listing, classify, parse_13f_list, yahoo_symbol
from tests.conftest import table_exists


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


# --- units --------------------------------------------------------------------

def test_parse_13f_list(tmp_path):
    p = tmp_path / "13flist2026q2.txt"
    p.write_text(
        "594918104*MICROSOFT CORP                COM                                    E\n"
        "G0R21F105 APEX TECH ACQUISITION INC     ORD SHS                    *A*         E\n"
        "not a data line\n"
    )
    df = parse_13f_list(p)
    assert df.rows() == [
        ("594918104", True, "MICROSOFT CORP", "COM", None, "2026Q2"),
        ("G0R21F105", False, "APEX TECH ACQUISITION INC", "ORD SHS", "A", "2026Q2"),
    ]


def test_best_listing_prefers_us_composite():
    resp = {"data": [{"exchCode": "UA", "ticker": "X1"}, {"exchCode": "US", "ticker": "X2"},
                     {"exchCode": "LN", "ticker": "X3"}]}
    assert best_listing(resp)["ticker"] == "X2"
    assert best_listing({"warning": "No identifier found."}) is None


def test_classify_etf_before_mutual_fund():
    assert classify({"securityType": "ETP", "securityType2": "Mutual Fund", "marketSector": "Equity"}, None) == "etf"
    assert classify({"securityType": "Common Stock", "securityType2": "Common Stock"}, None) == "common"
    assert classify({"securityType": "US DOMESTIC", "marketSector": "Corp"}, None) == "debt"
    assert classify(None, "COM") == "unmapped"


def test_yahoo_symbol():
    assert yahoo_symbol("BRK/B") == "BRK-B"
    assert yahoo_symbol(None) is None


# --- seed_positions -------------------------------------------------------------

@pytest.mark.data
def test_seed_positions_unique_key(sec_db):
    assert scalar(sec_db, """
        SELECT count(*) FROM (SELECT cik, quarter, cusip, put_call FROM seed_positions
                              GROUP BY ALL HAVING count(*) > 1)""") == 0


@pytest.mark.data
def test_seed_positions_weights_sum_to_one(sec_db):
    assert scalar(sec_db, """
        SELECT count(*) FROM (SELECT cik, quarter, sum(weight) w FROM seed_positions GROUP BY ALL)
        WHERE abs(w - 1) > 1e-9""") == 0


@pytest.mark.data
def test_seed_positions_no_null_cusips_and_known_at_after_quarter(sec_db):
    assert scalar(sec_db, "SELECT count(*) FROM seed_positions WHERE cusip IS NULL OR cusip = ''") == 0
    assert scalar(sec_db, """
        SELECT count(*) FROM seed_positions
        WHERE known_at <= last_day(make_date(left(quarter, 4)::INT, right(quarter, 1)::INT * 3, 1))""") == 0


# --- security_master ------------------------------------------------------------

@pytest.mark.data
def test_master_one_row_per_universe_cusip(sec_db):
    assert scalar(sec_db, "SELECT count(*) - count(DISTINCT cusip) FROM security_master") == 0
    assert scalar(sec_db, """
        SELECT count(*) FROM (SELECT DISTINCT cusip FROM seed_positions) ANTI JOIN security_master USING (cusip)
    """) == 0


@pytest.mark.data
def test_master_classes(sec_db):
    classes = {r[0] for r in sec_db.execute("SELECT DISTINCT security_class FROM security_master").fetchall()}
    assert classes <= {"common", "adr", "etf", "fund", "preferred", "warrant_right", "debt", "other", "unmapped"}
    assert scalar(sec_db, "SELECT count(*) FROM security_master WHERE (ticker IS NULL) <> (security_class = 'unmapped')"
                  " AND security_class <> 'debt'") == 0


@pytest.mark.data
def test_value_coverage_at_least_99_percent(sec_db):
    worst = scalar(sec_db, """
        SELECT min(c) FROM (
            SELECT quarter, sum(value_usd) FILTER (WHERE m.ticker IS NOT NULL) / sum(value_usd) c
            FROM seed_positions JOIN security_master m USING (cusip) GROUP BY quarter)""")
    assert worst >= 0.99


@pytest.mark.data
@pytest.mark.parametrize("cusip, symbol, cls", [
    ("037833100", "AAPL", "common"),
    ("H1467J104", "CB", "common"),     # CINS code: only maps as ID_CINS
    ("78462F103", "SPY", "etf"),
    ("874039100", "TSM", "adr"),
    ("084670702", "BRK-B", "common"),  # share class: BRK/B -> BRK-B
])
def test_known_mappings(sec_db, cusip, symbol, cls):
    if not table_exists(sec_db, "security_master"):
        pytest.skip("security_master not built")
    row = sec_db.execute("SELECT symbol, security_class FROM security_master WHERE cusip = ?", [cusip]).fetchone()
    assert row == (symbol, cls)
