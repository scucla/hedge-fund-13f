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

SEED_VIEWS = ["seed_positions", "seed_positions_pit"]


@pytest.mark.data
@pytest.mark.parametrize("table", SEED_VIEWS)
def test_seed_positions_unique_key(sec_db, table):
    assert scalar(sec_db, f"""
        SELECT count(*) FROM (SELECT cik, quarter, cusip, put_call FROM {table}
                              GROUP BY ALL HAVING count(*) > 1)""") == 0


@pytest.mark.data
@pytest.mark.parametrize("table", SEED_VIEWS)
def test_seed_positions_weights_sum_to_one(sec_db, table):
    assert scalar(sec_db, f"""
        SELECT count(*) FROM (SELECT cik, quarter, sum(weight) w FROM {table} GROUP BY ALL)
        WHERE abs(w - 1) > 1e-9""") == 0


@pytest.mark.data
@pytest.mark.parametrize("table", SEED_VIEWS)
def test_seed_positions_no_null_cusips_and_known_at_after_quarter(sec_db, table):
    assert scalar(sec_db, f"SELECT count(*) FROM {table} WHERE cusip IS NULL OR cusip = ''") == 0
    assert scalar(sec_db, f"""
        SELECT count(*) FROM {table}
        WHERE known_at <= last_day(make_date(left(quarter, 4)::INT, right(quarter, 1)::INT * 3, 1))""") == 0


@pytest.mark.data
def test_pit_view_is_never_later_than_latest_view(sec_db):
    """The point-in-time view is as of the original filing, so its known_at can
    only be on or before the latest view's, and the same filer-quarters exist."""
    assert scalar(sec_db, """
        SELECT count(*) FROM (SELECT cik, quarter, max(known_at) k FROM seed_positions GROUP BY ALL) a
        FULL JOIN (SELECT cik, quarter, max(known_at) k FROM seed_positions_pit GROUP BY ALL) b USING (cik, quarter)
        WHERE a.k IS NULL OR b.k IS NULL OR b.k > a.k""") == 0
    # and amendments do make a difference somewhere, otherwise the two views are pointless
    assert scalar(sec_db, """
        SELECT count(*) FROM (SELECT cik, quarter, max(known_at) k FROM seed_positions GROUP BY ALL) a
        JOIN (SELECT cik, quarter, max(known_at) k FROM seed_positions_pit GROUP BY ALL) b USING (cik, quarter)
        WHERE b.k < a.k""") > 0


# --- security_master ------------------------------------------------------------

@pytest.mark.data
def test_master_one_row_per_universe_cusip(sec_db):
    assert scalar(sec_db, "SELECT count(*) - count(DISTINCT cusip) FROM security_master") == 0
    assert scalar(sec_db, """
        SELECT count(*) FROM (SELECT cusip FROM seed_positions UNION SELECT cusip FROM seed_positions_pit)
        ANTI JOIN security_master USING (cusip)
    """) == 0


@pytest.mark.data
def test_master_classes(sec_db):
    classes = {r[0] for r in sec_db.execute("SELECT DISTINCT security_class FROM security_master").fetchall()}
    assert classes <= {"common", "adr", "etf", "fund", "preferred", "warrant_right", "debt", "other", "unmapped"}
    assert scalar(sec_db, "SELECT count(*) FROM security_master WHERE (ticker IS NULL) <> (security_class = 'unmapped')"
                  " AND security_class <> 'debt'") == 0


@pytest.mark.data
def test_value_coverage(sec_db):
    """>= 99% of value mapped from 2023 on; >= 90% in every quarter back to 2015,
    where retired CUSIPs (renames, mergers) are harder to map."""
    cov = dict(sec_db.execute("""
        SELECT quarter, sum(value_usd) FILTER (WHERE m.ticker IS NOT NULL) / sum(value_usd)
        FROM seed_positions JOIN security_master m USING (cusip) GROUP BY quarter""").fetchall())
    assert min(v for q, v in cov.items() if q >= "2023Q1") >= 0.99
    assert min(cov.values()) >= 0.90


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


# --- delisted issuers -----------------------------------------------------------

from src.securities.cusip_map import common_ticker, is_us_exchange, normalize_name  # noqa: E402


@pytest.mark.parametrize("a, b", [
    ("ELECTRONIC ARTS INC", "ELECTRONIC ARTS INC."),
    ("HOLOGIC INC", "HOLOGIC, INC"),
    ("NUVALENT INC-A", "NUVALENT, INC."),
    ("CYBERARK SOFTWARE LTD/ISRAEL", "CyberArk Software Ltd."),
])
def test_normalize_name_matches_filer_names(a, b):
    assert normalize_name(a) == normalize_name(b)


def test_normalize_name_keeps_distinct_companies_apart():
    assert normalize_name("BANK OF AMERICA CORP") != normalize_name("BANK OF AMERICA CORP /DE/ PREFERRED TRUST")


@pytest.mark.parametrize("code, us", [("US", True), ("UN", True), ("UW", True), ("MM", False), ("GR", False),
                                      ("X1", False), (None, False)])
def test_is_us_exchange(code, us):
    assert is_us_exchange(code) is us


def test_common_ticker_ignores_venue_suffixes():
    resp = {"data": [{"ticker": "HON"}, {"ticker": "HON"}, {"ticker": "HONGBP"}, {"ticker": "HON"}]}
    assert common_ticker(resp) == "HON"


@pytest.mark.data
@pytest.mark.parametrize("cusip, symbol, cik", [
    ("285512109", "EA", "0000712515"),     # taken private: US listing recovered, CIK by name
    ("934423104", "WBD", "0001437107"),    # was mapped to the Mexican listing "WBD*"
    ("436440101", "HOLX", "0000859737"),
])
def test_delisted_names_recovered(sec_db, cusip, symbol, cik):
    row = sec_db.execute("SELECT symbol, issuer_cik FROM security_master WHERE cusip = ?", [cusip]).fetchone()
    assert row == (symbol, cik)


# --- price store -------------------------------------------------------------------

from datetime import date as _date  # noqa: E402

import polars as _pl  # noqa: E402

from src.securities.prices import missing_pieces, stitch  # noqa: E402


def _px(days, closes):
    return _pl.DataFrame({"symbol": "X", "date": [_date(2026, 1, d) for d in days], "close": closes,
                          "volume": [0.0] * len(days)})


def test_stitch_rescales_stored_history_to_new_basis():
    # A dividend after day 5 made Yahoo scale all earlier prices by 0.98.
    old = _px([2, 3, 4, 5, 6], [100.0, 101.0, 102.0, 103.0, 104.0])
    new = _px([5, 6, 7, 8], [103.0 * 0.98, 104.0 * 0.98, 105.0, 106.0])
    out = stitch(old, new)
    assert out["close"].to_list() == pytest.approx([98.0, 98.98, 99.96, 100.94, 101.92, 105.0, 106.0])
    # returns are unchanged by the rescaling
    r = out["close"].pct_change().to_list()[1:5]
    assert r == pytest.approx(old["close"].pct_change().to_list()[1:])


def test_stitch_refuses_without_overlap_or_with_inconsistent_ratio():
    old = _px([2, 3], [100.0, 101.0])
    assert stitch(old, _px([7, 8], [50.0, 51.0])) is None
    assert stitch(old, _px([2, 3], [100.0, 90.0])) is None


def test_missing_pieces():
    entry = {"from": "2023-01-01", "to": "2026-10-15"}
    assert missing_pieces(entry, "2023-01-01", "2026-10-15") == []
    tail = missing_pieces(entry, "2023-01-01", "2026-10-20")
    assert tail == [("2026-10-05", "2026-10-20")]
    head_tail = missing_pieces(entry, "2013-06-01", "2026-10-20")
    assert head_tail[0] == ("2013-06-01", "2023-01-11") and head_tail[1][1] == "2026-10-20"
    assert missing_pieces({"from": "2023-01-01", "to": "2026-10-15", "no_data": True}, "2023-01-01", "2026-10-15") == []
    assert missing_pieces({"from": "2023-01-01", "to": "2026-10-15", "no_data": True}, "2013-06-01", "2026-10-15") \
        == [("2013-06-01", "2026-10-15")]
    assert missing_pieces(None, "2013-06-01", "2026-10-15") == [("2013-06-01", "2026-10-15")]


# --- unit errors in filings -------------------------------------------------------

from src.securities.universe import fix_unit_errors  # noqa: E402


def test_filing_in_dollars_instead_of_thousands_is_rescaled():
    rows = []
    for cik, mult in [("A", 1), ("B", 1), ("C", 1), ("BAD", 1000)]:
        for i in range(6):
            rows.append({"cik": cik, "fund_name": cik, "quarter": "2015Q1", "cusip": f"S{i}", "put_call": "",
                         "shares": 100.0, "value_usd": 100.0 * (10 + i) * mult})
    fixed, fixes = fix_unit_errors(_pl.DataFrame(rows))
    assert fixes["cik"].to_list() == ["BAD"] and fixes["power_of_1000"].to_list() == [1]
    bad = fixed.filter(_pl.col("cik") == "BAD").sort("cusip")["value_usd"].to_list()
    assert bad == pytest.approx([100.0 * (10 + i) for i in range(6)])
    assert fixed.filter(_pl.col("cik") == "A")["unit_fix_power"].to_list() == [0] * 6


def test_slow_fetch_backs_off_on_rate_limit_and_records_results(tmp_path, monkeypatch):
    from src.securities import prices as P

    monkeypatch.setattr(P, "STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(P, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(P, "RATE_LIMIT_PATH", tmp_path / "last_rate_limit.txt")
    monkeypatch.setattr(P, "SLOW_PAUSE", 0)
    monkeypatch.setattr(P, "SLOW_BACKOFF", 0)
    sleeps = []
    monkeypatch.setattr(P.time, "sleep", lambda s: sleeps.append(s))
    calls = []

    def fake_download(symbols, start, end):
        sym = symbols[0]
        calls.append(sym)
        if sym == "B" and calls.count("B") == 1:
            P.LAST_ERRORS = {"B": "YFRateLimitError('Too Many Requests. Rate limited.')"}
            return {}, set(), {"B"}
        if sym == "C":
            P.LAST_ERRORS = {"C": "possibly delisted; no price data found"}
            return {}, {"C"}, set()
        P.LAST_ERRORS = {}
        return {sym: _px([5, 6], [10.0, 11.0]).with_columns(_pl.lit(sym).alias("symbol"))}, set(), set()

    monkeypatch.setattr(P, "download_chunk", fake_download)
    P.fetch_slow(["A", "B", "C"], "2026-01-01", "2026-01-31")
    m = P.load_manifest()
    assert calls == ["A", "B", "B", "C"]                  # one symbol per request, in order; B retried
    assert m["A"] == {"from": "2026-01-01", "to": "2026-01-31"} and m["B"]["from"] == "2026-01-01"
    assert m["C"]["no_data"] is True
    assert (tmp_path / "last_rate_limit.txt").exists()     # the hit was recorded for the 2 h rule


def test_no_earlier_data_extends_window_instead_of_retrying(tmp_path, monkeypatch):
    """A 2023 IPO asked for 2014-2022: Yahoo says the data doesn't exist. That
    window is now covered, so the symbol isn't asked again next run."""
    from src.securities import prices as P

    monkeypatch.setattr(P, "STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(P, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(P, "RATE_LIMIT_PATH", tmp_path / "rl.txt")
    monkeypatch.setattr(P, "SLOW_PAUSE", 0)
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    P.save_manifest({"NEW": {"from": "2023-01-01", "to": "2026-10-15"}})
    calls = []

    def fake_download(symbols, start, end):
        calls.append((symbols[0], start, end))
        P.LAST_ERRORS = {symbols[0]: "Data doesn't exist for startDate = 1388552400, endDate = 1673413200"}
        return {}, {symbols[0]}, set()

    monkeypatch.setattr(P, "download_chunk", fake_download)
    P.fetch_slow(["NEW"], "2014-01-01", "2026-10-15")
    assert P.load_manifest()["NEW"] == {"from": "2014-01-01", "to": "2026-10-15"}
    assert P.missing_pieces(P.load_manifest()["NEW"], "2014-01-01", "2026-10-15") == []
    assert len(calls) == 1


def test_mixed_unit_filing_is_fixed_row_by_row():
    """Most rows in thousands (1000x too small as dollars) but one row already in
    dollars: only the thousands rows are rescaled (the Davidson Kempner case)."""
    rows = []
    for cik in ["A", "B", "C", "MIX"]:
        for i in range(6):
            wrong = cik == "MIX" and i != 0          # row S0 of MIX is already right
            rows.append({"cik": cik, "fund_name": cik, "quarter": "2022Q4", "cusip": f"S{i}", "put_call": "",
                         "shares": 100.0, "value_usd": 100.0 * (10 + i) / (1000 if wrong else 1)})
    fixed, fixes = fix_unit_errors(_pl.DataFrame(rows))
    mix = fixed.filter(_pl.col("cik") == "MIX").sort("cusip")
    assert mix["value_usd"].to_list() == pytest.approx([100.0 * (10 + i) for i in range(6)])
    assert mix["unit_fix_power"].to_list() == [0, -1, -1, -1, -1, -1]
    assert fixes.filter(_pl.col("cik") == "MIX")["rows_fixed"].to_list() == [5]



def test_lone_deviating_row_is_flagged_not_rescaled():
    """In a correct filing, one row's implied price is 1000x the consensus. That is
    usually a share count reported in thousands: keep the value, flag the row."""
    rows = []
    for cik in ["A", "B", "C", "D"]:
        for i in range(6):
            odd = cik == "D" and i == 0
            rows.append({"cik": cik, "fund_name": cik, "quarter": "2020Q1", "cusip": f"S{i}", "put_call": "",
                         "shares": 100.0 / (1000 if odd else 1), "value_usd": 100.0 * (10 + i)})
    fixed, _ = fix_unit_errors(_pl.DataFrame(rows))
    d0 = fixed.filter((_pl.col("cik") == "D") & (_pl.col("cusip") == "S0")).row(0, named=True)
    assert d0["value_usd"] == 1000.0 and d0["unit_fix_power"] == 0 and d0["shares_suspect"] is True
    assert not fixed.filter(_pl.col("cik") != "D")["shares_suspect"].any()


def test_common_ticker_ties_are_deterministic():
    # two listings each: no majority. Must not depend on set order (string hashing is randomized per run)
    resp = {"data": [{"ticker": "NDLSGBP"}, {"ticker": "NDLSUSD"}, {"ticker": "NDLSGBP"}, {"ticker": "NDLSUSD"}]}
    assert common_ticker(resp) == "NDLSGBP"
    resp = {"data": [{"ticker": "HONGBP"}, {"ticker": "HON"}]}
    assert common_ticker(resp) == "HON"
