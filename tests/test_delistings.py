"""Delisting gap (pre-Milestone 5): 13F-implied prices, SEC event index, deal-term parsing."""

import gzip

import pytest

from src.securities.deal_docs import parse_terms
from src.securities.sec_events import parse_events
from tests.conftest import table_exists

# --- deal terms: sentences from the definitive proxies of three known deals ------------

TWITTER = ("If the merger is completed, you will be entitled to receive $54.20 in cash, without interest "
           "and subject to any applicable withholding taxes, for each share of our common stock that you own")
ACTIVISION = ("Upon completion of the merger, you will be entitled to receive $95.00 in cash, without interest, "
              "for each share of our common stock, par value $0.000001 per share")
CELGENE = ("Celgene stockholders immediately prior to the completion of the merger will be entitled to receive "
           "$50.00 in cash, one share of Bristol-Myers Squibb common stock and one contingent value right "
           "(each, a “CVR”) for each share of Celgene common stock held by them")


@pytest.mark.parametrize("text, cash, ratio", [(TWITTER, 54.20, None), (ACTIVISION, 95.00, None), (CELGENE, 50.00, 1.0)])
def test_parse_terms_known_deals(text, cash, ratio):
    t = parse_terms(text)
    assert t["cash"] == cash and t["ratio"] == ratio


def test_parse_terms_stock_deal_and_cvr():
    t = parse_terms(CELGENE)
    assert t["acquirer_name"] == "Bristol-Myers Squibb" and t["has_cvr"]
    t = parse_terms("each share will be converted into the right to receive 0.8 shares of Acme Corp. common stock "
                    "for each share of Target common stock")
    assert t["cash"] is None and t["ratio"] == 0.8


def test_parse_terms_price_after_cash_and_traps():
    seagen = ("If the merger is completed, you will be entitled to receive an amount in cash equal to $229.00 "
              "(disregarding reduction for any applicable withholding taxes), for each eligible share of our common stock")
    assert parse_terms(seagen)["cash"] == 229.0
    pioneer = ("For example, if you own 110 shares of Pioneer common stock, you will be entitled to receive 255 shares "
               "of ExxonMobil common stock. The ratio of 2.3234 shares of ExxonMobil common stock for each share")
    assert parse_terms(pioneer) is None or parse_terms(pioneer)["ratio"] != 255
    alexion = ("holders entitled to receive notice of and to vote at the Alexion special meeting; there were "
               "220,912,612 shares of Alexion common stock issued and outstanding per share")
    t = parse_terms(alexion)
    assert t is None or t["ratio"] is None


def test_parse_terms_tender_offer_and_nothing():
    t = parse_terms("Purchaser is offering to purchase all outstanding Shares at a price of $12.50 per Share, net to the seller")
    assert t["cash"] == 12.50
    assert parse_terms("The annual meeting will be held on May 5.") is None


# --- EDGAR index parsing ----------------------------------------------------------------

def test_parse_events_handles_spaced_forms_and_names(tmp_path):
    lines = [
        "Form Type   Company Name                                                  CIK         Date Filed  File Name",
        "-" * 100,
        "SC TO-T          CELGENE CORP /DE/                                             816284      2015-07-28  edgar/data/816284/0001193125-15-266000.txt",
        "SC TO-T/A        CELGENE CORP /DE/                                             816284      2015-08-03  edgar/data/816284/0001193125-15-270000.txt",
        "25-NSE           TWITTER, INC.                                                 1418091     2022-10-28  edgar/data/1418091/0000876661-22-000600.txt",
        "10-K             SOMETHING ELSE INC                                            1           2022-03-01  edgar/data/1/0000000001-22-000001.txt",
    ]
    path = tmp_path / "form.gz"
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode("latin-1")))
    df = parse_events(path)
    assert df["form"].to_list() == ["SC TO-T", "25-NSE"]                  # the /A amendment and 10-K are not wanted
    assert df["company_name"].to_list() == ["CELGENE CORP /DE/", "TWITTER, INC."]   # not cut at the CIK in the path
    assert df["cik"].to_list() == ["0000816284", "0001418091"]


# --- tables on the real database ----------------------------------------------------------

def _need(con, *tables):
    for t in tables:
        if not table_exists(con, t):
            pytest.skip(f"{t} not built")


def test_implied_prices_keys_and_disagreement_flag(sec_db):
    _need(sec_db, "implied_prices", "implied_returns")
    assert sec_db.execute("SELECT count(*) - count(DISTINCT (cusip, quarter)) FROM implied_prices").fetchone()[0] == 0
    assert sec_db.execute("SELECT count(*) FROM implied_prices WHERE cusip IS NULL OR NOT px > 0").fetchone()[0] == 0
    # the flag is exactly "funds more than 5% apart"
    assert sec_db.execute("""SELECT count(*) FROM implied_prices
                             WHERE px_disagree <> (px_max / px_min - 1 > 0.05)""").fetchone()[0] == 0
    assert sec_db.execute("SELECT count(*) FROM implied_prices WHERE n_funds = 1 AND px_disagree").fetchone()[0] == 0


def test_implied_returns_after_unit_fix_are_plausible(sec_db):
    _need(sec_db, "implied_returns")
    assert sec_db.execute("SELECT count(*) - count(DISTINCT (cusip, quarter)) FROM implied_returns").fetchone()[0] == 0
    # a units error left in would show up as a ~1000x return
    assert sec_db.execute("SELECT max(abs(log10(1 + ret))) FROM implied_returns").fetchone()[0] < 2.0


def test_implied_returns_match_yahoo(sec_db):
    _need(sec_db, "implied_returns", "stock_quarter")
    from src.securities.implied_prices import validation
    v = validation(sec_db).row(0, named=True)
    assert v["corr_unflagged"] > 0.9 and v["median_abs_err"] < 0.005


def test_sec_events_and_cik_fill_keys(sec_db):
    _need(sec_db, "sec_events", "issuer_cik_fill")
    assert sec_db.execute("SELECT count(*) - count(DISTINCT (cik, form, accession)) FROM sec_events").fetchone()[0] == 0
    assert sec_db.execute("SELECT count(*) FROM sec_events WHERE cik IS NULL OR known_at IS NULL").fetchone()[0] == 0
    assert sec_db.execute("SELECT count(*) - count(DISTINCT cusip) FROM issuer_cik_fill").fetchone()[0] == 0
    # only fills CUSIPs the master left without a CIK
    assert sec_db.execute("""SELECT count(*) FROM issuer_cik_fill f JOIN security_master m USING (cusip)
                             WHERE m.issuer_cik IS NOT NULL""").fetchone()[0] == 0


def test_sec_events_known_deals(sec_db):
    _need(sec_db, "sec_events")
    rows = dict(sec_db.execute("""SELECT form, min(known_at)::VARCHAR FROM sec_events
                                  WHERE cik = '0001418091' GROUP BY 1""").fetchall())   # Twitter
    assert rows["DEFM14A"] == "2022-07-26" and rows["25-NSE"] == "2022-10-28"


# --- delisting events and filled labels -------------------------------------------------

EXIT_CLASSES = {"cash_acquisition", "stock_acquisition", "performance", "other",
                "sold_out", "cusip_change", "no_cik"}


def test_delisting_events_keys_and_classes(sec_db):
    _need(sec_db, "delisting_events")
    assert sec_db.execute("SELECT count(*) - count(DISTINCT (cusip, quarter)) FROM delisting_events").fetchone()[0] == 0
    classes = {r[0] for r in sec_db.execute("SELECT DISTINCT exit_class FROM delisting_events").fetchall()}
    assert classes <= EXIT_CLASSES
    # a delisting return only for acquisitions and performance delistings, and inside the sanity band
    assert sec_db.execute("""SELECT count(*) FROM delisting_events WHERE delist_ret IS NOT NULL
                             AND exit_class NOT IN ('cash_acquisition', 'stock_acquisition', 'performance')""").fetchone()[0] == 0
    # deal returns inside the band the code uses: 0.9x-1.5x if announced by q's end, 0.5x-4x if announced in q+1
    # (acquirer-return fallback: 0.5x-2x)
    assert sec_db.execute("""SELECT count(*) FROM delisting_events WHERE ret_source IN ('delist_cash', 'delist_stock', 'delist_mixed')
                             AND NOT CASE WHEN announced_in_next_q THEN delist_ret + 1 BETWEEN 0.5 AND 4.0
                                          ELSE delist_ret + 1 BETWEEN 0.9 AND 1.5 END""").fetchone()[0] == 0
    assert sec_db.execute("""SELECT count(*) FROM delisting_events WHERE ret_source = 'delist_stock_acq_return'
                             AND NOT delist_ret + 1 BETWEEN 0.5 AND 2.0""").fetchone()[0] == 0
    assert sec_db.execute("""SELECT count(*) FROM delisting_events
                             WHERE exit_class = 'performance' AND delist_ret <> -0.30""").fetchone()[0] == 0


def test_delisting_dates_are_ordered(sec_db):
    _need(sec_db, "delisting_events")
    assert sec_db.execute("""SELECT count(*) FROM delisting_events
                             WHERE delist_date < first_deal_filing OR ret_known_at < delist_date""").fetchone()[0] == 0
    # the delisting falls in (or right around) the quarter after the last holding
    assert sec_db.execute("""SELECT count(*) FROM delisting_events WHERE delist_date IS NOT NULL
                             AND NOT delist_date BETWEEN q_end - INTERVAL 30 DAY AND next_q_end + INTERVAL 15 DAY""").fetchone()[0] == 0


def test_known_deals_classified(sec_db):
    _need(sec_db, "delisting_events")
    rows = {r[0]: r[1:] for r in sec_db.execute("""
        SELECT issuer_cik, exit_class, exit_subclass, ret_source FROM delisting_events
        WHERE issuer_cik IN ('0001418091', '0000718877', '0000816284') AND delist_date IS NOT NULL""").fetchall()}
    if not rows:
        pytest.skip("known deals not among the exits")
    for cik in ("0001418091", "0000718877"):   # Twitter, Activision: cash
        if cik in rows:
            assert rows[cik][:2] == ("cash_acquisition", "cash")
    if "0000816284" in rows:                   # Celgene: cash + BMY stock
        assert rows["0000816284"][:2] == ("stock_acquisition", "mixed")


def _need_cols(con, table, col):
    _need(con, table)
    if col not in {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}:
        pytest.skip(f"{table}.{col} not built (run src.securities.labels)")


def test_every_missing_label_has_a_reason(sec_db):
    _need_cols(sec_db, "stock_quarter", "ret_source")
    assert sec_db.execute("SELECT count(*) FROM stock_quarter WHERE ret_source IS NULL").fetchone()[0] == 0
    assert sec_db.execute("""SELECT count(*) FROM stock_quarter
                             WHERE (ret_next_q IS NULL) <> (ret_source LIKE 'missing:%')""").fetchone()[0] == 0
    # Yahoo rows are untouched
    assert sec_db.execute("""SELECT count(*) FROM stock_quarter WHERE ret_source = 'yahoo'
                             AND ret_next_q IS DISTINCT FROM ret_next_q_yahoo""").fetchone()[0] == 0


def test_filled_labels_are_point_in_time(sec_db):
    _need_cols(sec_db, "stock_quarter", "ret_source")
    assert sec_db.execute("""SELECT count(*) FROM stock_quarter
                             WHERE ret_next_q IS NOT NULL AND (label_known_at IS NULL OR label_known_at <= known_at)""").fetchone()[0] == 0


# --- Task C exits never depend on prices ---------------------------------------------------

def _moves_db():
    import duckdb
    con = duckdb.connect()
    con.execute("""CREATE TABLE fund_holdings AS SELECT * FROM (VALUES
        ('f1', 'seed', 'AAA', '2020Q1', DATE '2020-05-10', '', 100.0, NULL, false),
        ('f1', 'seed', 'BBB', '2020Q1', DATE '2020-05-10', '', 100.0, NULL, false),
        ('f1', 'seed', 'CCC', '2020Q1', DATE '2020-05-10', '', 100.0, NULL, false),
        ('f1', 'seed', 'DDD', '2020Q1', DATE '2020-05-10', '', 100.0, NULL, false),
        ('f1', 'seed', 'AAA', '2020Q2', DATE '2020-08-10', '', 200.0, NULL, false),   -- 2:1 split: hold
        ('f1', 'seed', 'BBB', '2020Q2', DATE '2020-08-10', '', 150.0, NULL, false),   -- add
        ('f1', 'seed', 'CCC', '2020Q2', DATE '2020-08-10', '', 50.0, NULL, false),    -- trim
        ('f2', 'seed', 'AAA', '2020Q1', DATE '2020-05-12', '', 10.0, NULL, false)     -- f2 never files again
        ) t(fund_id, cohort, cusip, quarter, known_at, put_call, shares, principal, shares_suspect)""")
    con.execute("CREATE TABLE empty_books (fund_id VARCHAR, quarter VARCHAR, known_at DATE)")
    con.execute("CREATE TABLE stock_splits (cusip VARCHAR, quarter VARCHAR, split_ratio DOUBLE)")
    con.execute("CREATE TABLE implied_splits AS SELECT 'AAA' AS cusip, '2020Q2' AS quarter, 2.0 AS split_ratio")
    con.execute("""CREATE TABLE delisting_events AS
                   SELECT 'DDD' AS cusip, '2020Q1' AS quarter, 'cash_acquisition' AS exit_class""")
    return con


def test_position_moves_need_no_price_tables():
    from src.features import position_moves
    con = _moves_db()   # no prices_daily, returns_quarterly or stock_quarter exist at all
    position_moves.build(con)
    got = {(f, c): (a, fe) for f, c, a, fe in con.execute(
        "SELECT fund_id, cusip, action, forced_exit FROM position_moves").fetchall()}
    assert got == {("f1", "AAA"): ("hold", False), ("f1", "BBB"): ("add", False),
                   ("f1", "CCC"): ("trim", False), ("f1", "DDD"): ("exit", True)}   # f2 didn't file: no label


def test_position_moves_sql_reads_no_prices():
    import inspect
    from src.features import position_moves
    src = inspect.getsource(position_moves.build)
    for t in ("prices_daily", "returns_quarterly", "stock_quarter", "implied_prices", "implied_returns"):
        assert t not in src


def test_position_moves_real(sec_db):
    _need(sec_db, "position_moves")
    assert sec_db.execute("SELECT count(*) - count(DISTINCT (fund_id, cusip, quarter)) FROM position_moves").fetchone()[0] == 0
    assert sec_db.execute("SELECT count(*) FROM position_moves WHERE forced_exit AND action <> 'exit'").fetchone()[0] == 0


# --- merger targets ------------------------------------------------------------------------

def test_merger_target_share(sec_db):
    _need(sec_db, "merger_target_share", "merger_targets")
    assert sec_db.execute("SELECT count(*) - count(DISTINCT (fund_id, quarter)) FROM merger_target_share").fetchone()[0] == 0
    assert sec_db.execute("SELECT count(*) FROM merger_target_share WHERE NOT target_share BETWEEN 0 AND 1.0001").fetchone()[0] == 0
    assert sec_db.execute("SELECT count(*) FROM merger_targets WHERE announce_date > close_date").fetchone()[0] == 0


def test_no_target_counted_before_announcement():
    import duckdb
    from src.features import merger_targets
    con = duckdb.connect()
    con.execute("""CREATE TABLE delisting_events AS SELECT * FROM (VALUES
        ('T1', '0001', 'TGT', 'cash_acquisition', 'cash', DATE '2020-05-15', DATE '2020-11-20', '2020Q3')
        ) t(cusip, issuer_cik, name, exit_class, exit_subclass, first_deal_filing, delist_date, quarter)""")
    con.execute("""CREATE TABLE positions AS SELECT * FROM (VALUES
        ('f1', 'seed', '2020Q1', 'T1', 0.5, false), ('f1', 'seed', '2020Q1', 'X', 0.5, false),
        ('f1', 'seed', '2020Q2', 'T1', 0.5, false), ('f1', 'seed', '2020Q2', 'X', 0.5, false),
        ('f1', 'seed', '2020Q4', 'X', 1.0, false)
        ) t(fund_id, cohort, quarter, cusip, weight_long, is_option)""")
    con.execute("""CREATE TABLE fund_fingerprint AS SELECT * FROM (VALUES
        ('f1', 'seed', '2020Q1', DATE '2020-05-10'), ('f1', 'seed', '2020Q2', DATE '2020-08-10'),
        ('f1', 'seed', '2020Q4', DATE '2021-02-10')) t(fund_id, cohort, quarter, known_at)""")
    merger_targets.build(con)
    got = dict(con.execute("SELECT quarter, target_share FROM merger_target_share").fetchall())
    assert got == {"2020Q1": 0.0, "2020Q2": 0.5, "2020Q4": 0.0}   # announced 05-15: not yet at 03-31
