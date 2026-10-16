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
