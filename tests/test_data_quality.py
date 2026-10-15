"""Data-quality checks on the real database (skipped if it isn't built)."""

import pytest

pytestmark = pytest.mark.data


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


# --- filings ---------------------------------------------------------------

def test_filings_unique_accession(real_db):
    assert scalar(real_db, "SELECT count(*) - count(DISTINCT accession_number) FROM filings") == 0


def test_filings_known_at_is_filing_date_and_after_period(real_db):
    assert scalar(real_db, """
        SELECT count(*) FROM filings
        WHERE known_at IS NULL OR known_at <> filing_date OR known_at < period_of_report
    """) == 0


def test_filings_only_holdings_forms(real_db):
    assert scalar(real_db, "SELECT count(*) FROM filings WHERE submission_type NOT IN ('13F-HR', '13F-HR/A')") == 0


# --- infotable_raw -----------------------------------------------------------

def test_raw_unique_key(real_db):
    assert scalar(real_db, """
        SELECT count(*) FROM (
            SELECT accession_number, infotable_sk FROM infotable_raw GROUP BY ALL HAVING count(*) > 1)
    """) == 0


def test_raw_no_null_cusips(real_db):
    assert scalar(real_db, "SELECT count(*) FROM infotable_raw WHERE cusip IS NULL OR cusip = ''") == 0


def test_raw_values_present_and_not_negative(real_db):
    # $0 is allowed: ~0.3% of rows are tiny or worthless positions reported at zero.
    assert scalar(real_db, "SELECT count(*) FROM infotable_raw WHERE value_usd IS NULL OR value_usd < 0") == 0


def test_raw_rows_belong_to_a_filing(real_db):
    assert scalar(real_db, "SELECT count(*) FROM infotable_raw ANTI JOIN filings USING (accession_number)") == 0


# --- holdings ----------------------------------------------------------------

def test_holdings_unique_accession_cusip_putcall(real_db):
    assert scalar(real_db, """
        SELECT count(*) FROM (
            SELECT accession_number, cusip, put_call FROM holdings GROUP BY ALL HAVING count(*) > 1)
    """) == 0


def test_holdings_no_null_cusips(real_db):
    assert scalar(real_db, "SELECT count(*) FROM holdings WHERE cusip IS NULL OR cusip = ''") == 0


def test_holdings_values_present_and_not_negative(real_db):
    assert scalar(real_db, "SELECT count(*) FROM holdings WHERE value_usd IS NULL OR value_usd < 0") == 0


def test_zero_values_are_rare(real_db):
    share = scalar(real_db, "SELECT avg((value_usd = 0)::INT) FROM holdings")
    assert share < 0.01


def test_holdings_preserve_raw_totals(real_db):
    assert scalar(real_db, """
        SELECT count(*) FROM (SELECT accession_number, sum(value_usd) v FROM infotable_raw GROUP BY 1) r
        JOIN (SELECT accession_number, sum(value_usd) v FROM holdings GROUP BY 1) h USING (accession_number)
        WHERE abs(r.v - h.v) > 0.5
    """) == 0


def test_weights_sum_to_one_per_filing(real_db):
    assert scalar(real_db, """
        SELECT count(*) FROM (
            SELECT accession_number, sum(value_usd / total) w
            FROM holdings JOIN (SELECT accession_number, sum(value_usd) total FROM holdings GROUP BY 1)
                 USING (accession_number)
            WHERE total > 0
            GROUP BY 1)
        WHERE abs(w - 1) > 1e-9
    """) == 0
