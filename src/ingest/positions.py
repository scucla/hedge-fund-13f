"""Point-in-time view of one fund's holdings for one quarter.

Which filings count, as of a given date:
1. Only filings with known_at <= as_of (nothing the public couldn't see yet).
2. The base report is the latest 13F-HR original or RESTATEMENT amendment.
   A restatement replaces the whole report.
3. NEW HOLDINGS amendments filed after that base are added on top. They list
   positions left out of the original, often ones kept confidential.
4. Amendments with no type are ambiguous and are skipped (with a warning).
"""

from datetime import date

import duckdb
import polars as pl

_FILINGS_SQL = """
SELECT accession_number, manager_name, submission_type, amendment_type, filing_date, known_at
FROM filings
WHERE cik = $cik AND quarter = $quarter AND known_at <= $as_of
ORDER BY known_at, accession_number
"""


def normalize_cik(cik: str | int) -> str:
    return str(int(cik)).zfill(10)


def select_filings(filings: pl.DataFrame) -> tuple[list[str], list[str]]:
    """Return (accessions to use, warnings) for one fund-quarter's filings,
    sorted by known_at."""
    warnings = []
    is_orig = pl.col("submission_type") == "13F-HR"
    # fill_null: a missing amendment_type must count as "not this type", not drop out of filters.
    amendment = pl.col("amendment_type").fill_null("")
    is_restatement = (pl.col("submission_type") == "13F-HR/A") & (amendment == "RESTATEMENT")
    is_addition = (pl.col("submission_type") == "13F-HR/A") & (amendment == "NEW HOLDINGS")

    untyped = filings.filter(~is_orig & ~is_restatement & ~is_addition)
    for acc in untyped["accession_number"]:
        warnings.append(f"skipped amendment {acc}: it doesn't say whether it restates or adds holdings")

    bases = filings.filter(is_orig | is_restatement)
    if bases.is_empty():
        return [], warnings
    base = bases.row(-1, named=True)
    additions = filings.filter(is_addition & (pl.col("known_at") > base["known_at"]))
    return [base["accession_number"], *additions["accession_number"]], warnings


def fund_holdings(
    con: duckdb.DuckDBPyConnection, cik: str | int, quarter: str, as_of: date | None = None
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Holdings for one fund-quarter as known on `as_of` (default: today).

    Returns (positions, filings used, warnings). Positions have one row per
    (CUSIP, put/call) with value_usd and weight; weights sum to 1.
    """
    params = {"cik": normalize_cik(cik), "quarter": quarter.upper(), "as_of": as_of or date.today()}
    filings = con.execute(_FILINGS_SQL, params).pl()
    accessions, warnings = select_filings(filings)
    used = filings.filter(pl.col("accession_number").is_in(accessions))
    if not accessions:
        return pl.DataFrame(), used, warnings

    positions = con.execute(
        """
        SELECT cusip, put_call,
               any_value(name_of_issuer) AS name_of_issuer,
               sum(shares) AS shares,
               sum(principal) AS principal,
               sum(value_usd) AS value_usd
        FROM holdings
        WHERE accession_number IN (SELECT unnest($accessions))
        GROUP BY ALL
        """,
        {"accessions": accessions},
    ).pl()
    total = positions["value_usd"].sum()
    positions = positions.with_columns(
        (pl.col("value_usd") / total if total else pl.lit(None, dtype=pl.Float64)).alias("weight")
    ).sort("value_usd", descending=True)
    return positions, used, warnings
