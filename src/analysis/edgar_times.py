"""EDGAR acceptance times of the seed funds' 13F-HR filings.

Usage:
    uv run python -m src.analysis.edgar_times     # -> data/interim/filing_times.parquet

The 13F data sets and the EDGAR full-index give a filing *date* only. The source
of truth for the acceptance time is each filing's own EDGAR header
(`<accession>-index-headers.html`, line ACCEPTANCE-DATETIME, Eastern time),
cached one file per filing under data/raw/edgar/headers/ so an interrupted run
resumes. Only the filings the event study uses are fetched: each seed CIK's
original 13F-HR whose filing date is the known_at of a point-in-time book
(seed_positions_pit).

**Why not the submissions API.** Its `acceptanceDateTime` (suffix "Z") is usually
UTC, but checked against the headers on 13 filings, 6 were shifted by twice the
UTC offset (8 h in summer, 10 h in winter), at any time of day, so it can't be
corrected by rule. It is kept only as `api_time` for the record, and the
submissions JSON (src.ingest.seed_ciks.submissions, cached) is used just to find
each filing's accession number.

A filing whose header can't be fetched or parsed gets `accepted_et` null and
`source` = 'fallback': the event study then uses its filing date with day 0 = the
next trading day, and flags those events.
"""

import re
from datetime import datetime
from pathlib import Path

import polars as pl

from src.ingest import sec
from src.ingest.seed_ciks import MAP_PATH, submissions
from src.securities import db

OUT = Path("data/interim/filing_times.parquet")
HEADERS_DIR = Path("data/raw/edgar/headers")
_ACCEPTED = re.compile(r"ACCEPTANCE-DATETIME>\s*(\d{14})")


def quarter_of(report_date: str) -> str | None:
    if not report_date:
        return None
    y, m = int(report_date[:4]), int(report_date[5:7])
    return f"{y}Q{(m - 1) // 3 + 1}"


def filing_rows(cik: str, block: dict) -> list[dict]:
    return [{"cik": cik, "accession": a, "quarter": quarter_of(p), "filing_date": d, "api_time": t or None}
            for a, f, d, p, t in zip(block["accessionNumber"], block["form"], block["filingDate"],
                                     block["reportDate"], block["acceptanceDateTime"])
            if f == "13F-HR"]


def header_time(cik: str, accession: str) -> datetime | None:
    """ACCEPTANCE-DATETIME from the filing's EDGAR header (cached); None if it can't be had."""
    path = HEADERS_DIR / f"{accession}.html"
    if not path.exists():
        try:
            text = sec.get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
                           f"{accession}-index-headers.html").text
        except Exception as e:   # network or 404: no cache, so a later run retries it
            print(f"  header failed for {accession}: {str(e)[:80]}")
            return None
        HEADERS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    m = _ACCEPTED.search(path.read_text())
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None


def needed_filings() -> pl.DataFrame:
    """Original 13F-HRs of seed CIKs that a point-in-time book was built from."""
    ciks = sorted({c.zfill(10) for c in pl.read_csv(MAP_PATH, schema_overrides={"cik": pl.Utf8})["cik"]})
    filings = pl.DataFrame([r for cik in ciks for r in filing_rows(cik, submissions(cik)["filings"]["recent"])])
    filings = filings.unique("accession").with_columns(pl.col("filing_date").str.to_date())
    with db.connect(read_only=True) as con:
        books = con.execute("""SELECT DISTINCT cik, quarter, known_at AS filing_date FROM seed_positions_pit
                               WHERE cohort = 'seed'""").pl()
    # several originals for one CIK-quarter on the same day: the first accession filed
    return (filings.join(books, on=["cik", "quarter", "filing_date"]).sort("accession")
            .unique(["cik", "quarter", "filing_date"], keep="first"))


def build() -> pl.DataFrame:
    need = needed_filings()
    print(f"{len(need):,} filings; {sum((HEADERS_DIR / f'{a}.html').exists() for a in need['accession']):,} headers cached")
    times = [header_time(c, a) for c, a in need.select("cik", "accession").iter_rows()]
    df = need.with_columns(pl.Series("accepted_et", times, dtype=pl.Datetime),
                           pl.when(pl.Series(times).is_null()).then(pl.lit("fallback")).otherwise(pl.lit("header"))
                           .alias("source"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT)
    return df


if __name__ == "__main__":
    df = build()
    print(df.group_by("source").len())
    print(df.group_by(pl.col("filing_date").dt.year().alias("year")).agg(
        pl.len(), (pl.col("source") == "header").mean().alias("header_share"),
        (pl.col("accepted_et").dt.hour() >= 16).mean().alias("after_close")).sort("year"))
