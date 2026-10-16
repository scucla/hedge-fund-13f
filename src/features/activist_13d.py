"""13D activist heuristic (plan §5, Task A weak label): one row per fund x quarter.

Usage:
    uv run python -m src.features.activist_13d

A Schedule 13D is filed by an investor who owns more than 5% of a company and
may seek to influence it (a passive holder files the short 13G instead). The
heuristic: a fund is "activist" in a quarter if any of its CIKs filed a 13D
(original or amendment, "SC 13D" before Dec 2024 and "SCHEDULE 13D" after the
EDGAR form change) in the two years up to the fund-quarter's 13F filing date.
Point-in-time: only 13Ds filed on or before `known_at` count.

Source: the EDGAR quarterly full-index (form.gz, cached in
data/raw/edgar/full-index/), which lists a 13D under every entity on it,
including the filer. Writes `filings_13d` (every 13D row, 2016 on) and
`activist_13d` (fund_id, quarter, known_at, n_13d_2y, n_13d_orig_2y,
activist_13d) for the seed funds and the holdout funds.

Known blind spot: a fund whose 13Ds are filed under a different CIK than its
13F (e.g. a founder filing personally) is missed. Recall against the
activist label is reported, not assumed.
"""

import gzip
import re
from pathlib import Path

import polars as pl

from src.features.books import fund_id
from src.ingest.edgar_index import download_quarter
from src.ingest.seed_ciks import SEED_PATH
from src.securities import db

HOLDOUT_PATH = Path("data/reference/holdout_funds.csv")
FIRST_YEAR = 2016          # two years before the first modelling quarter (2018Q1)
LOOKBACK = "2 years"
FORM = re.compile(r"^(SC 13D/A|SC 13D|SCHEDULE 13D/A|SCHEDULE 13D)\s+(.*?)\s+(\d+)\s+(\d{4}-\d{2}-\d{2})\s+(\S+)\s*$")


ACTIVIST_SQL = f"""
    CREATE OR REPLACE TABLE activist_13d AS
    SELECT f.fund_id, f.quarter, f.known_at,
           count(DISTINCT d.file_name) AS n_13d_2y,
           count(DISTINCT d.file_name) FILTER (WHERE d.is_original) AS n_13d_orig_2y,
           count(DISTINCT d.file_name) > 0 AS activist_13d
    FROM fund_fingerprint f
    LEFT JOIN fund_ciks c ON c.fund_id = f.fund_id
    LEFT JOIN filings_13d d ON d.cik = c.cik
         AND d.date_filed <= f.known_at                                  -- point-in-time
         AND d.date_filed > f.known_at - INTERVAL {LOOKBACK}
    GROUP BY ALL
"""


def parse_13d(path: Path) -> list[tuple]:
    """13D rows of a form.idx file. The form type has spaces, so it is matched
    as a prefix; CIK, date and file name are the last three tokens."""
    rows = []
    with gzip.open(path, "rt", encoding="latin-1") as f:
        for line in f:
            if "13D" not in line[:16]:
                continue
            m = FORM.match(line)
            if m:
                form, name, cik, filed, fname = m.groups()
                rows.append((form, name, cik.zfill(10), filed, fname))
    return rows


def load_filings(last_year: int) -> pl.DataFrame:
    rows = []
    for y in range(FIRST_YEAR, last_year + 1):
        for q in range(1, 5):
            path = download_quarter(y, q)
            if path:
                rows += parse_13d(path)
    return (pl.DataFrame(rows, schema=["form", "company_name", "cik", "date_filed", "file_name"], orient="row")
            .with_columns(pl.col("date_filed").str.to_date(),
                          pl.col("form").str.ends_with("/A").not_().alias("is_original"))
            .unique())


def fund_ciks() -> pl.DataFrame:
    seed = (pl.read_csv(SEED_PATH, schema_overrides={"cik": pl.Utf8}).select("fund_name", "cik")
            .with_columns(pl.lit("seed").alias("cohort")))
    hold = (pl.read_csv(HOLDOUT_PATH, schema_overrides={"cik": pl.Utf8}).select("fund_name", "cik")
            .with_columns(pl.lit("holdout").alias("cohort")))
    return (pl.concat([seed, hold])
            .with_columns(pl.col("cik").str.zfill(10),
                          pl.col("fund_name").map_elements(fund_id, return_dtype=pl.Utf8).alias("fund_id"))
            .unique())


def build(con, last_year: int | None = None) -> None:
    last_year = last_year or int(db.LAST_QUARTER[:4])
    filings = load_filings(last_year)
    ciks = fund_ciks()
    con.execute("CREATE OR REPLACE TABLE filings_13d AS SELECT * FROM filings")
    con.execute("CREATE OR REPLACE TABLE fund_ciks AS SELECT * FROM ciks")
    con.execute(ACTIVIST_SQL)


if __name__ == "__main__":
    with db.connect() as con:
        build(con)
        print(con.execute("SELECT count(*) AS rows_13d, count(DISTINCT cik) AS filers FROM filings_13d").pl())
        labels = pl.read_csv(SEED_PATH).select("fund_name", "style_guess").unique().with_columns(
            pl.col("fund_name").map_elements(fund_id, return_dtype=pl.Utf8).alias("fund_id"))
        con.register("labels", labels)
        with pl.Config(tbl_rows=20):
            print(con.execute("""
                SELECT l.style_guess, count(*) AS fund_quarters,
                       round(avg(a.activist_13d::INT), 2) AS share_flagged
                FROM activist_13d a JOIN labels l USING (fund_id)
                WHERE a.quarter >= '2018Q1' GROUP BY 1 ORDER BY 3 DESC
            """).pl())
