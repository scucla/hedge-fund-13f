"""Fama-French 5 factors + momentum (daily and monthly) and the FF12 industry
definitions, from Ken French's data library.

Usage:
    uv run python -m src.securities.factors

Zips are cached under data/raw/french/. Values are converted from percent to
decimals. French revises past factor values when he updates the files, so
these are a fixed-after-the-fact series: fine for risk adjustment and beta,
not a point-in-time record of what was published when.

Writes `ff_factors_daily`, `ff_factors_monthly` and `ff12_sic_ranges`.
"""

import io
import re
import zipfile
from pathlib import Path

import polars as pl
import requests

from src.securities import db

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
RAW_DIR = Path("data/raw/french")
FILES = {
    "ff5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom_daily": "F-F_Momentum_Factor_daily_CSV.zip",
    "ff5_monthly": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom_monthly": "F-F_Momentum_Factor_CSV.zip",
    "ff12": "Siccodes12.zip",
}


def fetch(name: str) -> bytes:
    path = RAW_DIR / FILES[name]
    if not path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        resp = requests.get(BASE + FILES[name], timeout=60)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as zf:
        return zf.read(zf.namelist()[0])


def parse_french_csv(raw: bytes, date_digits: int) -> pl.DataFrame:
    """French's CSVs have a text preamble, then a header row starting with a
    comma, then rows keyed by YYYYMMDD (daily) or YYYYMM (monthly). Monthly
    files append an annual section after a blank line; it's ignored because
    its keys have 4 digits."""
    header, rows = None, []
    for line in raw.decode("latin-1").splitlines():
        cells = [c.strip() for c in line.split(",")]
        if header is None and len(cells) > 1 and cells[0] == "" and cells[1]:
            header = [c.replace("-", "_") for c in cells[1:]]
            continue
        if header and re.fullmatch(rf"\d{{{date_digits}}}", cells[0]):
            rows.append([cells[0]] + [float(c) / 100 for c in cells[1:len(header) + 1]])
        elif header and rows and cells[0] == "" and len(cells) > 1 and cells[1]:
            break  # a second table starts
    df = pl.DataFrame(rows, schema=["key", *header], orient="row")
    fmt = "%Y%m%d" if date_digits == 8 else "%Y%m"
    return df.with_columns(pl.col("key").str.to_date(fmt).alias("date")).drop("key")


def parse_ff12(raw: bytes) -> pl.DataFrame:
    """Siccodes12.txt: an industry line ('1 NoDur  Consumer NonDurables ...')
    followed by SIC ranges ('0100-0999'). Industry 12 (Other) has no ranges:
    it is everything not listed."""
    rows, current = [], None
    for line in raw.decode("latin-1").splitlines():
        if m := re.match(r"^\s*(\d{1,2})\s+(\w+)\s+(.*)$", line):
            current = (int(m.group(1)), m.group(2), m.group(3).strip())
        elif (m := re.match(r"^\s*(\d{4})-(\d{4})", line)) and current:
            rows.append((*current, int(m.group(1)), int(m.group(2))))
    return pl.DataFrame(rows, schema=["ff12", "ff12_code", "ff12_name", "sic_from", "sic_to"], orient="row")


def build() -> None:
    daily = parse_french_csv(fetch("ff5_daily"), 8).join(parse_french_csv(fetch("mom_daily"), 8), on="date")
    monthly = parse_french_csv(fetch("ff5_monthly"), 6).join(parse_french_csv(fetch("mom_monthly"), 6), on="date")
    # Monthly rows are keyed to the first of the month; store the month end too.
    monthly = monthly.with_columns(pl.col("date").dt.month_end().alias("month_end"))
    ff12 = parse_ff12(fetch("ff12"))
    with db.connect() as con:
        con.execute("CREATE OR REPLACE TABLE ff_factors_daily AS SELECT * FROM daily")
        con.execute("CREATE OR REPLACE TABLE ff_factors_monthly AS SELECT * FROM monthly")
        con.execute("CREATE OR REPLACE TABLE ff12_sic_ranges AS SELECT * FROM ff12")
    print(f"ff_factors_daily: {len(daily):,} days {daily['date'].min()}..{daily['date'].max()}; "
          f"monthly: {len(monthly):,} months to {monthly['date'].max()}; ff12: {len(ff12)} SIC ranges, "
          f"{ff12['ff12'].n_unique()} industries")
    print(daily.columns)


if __name__ == "__main__":
    build()
