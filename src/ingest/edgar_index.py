"""EDGAR quarterly full-index: every 13F filing with filer name, CIK and date.

Usage:
    uv run python -m src.ingest.edgar_index --start 1999 --end 2026

Downloads form.gz for each quarter into data/raw/edgar/full-index/ (cached)
and writes the 13F rows to data/interim/edgar_13f_index.parquet. Used to
discover which CIKs a fund has filed under, including old entity names.
"""

import argparse
import gzip
from datetime import date
from pathlib import Path

import polars as pl

from src.ingest import sec

RAW_DIR = Path("data/raw/edgar/full-index")
OUT_PATH = Path("data/interim/edgar_13f_index.parquet")


def download_quarter(year: int, q: int, raw_dir: Path = RAW_DIR) -> Path | None:
    target = raw_dir / f"{year}QTR{q}_form.gz"
    if target.exists():
        return target
    if date(year, 3 * q - 2, 1) > date.today():
        return None
    raw_dir.mkdir(parents=True, exist_ok=True)
    data = sec.get(f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/form.gz").content
    current = (year, q) == (date.today().year, (date.today().month - 1) // 3 + 1)
    if not current:  # the current quarter's index is still growing; don't cache it
        partial = target.with_suffix(".part")
        partial.write_bytes(data)
        partial.rename(target)
        return target
    tmp = raw_dir / f"{year}QTR{q}_form.current.gz"
    tmp.write_bytes(data)
    return tmp


def parse_form_index(path: Path) -> pl.DataFrame:
    """Parse the 13F rows of a form.idx file.

    The file looks fixed-width, but in older years the header doesn't line up
    with the data. 13F form types never contain spaces, so split instead: form
    type first, then CIK, date and file name from the right; the company name
    is whatever is in between."""
    rows = []
    with gzip.open(path, "rt", encoding="latin-1") as f:
        for line in f:
            if not line.startswith("13F"):
                continue
            tokens = line.split()
            form, (cik, filed, fname) = tokens[0], tokens[-3:]
            rows.append((form, " ".join(tokens[1:-3]), cik.zfill(10), filed, fname))
    return pl.DataFrame(
        rows, schema=["form", "company_name", "cik", "date_filed", "file_name"], orient="row"
    ).with_columns(pl.col("date_filed").str.to_date())


def build(start: int, end: int) -> pl.DataFrame:
    frames = []
    for year in range(start, end + 1):
        for q in range(1, 5):
            path = download_quarter(year, q)
            if path:
                frames.append(parse_form_index(path))
    df = pl.concat(frames).unique()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)
    print(f"{len(df):,} 13F index rows, {df['cik'].n_unique():,} filers -> {OUT_PATH}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=1999)
    p.add_argument("--end", type=int, default=date.today().year)
    args = p.parse_args()
    build(args.start, args.end)
