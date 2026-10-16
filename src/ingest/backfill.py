"""Backfill 13F holdings for the seed funds only, one data set file at a time.

Usage:
    uv run python -m src.ingest.backfill --start 2015Q1 --end 2025Q2
    uv run python -m src.ingest.backfill --holdout --start 2020Q1 --end 2025Q2

For each SEC data set file the holdings quarters need: download it, load only
the rows filed by a seed-fund CIK (every CIK in data/reference/seed_cik_map.csv,
predecessors included) into data/interim/13f_history.duckdb, commit, then delete
the zip. `loaded_files` records each finished file, so an interrupted run
resumes where it stopped. With --holdout the same is done for the hand-labelled
holdout funds (data/reference/holdout_funds.csv) into a separate database,
data/interim/13f_holdout_history.duckdb, so they can never mix with the seed
funds' history. Before each file, free disk space is checked.

Same tables and rules as src.ingest.load (values normalized to dollars by
filing date, notices dropped, known_at = filing date). Filings that arrived
late, inside the four recent files, live in data/interim/13f.duckdb; readers
combine both (see src.securities.universe).
"""

import argparse
import shutil
from pathlib import Path

import duckdb
import polars as pl

from src.ingest.download import dataset_for_quarter, download, list_datasets, quarters_between
from src.ingest.load import HOLDINGS_SQL, SCHEMA, load_zip

HIST_DB = Path("data/interim/13f_history.duckdb")
HOLDOUT_DB = Path("data/interim/13f_holdout_history.duckdb")
HOLDOUT_PATH = Path("data/reference/holdout_funds.csv")
SEED_MAP = Path("data/reference/seed_cik_map.csv")
MIN_FREE_GB = 3.0


def seed_ciks() -> list[str]:
    return sorted({c.zfill(10) for c in pl.read_csv(SEED_MAP, schema_overrides={"cik": pl.Utf8})["cik"]})


def holdout_ciks() -> list[str]:
    return sorted({c.zfill(10) for c in pl.read_csv(HOLDOUT_PATH, schema_overrides={"cik": pl.Utf8})["cik"]})


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def open_db(path: Path = HIST_DB) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    have = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    if "filings" not in have:
        con.execute(SCHEMA)
        con.execute("""CREATE TABLE loaded_files (source_file VARCHAR PRIMARY KEY, loaded_at TIMESTAMP,
                       n_filings INTEGER, n_rows BIGINT)""")
        # empty holdings table with the right types
        con.execute(f"CREATE TABLE holdings AS {HOLDINGS_SQL.split('INSERT INTO holdings', 1)[1]} LIMIT 0",
                    {"source_file": ""})
    return con


def build(start: str, end: str, holdout: bool = False) -> None:
    datasets = list_datasets()
    files = []
    for y, q in quarters_between(start, end):
        ds = dataset_for_quarter(datasets, y, q)
        if ds not in files:
            files.append(ds)
    ciks = holdout_ciks() if holdout else seed_ciks()
    db_path = HOLDOUT_DB if holdout else HIST_DB
    con = open_db(db_path)
    done = {r[0] for r in con.execute("SELECT source_file FROM loaded_files").fetchall()}
    print(f"{len(files)} data set files for {start}..{end}; {len(done & {f.filename for f in files})} already loaded; "
          f"{len(ciks)} {'holdout' if holdout else 'seed'} CIKs -> {db_path}")
    for i, ds in enumerate(files, 1):
        if ds.filename in done:
            continue
        if free_gb(HIST_DB.parent) < MIN_FREE_GB:
            raise SystemExit(f"Stopping: only {free_gb(HIST_DB.parent):.1f} GB free (< {MIN_FREE_GB} GB). "
                             f"Resume after freeing space; finished files are kept.")
        zip_path = download(ds)
        try:
            con.execute("BEGIN")
            load_zip(con, zip_path, HIST_DB.parent, ciks=ciks)
            con.execute(HOLDINGS_SQL, {"source_file": ds.filename})
            n_f, n_r = con.execute("""
                SELECT count(*), (SELECT count(*) FROM infotable_raw i JOIN filings f USING (accession_number)
                                  WHERE f.source_file = $f)
                FROM filings WHERE source_file = $f""", {"f": ds.filename}).fetchone()
            con.execute("INSERT INTO loaded_files VALUES (?, now(), ?, ?)", [ds.filename, n_f, n_r])
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        zip_path.unlink()  # loaded and committed: the zip is no longer needed
        print(f"[{i}/{len(files)}] {ds.filename}: {n_f} filings, {n_r:,} rows; "
              f"{free_gb(HIST_DB.parent):.1f} GB free")
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2015Q1")
    ap.add_argument("--end", default="2025Q2")
    ap.add_argument("--holdout", action="store_true", help="load the holdout funds into their own database")
    args = ap.parse_args()
    build(args.start, args.end, args.holdout)
