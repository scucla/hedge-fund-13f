"""The securities database: Milestone 2 tables, kept apart from the 13F
database because src.ingest.load rebuilds that one from scratch."""

from pathlib import Path

import duckdb

from src.ingest.load import DB_PATH as FILINGS_DB_PATH

DB_PATH = Path("data/processed/securities.duckdb")
QUARTERS = ["2025Q3", "2025Q4", "2026Q1", "2026Q2"]


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    con.execute(f"ATTACH IF NOT EXISTS '{FILINGS_DB_PATH}' AS filings_db (READ_ONLY)")
    return con
