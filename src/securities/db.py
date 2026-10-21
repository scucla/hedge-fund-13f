"""The securities database: Milestone 2 tables, kept apart from the 13F
database because src.ingest.load rebuilds that one from scratch."""

import os
from pathlib import Path

import duckdb

from src.ingest.load import DB_PATH as FILINGS_DB_PATH

# overridable so a seed-only copy can be built side by side (holdout invariance check)
DB_PATH = Path(os.environ.get("SECURITIES_DB", "data/processed/securities.duckdb"))
# LAST_QUARTER is overridable so the quarterly forward test (src.live.quarter) can
# extend the pipeline to a newly filed quarter; it must be set before src modules import
FIRST_QUARTER = "2015Q1"
LAST_QUARTER = os.environ.get("LAST_QUARTER", "2026Q2")


def _quarters(start: str, end: str) -> list[str]:
    y, q = int(start[:4]), int(start[-1])
    out = []
    while f"{y}Q{q}" <= end:
        out.append(f"{y}Q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


QUARTERS = _quarters(FIRST_QUARTER, LAST_QUARTER)

# Both views of seed holdings (latest restated and as of the original filing;
# see src.securities.universe). Securities are mapped and priced for either,
# so a CUSIP that only an original filing listed still gets characteristics.
ALL_SEED_POSITIONS = "(SELECT * FROM seed_positions UNION ALL BY NAME SELECT * FROM seed_positions_pit)"


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    con.execute(f"ATTACH IF NOT EXISTS '{FILINGS_DB_PATH}' AS filings_db (READ_ONLY)")
    return con
