"""Point-in-time positions of the seed funds: the universe for Milestone 2.

Usage:
    uv run python -m src.securities.universe

Writes `seed_positions` (one row per fund CIK, quarter, CUSIP, put/call) to
the securities database, using the filing-selection rules in
src.ingest.positions as of today. Every mapped CIK is included, affiliates
too; de-duplicating parallel filers is left to the feature milestone.
"""

import duckdb
import polars as pl

from src.ingest.load import DB_PATH as FILINGS_DB_PATH
from src.ingest.positions import fund_holdings
from src.ingest.seed_ciks import SEED_PATH
from src.securities import db


def build() -> pl.DataFrame:
    seed = pl.read_csv(SEED_PATH, schema_overrides={"cik": pl.Utf8}).select("fund_name", "cik", "role")
    filings = duckdb.connect(str(FILINGS_DB_PATH), read_only=True)
    frames = []
    for fund, cik, role in seed.iter_rows():
        for quarter in db.QUARTERS:
            positions, used, _ = fund_holdings(filings, cik, quarter)
            if positions.is_empty():
                continue
            frames.append(positions.with_columns(
                pl.lit(fund).alias("fund_name"), pl.lit(cik).alias("cik"), pl.lit(role).alias("role"),
                pl.lit(quarter).alias("quarter"), pl.lit(used["known_at"].max()).alias("known_at"),
            ))
    out = pl.concat(frames).select(
        "fund_name", "cik", "role", "quarter", "known_at", "cusip", "put_call", "name_of_issuer",
        "shares", "principal", "value_usd", "weight",
    )
    con = db.connect()
    con.execute("CREATE OR REPLACE TABLE seed_positions AS SELECT * FROM out")
    return out


if __name__ == "__main__":
    out = build()
    print(f"seed_positions: {len(out):,} rows, {out['cik'].n_unique()} CIKs, {out['cusip'].n_unique():,} CUSIPs")
    print(out.group_by("quarter").agg(pl.col("cik").n_unique().alias("ciks"),
                                      (pl.col("value_usd").sum() / 1e12).round(2).alias("value_tn")).sort("quarter"))
