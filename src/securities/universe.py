"""Point-in-time positions of the seed funds: the universe for Milestone 2.

Usage:
    uv run python -m src.securities.universe

Writes two views of the same holdings (one row per fund CIK, quarter, CUSIP,
put/call) to the securities database, using the filing-selection rules in
src.ingest.positions:
  seed_positions      as of today: the latest restated truth, including
                      amendments filed months or years after the quarter
  seed_positions_pit  as of the filer-quarter's original 13F-HR filing date:
                      what the public could see when the quarter was first
                      reported. Features are built from this one, so a later
                      amendment never leaks into an earlier quarter.
`known_at` is the latest filing used in each view.

Both views also hold the hand-labelled holdout funds (data/reference/
holdout_funds.csv, `cohort` = 'holdout', from HOLDOUT_FIRST_QUARTER on) so
their features are computed exactly like the seed funds'. Everything that
defines a cross-section (crowding, factor z-scores, the active-share
benchmark) uses seed funds only, and models never train on holdout rows.

Seed-fund filings come from both the
history backfill (data/interim/13f_history.duckdb, 2015 on) and the current
full database (data/interim/13f.duckdb, the latest four filing windows,
which also holds late filings and amendments for older quarters). Every mapped CIK is included, affiliates
too; de-duplicating parallel filers is left to the feature milestone.
"""

import duckdb
import polars as pl

from src.ingest.backfill import HIST_DB, HOLDOUT_DB, HOLDOUT_PATH
from src.ingest.load import DB_PATH as FILINGS_DB_PATH
from src.ingest.positions import fund_holdings, normalize_cik
from src.ingest.seed_ciks import SEED_PATH
from src.securities import db


HOLDOUT_FIRST_QUARTER = "2020Q1"   # their history backfill starts here (12 quarters before the 2023 test years)


def funds() -> pl.DataFrame:
    """fund_name, cik, role, cohort for the seed funds and the holdout funds."""
    seed = (pl.read_csv(SEED_PATH, schema_overrides={"cik": pl.Utf8}).select("fund_name", "cik", "role")
            .with_columns(pl.lit("seed").alias("cohort")))
    hold = (pl.read_csv(HOLDOUT_PATH, schema_overrides={"cik": pl.Utf8})
            .select("fund_name", "cik", pl.lit("primary").alias("role"), pl.lit("holdout").alias("cohort")))
    overlap = set(seed["cik"].str.zfill(10)) & set(hold["cik"].str.zfill(10))
    if overlap:
        raise ValueError(f"holdout CIKs also in the seed list: {sorted(overlap)}")
    return pl.concat([seed, hold])


def seed_filings_db(ciks: list[str]) -> duckdb.DuckDBPyConnection:
    """In-memory `filings` and `holdings` for the seed CIKs, combining the
    history backfill and the current full database. The two cover disjoint
    filing-date windows, so an accession appears in only one of them."""
    con = duckdb.connect()
    con.execute(f"ATTACH '{FILINGS_DB_PATH}' AS cur (READ_ONLY)")
    sources = ["cur"]
    if HIST_DB.exists():
        con.execute(f"ATTACH '{HIST_DB}' AS hist (READ_ONLY)")
        sources.append("hist")
    if HOLDOUT_DB.exists():
        con.execute(f"ATTACH '{HOLDOUT_DB}' AS hold (READ_ONLY)")
        sources.append("hold")
    con.execute("CREATE TEMP TABLE seed_ciks AS SELECT unnest($c) AS cik", {"c": ciks})
    for t in ("filings", "holdings"):
        union = " UNION ALL ".join(
            f"SELECT x.* FROM {s}.{t} x JOIN {s}.filings f USING (accession_number) "
            f"WHERE f.cik IN (SELECT cik FROM seed_ciks)" if t == "holdings" else
            f"SELECT * FROM {s}.filings WHERE cik IN (SELECT cik FROM seed_ciks)" for s in sources)
        con.execute(f"CREATE TEMP TABLE {t} AS {union}")
    dup = con.execute("SELECT count(*) - count(DISTINCT accession_number) FROM filings").fetchone()[0]
    if dup:
        raise RuntimeError(f"{dup} filings appear in both databases; the filing windows should be disjoint")
    return con


UNIT_ERROR_LOG = 2.5   # |log10 of median price ratio| above this (~316x) = a unit error


def fix_unit_errors(positions: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Find values off by 1000x and rescale them, row by row.

    Some filers report dollars where the form asks for thousands (or the
    reverse after 2023). Usually a whole filing is off, but some filings mix
    units (seen: an amended filing with most rows in thousands and one row in
    dollars). Every stock position implies a price per share (value / shares);
    for each CUSIP and quarter the median implied price across all seed filers
    is the consensus.
    - Filing-level error: the filing's median row ratio is ~1000x (or 1/1000)
      with >= 5 rows compared. All its rows are rescaled by that power, except
      rows whose own ratio shows they are already right (mixed-unit filings).
    - A lone deviating row in an otherwise correct filing is NOT rescaled:
      checked against market prices, ~90% of those had a correct value and a
      share count off by 1000 (shares reported in thousands). The row is
      flagged `shares_suspect` instead, so share-based features skip it.
    Returns (fixed positions, per-filing summary of corrections)."""
    px = (positions.filter((pl.col("put_call") == "") & (pl.col("shares") > 0) & (pl.col("value_usd") > 0))
          .with_columns((pl.col("value_usd") / pl.col("shares")).log10().alias("lp")))
    # consensus from seed filers only, so adding holdout funds can't move a seed correction
    seed_px = px.filter(pl.col("cohort") == "seed") if "cohort" in px.columns else px
    consensus = (seed_px.group_by("cusip", "quarter").agg(pl.col("lp").median().alias("lp_cons"),
                                                     pl.col("cik").n_unique().alias("n_filers"))
                 .filter(pl.col("n_filers") >= 3))
    rows = (px.join(consensus, on=["cusip", "quarter"])
            .with_columns((pl.col("lp") - pl.col("lp_cons")).alias("log_ratio"))
            .select("cik", "quarter", "cusip", "put_call", "log_ratio"))
    row_power = rows.with_columns(
        pl.when(pl.col("log_ratio").abs() > UNIT_ERROR_LOG).then((pl.col("log_ratio") / 3).round())
        .otherwise(0).cast(pl.Int32).alias("row_power")).select("cik", "quarter", "cusip", "put_call", "row_power")
    filing = (rows.group_by("cik", "quarter").agg(pl.col("log_ratio").median().alias("log_ratio"),
                                                  pl.len().alias("n_compared"))
              .with_columns(pl.when((pl.col("log_ratio").abs() > UNIT_ERROR_LOG) & (pl.col("n_compared") >= 5))
                            .then((pl.col("log_ratio") / 3).round()).otherwise(0).cast(pl.Int32)
                            .alias("filing_power")))
    fixed = (positions
             .join(row_power, on=["cik", "quarter", "cusip", "put_call"], how="left")
             .join(filing.select("cik", "quarter", "filing_power"), on=["cik", "quarter"], how="left")
             .with_columns(pl.col("filing_power").fill_null(0))
             .with_columns(
                 # filing-level fix, unless the row's own evidence says it is already right
                 pl.when(pl.col("filing_power") != 0)
                 .then(pl.when(pl.col("row_power").is_null() | (pl.col("row_power") == pl.col("filing_power")))
                       .then(pl.col("filing_power")).otherwise(0))
                 .otherwise(0).cast(pl.Int32).alias("unit_fix_power"),
                 # a lone deviating row in a correct filing: suspect share count
                 ((pl.col("filing_power") == 0) & (pl.col("row_power").fill_null(0) != 0)).alias("shares_suspect"))
             .with_columns((pl.col("value_usd") / (1000.0 ** pl.col("unit_fix_power"))).alias("value_usd"))
             .drop("row_power", "filing_power"))
    fixes = (fixed.filter(pl.col("unit_fix_power") != 0)
             .group_by("cik", "quarter").agg(pl.len().alias("rows_fixed"),
                                             pl.col("unit_fix_power").min().alias("min_power"),
                                             pl.col("unit_fix_power").max().alias("max_power"))
             .join(filing.select("cik", "quarter", "log_ratio", "n_compared", pl.col("filing_power").alias("power_of_1000")),
                   on=["cik", "quarter"], how="left"))
    return fixed, fixes


def original_filing_dates(filings) -> dict[tuple[str, str], object]:
    """(cik, quarter) -> date of the first original 13F-HR for that quarter."""
    return {(c, q): d for c, q, d in filings.execute("""
        SELECT cik, quarter, min(known_at) FROM filings WHERE submission_type = '13F-HR' GROUP BY ALL
    """).fetchall()}


def collect(seed: pl.DataFrame, filings, as_of: dict | None) -> pl.DataFrame:
    """Holdings of every seed CIK x quarter, as of today (as_of None) or as of
    the date that `as_of` gives for each (cik, quarter)."""
    frames = []
    for fund, cik, role, cohort in seed.iter_rows():
        for quarter in db.QUARTERS:
            if cohort == "holdout" and quarter < HOLDOUT_FIRST_QUARTER:
                continue
            when = None if as_of is None else as_of.get((normalize_cik(cik), quarter))
            if as_of is not None and when is None:
                continue   # no original filing: nothing was public at the first report
            positions, used, _ = fund_holdings(filings, cik, quarter, when)
            if positions.is_empty():
                continue
            frames.append(positions.with_columns(
                pl.lit(fund).alias("fund_name"), pl.lit(cik).alias("cik"), pl.lit(role).alias("role"),
                pl.lit(cohort).alias("cohort"),
                pl.lit(quarter).alias("quarter"), pl.lit(used["known_at"].max()).alias("known_at"),
            ))
    return pl.concat(frames).select(
        "fund_name", "cik", "role", "cohort", "quarter", "known_at", "cusip", "put_call", "name_of_issuer",
        "shares", "principal", "value_usd", "weight",
    )


def build() -> pl.DataFrame:
    seed = funds()
    filings = seed_filings_db(seed["cik"].unique().to_list())
    con = db.connect()
    out = None
    for table, as_of in [("seed_positions", None), ("seed_positions_pit", original_filing_dates(filings))]:
        view, fixes = fix_unit_errors(collect(seed, filings, as_of))
        if len(fixes):
            print(f"{table}: unit errors corrected (value divided by 1000^power):")
            with pl.Config(tbl_rows=50):
                print(fixes.join(view.select("cik", "fund_name").unique(), on="cik").sort("fund_name", "quarter"))
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM view")
        con.execute(f"CREATE OR REPLACE TABLE {table.replace('seed_positions', 'unit_fixes')} AS SELECT * FROM fixes")
        out = out if out is not None else view
    return out


if __name__ == "__main__":
    out = build()
    print(f"seed_positions: {len(out):,} rows, {out['cik'].n_unique()} CIKs, {out['cusip'].n_unique():,} CUSIPs")
    print(out.group_by("quarter").agg(pl.col("cik").n_unique().alias("ciks"),
                                      (pl.col("value_usd").sum() / 1e12).round(2).alias("value_tn")).sort("quarter"))
