"""Position features: one row per fund x security x put/call x quarter.

Usage:
    uv run python -m src.features.positions

Columns:
  weight         value / the fund's total reported 13F value (all rows, options
                 included: options are reported at the underlying's value)
  weight_long    value / total of the non-option rows; NULL for option rows
  rank, top10    rank by value within the fund-quarter (1 = largest), all rows
  share_change   split-adjusted change in shares (or bond principal) vs the
                 previous quarter; NULL for new positions and in the first
                 quarter of data
  is_new         not held (same put/call) in the previous quarter
  age_q          consecutive quarters held, including this one
  age_censored   the run starts in the first quarter of data, so age_q is a
                 lower bound
  crowding       number of *other* seed funds holding the same security long
  known_at       the fund-quarter's filing date (holdings are public then)

Splits are detected from the data, not a corporate-actions feed: for each
stock and quarter, compare every fund's reported share ratio (t / t-1) with
its value-implied ratio (value / split-adjusted price). Funds that just held
on show the split ratio; the median across funds is rounded to a standard
split ratio. Writes `stock_splits` and `positions`.
"""

import math

import polars as pl

from src.securities import db

SPLIT_MIN_LOG = math.log(1.25)   # ignore median ratios closer to 1 than a 5-for-4
SPLIT_MAX_LOG_ERROR = 0.012      # real splits land within ~0.5%; spin-offs and special dividends drift further
STANDARD_RATIOS = sorted({1.25, 1.5, *range(2, 51), 100})


def snap_ratio(r: float) -> float | None:
    """Nearest standard split ratio (or its reciprocal) in log space, or None."""
    if r is None or r <= 0:
        return None
    candidates = STANDARD_RATIOS + [1 / x for x in STANDARD_RATIOS]
    best = min(candidates, key=lambda c: abs(math.log(r / c)))
    return float(best) if abs(math.log(r / best)) <= SPLIT_MAX_LOG_ERROR else None


QIDX = "(left({q}, 4)::INT * 4 + right({q}, 1)::INT)"


def build(con) -> None:
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE h AS
        SELECT h.*, {QIDX.format(q='h.quarter')} AS q_idx,
               coalesce(m.symbol, h.cusip) AS security_key,
               m.symbol, m.security_class, m.issuer_cik,
               CASE WHEN h.shares > 0 THEN h.shares ELSE h.principal END AS qty
        FROM fund_holdings h LEFT JOIN security_master m USING (cusip)
    """)

    # --- splits ---------------------------------------------------------------
    ratios = con.execute(f"""
        WITH px AS (SELECT symbol, {QIDX.format(q='quarter')} AS q_idx, close_q FROM returns_quarterly),
        cur AS (SELECT h.*, px.close_q FROM h JOIN px USING (symbol, q_idx)
                WHERE h.put_call = '' AND h.shares > 0 AND h.value_usd > 0 AND NOT h.shares_suspect)
        SELECT t.cusip, t.quarter, t.q_idx,
               median((t.shares / p.shares) / ((t.value_usd / t.close_q) / (p.value_usd / p.close_q))) AS median_ratio,
               count(*) AS n_funds
        FROM cur t JOIN cur p ON p.fund_id = t.fund_id AND p.cusip = t.cusip AND p.q_idx = t.q_idx - 1
        GROUP BY ALL
    """).pl()
    splits = (
        ratios.filter(pl.col("median_ratio").log().abs() >= SPLIT_MIN_LOG)
        .with_columns(pl.col("median_ratio").map_elements(snap_ratio, return_dtype=pl.Float64).alias("split_ratio"))
    )
    con.execute("CREATE OR REPLACE TABLE stock_splits AS SELECT * FROM splits")

    # --- positions ------------------------------------------------------------
    con.execute(f"""
        CREATE OR REPLACE TABLE positions AS
        WITH t AS (
            SELECT h.*,
                   sum(value_usd) OVER (PARTITION BY fund_id, quarter) AS fund_value,
                   sum(value_usd) FILTER (WHERE put_call = '') OVER (PARTITION BY fund_id, quarter) AS fund_long_value,
                   min(q_idx) OVER () AS first_q_idx
            FROM h
        ),
        prev AS (SELECT fund_id, cusip, put_call, q_idx + 1 AS q_idx, qty AS prev_qty,
                        shares_suspect AS prev_suspect FROM h),
        runs AS (
            -- gaps-and-islands: positions held in consecutive quarters share a run id
            SELECT fund_id, cusip, put_call, q_idx,
                   q_idx - row_number() OVER (PARTITION BY fund_id, cusip, put_call ORDER BY q_idx) AS run_id
            FROM h
        ),
        ages AS (
            SELECT fund_id, cusip, put_call, q_idx,
                   row_number() OVER (PARTITION BY fund_id, cusip, put_call, run_id ORDER BY q_idx) AS age_q,
                   min(q_idx) OVER (PARTITION BY fund_id, cusip, put_call, run_id) AS run_start
            FROM runs
        ),
        crowd AS (
            SELECT quarter, security_key, count(DISTINCT fund_id) AS n_funds
            FROM h WHERE put_call = '' GROUP BY ALL
        )
        SELECT
            t.fund_id, t.fund_name, t.quarter, t.cusip, t.put_call, t.symbol, t.security_key,
            t.security_class, t.issuer_cik, t.name_of_issuer,
            t.shares, t.principal, t.value_usd,
            t.value_usd / t.fund_value AS weight,
            CASE WHEN t.put_call = '' THEN t.value_usd / t.fund_long_value END AS weight_long,
            row_number() OVER (PARTITION BY t.fund_id, t.quarter ORDER BY t.value_usd DESC, t.cusip, t.put_call) AS rank,
            p.prev_qty IS NULL AND t.q_idx > t.first_q_idx AS is_new,
            coalesce(s.split_ratio, 1.0) AS split_ratio,
            -- no share change from a share count flagged as suspect (see universe.fix_unit_errors)
            CASE WHEN p.prev_qty > 0 AND NOT t.shares_suspect AND NOT p.prev_suspect
                 THEN t.qty / (p.prev_qty * coalesce(s.split_ratio, 1.0)) - 1 END AS share_change,
            t.shares_suspect,
            a.age_q,
            a.run_start = t.first_q_idx AS age_censored,
            CASE WHEN t.put_call = '' THEN c.n_funds - 1 END AS crowding,
            t.put_call <> '' AS is_option,
            t.known_at
        FROM t
        LEFT JOIN prev p USING (fund_id, cusip, put_call, q_idx)
        LEFT JOIN ages a USING (fund_id, cusip, put_call, q_idx)
        LEFT JOIN crowd c USING (quarter, security_key)
        LEFT JOIN stock_splits s ON s.cusip = t.cusip AND s.quarter = t.quarter AND s.split_ratio IS NOT NULL
    """)
    con.execute("ALTER TABLE positions ADD COLUMN top10 BOOLEAN")
    con.execute("UPDATE positions SET top10 = rank <= 10")


if __name__ == "__main__":
    with db.connect() as con:
        build(con)
        print(con.execute("SELECT count(*), count(DISTINCT fund_id), count(DISTINCT quarter) FROM positions").fetchone())
        print("Detected splits:")
        print(con.execute("""
            SELECT any_value(m.symbol) AS symbol, s.quarter, round(any_value(s.median_ratio), 3) AS median_ratio,
                   any_value(s.split_ratio) AS split_ratio, max(s.n_funds) AS n_funds
            FROM stock_splits s JOIN security_master m USING (cusip)
            GROUP BY s.cusip, s.quarter ORDER BY n_funds DESC LIMIT 25
        """).pl())
