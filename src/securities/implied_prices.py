"""13F-implied quarter-end prices for stocks Yahoo doesn't price.

Usage:
    uv run python -m src.securities.implied_prices

Every 13F row reports a value and a share count as of the quarter end, so
value / shares is the filer's quarter-end price. For a CUSIP held by several
seed funds the price is the **median across funds**; `px_disagree` flags
quarters where the funds' prices are more than PX_DISAGREE apart (max / min - 1),
which catches stale marks, unit errors and share-class mix-ups.

Checked against stocks Yahoo does price: quarterly returns from implied prices
match Yahoo's with correlation 0.95 and a median absolute error of 0.16
percentage points (docs/decisions.md). They are price-only (no dividends).

Splits: the existing `stock_splits` table needs a Yahoo price, so it misses
exactly these stocks. Here a split at q is the median per-fund share ratio
(shares_q / shares_q-1, funds that held both quarters) snapped to a standard
ratio, accepted only when it makes the price change smaller, i.e. the
split-adjusted return is more plausible than the raw one. A fund that doubled
its position shows the same share ratio but no price halving, so it fails.

Units: small names with one or two filers escape the >= 3-filer unit fix in
universe.fix_unit_errors, so ~2% of implied prices (9% in 2022Q4, the first
quarter filed in dollars) are off by 1000x. A quarter-on-quarter ratio above
100x is rescaled by the nearest power of 1000 (`unit_power`). Moves beyond 3x
(or below 1/3) seen by fewer than 3 funds are flagged `big_move_few_funds`.

Seed cohort only, like `crowding`: holdout funds never change a seed number.

Writes `implied_prices` (cusip, quarter), `implied_returns` (cusip,
quarter -> next quarter) and `implied_splits` (cusip, quarter of the split).
"""

import math

import polars as pl

from src.features.positions import QIDX, SPLIT_MIN_LOG, snap_ratio
from src.securities import db

PX_DISAGREE = 0.05
UNIT_ERROR_LOG10 = 2.0   # same threshold idea as universe.UNIT_ERROR_LOG: a ~1000x jump, not a return


def build(con) -> None:
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ip_rows AS
        SELECT fund_id, cusip, quarter, {QIDX.format(q='quarter')} AS q_idx, known_at,
               shares, value_usd / shares AS px
        FROM fund_holdings
        WHERE cohort = 'seed' AND put_call = '' AND shares > 0 AND value_usd > 0
          AND NOT coalesce(shares_suspect, false)
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE implied_prices AS
        SELECT cusip, quarter, q_idx,
               median(px) AS px, min(px) AS px_min, max(px) AS px_max,
               count(DISTINCT fund_id) AS n_funds,
               max(px) / min(px) - 1 > {PX_DISAGREE} AS px_disagree,
               max(known_at) AS known_at       -- the last filing that went into the median
        FROM ip_rows GROUP BY ALL
    """)

    # share ratios of funds that held the stock in both quarters
    ratios = con.execute("""
        SELECT t.cusip, t.quarter, t.q_idx, median(t.shares / p.shares) AS median_share_ratio,
               count(*) AS n_funds
        FROM ip_rows t JOIN ip_rows p ON p.fund_id = t.fund_id AND p.cusip = t.cusip AND p.q_idx = t.q_idx - 1
        GROUP BY ALL
    """).pl()
    cand = (
        ratios.filter(pl.col("median_share_ratio").log().abs() >= SPLIT_MIN_LOG)
        .with_columns(pl.col("median_share_ratio").map_elements(snap_ratio, return_dtype=pl.Float64).alias("split_ratio"))
        .filter(pl.col("split_ratio").is_not_null())
    )
    con.execute("CREATE OR REPLACE TEMP TABLE split_cand AS SELECT * FROM cand")

    con.execute(f"""
        CREATE OR REPLACE TABLE implied_returns AS
        WITH pair AS (
            SELECT a.cusip, a.quarter, a.px AS px0, b.px AS px1, b.quarter AS next_quarter,
                   a.px_disagree OR b.px_disagree AS px_disagree,
                   least(a.n_funds, b.n_funds) AS n_funds, b.known_at,
                   -- Yahoo-detected split first, then the 13F-only one
                   coalesce(ys.split_ratio, sc.split_ratio) AS cand_split
            FROM implied_prices a
            JOIN implied_prices b ON b.cusip = a.cusip AND b.q_idx = a.q_idx + 1
            LEFT JOIN stock_splits ys ON ys.cusip = a.cusip AND ys.quarter = b.quarter AND ys.split_ratio IS NOT NULL
            LEFT JOIN split_cand sc ON sc.cusip = a.cusip AND sc.quarter = b.quarter
        ),
        split AS (
            SELECT *, CASE WHEN cand_split IS NOT NULL AND abs(ln(px1 * cand_split / px0)) < abs(ln(px1 / px0))
                           THEN cand_split ELSE 1.0 END AS split_ratio
            FROM pair
        ),
        units AS (
            -- a quarter-on-quarter price ratio near 1000x is a units error in one of the
            -- two quarters (thousands vs dollars) that the >= 3-filer fix upstream could
            -- not see; no real quarterly return is that large
            SELECT *, CASE WHEN abs(log10(px1 * split_ratio / px0)) > {UNIT_ERROR_LOG10}
                           THEN round(log10(px1 * split_ratio / px0) / 3)::INT ELSE 0 END AS unit_power
            FROM split
        )
        SELECT cusip, quarter, next_quarter, px0, px1, px_disagree, n_funds, known_at,
               split_ratio, unit_power,
               px1 * split_ratio / pow(1000, unit_power) / px0 - 1 AS ret,
               -- kept, but flagged: against Yahoo, ~80% of these are real moves and ~20% a missed
               -- (reverse) split; dropping them would bias the label against extreme returns
               n_funds < 3 AND abs(ln(px1 * split_ratio / pow(1000, unit_power) / px0)) > ln(3) AS big_move_few_funds
        FROM units
    """)
    # 13F-only splits, for share-change labels of stocks Yahoo doesn't price (position_moves)
    con.execute("""
        CREATE OR REPLACE TABLE implied_splits AS
        SELECT cusip, next_quarter AS quarter, split_ratio FROM implied_returns WHERE split_ratio <> 1
    """)

def validation(con) -> pl.DataFrame:
    """Implied vs Yahoo quarterly returns on stocks both cover."""
    cols = {r[0] for r in con.execute("DESCRIBE stock_quarter").fetchall()}
    yahoo = "ret_next_q_yahoo" if "ret_next_q_yahoo" in cols else "ret_next_q"   # labels.fill keeps the Yahoo label there
    return con.execute(f"""
        SELECT count(*) AS n, corr(r.ret, s.ret_next_q) AS corr,
               corr(r.ret, s.ret_next_q) FILTER (WHERE NOT r.big_move_few_funds) AS corr_unflagged,
               median(abs(r.ret - s.ret_next_q)) AS median_abs_err,
               avg((abs(r.ret - s.ret_next_q) > 0.10)::INT) AS share_off_10pp,
               avg(r.px_disagree::INT) AS share_disagree,
               median(abs(r.ret - s.ret_next_q)) FILTER (WHERE r.px_disagree) AS median_abs_err_disagree,
               median(abs(r.ret - s.ret_next_q)) FILTER (WHERE NOT r.px_disagree) AS median_abs_err_agree
        FROM implied_returns r JOIN (SELECT cusip, quarter, security_class, {yahoo} AS ret_next_q FROM stock_quarter) s
             USING (cusip, quarter)
        WHERE s.ret_next_q IS NOT NULL AND s.security_class = 'common'
    """).pl()


if __name__ == "__main__":
    with db.connect() as con:
        build(con)
        print(con.execute("""SELECT count(*) AS rows, avg(px_disagree::INT) AS disagree, avg(n_funds) AS funds
                             FROM implied_prices""").pl())
        print(con.execute("""SELECT count(*) AS pairs, sum((split_ratio <> 1)::INT) AS splits_applied
                             FROM implied_returns""").pl())
        print(validation(con))
