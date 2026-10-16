"""Task C1 targets: what each fund did with each long position at the next quarter.

Usage:
    uv run python -m src.features.position_moves

One row per (fund, CUSIP, quarter q) for long, non-option positions, when the
fund also filed for q+1 (a fund that stopped filing did not exit anything):
  action        'exit' (not held at q+1), 'add' (> +MOVE_THRESHOLD split-adjusted
                shares), 'trim' (< -MOVE_THRESHOLD), 'hold'
  forced_exit   the exit coincides with the stock being delisted in q+1 for an
                acquisition or a performance delisting (delisting_events): not a
                decision of the fund, so C1 should drop or separate these
  known_at      the fund's q+1 13F filing date (the label is public then)

**Exit never depends on a price.** It is decided from the fund's holdings alone
(held at q, filed at q+1, not held at q+1); `forced_exit` comes from SEC
delisting filings, not from a price series ending. Nothing here reads
prices_daily, returns_quarterly or stock_quarter, and a test rebuilds the
table with those tables empty and checks the exits are unchanged. Add/trim use
the split ratio from `stock_splits` (Yahoo-detected) or, for stocks Yahoo
doesn't price, `implied_splits` (13F share counts only).
"""

from src.features.positions import QIDX
from src.securities import db

MOVE_THRESHOLD = 0.10   # plan §5 Task C: tune later; first a fixed 10% band


def build(con) -> None:
    con.execute(f"""
        CREATE OR REPLACE TABLE position_moves AS
        WITH h AS (
            SELECT fund_id, cohort, cusip, quarter, {QIDX.format(q='quarter')} AS q_idx, known_at,
                   CASE WHEN shares > 0 THEN shares ELSE principal END AS qty,
                   coalesce(shares_suspect, false) AS shares_suspect
            FROM fund_holdings WHERE put_call = ''
        ),
        -- every fund-quarter that was filed, including empty books (everything exited)
        filed AS (
            SELECT fund_id, {QIDX.format(q='quarter')} AS q_idx, max(known_at) AS known_at
            FROM (SELECT fund_id, quarter, known_at FROM fund_holdings
                  UNION ALL BY NAME SELECT fund_id, quarter, known_at FROM empty_books)
            GROUP BY ALL
        ),
        splits AS (
            SELECT cusip, quarter, split_ratio FROM stock_splits WHERE split_ratio IS NOT NULL
            UNION ALL
            SELECT cusip, quarter, split_ratio FROM implied_splits
            WHERE (cusip, quarter) NOT IN (SELECT (cusip, quarter) FROM stock_splits WHERE split_ratio IS NOT NULL)
        )
        SELECT t.fund_id, t.cohort, t.cusip, t.quarter, f.known_at,
               CASE WHEN n.cusip IS NULL THEN NULL
                    WHEN t.shares_suspect OR n.shares_suspect OR NOT (t.qty > 0) THEN NULL
                    ELSE n.qty / (t.qty * coalesce(sp.split_ratio, 1.0)) - 1 END AS share_change_next,
               CASE WHEN n.cusip IS NULL THEN 'exit'
                    WHEN t.shares_suspect OR n.shares_suspect OR NOT (t.qty > 0) THEN NULL
                    WHEN n.qty / (t.qty * coalesce(sp.split_ratio, 1.0)) - 1 > {MOVE_THRESHOLD} THEN 'add'
                    WHEN n.qty / (t.qty * coalesce(sp.split_ratio, 1.0)) - 1 < -{MOVE_THRESHOLD} THEN 'trim'
                    ELSE 'hold' END AS action,
               coalesce(n.cusip IS NULL AND d.exit_class IN ('cash_acquisition', 'stock_acquisition', 'performance'),
                        false) AS forced_exit
        FROM h t
        JOIN filed f ON f.fund_id = t.fund_id AND f.q_idx = t.q_idx + 1
        LEFT JOIN h n ON n.fund_id = t.fund_id AND n.cusip = t.cusip AND n.q_idx = t.q_idx + 1
        LEFT JOIN splits sp ON sp.cusip = t.cusip AND sp.quarter = n.quarter
        LEFT JOIN delisting_events d ON d.cusip = t.cusip AND d.quarter = t.quarter
    """)


if __name__ == "__main__":
    with db.connect() as con:
        build(con)
        print(con.execute("""
            SELECT action, count(*) AS n, sum(forced_exit::INT) AS forced
            FROM position_moves GROUP BY 1 ORDER BY 2 DESC
        """).pl())
