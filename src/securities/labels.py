"""Fill `stock_quarter.ret_next_q` beyond Yahoo, and say where every value came from.

Usage:
    uv run python -m src.securities.labels      # after stock_quarter, implied_prices, delistings

Order, first that exists:
  yahoo                Yahoo total return (unchanged from stock_quarter)
  delist_cash / delist_stock / delist_mixed / delist_last_13f_price / delist_shumway
                       the stock was delisted during q+1 (src.securities.delistings)
  13f_implied          price-only return from median 13F-implied prices at q and q+1
                       (src.securities.implied_prices); exists only if a seed fund
                       still holds the stock at q+1
  missing:<reason>     no return, with the reason

`ret_next_q_yahoo` keeps the Yahoo-only label, so every model can be rerun on it.
`label_known_at` is the date the filled return became public (the last 13F filing
behind an implied price, or the delisting). `label_flag` marks implied returns that
are less certain (`px_disagree`: funds' prices more than 5% apart;
`big_move_few_funds`).

The selection problem the user flagged: a 13F-implied return needs a holder at
q+1, so a stock that every seed fund sold has none. `bias_report` measures how
much that matters on stocks Yahoo does price, where the return is known either
way (docs/delisting_returns.md).
"""

from pathlib import Path

import polars as pl

from src.securities import db
from src.securities.delistings import NEXTQ

REPORT = Path("docs/delisting_returns.md")


def fill(con) -> None:
    cols = {r[0] for r in con.execute("DESCRIBE stock_quarter").fetchall()}
    if "ret_next_q_yahoo" not in cols:      # first run on a fresh stock_quarter
        con.execute("ALTER TABLE stock_quarter ADD COLUMN ret_next_q_yahoo DOUBLE")
        con.execute("UPDATE stock_quarter SET ret_next_q_yahoo = ret_next_q")
        for c in ("ret_source", "label_flag"):
            con.execute(f"ALTER TABLE stock_quarter ADD COLUMN {c} VARCHAR")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fill AS
        WITH sq AS (
            SELECT s.cusip, s.quarter, s.security_class, s.ret_next_q_yahoo,
                   s.label_known_at AS yahoo_known_at, {NEXTQ.format(q='s.quarter')} AS next_quarter
            FROM stock_quarter s
        ),
        seed_held AS (SELECT DISTINCT cusip, quarter FROM fund_holdings WHERE cohort = 'seed' AND put_call = ''),
        holdout_only AS (
            SELECT cusip, quarter FROM fund_holdings WHERE put_call = ''
            GROUP BY ALL HAVING bool_and(cohort = 'holdout')
        )
        SELECT sq.cusip, sq.quarter,
            CASE
                WHEN sq.ret_next_q_yahoo IS NOT NULL THEN sq.ret_next_q_yahoo
                WHEN d.ret_source IS NOT NULL THEN d.delist_ret
                WHEN r.ret IS NOT NULL THEN r.ret
            END AS ret_next_q,
            CASE
                WHEN sq.ret_next_q_yahoo IS NOT NULL THEN sq.yahoo_known_at
                WHEN d.ret_source IS NOT NULL THEN d.ret_known_at
                WHEN r.ret IS NOT NULL THEN r.known_at
            END AS label_known_at,
            CASE
                WHEN sq.ret_next_q_yahoo IS NOT NULL THEN 'yahoo'
                WHEN d.ret_source IS NOT NULL THEN d.ret_source
                WHEN r.ret IS NOT NULL THEN '13f_implied'
                WHEN sq.quarter = '{db.LAST_QUARTER}' THEN 'missing:no_next_quarter'
                WHEN sq.security_class IN ('etf', 'fund') THEN 'missing:etf_or_fund'
                WHEN ho.cusip IS NOT NULL THEN 'missing:holdout_only'
                WHEN d.exit_class IS NOT NULL THEN 'missing:' || d.exit_class
                    || CASE WHEN d.exit_subclass = 'spac' THEN '_spac' ELSE '' END
                WHEN sh.cusip IS NULL THEN 'missing:not_seed_held'
                ELSE 'missing:no_implied_price'
            END AS ret_source,
            CASE
                WHEN sq.ret_next_q_yahoo IS NOT NULL OR d.ret_source IS NOT NULL OR r.ret IS NULL THEN NULL
                WHEN r.big_move_few_funds THEN 'big_move_few_funds'
                WHEN r.px_disagree THEN 'px_disagree'
            END AS label_flag
        FROM sq
        LEFT JOIN delisting_events d USING (cusip, quarter)
        LEFT JOIN implied_returns r USING (cusip, quarter)
        LEFT JOIN holdout_only ho USING (cusip, quarter)
        LEFT JOIN seed_held sh USING (cusip, quarter)
    """)
    con.execute("""
        UPDATE stock_quarter s SET ret_next_q = f.ret_next_q, label_known_at = f.label_known_at,
                                   ret_source = f.ret_source, label_flag = f.label_flag
        FROM fill f WHERE f.cusip = s.cusip AND f.quarter = s.quarter
    """)


def coverage(con) -> pl.DataFrame:
    """Share of seed-fund long value (common + ADR) by label source and year."""
    return con.execute("""
        WITH v AS (SELECT cusip, quarter, sum(value_usd) AS val FROM fund_holdings
                   WHERE cohort = 'seed' AND put_call = '' GROUP BY ALL)
        SELECT left(s.quarter, 4) AS year,
               round(sum(v.val) FILTER (WHERE s.ret_source = 'yahoo') / sum(v.val), 3) AS yahoo,
               round(sum(v.val) FILTER (WHERE s.ret_source = '13f_implied') / sum(v.val), 3) AS implied,
               round(sum(v.val) FILTER (WHERE s.ret_source LIKE 'delist_%') / sum(v.val), 4) AS delist,
               round(sum(v.val) FILTER (WHERE s.ret_source LIKE 'missing:%') / sum(v.val), 4) AS missing,
               round(sum(v.val) FILTER (WHERE s.ret_next_q_yahoo IS NOT NULL) / sum(v.val), 3) AS yahoo_only_before
        FROM stock_quarter s JOIN v USING (cusip, quarter)
        WHERE s.security_class IN ('common', 'adr') AND s.quarter < (SELECT max(quarter) FROM stock_quarter)
        GROUP BY 1 ORDER BY 1
    """).pl()


def by_source(con) -> pl.DataFrame:
    return con.execute("""
        SELECT ret_source, count(*) AS rows, count(DISTINCT cusip) AS cusips,
               round(avg(ret_next_q), 4) AS mean_ret, round(median(ret_next_q), 4) AS median_ret,
               round(avg((label_flag IS NOT NULL)::INT), 3) AS flagged
        FROM stock_quarter WHERE security_class IN ('common', 'adr')
        GROUP BY 1 ORDER BY rows DESC
    """).pl()


def bias_report(con) -> pl.DataFrame:
    """On Yahoo-priced common stock, where the next-quarter return is known either way:
    returns of rows that a 13F-implied return could cover (some seed fund still holds
    the stock at q+1) vs rows it could not (every seed fund sold). The gap is the bias
    that conditioning on continued holding puts into the 13f_implied rows."""
    return con.execute(f"""
        WITH seed_held AS (SELECT DISTINCT cusip, quarter FROM fund_holdings WHERE cohort = 'seed' AND put_call = '')
        SELECT CASE WHEN n.cusip IS NOT NULL THEN 'still held at q+1 (implied possible)'
                    ELSE 'all seed funds sold (implied impossible)' END AS group_,
               count(*) AS rows,
               round(avg(s.ret_next_q_yahoo), 4) AS mean_ret,
               round(median(s.ret_next_q_yahoo), 4) AS median_ret,
               round(avg((s.ret_next_q_yahoo < -0.3)::INT), 4) AS share_below_m30,
               round(avg((s.ret_next_q_yahoo > 0.3)::INT), 4) AS share_above_p30
        FROM stock_quarter s
        JOIN seed_held h USING (cusip, quarter)
        LEFT JOIN seed_held n ON n.cusip = s.cusip AND n.quarter = {NEXTQ.format(q='s.quarter')}
        WHERE s.ret_next_q_yahoo IS NOT NULL AND s.security_class = 'common'
        GROUP BY 1 ORDER BY 1
    """).pl()


def unpriced_exit_share(con) -> pl.DataFrame:
    """Among unpriced seed-held common/ADR rows: how often is it an exit (no holder at q+1)?"""
    return con.execute("""
        SELECT left(quarter, 4) AS year, count(*) AS unpriced_rows,
               round(avg((ret_source = '13f_implied')::INT), 3) AS implied,
               round(avg((ret_source LIKE 'delist_%')::INT), 3) AS delisted,
               round(avg((ret_source LIKE 'missing:%')::INT), 3) AS missing
        FROM stock_quarter
        WHERE ret_next_q_yahoo IS NULL AND security_class IN ('common', 'adr')
          AND quarter < (SELECT max(quarter) FROM stock_quarter)
        GROUP BY 1 ORDER BY 1
    """).pl()


def _md(df: pl.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in df.iter_rows())


def write_report(con) -> None:
    ev = con.execute("""
        SELECT exit_class, coalesce(exit_subclass, '') AS subclass, coalesce(ret_source, '(none)') AS ret_source,
               count(*) AS exits, round(median(delist_ret), 3) AS median_ret, round(avg(delist_ret), 3) AS mean_ret,
               sum(terms_suspect::INT) AS terms_rejected
        FROM delisting_events GROUP BY ALL ORDER BY exits DESC
    """).pl()
    sens = con.execute("""
        SELECT count(*) AS performance_exits, round(avg(delist_ret), 3) AS mean_used,
               round(avg(delist_ret_sensitivity), 3) AS mean_shumway_warther
        FROM delisting_events WHERE exit_class = 'performance'
    """).pl()
    REPORT.write_text(f"""# Delisting returns and the filled `ret_next_q`

Generated by `uv run python -m src.securities.labels`. Method and decisions: docs/decisions.md
(2026-10-16, delisting gap).

## Label coverage by source, share of seed-fund long value (common + ADR)

`yahoo_only_before` is the coverage before this change.

{_md(coverage(con))}

## Rows by source (common + ADR, all quarters)

{_md(by_source(con))}

## Exits of unpriced stocks, by class

One row per (CUSIP, quarter q) held by a seed fund at q and by none at q+1, for stocks
Yahoo does not price. Only exits with a delisting filing in q+1 get a delisting return.

{_md(ev)}

Performance delistings use Shumway's -30%. Sensitivity with Shumway & Warther's Nasdaq
estimate (-55%), not used in the label:

{_md(sens)}

## Selection bias of 13F-implied returns

An implied return exists only if some seed fund still holds the stock at q+1. On stocks
Yahoo prices, where the return is known either way:

{_md(bias_report(con))}

Unpriced seed-held rows by outcome:

{_md(unpriced_exit_share(con))}
""")


if __name__ == "__main__":
    with db.connect() as con:
        fill(con)
        print(coverage(con))
        print(by_source(con))
        print(bias_report(con))
        write_report(con)
        print(f"wrote {REPORT}")
