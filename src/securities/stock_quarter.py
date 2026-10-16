"""The stock_quarter table: point-in-time characteristics per (CUSIP, quarter).

Usage:
    uv run python -m src.securities.stock_quarter

One row per CUSIP held by a seed fund in a quarter (common stock, ADRs, ETFs
and closed-end funds; bonds, warrants and preferreds are out of scope).

Every feature is as of the calendar quarter end (`known_at`): prices on or
before that day, and fundamentals filed on or before it. The label
`ret_next_q` is the next calendar quarter's total return; it has its own
`label_known_at` (the last trading day of that quarter).

Features:
  me, log_me     market equity = shares outstanding x quarter-end price
                 (common stock only: ADR share counts are in ordinary shares,
                 not ADRs, and ETFs/funds have no XBRL shares)
  be, bm         book equity and book-to-market; bm is NULL when book equity
                 is negative, zero or missing, and the stock is kept
  gp_at          gross profitability (Novy-Marx 2013): latest annual gross
                 profit / total assets at the same fiscal year end
  asset_growth   total assets at the latest fiscal year end / one year earlier - 1
  mom_12_1       return from month-end t-12 to month-end t-1
  beta, vol      from daily returns over the past 365 calendar days (>= 120
                 days): slope on the FF market excess return, and annualized
                 volatility. Days after French's data ends use SPY minus the
                 last RF instead; beta_market = 'ff' or 'ff+spy' says which
                 was used and n_proxy_days how many SPY days
  sic, ff12      SIC code from EDGAR, mapped to Fama-French 12 industries
"""

import shutil

import duckdb

from src.securities import db

PRICED_CLASSES = ("common", "adr", "etf", "fund")
STALE_PRICE_DAYS = 10   # a "quarter-end price" must be from the last 10 days
MIN_BETA_DAYS = 120
STALE_FUNDAMENTALS_DAYS = 550  # ignore balance sheets older than ~18 months

DAILY_SQL = """
CREATE OR REPLACE TEMP TABLE daily AS
SELECT symbol, date, close,
       close / lag(close) OVER w - 1 AS ret,
       date - lag(date) OVER w AS gap
FROM prices_daily WINDOW w AS (PARTITION BY symbol ORDER BY date)
"""

SQL = f"""
CREATE OR REPLACE TEMP TABLE sq_chunk AS
WITH
universe AS (
    SELECT DISTINCT p.cusip, p.quarter, m.symbol, m.security_class, m.issuer_cik,
           last_day(make_date(left(p.quarter, 4)::INT, right(p.quarter, 1)::INT * 3, 1)) AS q_end
    FROM {db.ALL_SEED_POSITIONS} p JOIN security_master m USING (cusip)
    WHERE m.symbol IS NOT NULL AND m.security_class IN {PRICED_CLASSES}
      AND left(p.quarter, 4) = '{{year}}'      -- built one year at a time (see build)
),
-- price-based features depend only on (symbol, quarter): compute them once per
-- pair, not once per CUSIP (share classes and retired CUSIPs share a ticker)
symq AS (SELECT DISTINCT symbol, quarter, q_end FROM universe),

-- prices ---------------------------------------------------------------------
px AS (
    SELECT u.symbol, u.quarter,
           arg_max(d.close, d.date) AS price, max(d.date) AS price_date
    FROM symq u JOIN prices_daily d
      ON d.symbol = u.symbol AND d.date <= u.q_end AND d.date > u.q_end - INTERVAL {STALE_PRICE_DAYS} DAY
    GROUP BY ALL
),
mom AS (
    SELECT u.symbol, u.quarter,
           arg_max(d.close, d.date) FILTER (WHERE d.date <= last_day(u.q_end - INTERVAL 1 MONTH)
                                          AND d.date > last_day(u.q_end - INTERVAL 1 MONTH) - INTERVAL {STALE_PRICE_DAYS} DAY) AS p_t1,
           arg_max(d.close, d.date) FILTER (WHERE d.date <= last_day(u.q_end - INTERVAL 12 MONTH)
                                          AND d.date > last_day(u.q_end - INTERVAL 12 MONTH) - INTERVAL {STALE_PRICE_DAYS} DAY) AS p_t12
    FROM symq u JOIN prices_daily d
      ON d.symbol = u.symbol AND d.date <= u.q_end AND d.date > u.q_end - INTERVAL 13 MONTH
    GROUP BY ALL
),
-- Market excess return: Fama-French Mkt-RF while it exists; after French's last
-- date, SPY's return minus the last published RF (flagged via n_proxy_days).
market AS (
    SELECT date, Mkt_RF, RF, FALSE AS is_proxy FROM ff_factors_daily
    UNION ALL
    SELECT s.date, s.ret - rf.RF, rf.RF, TRUE
    FROM daily s, (SELECT arg_max(RF, date) AS RF, max(date) AS last_ff FROM ff_factors_daily) rf
    WHERE s.symbol = 'SPY' AND s.date > rf.last_ff AND s.ret IS NOT NULL
),
risk AS (
    SELECT u.symbol, u.quarter,
           count(*) AS n_days,
           count(*) FILTER (WHERE f.is_proxy) AS n_proxy_days,
           regr_slope(d.ret - f.RF, f.Mkt_RF) AS beta,
           stddev_samp(d.ret) * sqrt(252) AS vol
    FROM symq u
    JOIN daily d ON d.symbol = u.symbol AND d.date <= u.q_end AND d.date > u.q_end - INTERVAL 365 DAY
                AND d.ret IS NOT NULL AND d.gap <= 7
    JOIN market f ON f.date = d.date
    GROUP BY ALL
),

-- fundamentals: only facts filed on or before the quarter end -------------------
issuers AS (SELECT DISTINCT issuer_cik, quarter, q_end FROM universe WHERE issuer_cik IS NOT NULL),
pit AS (
    -- for each concept and period, the latest version filed by q_end
    SELECT i.quarter, i.q_end, f.issuer_cik, f.item, f.priority, f.start, f."end",
           -- latest filing wins; same-day ties: the larger value (explicit, so reruns agree)
           first(f.val ORDER BY f.known_at DESC, f.val DESC) AS val, max(f.known_at) AS known_at
    FROM issuers i JOIN fundamentals_facts f ON f.issuer_cik = i.issuer_cik AND f.known_at <= i.q_end
                                               AND f."end" <= i.q_end  -- some filings carry periods ending after their filing date
    WHERE NOT (f.item = 'shares_out' AND f.val <= 0)                   -- a zero share count is a tagging error
    GROUP BY ALL
),
pit_best AS (
    -- across fallback concepts for the same item and period, the highest priority
    SELECT quarter, q_end, issuer_cik, item, start, "end",
           first(val ORDER BY priority, val DESC) AS val, first(known_at ORDER BY priority, val DESC) AS known_at,
           min(priority) AS priority
    FROM pit GROUP BY ALL
),
instant AS (SELECT * FROM pit_best WHERE start IS NULL),
annual AS (SELECT * FROM pit_best WHERE start IS NOT NULL AND "end" - start BETWEEN 350 AND 380),
equity AS (
    SELECT issuer_cik, quarter, arg_max(val, "end") AS be, max("end") AS be_end, arg_max(known_at, "end") AS be_known_at
    FROM instant WHERE item = 'equity' AND "end" > q_end - INTERVAL {STALE_FUNDAMENTALS_DAYS} DAY
    GROUP BY ALL
),
shares AS (
    -- take the highest-priority concept that has a recent value, then its latest value
    SELECT issuer_cik, quarter,
           first(val ORDER BY priority, "end" DESC, known_at DESC, val DESC) AS shares_out,
           first(known_at ORDER BY priority, "end" DESC, known_at DESC, val DESC) AS shares_known_at
    FROM pit WHERE item = 'shares_out' AND "end" > q_end - INTERVAL 400 DAY
    GROUP BY ALL
),
fy AS (
    -- annual gross profit; revenue minus cost of revenue when gross profit isn't tagged
    SELECT issuer_cik, quarter, "end",
           coalesce(max(val) FILTER (WHERE item = 'gross_profit'),
                    max(val) FILTER (WHERE item = 'revenue') - max(val) FILTER (WHERE item = 'cost_of_revenue')) AS gp,
           max(known_at) AS known_at
    FROM annual WHERE item IN ('gross_profit', 'revenue', 'cost_of_revenue')
    GROUP BY ALL
),
fy_assets AS (
    SELECT fy.issuer_cik, fy.quarter, fy."end", fy.gp, a.val AS assets,
           greatest(fy.known_at, a.known_at) AS known_at
    FROM fy JOIN instant a ON a.issuer_cik = fy.issuer_cik AND a.quarter = fy.quarter
                          AND a.item = 'assets' AND a."end" = fy."end"
),
latest_fy AS (
    SELECT issuer_cik, quarter, arg_max(gp, "end") AS gp, arg_max(assets, "end") AS assets,
           max("end") AS fy_end, arg_max(known_at, "end") AS fy_known_at
    FROM fy_assets GROUP BY ALL
),
growth AS (
    SELECT l.issuer_cik, l.quarter, l.gp, l.assets, l.fy_end, l.fy_known_at,
           arg_max(a.val, a."end") AS assets_prev, arg_max(a.known_at, a."end") AS prev_known_at
    FROM latest_fy l
    LEFT JOIN instant a ON a.issuer_cik = l.issuer_cik AND a.quarter = l.quarter AND a.item = 'assets'
                       AND a."end" BETWEEN l.fy_end - INTERVAL 375 DAY AND l.fy_end - INTERVAL 355 DAY
    GROUP BY ALL
),

-- sector ---------------------------------------------------------------------------
sector AS (
    SELECT p.issuer_cik, p.sic,
           CASE WHEN p.sic IS NULL THEN NULL ELSE coalesce(r.ff12, 12) END AS ff12,
           CASE WHEN p.sic IS NULL THEN NULL ELSE coalesce(r.ff12_code, 'Other') END AS ff12_code
    FROM issuer_profile p
    LEFT JOIN ff12_sic_ranges r ON p.sic BETWEEN r.sic_from AND r.sic_to
),

-- labels ---------------------------------------------------------------------------
rq AS (SELECT symbol, quarter, ret_q, known_at FROM returns_quarterly)

SELECT
    u.cusip, u.quarter, u.symbol, u.security_class, u.issuer_cik,
    u.q_end AS known_at,
    px.price, px.price_date,
    CASE WHEN u.security_class = 'common' THEN sh.shares_out * px.price END AS me,
    CASE WHEN u.security_class = 'common' AND sh.shares_out * px.price > 0
         THEN ln(sh.shares_out * px.price / 1e6) END AS log_me,
    eq.be,
    CASE WHEN u.security_class = 'common' AND eq.be > 0 AND sh.shares_out * px.price > 0
         THEN eq.be / (sh.shares_out * px.price) END AS bm,
    CASE WHEN g.assets > 0 THEN g.gp / g.assets END AS gp_at,
    CASE WHEN g.assets_prev > 0 THEN g.assets / g.assets_prev - 1 END AS asset_growth,
    mom.p_t1 / mom.p_t12 - 1 AS mom_12_1,
    CASE WHEN risk.n_days >= {MIN_BETA_DAYS} THEN risk.beta END AS beta,
    CASE WHEN risk.n_days >= {MIN_BETA_DAYS} THEN risk.vol END AS vol,
    risk.n_days,
    risk.n_proxy_days,
    CASE WHEN risk.n_days >= {MIN_BETA_DAYS}
         THEN CASE WHEN risk.n_proxy_days > 0 THEN 'ff+spy' ELSE 'ff' END END AS beta_market,
    s.sic, s.ff12, s.ff12_code,
    rq.ret_q,
    nq.ret_q AS ret_next_q,
    nq.known_at AS label_known_at,
    -- audit trail: the newest filing date behind any fundamental feature
    greatest(eq.be_known_at, sh.shares_known_at, g.fy_known_at, g.prev_known_at) AS fundamentals_known_at,
    eq.be_end, g.fy_end
FROM universe u
LEFT JOIN px   ON px.symbol = u.symbol AND px.quarter = u.quarter
LEFT JOIN mom  ON mom.symbol = u.symbol AND mom.quarter = u.quarter
LEFT JOIN risk ON risk.symbol = u.symbol AND risk.quarter = u.quarter
LEFT JOIN equity eq ON eq.issuer_cik = u.issuer_cik AND eq.quarter = u.quarter AND u.security_class IN ('common', 'adr')
LEFT JOIN shares sh ON sh.issuer_cik = u.issuer_cik AND sh.quarter = u.quarter AND u.security_class = 'common'
LEFT JOIN growth g  ON g.issuer_cik = u.issuer_cik AND g.quarter = u.quarter AND u.security_class IN ('common', 'adr')
LEFT JOIN sector s  ON s.issuer_cik = u.issuer_cik AND u.security_class IN ('common', 'adr')
LEFT JOIN rq ON rq.symbol = u.symbol AND rq.quarter = u.quarter
LEFT JOIN rq nq ON nq.symbol = u.symbol AND nq.quarter =
    CASE WHEN right(u.quarter, 1) = '4' THEN (left(u.quarter, 4)::INT + 1)::VARCHAR || 'Q1'
         ELSE left(u.quarter, 5) || (right(u.quarter, 1)::INT + 1)::VARCHAR END
"""


def build(con: duckdb.DuckDBPyConnection) -> None:
    """One year of quarters at a time: the price windows for a year need only
    ~13 months of prices, so memory and temp space stay small (the whole
    history at once spilled more than the free disk). The new table replaces
    the old one only after every year is built."""
    con.execute("SET preserve_insertion_order = false; SET threads = 4")
    # never fill the disk: leave 1.5 GB free and fail cleanly if a spill needs more
    free_gb = shutil.disk_usage(db.DB_PATH.parent).free / 2**30
    con.execute(f"SET max_temp_directory_size = '{max(free_gb - 1.5, 0.5):.1f}GiB'")
    con.execute(DAILY_SQL)
    years = [r[0] for r in con.execute(
        f"SELECT DISTINCT left(quarter, 4) FROM {db.ALL_SEED_POSITIONS} ORDER BY 1").fetchall()]
    con.execute("DROP TABLE IF EXISTS stock_quarter_new")
    for i, year in enumerate(years):
        con.execute(SQL.replace("{year}", year))
        if i == 0:
            con.execute("CREATE TABLE stock_quarter_new AS SELECT * FROM sq_chunk")
        else:
            con.execute("INSERT INTO stock_quarter_new SELECT * FROM sq_chunk")
    con.execute("DROP TABLE IF EXISTS stock_quarter")
    con.execute("ALTER TABLE stock_quarter_new RENAME TO stock_quarter")


def coverage_report(con) -> None:
    print(con.execute("""
        SELECT quarter, security_class, count(*) AS n,
               round(avg((price IS NOT NULL)::INT), 3) AS price,
               round(avg((log_me IS NOT NULL)::INT), 3) AS size,
               round(avg((bm IS NOT NULL)::INT), 3) AS bm,
               round(avg((gp_at IS NOT NULL)::INT), 3) AS gp_at,
               round(avg((asset_growth IS NOT NULL)::INT), 3) AS ag,
               round(avg((mom_12_1 IS NOT NULL)::INT), 3) AS mom,
               round(avg((beta IS NOT NULL)::INT), 3) AS beta,
               round(avg((ff12 IS NOT NULL)::INT), 3) AS ff12,
               round(avg((ret_next_q IS NOT NULL)::INT), 3) AS label
        FROM stock_quarter GROUP BY ALL ORDER BY quarter, n DESC
    """).pl())


if __name__ == "__main__":
    with db.connect() as con:
        build(con)
        coverage_report(con)
