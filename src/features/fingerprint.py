"""Strategy fingerprint (Milestone 4, Task A): one row per fund x quarter.

Usage:
    uv run python -m src.features.fingerprint

Point-in-time: a fund-quarter's fingerprint is as of its filing date
(`known_at`, the original 13F-HR; holdings come from `seed_positions_pit`, so
later amendments are never used). Every input row must have known_at on or
before that date: holdings (the fund-quarter's own filing and earlier
quarters' filings, for turnover and holding period) and stock characteristics
(`stock_quarter.known_at`, the quarter end, and `fundamentals_known_at`, the
latest 10-K/10-Q fact used). A stock whose characteristics are not yet known
counts as missing, exactly like an unpriced stock. `max_input_known_at`
records the latest input date actually used; the tests check it never passes
`known_at`.

Columns (long book: non-option rows, `weight_long`):
  n_positions, log_n_positions, top10_weight, hhi     concentration
  active_share                                        vs the 500 largest issuers
  turnover, gross_turnover                            from fund_quarter
  holding_period_q   value-weighted average number of consecutive quarters
                     the current positions have been held (age_q), each
                     capped at AGE_CAP quarters
  long_held_share    weight of positions held AGE_CAP quarters or more
  Why the cap: the data start in 2015Q1, so uncapped ages can reach 13
  quarters in 2018 but 46 in 2026. A model trained on early years and tested
  on later ones would see long holders drift "older" for no reason but the
  calendar. With a 12-quarter cap the feature means the same thing in every
  quarter from 2017Q4 on (a 12-quarter run then starts no earlier than 2015Q1).
  sector_*           FF12 industry weights; sector_none = no industry (ETFs,
                     funds, unmapped)
  tilt_size, tilt_value, tilt_momentum, tilt_profitability
                     value-weighted z-scores (stock_zscores); cov_* = share of
                     the long book with a known value
"""

import polars as pl

from src.features.fund_quarter import FF12
from src.securities import db

TILTS = ["size", "value", "momentum", "profitability"]
AGE_CAP = 12


def build(con) -> None:
    tilt_sel = ", ".join(
        f"sum(p.weight_long * z.z_{n}) / nullif(sum(p.weight_long) FILTER (WHERE z.z_{n} IS NOT NULL), 0) AS tilt_{n}, "
        f"coalesce(sum(p.weight_long) FILTER (WHERE z.z_{n} IS NOT NULL), 0) AS cov_{n}"
        for n in TILTS
    )
    sector_sel = ", ".join(
        f"coalesce(sum(p.weight_long) FILTER (WHERE z.ff12_code = '{s}'), 0) AS sector_{s}" for s in FF12
    )
    con.execute(f"""
        CREATE OR REPLACE TABLE fund_fingerprint AS
        WITH p AS (
            SELECT p.fund_id, p.quarter, p.known_at, p.symbol, p.cusip, p.weight_long, p.age_q,
                   s.known_at AS stock_known_at, s.fundamentals_known_at
            FROM positions p
            LEFT JOIN stock_quarter s USING (cusip, quarter)
            WHERE NOT p.is_option
        ),
        pit AS (
            -- characteristics count only if known by the fund's filing date
            SELECT *, stock_known_at IS NOT NULL AND stock_known_at <= known_at
                      AND coalesce(fundamentals_known_at, stock_known_at) <= known_at AS chars_known
            FROM p
        ),
        agg AS (
            SELECT p.fund_id, p.quarter,
                   sum(p.weight_long * least(p.age_q, {AGE_CAP})) / nullif(sum(p.weight_long), 0) AS holding_period_q,
                   coalesce(sum(p.weight_long) FILTER (WHERE p.age_q >= {AGE_CAP}), 0) AS long_held_share,
                   {tilt_sel},
                   {sector_sel},
                   coalesce(sum(p.weight_long) FILTER (WHERE z.ff12_code IS NULL), 0) AS sector_none,
                   max(greatest(p.stock_known_at, p.fundamentals_known_at)) FILTER (WHERE p.chars_known)
                       AS max_stock_known_at
            FROM pit p
            LEFT JOIN stock_zscores z ON z.symbol = p.symbol AND z.quarter = p.quarter AND p.chars_known
            GROUP BY ALL
        ),
        prior AS (
            -- turnover and holding period look back at earlier quarters' filings
            SELECT a.fund_id, a.quarter, max(b.known_at) AS max_prior_known_at
            FROM (SELECT DISTINCT fund_id, quarter FROM fund_quarter) a
            JOIN fund_quarter b ON b.fund_id = a.fund_id AND b.quarter < a.quarter
            GROUP BY ALL
        )
        SELECT f.fund_id, f.quarter, f.fund_name, f.known_at,
               f.n_positions, ln(greatest(f.n_positions, 1)) AS log_n_positions, f.top10_weight, f.hhi,
               f.active_share, f.turnover, f.gross_turnover,
               a.* EXCLUDE (fund_id, quarter, max_stock_known_at),
               greatest(a.max_stock_known_at, pr.max_prior_known_at, f.known_at) AS max_input_known_at
        FROM fund_quarter f
        JOIN agg a USING (fund_id, quarter)
        LEFT JOIN prior pr USING (fund_id, quarter)
        WHERE f.n_positions > 0
    """)


if __name__ == "__main__":
    with db.connect() as con:
        build(con)
        print(con.execute("""
            SELECT count(*) AS rows, count(DISTINCT fund_id) AS funds, count(DISTINCT quarter) AS quarters,
                   count(*) FILTER (WHERE max_input_known_at > known_at) AS pit_violations,
                   round(avg(cov_size), 3) AS avg_cov_size, round(avg(holding_period_q), 2) AS avg_holding_q
            FROM fund_fingerprint
        """).pl())
        with pl.Config(tbl_rows=12, tbl_width_chars=200, tbl_cols=12):
            print(con.execute("""
                SELECT fund_id, n_positions, round(hhi, 3) AS hhi, round(active_share, 2) AS act,
                       round(gross_turnover, 2) AS gturn, round(holding_period_q, 1) AS hold_q,
                       round(tilt_size, 2) AS size, round(tilt_value, 2) AS value,
                       round(tilt_momentum, 2) AS mom, round(tilt_profitability, 2) AS prof
                FROM fund_fingerprint WHERE quarter = '2026Q2' ORDER BY n_positions LIMIT 12
            """).pl())
