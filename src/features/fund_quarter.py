"""Fund-level features: one row per fund x quarter.

Usage:
    uv run python -m src.features.fund_quarter

Concentration, turnover, active share and factor tilts use the long book
(non-option rows, `weight_long`); options are summarized by `options_share`.

  n_positions        long positions; n_option_rows separately
  top10_weight       weight of the 10 largest long positions
  hhi                Herfindahl index of long weights
  turnover           min(buys, sells) / average long book value over t-1 and t.
                     Buys and sells are split-adjusted quantity changes valued
                     at this quarter's price (value / quantity); a new position
                     is a buy of its value, an exit a sale of last quarter's
                     value grown by the stock's return. NULL in the first
                     quarter of data. Clipped to [0, 1] (`turnover_clipped`).
  gross_turnover     (buys + sells) / 2 / the same average book value. Unlike
                     turnover it sees one-sided trading: a fund that only sells
                     is not 0. Also clipped to [0, 1] (`gross_turnover_clipped`).
  active_share       0.5 * sum |w_fund - w_bench| at the issuer level, against
                     a cap-weighted proxy for the S&P 500: the 500 largest
                     issuers by market cap in stock_quarter (no free
                     constituent weights; see docs/decisions.md)
  tilt_*             value-weighted average of stock characteristics, each
                     winsorized at 1%/99% and z-scored across the quarter's
                     stocks: size (log ME), value (log B/M), momentum (12-1),
                     profitability (GP/A), investment (asset growth), beta.
                     cov_* = share of the long book with a value.
  sector_*           long weight per Fama-French 12 industry; sector_none for
                     securities with no industry (ETFs, funds, unmapped)
  options_share      value of put/call rows / total reported value
  known_at           the fund-quarter's filing date
"""

import polars as pl

from src.securities import db

FACTORS = {
    "size": "log_me",
    "value": "ln(bm)",
    "momentum": "mom_12_1",
    "profitability": "gp_at",
    "investment": "asset_growth",
    "beta": "beta",
}
FF12 = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]
BENCH_SIZE = 500


def build(con) -> None:
    # --- stock z-scores per quarter ---------------------------------------------------
    raw = ", ".join(f"{expr} AS {name}" for name, expr in FACTORS.items())
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE stock_x AS
        SELECT DISTINCT ON (symbol, quarter) symbol, quarter, {raw}, ff12_code
        FROM stock_quarter WHERE security_class = 'common'
        ORDER BY symbol, quarter, me DESC NULLS LAST
    """)
    zcols = []
    for name in FACTORS:
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE bounds_{name} AS
            SELECT quarter, quantile_cont({name}, 0.01) AS lo, quantile_cont({name}, 0.99) AS hi
            FROM stock_x GROUP BY quarter
        """)
        zcols.append(f"""(least(greatest(x.{name}, b_{name}.lo), b_{name}.hi)
              - avg(least(greatest(x.{name}, b_{name}.lo), b_{name}.hi)) OVER (PARTITION BY x.quarter))
              / stddev_samp(least(greatest(x.{name}, b_{name}.lo), b_{name}.hi)) OVER (PARTITION BY x.quarter) AS z_{name}""")
    joins = " ".join(f"JOIN bounds_{n} b_{n} USING (quarter)" for n in FACTORS)
    con.execute(f"""
        CREATE OR REPLACE TABLE stock_zscores AS
        SELECT x.symbol, x.quarter, x.ff12_code, {", ".join(zcols)}
        FROM stock_x x {joins}
    """)

    # --- benchmark: 500 largest issuers ------------------------------------------------
    con.execute(f"""
        CREATE OR REPLACE TABLE benchmark_weights AS
        WITH issuers AS (
            SELECT quarter, issuer_cik, max(me) AS me FROM stock_quarter
            WHERE security_class = 'common' AND me > 0 AND issuer_cik IS NOT NULL GROUP BY ALL
        ),
        top AS (SELECT *, row_number() OVER (PARTITION BY quarter ORDER BY me DESC) AS rk FROM issuers)
        SELECT quarter, issuer_cik AS issuer_key, me / sum(me) OVER (PARTITION BY quarter) AS w_bench
        FROM top WHERE rk <= {BENCH_SIZE}
    """)

    # --- per fund-quarter ---------------------------------------------------------------
    tilt_sel = ", ".join(
        f"sum(p.weight_long * z.z_{n}) / nullif(sum(p.weight_long) FILTER (WHERE z.z_{n} IS NOT NULL), 0) AS tilt_{n}, "
        f"sum(p.weight_long) FILTER (WHERE z.z_{n} IS NOT NULL) AS cov_{n}"
        for n in FACTORS
    )
    sector_sel = ", ".join(
        f"coalesce(sum(p.weight_long) FILTER (WHERE z.ff12_code = '{s}'), 0) AS sector_{s}" for s in FF12
    )
    con.execute(f"""
        CREATE OR REPLACE TABLE fund_quarter AS
        WITH p AS (SELECT * FROM positions),
        longp AS (
            SELECT *, row_number() OVER (PARTITION BY fund_id, quarter ORDER BY value_usd DESC, cusip) AS long_rank
            FROM p WHERE NOT is_option
        ),
        basic AS (
            SELECT fund_id, quarter, any_value(fund_name) AS fund_name, max(known_at) AS known_at,
                   count(*) FILTER (WHERE NOT is_option) AS n_positions,
                   count(*) FILTER (WHERE is_option) AS n_option_rows,
                   sum(value_usd) AS total_value,
                   sum(value_usd) FILTER (WHERE NOT is_option) AS long_value,
                   coalesce(sum(value_usd) FILTER (WHERE is_option), 0) / sum(value_usd) AS options_share
            FROM p GROUP BY ALL
        ),
        conc AS (
            SELECT fund_id, quarter,
                   sum(weight_long) FILTER (WHERE long_rank <= 10) AS top10_weight,
                   sum(weight_long * weight_long) AS hhi
            FROM longp GROUP BY ALL
        ),
        tilts AS (
            SELECT p.fund_id, p.quarter, {tilt_sel},
                   {sector_sel},
                   coalesce(sum(p.weight_long) FILTER (WHERE z.ff12_code IS NULL), 0) AS sector_none
            FROM longp p LEFT JOIN stock_zscores z USING (symbol, quarter)
            GROUP BY ALL
        ),
        fund_issuer AS (
            SELECT fund_id, quarter,
                   CASE WHEN security_class IN ('common', 'adr') AND issuer_cik IS NOT NULL THEN issuer_cik
                        ELSE security_key END AS issuer_key,
                   sum(weight_long) AS w_fund
            FROM longp GROUP BY ALL
        ),
        both_sides AS (
            -- every issuer in the fund or the benchmark, with both weights (0 if absent)
            SELECT fund_id, quarter, issuer_key, w_fund, 0.0 AS w_bench FROM fund_issuer
            UNION ALL
            SELECT fq.fund_id, b.quarter, b.issuer_key, 0.0, b.w_bench
            FROM (SELECT DISTINCT fund_id, quarter FROM fund_issuer) fq JOIN benchmark_weights b USING (quarter)
        ),
        act AS (
            SELECT fund_id, quarter, 0.5 * sum(abs(wf - wb)) AS active_share
            FROM (SELECT fund_id, quarter, issuer_key, sum(w_fund) AS wf, sum(w_bench) AS wb
                  FROM both_sides GROUP BY ALL)
            GROUP BY ALL
        ),
        -- turnover ------------------------------------------------------------------
        q AS (
            SELECT fund_id, quarter, left(quarter, 4)::INT * 4 + right(quarter, 1)::INT AS q_idx,
                   cusip, symbol, value_usd, split_ratio,
                   CASE WHEN shares > 0 THEN shares ELSE principal END AS qty,
                   NOT (shares > 0) AS by_value   -- bonds: principal units vary by filer, use values
            FROM longp
        ),
        rq AS (SELECT symbol, left(quarter, 4)::INT * 4 + right(quarter, 1)::INT AS q_idx, ret_q FROM returns_quarterly),
        flows AS (
            SELECT coalesce(t.fund_id, pr.fund_id) AS fund_id, coalesce(t.q_idx, pr.q_idx + 1) AS q_idx,
                   CASE WHEN pr.cusip IS NULL THEN t.value_usd                                   -- new position
                        WHEN t.cusip IS NULL THEN 0
                        WHEN t.by_value OR pr.by_value THEN greatest(t.value_usd - pr.value_usd, 0)
                        ELSE greatest(t.qty - pr.qty * t.split_ratio, 0) * t.value_usd / nullif(t.qty, 0) END AS buy,
                   CASE WHEN t.cusip IS NULL THEN pr.value_usd * (1 + coalesce(r.ret_q, 0))       -- exit
                        WHEN pr.cusip IS NULL THEN 0
                        WHEN t.by_value OR pr.by_value THEN greatest(pr.value_usd - t.value_usd, 0)
                        -- can't sell more than last quarter's holding, at this quarter's prices
                        ELSE least(greatest(pr.qty * t.split_ratio - t.qty, 0) * t.value_usd / nullif(t.qty, 0),
                                   pr.value_usd * (1 + coalesce(r.ret_q, 0))) END AS sell
            FROM q t
            FULL OUTER JOIN q pr ON pr.fund_id = t.fund_id AND pr.cusip = t.cusip AND pr.q_idx = t.q_idx - 1
            LEFT JOIN rq r ON r.symbol = pr.symbol AND r.q_idx = pr.q_idx + 1
        ),
        vals AS (SELECT fund_id, left(quarter, 4)::INT * 4 + right(quarter, 1)::INT AS q_idx, sum(value_usd) AS v
                 FROM longp GROUP BY ALL),
        turn AS (
            SELECT f.fund_id, f.q_idx,
                   least(sum(f.buy), sum(f.sell)) / ((cur.v + prv.v) / 2) AS turnover_raw,
                   (sum(f.buy) + sum(f.sell)) / 2 / ((cur.v + prv.v) / 2) AS gross_turnover_raw
            FROM flows f
            JOIN vals cur ON cur.fund_id = f.fund_id AND cur.q_idx = f.q_idx
            JOIN vals prv ON prv.fund_id = f.fund_id AND prv.q_idx = f.q_idx - 1   -- both quarters filed
            GROUP BY f.fund_id, f.q_idx, cur.v, prv.v
        )
        SELECT b.*, c.top10_weight, c.hhi,
               -- CASE first: DuckDB's greatest() skips NULLs, which would turn "no prior quarter" into 0
               CASE WHEN t.turnover_raw IS NOT NULL THEN least(greatest(t.turnover_raw, 0), 1) END AS turnover,
               t.turnover_raw > 1 AS turnover_clipped,
               CASE WHEN t.gross_turnover_raw IS NOT NULL THEN least(greatest(t.gross_turnover_raw, 0), 1) END AS gross_turnover,
               t.gross_turnover_raw > 1 AS gross_turnover_clipped,
               CASE WHEN a.active_share IS NOT NULL THEN least(greatest(a.active_share, 0), 1) END AS active_share,
               tl.* EXCLUDE (fund_id, quarter)
        FROM basic b
        LEFT JOIN conc c USING (fund_id, quarter)
        LEFT JOIN tilts tl USING (fund_id, quarter)
        LEFT JOIN act a USING (fund_id, quarter)
        LEFT JOIN turn t ON t.fund_id = b.fund_id
                        AND t.q_idx = left(b.quarter, 4)::INT * 4 + right(b.quarter, 1)::INT
    """)
    # Funds that filed an empty book: a row with zero positions, weight-based metrics NULL.
    has_empty = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'empty_books'").fetchone()[0]
    if has_empty:
        # Selling the whole book: buys 0, sells = last book, average book = half of it.
        # So turnover (min) is 0 and gross turnover is 1, when last quarter was filed.
        con.execute("""
            INSERT INTO fund_quarter BY NAME
            SELECT e.fund_id, e.quarter, e.fund_name, e.known_at, 0 AS n_positions, 0 AS n_option_rows,
                   0.0 AS total_value, 0.0 AS long_value,
                   CASE WHEN p.long_value > 0 THEN 0.0 END AS turnover,
                   CASE WHEN p.long_value > 0 THEN 1.0 END AS gross_turnover
            FROM empty_books e
            LEFT JOIN fund_quarter p ON p.fund_id = e.fund_id AND p.quarter =
                CASE WHEN right(e.quarter, 1) = '1' THEN (left(e.quarter, 4)::INT - 1)::VARCHAR || 'Q4'
                     ELSE left(e.quarter, 5) || (right(e.quarter, 1)::INT - 1)::VARCHAR END
        """)


if __name__ == "__main__":
    with db.connect() as con:
        build(con)
        with pl.Config(tbl_rows=25, tbl_width_chars=220, tbl_cols=16):
            print(con.execute("""
                SELECT fund_id, quarter, n_positions, round(top10_weight, 2) AS top10, round(hhi, 3) AS hhi,
                       round(turnover, 2) AS turnover, round(active_share, 2) AS act_share,
                       round(tilt_size, 2) AS size, round(tilt_value, 2) AS value, round(tilt_momentum, 2) AS mom,
                       round(tilt_beta, 2) AS beta, round(options_share, 3) AS opt
                FROM fund_quarter WHERE quarter = '2026Q2'
                  AND fund_id IN ('pershing_square_capital_management', 'berkshire_hathaway', 'renaissance_technologies',
                                  'tiger_global_management', 'citadel_advisors', 'baupost_group', 'baker_bros_advisors',
                                  'icahn_capital', 'two_sigma_investments', 'scion_asset_management')
                ORDER BY n_positions
            """).pl())
