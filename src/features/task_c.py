"""Task C panel: features and targets for next-quarter moves (C1) and new buys (C2).

Usage:
    uv run python -m src.features.task_c

**Prediction date.** A row (fund f, stock s, quarter q) is predicted on f's own
quarter-q filing date `pred_date` (the original 13F-HR; `fund_holdings` is the
point-in-time view, so a later 13F-HR/A never feeds a feature). Every feature
group carries its own `<group>_known_at`, and `src.models.leakage_audit` checks
each one against `pred_date` row by row.

Feature groups (FEATURE_GROUPS lists the columns):
  own      f's own quarter-q position and book (known at pred_date)
  stock    stock characteristics at q's end (stock_quarter) plus the stock's
           return from q's end to pred_date (daily closes on or before it)
  prior    f's historical action rates, from labels known by pred_date
  crowd    how many *other* seed funds hold s in their latest filing known by
           pred_date. Not `positions.crowding`: that counts other funds'
           quarter-q filings, many of which come out after f's filing date.
  peer     what other seed funds did with s in that latest filing
           (new / add / trim / exit), and how many of them had already filed q
  cluster  the same counts restricted to f's Milestone 5 cluster (fund2vec +
           HDBSCAN), taken from the latest clustering with known_at <= pred_date.
           A quarter's clusters are fitted on filings up to ~60 days after its
           end, so a fund filing on day 40 gets the q-1 clusters. Holdout funds
           were never clustered, so their cluster features are null.

Peers are seed funds only, so a holdout fund never moves a seed feature. A
peer's "latest filing" is the latest quarter r <= q it filed with known_at <=
pred_date; peers whose latest is older than q-1 (stopped filing) are ignored.

Targets: `action` from position_moves (forced exits dropped: deals and
delistings are not the fund's decision). `label_known_at` = f's q+1 filing.
Only securities in the stock universe (a `stock_quarter` row: common, ADR, ETF,
fund) are kept; warrants, SPAC units, debt, preferreds and unmapped CUSIPs
(~9.5% of long rows) have no characteristics and aren't the stock decisions
Task C is about.
`ret_clean` is false when the stock's return over the quarter of the move is a
0% deal fallback (`delist_last_13f_price`) or missing; it is a reporting
filter only and is never a feature. For a new buy of a stock no seed fund held
at q (no quarter-q row), the quarter-(q+1) return `ret_q` is used instead.

Tables written: c1_all (every stock-universe long position, labelled or not: what
a backtest scores, since whether a fund files next quarter is future information),
c1_panel (its labelled, non-forced rows: what C1 trains and is scored on), new_buys, task_c_peer_latest (each target fund-quarter's
visible peer filings), task_c_cluster (its cluster as of pred_date).
"""

from src.features.positions import QIDX
from src.securities import db

CLUSTER_METHOD, CLUSTER_ALGO = "fund2vec", "hdbscan"   # the Milestone 5 choice (data/reference/embedding_choice.json)
FIRST_PANEL_QUARTER = "2018Q1"   # first training quarter; earlier quarters still feed lags and fund priors

FEATURE_GROUPS: dict[str, list[str]] = {
    "own": ["weight_long", "log_rank", "top10", "is_new", "share_change", "share_change_prev",
            "age_q", "age_censored", "log_value", "unrealized_gain",
            "log_n_positions", "top10_weight", "hhi", "turnover_clipped", "gross_turnover_clipped",
            "options_share", "log_long_value"],
    "stock": ["log_me", "bm", "gp_at", "asset_growth", "mom_12_1", "beta", "vol", "ret_q", "ff12",
              "ret_since_qend"],
    "prior": ["prior_exit", "prior_trim", "prior_hold", "prior_add", "prior_n",
              "last_exit", "last_trim", "last_hold", "last_add"],
    "crowd": ["n_peer_hold", "n_peers_active"],
    "peer": ["n_peer_new", "n_peer_add", "n_peer_trim", "n_peer_exit", "share_peers_filed_q"],
    "cluster": ["cl_noise", "cl_size", "cl_n_hold", "cl_n_new", "cl_n_add", "cl_n_trim", "cl_n_exit",
                "cl_frac_hold", "cl_net_buy"],
}
KNOWN_AT = {g: f"{g}_known_at" for g in FEATURE_GROUPS}
# columns that describe the outcome; a model must never see them (checked by the audit)
FORBIDDEN = {"action", "share_change_next", "label_known_at", "ret_next_q", "ret_source", "ret_clean",
             "label_flag", "forced_exit", "crowding"}
BAD_RET = "(ret_source IS NULL OR ret_source LIKE 'missing:%' OR ret_source = 'delist_last_13f_price')"


def _filings(con) -> None:
    """One row per filed fund-quarter (empty books included) and every seed fund's moves in it."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE filings AS
        SELECT fund_id, any_value(cohort) AS cohort, quarter, {QIDX.format(q='quarter')} AS q_idx,
               max(known_at) AS known_at
        FROM (SELECT fund_id, cohort, quarter, known_at FROM positions
              UNION ALL BY NAME SELECT fund_id, cohort, quarter, known_at FROM empty_books)
        GROUP BY fund_id, quarter
    """)
    # a seed fund's long-book moves *into* quarter r (what its quarter-r filing revealed)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE peer_moves AS
        SELECT fund_id, {QIDX.format(q='quarter')} AS q_idx, cusip,
               CASE WHEN is_new THEN 'new'
                    WHEN share_change > 0.10 THEN 'add' WHEN share_change < -0.10 THEN 'trim'
                    ELSE 'hold' END AS move
        FROM positions WHERE cohort = 'seed' AND put_call = ''
        UNION ALL
        SELECT fund_id, {QIDX.format(q='quarter')} + 1 AS q_idx, cusip, 'exit' AS move
        FROM position_moves WHERE cohort = 'seed' AND action = 'exit' AND NOT forced_exit
    """)


def _peer_latest(con, quarters_from: str) -> None:
    """For each target fund-quarter (f, q, pred_date): every other seed fund's latest
    filing for a quarter <= q with known_at <= pred_date, and f's cluster then."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE targets AS
        SELECT fund_id, cohort, quarter, q_idx, known_at AS pred_date FROM filings
        WHERE quarter >= '{quarters_from}'
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE peer_latest AS
        WITH pairs AS (
            SELECT t.fund_id, t.quarter, t.q_idx, t.pred_date, p.fund_id AS peer_id
            FROM targets t CROSS JOIN (SELECT DISTINCT fund_id FROM filings WHERE cohort = 'seed') p
            WHERE p.fund_id <> t.fund_id
        ),
        cand AS (
            SELECT pr.*, f.q_idx AS peer_q_idx, f.known_at AS peer_known_at,
                   row_number() OVER (PARTITION BY pr.fund_id, pr.quarter, pr.peer_id
                                      ORDER BY f.q_idx DESC, f.known_at DESC) AS rn
            FROM pairs pr JOIN filings f
              ON f.fund_id = pr.peer_id AND f.known_at <= pr.pred_date AND f.q_idx <= pr.q_idx
        )
        SELECT fund_id, quarter, q_idx, pred_date, peer_id, peer_q_idx, peer_known_at
        FROM cand WHERE rn = 1 AND peer_q_idx >= q_idx - 1
    """)
    # f's cluster: latest clustering with known_at <= pred_date (and quarter <= q)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE target_cluster AS
        WITH c AS (
            SELECT t.fund_id, t.quarter, c.quarter AS cl_quarter, c.cluster, c.known_at AS cl_known_at,
                   row_number() OVER (PARTITION BY t.fund_id, t.quarter ORDER BY c.quarter DESC) AS rn
            FROM targets t JOIN fund_clusters c
              ON c.fund_id = t.fund_id AND c.known_at <= t.pred_date AND c.quarter <= t.quarter
             AND c.method = '{CLUSTER_METHOD}' AND c.algo = '{CLUSTER_ALGO}'
        )
        SELECT fund_id, quarter, cl_quarter, cluster, cl_known_at FROM c WHERE rn = 1
    """)
    # peers in f's cluster (same clustering quarter); noise (-1) has no cluster peers
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cluster_peers AS
        SELECT tc.fund_id, tc.quarter, c.fund_id AS peer_id
        FROM target_cluster tc JOIN fund_clusters c
          ON c.quarter = tc.cl_quarter AND c.cluster = tc.cluster AND tc.cluster >= 0
         AND c.method = '{CLUSTER_METHOD}' AND c.algo = '{CLUSTER_ALGO}' AND c.fund_id <> tc.fund_id
    """)


def _unrealized_gain(con) -> None:
    """Price now vs the split-adjusted average price paid on the adds of the
    current holding run, from the fund's own filings (value / shares)."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gain AS
        WITH p AS (
            SELECT fund_id, cusip, quarter, {QIDX.format(q='quarter')} AS q_idx, shares, value_usd, split_ratio,
                   {QIDX.format(q='quarter')} - age_q + 1 AS run_start
            FROM positions
            WHERE put_call = '' AND shares > 0 AND value_usd > 0 AND NOT shares_suspect
        ),
        s AS (
            -- cumulative split factor within the run (the first quarter's split is before the run)
            SELECT *, exp(sum(CASE WHEN q_idx = run_start THEN 0 ELSE ln(split_ratio) END)
                          OVER (PARTITION BY fund_id, cusip, run_start ORDER BY q_idx)) AS cum_split
            FROM p
        ),
        n AS (
            SELECT *, shares / cum_split AS n_shares, value_usd / shares * cum_split AS n_price
            FROM s
        ),
        d AS (
            SELECT *, greatest(n_shares - coalesce(lag(n_shares) OVER w, 0), 0) AS bought
            FROM n WINDOW w AS (PARTITION BY fund_id, cusip, run_start ORDER BY q_idx)
        ),
        c AS (
            SELECT *, sum(bought * n_price) OVER w / nullif(sum(bought) OVER w, 0) AS cost
            FROM d WINDOW w AS (PARTITION BY fund_id, cusip, run_start ORDER BY q_idx)
        )
        SELECT fund_id, cusip, quarter, least(greatest(n_price / cost - 1, -1), 10) AS unrealized_gain
        FROM c
    """)


def build(con) -> None:
    _filings(con)
    _peer_latest(con, FIRST_PANEL_QUARTER)
    _unrealized_gain(con)

    # f's own action counts per quarter r (moves r -> r+1, public at its r+1 filing)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE own_rates AS
        SELECT fund_id, {QIDX.format(q='quarter')} AS q_idx, max(known_at) AS known_at,
               count(*) FILTER (WHERE action = 'exit') AS n_exit, count(*) FILTER (WHERE action = 'trim') AS n_trim,
               count(*) FILTER (WHERE action = 'hold') AS n_hold, count(*) FILTER (WHERE action = 'add') AS n_add
        FROM position_moves WHERE NOT forced_exit AND action IS NOT NULL GROUP BY ALL
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE prior AS
        SELECT t.fund_id, t.quarter,
               sum(r.n_exit) / sum(r.n_exit + r.n_trim + r.n_hold + r.n_add) AS prior_exit,
               sum(r.n_trim) / sum(r.n_exit + r.n_trim + r.n_hold + r.n_add) AS prior_trim,
               sum(r.n_hold) / sum(r.n_exit + r.n_trim + r.n_hold + r.n_add) AS prior_hold,
               sum(r.n_add)  / sum(r.n_exit + r.n_trim + r.n_hold + r.n_add) AS prior_add,
               sum(r.n_exit + r.n_trim + r.n_hold + r.n_add) AS prior_n,
               max(r.known_at) AS prior_known_at,
               max(r.n_exit / (r.n_exit + r.n_trim + r.n_hold + r.n_add)) FILTER (WHERE r.q_idx = t.q_idx - 1) AS last_exit,
               max(r.n_trim / (r.n_exit + r.n_trim + r.n_hold + r.n_add)) FILTER (WHERE r.q_idx = t.q_idx - 1) AS last_trim,
               max(r.n_hold / (r.n_exit + r.n_trim + r.n_hold + r.n_add)) FILTER (WHERE r.q_idx = t.q_idx - 1) AS last_hold,
               max(r.n_add  / (r.n_exit + r.n_trim + r.n_hold + r.n_add)) FILTER (WHERE r.q_idx = t.q_idx - 1) AS last_add
        FROM targets t JOIN own_rates r ON r.fund_id = t.fund_id AND r.known_at <= t.pred_date
        GROUP BY t.fund_id, t.quarter
    """)

    # peer counts on the stocks f holds at q
    con.execute("""
        CREATE OR REPLACE TEMP TABLE peer_counts AS
        WITH tp AS (SELECT fund_id, quarter, cusip FROM positions WHERE put_call = ''),
        j AS (
            SELECT pl.fund_id, pl.quarter, m.cusip, m.move, pl.peer_id, pl.peer_known_at,
                   cp.peer_id IS NOT NULL AS in_cluster
            FROM peer_latest pl
            JOIN peer_moves m ON m.fund_id = pl.peer_id AND m.q_idx = pl.peer_q_idx
            JOIN tp ON tp.fund_id = pl.fund_id AND tp.quarter = pl.quarter AND tp.cusip = m.cusip
            LEFT JOIN cluster_peers cp ON cp.fund_id = pl.fund_id AND cp.quarter = pl.quarter AND cp.peer_id = pl.peer_id
        )
        SELECT fund_id, quarter, cusip,
               count(*) FILTER (WHERE move <> 'exit') AS n_peer_hold,
               count(*) FILTER (WHERE move = 'new') AS n_peer_new,
               count(*) FILTER (WHERE move = 'add') AS n_peer_add,
               count(*) FILTER (WHERE move = 'trim') AS n_peer_trim,
               count(*) FILTER (WHERE move = 'exit') AS n_peer_exit,
               count(*) FILTER (WHERE in_cluster AND move <> 'exit') AS cl_n_hold,
               count(*) FILTER (WHERE in_cluster AND move = 'new') AS cl_n_new,
               count(*) FILTER (WHERE in_cluster AND move = 'add') AS cl_n_add,
               count(*) FILTER (WHERE in_cluster AND move = 'trim') AS cl_n_trim,
               count(*) FILTER (WHERE in_cluster AND move = 'exit') AS cl_n_exit,
               max(peer_known_at) AS max_peer_known_at
        FROM j GROUP BY ALL
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE peer_fq AS
        SELECT pl.fund_id, pl.quarter, count(*) AS n_peers_active,
               avg((pl.peer_q_idx = pl.q_idx)::INT) AS share_peers_filed_q,
               max(pl.peer_known_at) AS peer_known_at,
               count(cp.peer_id) AS cl_size_active
        FROM peer_latest pl
        LEFT JOIN cluster_peers cp ON cp.fund_id = pl.fund_id AND cp.quarter = pl.quarter AND cp.peer_id = pl.peer_id
        GROUP BY ALL
    """)

    # stock return from q's end to the last close on or before pred_date
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE since AS
        WITH k AS (
            SELECT DISTINCT p.symbol, t.quarter, t.pred_date,
                   make_date(left(t.quarter, 4)::INT, right(t.quarter, 1)::INT * 3, 1) + INTERVAL 1 MONTH - INTERVAL 1 DAY AS q_end
            FROM targets t JOIN positions p USING (fund_id, quarter) WHERE p.symbol IS NOT NULL AND p.put_call = ''
        ),
        px AS (
            SELECT k.symbol, k.quarter, k.pred_date,
                   arg_max(d.close, d.date) FILTER (WHERE d.date <= k.q_end) AS c0,
                   arg_max(d.close, d.date) FILTER (WHERE d.date <= k.pred_date) AS c1,
                   max(d.date) FILTER (WHERE d.date <= k.pred_date) AS c1_date
            FROM k JOIN prices_daily d ON d.symbol = k.symbol AND d.date BETWEEN k.q_end - INTERVAL 10 DAY AND k.pred_date
            GROUP BY ALL
        )
        SELECT symbol, quarter, pred_date, c1 / c0 - 1 AS ret_since_qend, c1_date FROM px WHERE c0 > 0
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE c1_all AS
        SELECT
            p.fund_id, p.cohort, p.quarter, p.cusip, p.symbol,
            p.known_at AS pred_date,
            m.known_at AS label_known_at, m.action, m.share_change_next, m.forced_exit,
            NOT {BAD_RET.replace('ret_source', 'sq.ret_source')} AS ret_clean, sq.ret_source,
            -- own
            p.weight_long, ln(p.rank) AS log_rank, p.top10::INT AS top10, p.is_new::INT AS is_new, p.share_change,
            pp.share_change AS share_change_prev, p.age_q, p.age_censored::INT AS age_censored,
            ln(1 + p.value_usd) AS log_value, g.unrealized_gain,
            ln(fq.n_positions) AS log_n_positions, fq.top10_weight, fq.hhi, fq.turnover_clipped,
            fq.gross_turnover_clipped, fq.options_share, ln(1 + fq.long_value) AS log_long_value,
            greatest(p.known_at, fq.known_at, coalesce(pp.known_at, p.known_at)) AS own_known_at,
            -- stock
            sq.log_me, sq.bm, sq.gp_at, sq.asset_growth, sq.mom_12_1, sq.beta, sq.vol, sq.ret_q, sq.ff12,
            least(si.ret_since_qend, 3) AS ret_since_qend,   -- a handful of Yahoo glitches (999x)
            greatest(sq.known_at, sq.fundamentals_known_at, si.c1_date) AS stock_known_at,
            -- prior
            pr.prior_exit, pr.prior_trim, pr.prior_hold, pr.prior_add, pr.prior_n,
            pr.last_exit, pr.last_trim, pr.last_hold, pr.last_add, pr.prior_known_at,
            -- crowd / peer
            coalesce(pc.n_peer_hold, 0) AS n_peer_hold, coalesce(pf.n_peers_active, 0) AS n_peers_active,
            pf.peer_known_at AS crowd_known_at,
            coalesce(pc.n_peer_new, 0) AS n_peer_new, coalesce(pc.n_peer_add, 0) AS n_peer_add,
            coalesce(pc.n_peer_trim, 0) AS n_peer_trim, coalesce(pc.n_peer_exit, 0) AS n_peer_exit,
            pf.share_peers_filed_q, pf.peer_known_at AS peer_known_at, pc.max_peer_known_at,
            -- cluster (null for holdout funds, which were never clustered)
            CASE WHEN tc.cluster IS NOT NULL THEN (tc.cluster < 0)::INT END AS cl_noise,
            CASE WHEN tc.cluster >= 0 THEN pf.cl_size_active END AS cl_size,
            CASE WHEN tc.cluster >= 0 THEN coalesce(pc.cl_n_hold, 0) END AS cl_n_hold,
            CASE WHEN tc.cluster >= 0 THEN coalesce(pc.cl_n_new, 0) END AS cl_n_new,
            CASE WHEN tc.cluster >= 0 THEN coalesce(pc.cl_n_add, 0) END AS cl_n_add,
            CASE WHEN tc.cluster >= 0 THEN coalesce(pc.cl_n_trim, 0) END AS cl_n_trim,
            CASE WHEN tc.cluster >= 0 THEN coalesce(pc.cl_n_exit, 0) END AS cl_n_exit,
            CASE WHEN tc.cluster >= 0 THEN coalesce(pc.cl_n_hold, 0) / nullif(pf.cl_size_active, 0) END AS cl_frac_hold,
            CASE WHEN tc.cluster >= 0 THEN coalesce(pc.cl_n_new, 0) + coalesce(pc.cl_n_add, 0)
                                          - coalesce(pc.cl_n_trim, 0) - coalesce(pc.cl_n_exit, 0) END AS cl_net_buy,
            tc.cl_quarter, tc.cl_known_at AS cluster_known_at
        FROM positions p
        LEFT JOIN position_moves m ON m.fund_id = p.fund_id AND m.cusip = p.cusip AND m.quarter = p.quarter
        JOIN targets t ON t.fund_id = p.fund_id AND t.quarter = p.quarter
        JOIN fund_quarter fq ON fq.fund_id = p.fund_id AND fq.quarter = p.quarter
        LEFT JOIN positions pp ON pp.fund_id = p.fund_id AND pp.cusip = p.cusip AND pp.put_call = ''
             AND {QIDX.format(q='pp.quarter')} = t.q_idx - 1
        LEFT JOIN gain g ON g.fund_id = p.fund_id AND g.cusip = p.cusip AND g.quarter = p.quarter
        LEFT JOIN stock_quarter sq ON sq.cusip = p.cusip AND sq.quarter = p.quarter
        LEFT JOIN since si ON si.symbol = p.symbol AND si.quarter = p.quarter AND si.pred_date = t.pred_date
        LEFT JOIN prior pr ON pr.fund_id = p.fund_id AND pr.quarter = p.quarter
        LEFT JOIN peer_counts pc ON pc.fund_id = p.fund_id AND pc.quarter = p.quarter AND pc.cusip = p.cusip
        LEFT JOIN peer_fq pf ON pf.fund_id = p.fund_id AND pf.quarter = p.quarter
        LEFT JOIN target_cluster tc ON tc.fund_id = p.fund_id AND tc.quarter = p.quarter
        WHERE p.put_call = '' AND sq.cusip IS NOT NULL
    """)
    con.execute("""
        CREATE OR REPLACE TABLE c1_panel AS
        SELECT * EXCLUDE (forced_exit) FROM c1_all WHERE action IS NOT NULL AND NOT forced_exit
    """)

    # kept for C2 and the leakage audit: who each target fund-quarter could see, and its cluster
    con.execute("CREATE OR REPLACE TABLE task_c_peer_latest AS SELECT * FROM peer_latest")
    con.execute("CREATE OR REPLACE TABLE task_c_cluster AS SELECT * FROM target_cluster")

    # C2 targets: long positions held at q+1 and not at q (both quarters filed)
    con.execute(f"""
        CREATE OR REPLACE TABLE new_buys AS
        WITH h AS (SELECT DISTINCT fund_id, cohort, cusip, {QIDX.format(q='quarter')} AS q_idx
                   FROM fund_holdings WHERE put_call = '')
        SELECT f.fund_id, f.cohort, f.quarter, n.cusip, f.known_at AS pred_date, f1.known_at AS label_known_at,
               CASE WHEN sq.cusip IS NOT NULL THEN NOT {BAD_RET.replace('ret_source', 'sq.ret_source')}
                    ELSE sq1.ret_q IS NOT NULL END AS ret_clean
        FROM filings f
        JOIN filings f1 ON f1.fund_id = f.fund_id AND f1.q_idx = f.q_idx + 1
        JOIN h n ON n.fund_id = f.fund_id AND n.q_idx = f.q_idx + 1
        LEFT JOIN h c ON c.fund_id = f.fund_id AND c.cusip = n.cusip AND c.q_idx = f.q_idx
        JOIN stock_quarter sq1 ON sq1.cusip = n.cusip AND sq1.quarter = f1.quarter
        LEFT JOIN stock_quarter sq ON sq.cusip = n.cusip AND sq.quarter = f.quarter
        WHERE c.cusip IS NULL AND f.quarter >= '{FIRST_PANEL_QUARTER}'
    """)


if __name__ == "__main__":
    with db.connect() as con:
        build(con)
        print(con.execute("""
            SELECT left(quarter, 4) AS y, count(*) AS rows, avg(ret_clean::INT) AS clean,
                   avg((cl_quarter < quarter)::INT) AS cluster_from_earlier_q,
                   avg(share_peers_filed_q) AS peers_filed_q
            FROM c1_panel GROUP BY 1 ORDER BY 1
        """).pl())
        print(con.execute("SELECT left(quarter, 4) AS y, count(*) AS new_buys FROM new_buys GROUP BY 1 ORDER BY 1").pl())
