"""Leakage audit for Task C: every feature's timestamp against the prediction date.

Usage:
    uv run python -m src.models.leakage_audit    # -> docs/task_c_leakage_audit.md, exit 1 on any violation

The prediction date of a C1 row / C2 fund-quarter is the fund's own quarter-q
13F-HR filing date. Checks (each counts violations; all must be 0):

  known_at:<group>     every feature group's known_at <= pred_date, and a group
                       with any non-null feature has a known_at
  label_after_pred     label_known_at > pred_date
  peer_recompute       fix (1): n_peer_hold recomputed independently from raw
                       positions for a sample of fund-quarters, using only peer
                       filings with known_at <= pred_date, matches the panel
  peer_filing_dates    every peer filing behind a target is public by pred_date
  cluster_vintage      fix (2): the clustering used was public by pred_date (so a
                       quarter-q clustering fitted on later filings is never used)
  embargo:train|trainval  fix (3): no fit row's label is public after the first
                       prediction date of the period it is scored against
  fit_frames           fit frames hold seed rows of their own years only
  amendments           a fund-quarter with a later 13F-HR/A is predicted from its
                       original filing date, before the amendment
  forbidden_features   no model feature is an outcome column (ret_next_q,
                       ret_source, ret_clean, action, labels, forced_exit) or
                       `positions.crowding` (quarter-q, not as-of)
  c2_inputs            every book a C2 recommender used was public by pred_date

Also reported (not violations): how many C1 rows a naive quarter-q peer count
would have filled from filings published after the prediction date.
"""

import sys

import polars as pl

from src.features.task_c import FEATURE_GROUPS, FORBIDDEN, KNOWN_AT
from src.models.task_c_common import frames, load_c1, md_table
from src.securities import db

DOC = "docs/task_c_leakage_audit.md"
C2_AUDIT = "data/processed/task_c2_results_audit.parquet"
N_PEER_SAMPLE = 300

# a group's known_at column, and extra timestamp columns that must also be <= pred_date
EXTRA_KNOWN_AT = {"peer": ["max_peer_known_at"], "cluster": []}
GROUP_KNOWN_AT = KNOWN_AT | {"cluster": "cluster_known_at"}


def check_known_at(df: pl.DataFrame) -> list[dict]:
    out = []
    for g, cols in FEATURE_GROUPS.items():
        k = GROUP_KNOWN_AT[g]
        stamps = [k] + EXTRA_KNOWN_AT.get(g, [])
        late = df.filter(pl.any_horizontal([pl.col(s) > pl.col("pred_date") for s in stamps]))
        has_feature = pl.any_horizontal([pl.col(c).is_not_null() for c in cols if c not in ("n_peer_hold", "n_peers_active")]
                                        or [pl.lit(False)])
        undated = df.filter(has_feature & pl.col(k).is_null())
        out.append({"check": f"known_at:{g}", "rows_checked": len(df), "violations": len(late) + len(undated),
                    "note": f"{len(late)} late, {len(undated)} with features but no {k}"})
    bad = df.filter(pl.col("label_known_at") <= pl.col("pred_date"))
    out.append({"check": "label_after_pred", "rows_checked": len(df), "violations": len(bad), "note": ""})
    return out


def check_embargo(fr: dict[str, pl.DataFrame], scored_all: dict[str, pl.DataFrame]) -> list[dict]:
    out = []
    for fit, period in (("train", "val"), ("trainval", "test")):
        first = scored_all[period]["pred_date"].min()
        late = fr[fit].filter(pl.col("label_known_at") > first)
        out.append({"check": f"embargo:{fit}", "rows_checked": len(fr[fit]), "violations": len(late),
                    "note": f"first {period} prediction {first}, latest {fit} label {fr[fit]['label_known_at'].max()}"})
    wrong = (fr["train"].filter((pl.col("quarter") > "2021Q4") | (pl.col("cohort") != "seed")).height
             + fr["trainval"].filter((pl.col("quarter") > "2022Q4") | (pl.col("cohort") != "seed")).height)
    out.append({"check": "fit_frames", "rows_checked": len(fr["train"]) + len(fr["trainval"]), "violations": wrong,
                "note": "fit rows outside their years or not seed"})
    return out


def peer_recompute(positions: pl.DataFrame, filings: pl.DataFrame, panel: pl.DataFrame) -> tuple[int, int]:
    """Independent n_peer_hold: for each panel fund-quarter, count other seed funds holding the
    stock in their latest filing for a quarter <= q, public by pred_date, and not older than q-1.
    Returns (rows compared, mismatches)."""
    seed_f = filings.filter(pl.col("cohort") == "seed")
    rows = mism = 0
    for (f, q, q_idx, T), grp in panel.group_by("fund_id", "quarter", "q_idx", "pred_date"):
        vis = (seed_f.filter((pl.col("fund_id") != f) & (pl.col("known_at") <= T) & (pl.col("q_idx") <= q_idx))
               .sort("q_idx").group_by("fund_id").last().filter(pl.col("q_idx") >= q_idx - 1))
        held = (positions.join(vis.select("fund_id", "q_idx"), on=["fund_id", "q_idx"])
                .group_by("cusip").agg(pl.col("fund_id").n_unique().alias("n")))
        cmp = grp.join(held, on="cusip", how="left").with_columns(pl.col("n").fill_null(0))
        rows += len(cmp)
        mism += cmp.filter(pl.col("n") != pl.col("n_peer_hold")).height
    return rows, mism


def check_peers(con, panel: pl.DataFrame) -> list[dict]:
    q_idx = pl.col("quarter").str.slice(0, 4).cast(pl.Int32) * 4 + pl.col("quarter").str.slice(5, 1).cast(pl.Int32)
    positions = (con.execute("SELECT fund_id, quarter, cusip FROM positions WHERE put_call = '' AND cohort = 'seed'").pl()
                 .with_columns(q_idx.alias("q_idx")).unique())
    filings = (con.execute("""SELECT fund_id, any_value(cohort) AS cohort, quarter, max(known_at) AS known_at FROM
                              (SELECT fund_id, cohort, quarter, known_at FROM positions
                               UNION ALL BY NAME SELECT fund_id, cohort, quarter, known_at FROM empty_books)
                              GROUP BY fund_id, quarter""").pl().with_columns(q_idx.alias("q_idx")))
    fq = panel.select("fund_id", "quarter").unique().sort("fund_id", "quarter").sample(n=N_PEER_SAMPLE, seed=0)
    sample = panel.join(fq, on=["fund_id", "quarter"]).select("fund_id", "quarter", "pred_date", "cusip", "n_peer_hold") \
                  .with_columns(q_idx.alias("q_idx"))
    rows, mism = peer_recompute(positions, filings, sample)

    late = con.execute("""SELECT count(*) FROM task_c_peer_latest
                          WHERE peer_known_at > pred_date OR peer_q_idx > q_idx""").fetchone()[0]
    n_pl = con.execute("SELECT count(*) FROM task_c_peer_latest").fetchone()[0]
    # what a naive "other funds' quarter-q filings" count would have used
    naive = con.execute("""
        SELECT count(DISTINCT (c.fund_id, c.quarter, c.cusip)) FROM c1_panel c
        JOIN positions p ON p.quarter = c.quarter AND p.cusip = c.cusip AND p.put_call = ''
         AND p.cohort = 'seed' AND p.fund_id <> c.fund_id AND p.known_at > c.pred_date
    """).fetchone()[0]
    return [{"check": "peer_recompute", "rows_checked": rows, "violations": mism,
             "note": f"{N_PEER_SAMPLE} sampled fund-quarters, independent count from raw positions"},
            {"check": "peer_filing_dates", "rows_checked": n_pl, "violations": late,
             "note": f"info: a naive quarter-q peer count would use a not-yet-public filing on {naive:,} of {len(panel):,} rows"}]


def check_clusters(con, df: pl.DataFrame) -> list[dict]:
    late = df.filter((pl.col("cluster_known_at") > pl.col("pred_date")) | (pl.col("cl_quarter") > pl.col("quarter")))
    # was quarter q's own clustering used although it was published after pred_date?
    vintage = con.execute("SELECT quarter, min(known_at) AS q_cluster_known_at FROM fund_clusters GROUP BY 1").pl()
    d = df.filter(pl.col("cluster_known_at").is_not_null()).join(vintage, on="quarter", how="left")
    used_future = d.filter((pl.col("cl_quarter") == pl.col("quarter")) & (pl.col("q_cluster_known_at") > pl.col("pred_date")))
    share_prev = d.filter(pl.col("cl_quarter") < pl.col("quarter")).height / max(len(d), 1)
    return [{"check": "cluster_vintage", "rows_checked": len(d), "violations": len(late) + len(used_future),
             "note": f"{share_prev:.0%} of rows use an earlier quarter's clustering (q's was not yet public)"}]


def check_amendments(con, df: pl.DataFrame) -> list[dict]:
    """Fund-quarters whose restated view has a later filing than the point-in-time view."""
    am = con.execute("""
        WITH fid AS (SELECT DISTINCT fund_id, fund_name FROM positions),
        r AS (SELECT fund_name, quarter, max(known_at) AS restated_known_at FROM seed_positions GROUP BY ALL),
        p AS (SELECT fund_name, quarter, max(known_at) AS pit_known_at FROM seed_positions_pit GROUP BY ALL)
        SELECT fid.fund_id, r.quarter, r.restated_known_at, p.pit_known_at
        FROM r JOIN p USING (fund_name, quarter) JOIN fid USING (fund_name)
        WHERE r.restated_known_at > p.pit_known_at
    """).pl()
    fq = df.select("fund_id", "quarter", "pred_date").unique().join(am, on=["fund_id", "quarter"])
    bad = fq.filter((pl.col("pred_date") >= pl.col("restated_known_at")) | (pl.col("pred_date") > pl.col("pit_known_at")))
    return [{"check": "amendments", "rows_checked": len(fq), "violations": len(bad),
             "note": f"{len(fq)} panel fund-quarters have a later 13F-HR/A; all predicted from the original filing"}]


def check_features(variants: dict[str, list[str]]) -> list[dict]:
    used = {c for cols in variants.values() for c in cols}
    bad = sorted((used & FORBIDDEN) | {c for c in used if "ret_next" in c or "label" in c})
    return [{"check": "forbidden_features", "rows_checked": len(used), "violations": len(bad), "note": ", ".join(bad)}]


def check_c2(path: str = C2_AUDIT) -> list[dict]:
    try:
        a = pl.read_parquet(path)
    except FileNotFoundError:
        return [{"check": "c2_inputs", "rows_checked": 0, "violations": 1, "note": f"{path} missing: run task_c2 first"}]
    late = a.filter(pl.col("max_input_known_at") > pl.col("pred_date"))
    return [{"check": "c2_inputs", "rows_checked": len(a), "violations": len(late),
             "note": "recommender inputs per (method, quarter, prediction date)"}]


def audit(con) -> pl.DataFrame:
    from src.models.task_c1 import VARIANTS, features

    df = load_c1(con)
    fr = frames(df)
    from src.models.task_c_common import split_of
    scored_all = {s: df.with_columns(split_of(pl.col("quarter")).alias("split")).filter(pl.col("split") == s)
                  for s in ("val", "test")}
    rows = (check_known_at(df) + check_peers(con, df) + check_clusters(con, df) + check_embargo(fr, scored_all)
            + check_amendments(con, df) + check_features({n: features(g) for n, (_, g) in VARIANTS.items()})
            + check_c2())
    return pl.DataFrame(rows)


def main() -> int:
    with db.connect(read_only=True) as con:
        res = audit(con)
    text = f"""# Task C leakage audit

Written by `src/models/leakage_audit.py` (also run by `tests/test_task_c.py`). The prediction date is the
fund's own quarter-q 13F-HR filing date; every check counts violations and must be 0.

{md_table(res)}
"""
    with open(DOC, "w") as f:
        f.write(text)
    print(text)
    return int(res["violations"].sum() > 0)


if __name__ == "__main__":
    sys.exit(main())
