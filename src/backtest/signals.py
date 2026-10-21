"""Frozen Task C signals for the backtest (test years 2023+ only).

Usage:
    uv run python -m src.backtest.signals    # -> tables backtest_c1_scores, backtest_c2_lists

C1: `lgbm_full` refitted on the embargoed train + val frame with the
hyperparameters chosen on 2022 (data/processed/task_c1_choices.json), exactly
as Milestone 6 scored the test years; the booster is saved next to it. It
scores *every* seed stock-universe position from 2023Q1 (`c1_all`), not only
the labelled ones: which positions get a next-quarter label depends on whether
the fund files again, which a trader can't know on the filing date.

C2: each seed fund-quarter's ranked list from ALS (settings chosen on 2022,
read from the Milestone 6 results) and from the re-entry baseline, for every
seed fund-quarter that filed, not only those that later made a new buy.
"""

import ast
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from src.models import task_c2
from src.models.task_c1 import OUT as C1_OUT, VARIANTS, X_of, features, fit_final
from src.models.task_c_common import CLASSES, frames, load_c1
from src.securities import db

FIRST_QUARTER = "2023Q1"
C1_VARIANT = "lgbm_full"
LIST_DEPTH = 60   # kept per fund-quarter, so the backtest can drop ETFs and still have 20 names


def c1_scores(con) -> pl.DataFrame:
    choice = json.load(open(f"{C1_OUT}_choices.json"))[C1_VARIANT]["params"]
    cols = features(VARIANTS[C1_VARIANT][1])
    model = fit_final("lgbm", cols, choice, frames(load_c1(con))["trainval"])
    model.save_model(f"{C1_OUT}_{C1_VARIANT}.txt")
    df = con.execute(f"""SELECT * FROM c1_all WHERE cohort = 'seed' AND quarter >= '{FIRST_QUARTER}'""").pl()
    p = model.predict(X_of(df, cols))
    return df.select("fund_id", "quarter", "cusip", "pred_date").with_columns(
        *[pl.Series(f"p_{c}", p[:, i]) for i, c in enumerate(CLASSES)])


def c2_lists(con, cohort: str = "seed", quarter: str | None = None, methods: dict | None = None,
             depth: int = LIST_DEPTH) -> pl.DataFrame:
    """Ranked lists for every fund-quarter of `cohort` that filed (the app also scores holdouts),
    from FIRST_QUARTER on or for one `quarter` (the forward test). `methods` maps method -> settings;
    by default ALS with the Milestone 6 choice, and re-entry."""
    first = quarter or FIRST_QUARTER
    data = task_c2.load(con, first)
    if methods is None:
        res = pl.read_parquet(task_c2.RESULTS_PATH)
        methods = {"als": ast.literal_eval(res.filter((pl.col("method") == "als") & (pl.col("split") == "test"))["params"][0]),
                   "reentry": {}}
    # every seed fund-quarter that filed, with no targets attached
    fq = con.execute(f"""
        SELECT DISTINCT t.fund_id, c.cohort, t.quarter, left(t.quarter, 4)::INT * 4 + right(t.quarter, 1)::INT AS q_idx,
               t.pred_date
        FROM task_c_peer_latest t JOIN (SELECT DISTINCT fund_id, cohort FROM positions) c USING (fund_id)
        WHERE t.quarter {'=' if quarter else '>='} '{first}' AND c.cohort = '{cohort}'
    """).pl()
    targets = fq.with_columns(pl.lit(None, dtype=pl.Utf8).alias("cusip"), pl.lit(True).alias("ret_clean"),
                              pl.lit(None, dtype=pl.Utf8).alias("style"), pl.lit("test").alias("split"),
                              pl.lit(None, dtype=pl.Date).alias("label_known_at"))
    d = task_c2.Data(data.books, targets, data.peers, data.clusters)
    out = []
    for method, params in methods.items():
        for quarter, pred_date in targets.select("quarter", "pred_date").unique().sort("quarter", "pred_date").iter_rows():
            recs, audit = task_c2.score_group(d, quarter, pred_date, method, params)
            assert audit["max_input_known_at"] <= pred_date
            for r in recs:
                out += [{"method": method, "fund_id": r["fund_id"], "quarter": quarter, "pred_date": pred_date,
                         "cusip": c, "rank": i + 1} for i, c in enumerate(r["ranked"][:depth])]
    return pl.DataFrame(out)


def main() -> None:
    with db.connect(read_only=True) as con:
        s1 = c1_scores(con)
        s2 = c2_lists(con)
    with db.connect() as con:
        con.register("s1", s1)
        con.register("s2", s2)
        con.execute("CREATE OR REPLACE TABLE backtest_c1_scores AS SELECT * FROM s1")
        con.execute("CREATE OR REPLACE TABLE backtest_c2_lists AS SELECT * FROM s2")
    print(s1.group_by("quarter").agg(pl.len(), pl.col("p_add").mean()).sort("quarter"))
    print(s2.group_by("method", "quarter").agg(pl.col("fund_id").n_unique()).sort("method", "quarter"))


if __name__ == "__main__":
    main()
