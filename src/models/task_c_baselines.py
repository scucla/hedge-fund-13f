"""Task C baselines, written down before any model is trained (plan Milestone 6).

Usage:
    uv run python -m src.models.task_c2 baselines   # first: the C2 baseline numbers
    uv run python -m src.models.task_c_baselines    # -> docs/task_c_baselines.md

C1 baselines (class probabilities per row):
  persistence   "hold" for everything (probability 1, so its log-loss is huge)
  class_prior   the fit period's class shares for every row
  fund_prior    the fund's own historical action shares, from labels public by
                the prediction date (plan: "the fund's own historical turnover rates")
  fund_last_q   the fund's action shares on its previous quarter's book
Funds with no history fall back to the class prior. Validation is scored with
the train fit, test with the train + val fit (the class prior is the only fitted part).
"""

import numpy as np
import polars as pl

from src.models.task_c2 import RESULTS_PATH
from src.models.task_c_common import CLASSES, c1_metrics, frames, load_c1, md_table, subsets
from src.securities import db

DOC = "docs/task_c_baselines.md"


def baseline_proba(df: pl.DataFrame, name: str, class_prior: np.ndarray) -> np.ndarray:
    n = len(df)
    if name == "persistence":
        p = np.zeros((n, 4))
        p[:, CLASSES.index("hold")] = 1
        return p
    if name == "class_prior":
        return np.tile(class_prior, (n, 1))
    prefix = {"fund_prior": "prior_", "fund_last_q": "last_"}[name]
    p = df.select([f"{prefix}{c}" for c in CLASSES]).to_numpy().astype(float)
    if name == "fund_last_q":   # no previous quarter: use the fund's longer history
        alt = df.select([f"prior_{c}" for c in CLASSES]).to_numpy().astype(float)
        p = np.where(np.isnan(p), alt, p)
    return np.where(np.isnan(p), class_prior, p)


def c1_table(fr: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows = []
    for split, fit in (("val", "train"), ("test", "trainval"), ("test_holdout", "trainval")):
        prior = np.bincount(fr[fit]["y"].to_numpy(), minlength=4) / len(fr[fit])
        df = fr[split]
        for subset, mask in subsets(df).items():
            d = df.filter(mask)
            for name in ("persistence", "class_prior", "fund_prior", "fund_last_q"):
                m = c1_metrics(d["y"].to_numpy(), baseline_proba(d, name, prior))
                rows.append({"split": split, "subset": subset, "baseline": name} | m)
    return pl.DataFrame(rows)


def style_shares(df: pl.DataFrame) -> pl.DataFrame:
    return (df.filter(pl.col("cohort") == "seed").with_columns(pl.col("style").fill_null("unlabelled"))
            .group_by("style").agg(pl.len().alias("rows"), pl.col("fund_id").n_unique().alias("funds"),
                                   *[(pl.col("action") == c).mean().alias(c) for c in CLASSES])
            .sort("rows", descending=True))


def main() -> None:
    with db.connect(read_only=True) as con:
        df = load_c1(con)
    fr = frames(df)
    c1 = c1_table(fr)
    c2 = pl.read_parquet(RESULTS_PATH.replace(".parquet", "_baselines.parquet"))
    sizes = pl.DataFrame([{"frame": k, "rows": len(v), "funds": v["fund_id"].n_unique(),
                           "first_q": v["quarter"].min(), "last_q": v["quarter"].max(),
                           "max_label_known_at": str(v["label_known_at"].max()),
                           "first_pred_date": str(v["pred_date"].min())} for k, v in fr.items()])
    in_split = df.filter(pl.col("quarter") >= "2018Q1")
    by_split = (fr["train"].with_columns(pl.lit("train").alias("frame"))
                .vstack(fr["val"].with_columns(pl.lit("val").alias("frame")))
                .vstack(fr["test"].with_columns(pl.lit("test").alias("frame")))
                .group_by("frame").agg(*[(pl.col("action") == c).mean().alias(c) for c in CLASSES]).sort("frame"))
    embargoed = (len(df.filter((pl.col("cohort") == "seed") & (pl.col("quarter").is_between(pl.lit("2018Q1"), pl.lit("2021Q4")))))
                 - len(fr["train"]))

    cols = ["split", "subset", "baseline", "n", "macro_f1", "log_loss", "accuracy", "exit_auc",
            "R_exit", "R_trim", "R_hold", "R_add"]
    c2_cols = ["split", "cohort", "subset", "method", "fund_quarters", "recall@20", "ndcg@20", "hit_rate", "reachable"]
    text = f"""# Task C baselines

Written by `src/models/task_c_baselines.py` **before any C1 or C2 model was trained**. Splits by the
row's quarter: train 2018-2021, validate 2022, test 2023Q1-2026Q1 (the last quarter with a label). Seed
funds only unless the split says `holdout`. The embargo (see `src/models/task_c_common.py`) removed
{embargoed:,} train rows whose labels became public after the first 2022 prediction date.

## Frames

{md_table(sizes)}

## Class shares

Hold is the *smallest* class, not the 50-60% the plan guessed: these are active managers with 10% bands,
so "predict hold" is a weak baseline and accuracy is meaningless. Macro-F1 and log-loss are the metrics.

{md_table(by_split)}

By style (seed funds, all quarters 2018-2026Q1):

{md_table(style_shares(in_split))}

## C1 baselines

`ret_clean` drops rows whose stock's return over the quarter of the move is a 0% deal fallback or
missing (`ret_source`); `excl_multi_quant` drops multi-manager and quant funds. Persistence's log-loss is
from probability 1 on hold, clipped at 1e-6.

{md_table(c1.select(cols))}

## C2 baselines (new buys, top 20)

Per fund-quarter with at least one new buy, equal weight. `reachable` = share of new buys held in any
book visible on the prediction date (the most any recommender over those books could recall).

{md_table(c2.filter(pl.col("split") != "val_grid").select(c2_cols).sort("split", "cohort", "subset", "method"))}
"""
    with open(DOC, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
