"""Shared pieces for Task C: time splits with an embargo, data loading, metrics.

Splits are by the row's quarter q (the quarter whose filing the prediction is
made from):
  train   2018Q1-2021Q4   fit
  val     2022            every choice (hyperparameters, feature variants) is
                          made here; one market regime only (see decisions.md)
  test    2023Q1 on       scored once, by a model refitted on train + val with
                          the validation choices frozen

**Embargo.** A training row's label is public at the fund's q+1 filing, so the
2021Q4 labels arrive in Feb-May 2022, inside validation. Fitting on them would
use outcomes that were unknown when the first validation predictions were made.
`embargo` drops every training row whose label_known_at is after the first
prediction date of the period being scored (validation for the train fit, test
for the train + val refit).

Training uses seed funds only (the universe rule in src.securities.universe);
holdout funds are scored separately as funds the model never saw.
"""

from datetime import date

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, log_loss, precision_recall_fscore_support, roc_auc_score

from src.models import style_classifier

CLASSES = ["exit", "trim", "hold", "add"]
SPLITS = {"train": ("2018Q1", "2021Q4"), "val": ("2022Q1", "2022Q4"), "test": ("2023Q1", "2099Q4")}
MULTI = ("multi_manager", "quant")   # plan pitfall 6: thousands of positions, very different behaviour


def split_of(quarter: pl.Expr) -> pl.Expr:
    expr = pl.lit(None, dtype=pl.Utf8)
    for name, (lo, hi) in reversed(SPLITS.items()):
        expr = pl.when((quarter >= lo) & (quarter <= hi)).then(pl.lit(name)).otherwise(expr)
    return expr


def embargo(fit: pl.DataFrame, scored: pl.DataFrame) -> tuple[pl.DataFrame, date]:
    """Drop fit rows whose label was not yet public on the first prediction date of `scored`."""
    cutoff = scored["pred_date"].min()
    return fit.filter(pl.col("label_known_at") <= cutoff), cutoff


def frames(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """train / val / trainval (seed, embargoed) and the scored sets val, test, test_holdout."""
    df = df.with_columns(split_of(pl.col("quarter")).alias("split")).filter(pl.col("split").is_not_null())
    seed = df.filter(pl.col("cohort") == "seed")
    val = seed.filter(pl.col("split") == "val")
    test = seed.filter(pl.col("split") == "test")
    test_holdout = df.filter((pl.col("cohort") == "holdout") & (pl.col("split") == "test"))
    # the first prediction date of a period counts every fund scored in it, holdouts included
    train, _ = embargo(seed.filter(pl.col("split") == "train"), df.filter(pl.col("split") == "val"))
    trainval, _ = embargo(seed.filter(pl.col("split").is_in(["train", "val"])), df.filter(pl.col("split") == "test"))
    return {"train": train, "val": val, "trainval": trainval, "test": test, "test_holdout": test_holdout}


def load_c1(con) -> pl.DataFrame:
    styles = style_classifier.labels().select("fund_id", pl.col("label_v2").alias("style"))
    return (con.execute("SELECT * FROM c1_panel").pl()
            .join(styles, on="fund_id", how="left")
            .with_columns(pl.col("action").replace_strict({c: i for i, c in enumerate(CLASSES)},
                                                          return_dtype=pl.Int32).alias("y")))


def c1_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """macro-F1, per-class precision/recall, log-loss, accuracy, and AUC of exit vs not."""
    proba = np.clip(proba, 1e-6, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = proba.argmax(axis=1)
    p, r, _, _ = precision_recall_fscore_support(y, pred, labels=range(len(CLASSES)), zero_division=0)
    out = {"n": len(y), "macro_f1": f1_score(y, pred, average="macro", labels=range(len(CLASSES))),
           "log_loss": log_loss(y, proba, labels=range(len(CLASSES))), "accuracy": float((pred == y).mean())}
    exit_y = y == CLASSES.index("exit")
    out["exit_auc"] = roc_auc_score(exit_y, proba[:, 0]) if 0 < exit_y.sum() < len(y) else float("nan")
    for i, c in enumerate(CLASSES):
        out[f"P_{c}"], out[f"R_{c}"] = p[i], r[i]
    return out


def subsets(df: pl.DataFrame) -> dict[str, pl.Series]:
    """Row masks every result is reported on: all rows, without 0%-fallback/missing
    returns, and without the multi-manager / quant funds."""
    return {"all": pl.Series([True] * len(df)),
            "ret_clean": df["ret_clean"],
            "excl_multi_quant": ~df["style"].is_in(MULTI)}


def md_table(df: pl.DataFrame, digits: int = 3) -> str:
    cols = df.columns
    fmt = lambda v: f"{v:.{digits}f}" if isinstance(v, float) else ("" if v is None else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.iter_rows()]
    return "\n".join(lines)
