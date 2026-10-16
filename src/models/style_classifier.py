"""Style classifier (Milestone 4, Task A, layer 2): LightGBM on fund fingerprints.

Usage:
    uv run python -m src.models.style_classifier

Rows: fund-quarters from `fund_fingerprint`. Training uses seed funds only
(`cohort = 'seed'`); the 16 hand-labelled holdout funds
(data/reference/holdout_funds.csv) are scored once, at the end, and never used
to choose anything. Split by time only: train 2018Q1-2022Q4, test 2023Q1 on
(models start in 2018 because prices before that are survivor-skewed; see
docs/decisions.md). Hyperparameters are fixed below and were not tuned.

Label sets:
  v1   the seed list's `style_guess` (one weak label per fund)
  v2   v1 with the reviewed corrections in data/reference/style_labels_v2.csv
       applied when their confidence is high or medium
Variants: v1, v2, and v2 plus the 13D activist features (`activist_13d`).

Evaluations, for each variant:
  time split         same funds, later years: is a fund's style recognisable
                     later? (memorising funds also scores well here)
  funds held out     5 folds over seed funds, trained on 2018-2022 rows of the
                     other funds, scored on the held-out funds' 2023+ rows
  holdout            trained on all seed funds 2018-2022, scored on the
                     holdout funds' 2023+ rows against their own labels
The final model (figures, `style_predictions`, disagreements) is the variant
with the best funds-held-out macro-F1, a seed-only criterion fixed before
looking at the holdout.

Writes `style_predictions` (final model, seed and holdout fund-quarters
2018+), docs/figures/06_style_confusion.png, 07_style_shap.png and
docs/style_classifier.md.
"""

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from src.features.books import fund_id
from src.features.fund_quarter import FF12
from src.ingest.seed_ciks import SEED_PATH
from src.securities import db

TRAIN = ("2018Q1", "2022Q4")
TEST_FROM = "2023Q1"
V2_PATH = Path("data/reference/style_labels_v2.csv")
HOLDOUT_PATH = Path("data/reference/holdout_funds.csv")
V2_CONFIDENCE = ("high", "medium")   # low-confidence proposals stay at v1
BASE_FEATURES = (
    ["log_n_positions", "top10_weight", "hhi", "active_share", "turnover", "gross_turnover",
     "holding_period_q", "long_held_share",
     "tilt_size", "tilt_value", "tilt_momentum", "tilt_profitability"]
    + [f"sector_{s}" for s in FF12] + ["sector_none"]
)
ACTIVIST_FEATURES = ["log_n_13d_2y", "activist_13d"]
# Coverage columns (cov_*) are left out on purpose: price coverage rises with
# the calendar year, so they would let the model tell train years from test years.
PARAMS = dict(
    objective="multiclass", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    class_weight="balanced",   # 4 quant funds vs 18 growth/tech: don't let the big classes win by default
    random_state=0, verbose=-1,
)
VARIANTS = {  # name -> (label column, features)
    "v1": ("label_v1", BASE_FEATURES),
    "v2": ("label_v2", BASE_FEATURES),
    "v2 + 13D": ("label_v2", BASE_FEATURES + ACTIVIST_FEATURES),
}
FIG = Path("docs/figures")
REPORT = Path("docs/style_classifier.md")
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
BLUE = "#2a78d6"


def labels() -> pl.DataFrame:
    """fund_id, cohort, label_v1, label_v2 (holdout: both are the hand label)."""
    seed = (pl.read_csv(SEED_PATH).select("fund_name", "style_guess").unique()
            .rename({"style_guess": "label_v1"}))
    v2 = (pl.read_csv(V2_PATH).filter(pl.col("confidence").is_in(V2_CONFIDENCE))
          .select("fund_name", pl.col("style_v2").alias("override")))
    seed = (seed.join(v2, on="fund_name", how="left")
            .with_columns(pl.coalesce("override", "label_v1").alias("label_v2"), pl.lit("seed").alias("cohort"))
            .drop("override"))
    hold = (pl.read_csv(HOLDOUT_PATH).select("fund_name", pl.col("style").alias("label_v1"))
            .with_columns(pl.col("label_v1").alias("label_v2"), pl.lit("holdout").alias("cohort")))
    return (pl.concat([seed, hold], how="diagonal")
            .with_columns(pl.col("fund_name").map_elements(fund_id, return_dtype=pl.Utf8).alias("fund_id"))
            .select("fund_id", "cohort", "label_v1", "label_v2"))


def load(con) -> pl.DataFrame:
    rows = con.execute(f"""
        SELECT f.fund_id, f.fund_name, f.quarter, f.cohort, {', '.join('f.' + c for c in BASE_FEATURES)},
               ln(1 + a.n_13d_2y) AS log_n_13d_2y, a.activist_13d::INT AS activist_13d
        FROM fund_fingerprint f LEFT JOIN activist_13d a USING (fund_id, quarter)
        WHERE f.quarter >= '{TRAIN[0]}' ORDER BY f.fund_id, f.quarter
    """).pl()
    return rows.join(labels(), on=["fund_id", "cohort"], how="inner")


def time_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = df.filter(pl.col("quarter").is_between(pl.lit(TRAIN[0]), pl.lit(TRAIN[1])))
    test = df.filter(pl.col("quarter") >= TEST_FROM)
    return train, test


def fit(train: pl.DataFrame, label: str, features: list[str], classes: list[str]) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**PARAMS)
    y = np.array([classes.index(s) for s in train[label]])
    model.fit(train.select(features).to_pandas(), y)
    return model


def predict(model, df: pl.DataFrame, label: str, features: list[str], classes: list[str]) -> pl.DataFrame:
    proba = model.predict_proba(df.select(features).to_pandas())
    return df.select("fund_id", "fund_name", "quarter", "cohort", pl.col(label).alias("label")).with_columns(
        pl.Series("predicted", [classes[i] for i in proba.argmax(1)]),
        *[pl.Series(f"p_{c}", proba[:, k]) for k, c in enumerate(classes)],
    )


def by_fund(pred: pl.DataFrame) -> pl.DataFrame:
    """One row per fund: the most frequent prediction over its quarters (ties alphabetical)."""
    return (pred.group_by("fund_id", "fund_name", "label", "predicted").agg(pl.len().alias("n"))
            .sort("n", "predicted", descending=[True, False])
            .group_by("fund_id", "fund_name", "label", maintain_order=True)
            .agg(pl.col("predicted").first().alias("predicted"),
                 (pl.col("n").first() / pl.col("n").sum()).alias("share_of_quarters"),
                 pl.col("n").sum().alias("quarters")))


def scores(pred: pl.DataFrame) -> dict:
    return {"macro_f1": f1_score(pred["label"], pred["predicted"], average="macro", zero_division=0),
            "accuracy": accuracy_score(pred["label"], pred["predicted"]), "n": len(pred)}


def funds_held_out(seed: pl.DataFrame, label: str, features: list[str], classes: list[str], folds: int = 5) -> pl.DataFrame:
    train_all, test_all = time_split(seed)
    funds = train_all["fund_id"].unique().sort().to_list()
    out = []
    for _, te_idx in GroupKFold(n_splits=folds).split(funds, groups=funds):
        held = [funds[i] for i in te_idx]
        model = fit(train_all.filter(~pl.col("fund_id").is_in(held)), label, features, classes)
        out.append(predict(model, test_all.filter(pl.col("fund_id").is_in(held)), label, features, classes))
    return pl.concat(out)


def evaluate(df: pl.DataFrame, name: str, classes: list[str]) -> tuple[dict, dict]:
    label, features = VARIANTS[name]
    seed, hold = df.filter(pl.col("cohort") == "seed"), df.filter(pl.col("cohort") == "holdout")
    train, test = time_split(seed)
    model = fit(train, label, features, classes)
    test_pred = predict(model, test, label, features, classes)
    held = funds_held_out(seed, label, features, classes)
    hold_pred = predict(model, time_split(hold)[1], label, features, classes)
    res = {
        "time split": scores(test_pred), "time split, by fund": scores(by_fund(test_pred)),
        "funds held out": scores(held), "funds held out, by fund": scores(by_fund(held)),
        "holdout": scores(hold_pred), "holdout, by fund": scores(by_fund(hold_pred)),
    }
    return res, {"model": model, "train": train, "test": test, "test_pred": test_pred, "hold_pred": hold_pred,
                 "label": label, "features": features}


def shap_values(model, X) -> np.ndarray:
    """(rows, features, classes), whatever shape this shap version returns."""
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    return sv


def plot_confusion(pred: pl.DataFrame, classes: list[str], title: str, note: str, path: Path) -> None:
    cm = confusion_matrix(pred["label"], pred["predicted"], labels=classes)
    share = cm / cm.sum(1, keepdims=True).clip(1)
    cmap = LinearSegmentedColormap.from_list("blues", [SURFACE, "#9ec5f0", BLUE, "#104281"])
    fig, ax = plt.subplots(figsize=(7.5, 6.4), facecolor=SURFACE)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    names = [c.replace("_", " ") for c in classes]
    ax.set_xticks(range(len(classes)), names, rotation=35, ha="right", color=INK2)
    ax.set_yticks(range(len(classes)), names, color=INK2)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8.5,
                        color=SURFACE if share[i, j] > 0.5 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Predicted style", color=INK2)
    ax.set_ylabel("Label", color=INK2)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12, color=INK)
    fig.text(0.02, -0.1, note, color=MUTED, fontsize=8.5)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_shap(sv: np.ndarray, features: list[str], path: Path, top: int = 15) -> pl.DataFrame:
    imp = pl.DataFrame({"feature": features, "mean_abs_shap": np.abs(sv).mean(axis=(0, 2))}) \
        .sort("mean_abs_shap", descending=True)
    head = imp.head(top).reverse()
    fig, ax = plt.subplots(figsize=(7.5, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(head["feature"], head["mean_abs_shap"], color=BLUE, height=0.65, zorder=3)
    ax.grid(axis="x", color="#e6e5e1", zorder=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)
    ax.set_xlabel("Mean |SHAP| (log-odds), averaged over test rows and classes", color=INK2)
    ax.set_title("What the style classifier looks at", loc="left", fontweight="bold", fontsize=12, color=INK)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return imp


def disagreements(fund_pred: pl.DataFrame, test_pred: pl.DataFrame, sv: np.ndarray, features: list[str],
                  classes: list[str]) -> pl.DataFrame:
    """Funds whose majority test-period prediction differs from the label, with
    the three features pushing hardest toward the predicted style."""
    rows = []
    fund_ids = test_pred["fund_id"].to_list()
    for r in fund_pred.filter(pl.col("predicted") != pl.col("label")).iter_rows(named=True):
        idx = [i for i, f in enumerate(fund_ids) if f == r["fund_id"]]
        push = sv[idx, :, classes.index(r["predicted"])].mean(0)
        top = np.argsort(push)[::-1][:3]
        rows.append({**r, "why": ", ".join(f"{features[k]} ({push[k]:+.2f})" for k in top)})
    return pl.DataFrame(rows) if rows else fund_pred.head(0).with_columns(pl.lit("").alias("why"))


def md_table(df: pl.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def results_table(all_res: dict) -> pl.DataFrame:
    evals = ["time split", "time split, by fund", "funds held out", "funds held out, by fund",
             "holdout", "holdout, by fund"]
    rows = []
    for e in evals:
        row = {"evaluation": e, "n": all_res["v1"][e]["n"]}
        for v in VARIANTS:
            row[f"{v} macro-F1"] = round(all_res[v][e]["macro_f1"], 3)
        for v in VARIANTS:
            row[f"{v} acc"] = round(all_res[v][e]["accuracy"], 3)
        rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    con = db.connect()
    df = load(con)
    classes = sorted(set(df["label_v1"]) | set(df["label_v2"]))
    all_res, runs = {}, {}
    for name in VARIANTS:
        all_res[name], runs[name] = evaluate(df, name, classes)
    best = max(VARIANTS, key=lambda v: all_res[v]["funds held out"]["macro_f1"])   # seed-only criterion
    run = runs[best]
    seed_all = df.filter(pl.col("cohort") == "seed")
    majority = run["train"][run["label"]].mode().sort()[0]
    base = scores(run["test"].select(pl.col(run["label"]).alias("label"), pl.lit(majority).alias("predicted")))

    all_pred = pl.concat([
        predict(run["model"], run["train"], run["label"], run["features"], classes).with_columns(pl.lit("train").alias("split")),
        run["test_pred"].with_columns(pl.lit("test").alias("split")),
        run["hold_pred"].with_columns(pl.lit("holdout").alias("split")),
    ]).with_columns(pl.lit(best).alias("variant"))
    con.execute("CREATE OR REPLACE TABLE style_predictions AS SELECT * FROM all_pred")

    sv = shap_values(run["model"], run["test"].select(run["features"]).to_pandas())
    FIG.mkdir(parents=True, exist_ok=True)
    f1 = all_res[best]["time split"]["macro_f1"]
    plot_confusion(run["test_pred"], classes, f"Style classifier ({best}), test quarters {TEST_FROM}+ (macro-F1 {f1:.2f})",
                   f"Cells: fund-quarters of the seed funds; shading: share of the label's row. Trained on "
                   f"{TRAIN[0]}–{TRAIN[1]} of the same funds.", FIG / "06_style_confusion.png")
    imp = plot_shap(sv, run["features"], FIG / "07_style_shap.png")
    fund_test = by_fund(run["test_pred"])
    dis = disagreements(fund_test, run["test_pred"], sv, run["features"], classes)
    hold_funds = by_fund(run["hold_pred"]).sort("label", "fund_name")
    table = results_table(all_res)

    REPORT.write_text(f"""# Style classifier results

Generated by `uv run python -m src.models.style_classifier`. Features: `fund_fingerprint`
(+ `activist_13d` for the 13D variant). Trained on seed funds {TRAIN[0]}–{TRAIN[1]};
tested on {TEST_FROM}+. Hyperparameters fixed, not tuned. v2 labels apply the high- and
medium-confidence corrections in `data/reference/style_labels_v2.csv`.

## v1 vs v2 vs holdout

{md_table(table)}

- **time split**: same seed funds, later years (scored against that variant's labels).
- **funds held out**: 5 folds over seed funds; the honest number for an unseen fund among the seed funds.
- **holdout**: 16 hand-labelled funds outside the seed list, 2 per style, scored once with a model trained on all
  seed funds; the only evaluation not influenced by the label review.
- "by fund" = majority vote over a fund's quarters. Majority-class baseline (time split, {best} labels):
  macro-F1 {base['macro_f1']:.2f}, accuracy {base['accuracy']:.2f}.

**Caveat:** v2 corrected labels after looking at where the v1 model disagreed, so v2's seed-fund scores
(time split, funds held out) are optimistic by construction. Compare label sets on the holdout only.

Final model (figures, `style_predictions`): **{best}**, chosen by funds-held-out macro-F1 (seed funds only).

## Holdout funds ({best})

{md_table(hold_funds.select("fund_name", "label", "predicted", "share_of_quarters", "quarters"))}

## Top features (mean |SHAP|, seed test rows, {best})

{md_table(imp.head(15))}

![confusion](figures/06_style_confusion.png)
![shap](figures/07_style_shap.png)

## Seed funds whose predicted style still disagrees with the label ({best}, test quarters)

`why` = the three features with the largest mean SHAP toward the predicted style.

{md_table(dis.select("fund_name", "label", "predicted", "share_of_quarters", "quarters", "why")) if len(dis) else "None."}
""")
    with pl.Config(tbl_rows=40, fmt_str_lengths=90, tbl_width_chars=250, tbl_cols=20):
        print(f"seed rows {len(seed_all):,}; holdout rows {len(df) - len(seed_all):,}; classes {classes}")
        print(table)
        print(f"final model: {best}; majority baseline macro-F1 {base['macro_f1']:.3f}")
        print(hold_funds)
        print(imp.head(15))
        print(dis.select("fund_name", "label", "predicted", "share_of_quarters", "why") if len(dis) else "no disagreements")


if __name__ == "__main__":
    main()
