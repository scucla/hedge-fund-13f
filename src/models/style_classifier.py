"""Style classifier (Milestone 4, Task A, layer 2): LightGBM on fund fingerprints.

Usage:
    uv run python -m src.models.style_classifier

Label: the seed list's `style_guess`, one weak label per fund, constant over
time. Rows: fund-quarters from `fund_fingerprint`. Split by time only:
train 2018Q1-2022Q4, test 2023Q1 on (models start in 2018 because prices
before that are survivor-skewed; see docs/decisions.md). Hyperparameters are
fixed below and were not tuned on anything, least of all the test years.

Two evaluations, because the same funds appear in train and test:
  time split         train on 2018-2022, test on 2023+ (same funds): is a
                     fund's style recognisable in later years? A model that
                     memorises each fund's fingerprint also scores well here.
  funds held out     5 folds over funds; train on 2018-2022 rows of the other
                     funds, predict 2023+ rows of the held-out funds: does it
                     work on a fund it has never seen? This is the honest number.

Writes `style_predictions` (time-split model, every fund-quarter 2018+),
docs/figures/06_style_confusion.png, docs/figures/07_style_shap.png and
docs/style_classifier.md (metrics, disagreements).
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
FEATURES = (
    ["log_n_positions", "top10_weight", "hhi", "active_share", "turnover", "gross_turnover",
     "holding_period_q", "long_held_share",
     "tilt_size", "tilt_value", "tilt_momentum", "tilt_profitability"]
    + [f"sector_{s}" for s in FF12] + ["sector_none"]
)
# Coverage columns (cov_*) are left out on purpose: price coverage rises with
# the calendar year, so they would let the model tell train years from test years.
PARAMS = dict(
    objective="multiclass", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    class_weight="balanced",   # 4 quant funds vs 18 growth/tech: don't let the big classes win by default
    random_state=0, verbose=-1,
)
FIG = Path("docs/figures")
REPORT = Path("docs/style_classifier.md")
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
BLUE = "#2a78d6"


def load(con) -> pl.DataFrame:
    labels = (pl.read_csv(SEED_PATH).select("fund_name", "style_guess").unique()
              .with_columns(pl.col("fund_name").map_elements(fund_id, return_dtype=pl.Utf8).alias("fund_id"))
              .select("fund_id", "style_guess"))
    rows = con.execute(f"SELECT fund_id, fund_name, quarter, {', '.join(FEATURES)} FROM fund_fingerprint "
                       f"WHERE quarter >= '{TRAIN[0]}' ORDER BY fund_id, quarter").pl()
    return rows.join(labels, on="fund_id", how="inner")


def time_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = df.filter(pl.col("quarter").is_between(pl.lit(TRAIN[0]), pl.lit(TRAIN[1])))
    test = df.filter(pl.col("quarter") >= TEST_FROM)
    return train, test


def fit(train: pl.DataFrame, classes: list[str]) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**PARAMS)
    y = np.array([classes.index(s) for s in train["style_guess"]])
    model.fit(train.select(FEATURES).to_pandas(), y)
    return model


def predict(model, df: pl.DataFrame, classes: list[str]) -> pl.DataFrame:
    proba = model.predict_proba(df.select(FEATURES).to_pandas())
    return df.select("fund_id", "fund_name", "quarter", "style_guess").with_columns(
        pl.Series("predicted", [classes[i] for i in proba.argmax(1)]),
        *[pl.Series(f"p_{c}", proba[:, k]) for k, c in enumerate(classes)],
    )


def by_fund(pred: pl.DataFrame) -> pl.DataFrame:
    """One row per fund: the most frequent prediction over its quarters."""
    return (pred.group_by("fund_id", "fund_name", "style_guess", "predicted").agg(pl.len().alias("n"))
            .sort("n", "predicted", descending=[True, False])
            .group_by("fund_id", "fund_name", "style_guess", maintain_order=True)
            .agg(pl.col("predicted").first().alias("predicted"),
                 (pl.col("n").first() / pl.col("n").sum()).alias("share_of_quarters"),
                 pl.col("n").sum().alias("quarters")))


def scores(y_true, y_pred) -> dict:
    return {"macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred)}


def funds_held_out(df: pl.DataFrame, classes: list[str], folds: int = 5) -> pl.DataFrame:
    train_all, test_all = time_split(df)
    funds = train_all["fund_id"].unique().sort().to_list()
    out = []
    for tr_idx, te_idx in GroupKFold(n_splits=folds).split(funds, groups=funds):
        held = {funds[i] for i in te_idx}
        model = fit(train_all.filter(~pl.col("fund_id").is_in(list(held))), classes)
        out.append(predict(model, test_all.filter(pl.col("fund_id").is_in(list(held))), classes))
    return pl.concat(out)


def shap_values(model, X) -> np.ndarray:
    """(rows, features, classes), whatever shape this shap version returns."""
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    return sv


def plot_confusion(pred: pl.DataFrame, classes: list[str], path: Path) -> None:
    cm = confusion_matrix(pred["style_guess"], pred["predicted"], labels=classes)
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
    ax.set_ylabel("Label (seed list style_guess)", color=INK2)
    f1 = f1_score(pred["style_guess"], pred["predicted"], average="macro", zero_division=0)
    ax.set_title(f"Style classifier, test quarters {TEST_FROM}+ (macro-F1 {f1:.2f})",
                 loc="left", fontweight="bold", fontsize=12, color=INK)
    fig.text(0.02, -0.1, "Cells: fund-quarters; shading: share of the label's row. Trained on 2018Q1–2022Q4 "
             "of the same funds.", color=MUTED, fontsize=8.5)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_shap(sv: np.ndarray, path: Path, top: int = 15) -> pl.DataFrame:
    imp = pl.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(sv).mean(axis=(0, 2))}) \
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


def disagreements(fund_pred: pl.DataFrame, test_pred: pl.DataFrame, sv: np.ndarray, classes: list[str]) -> pl.DataFrame:
    """Funds whose majority test-period prediction differs from the label, with
    the three features pushing hardest toward the predicted style."""
    rows = []
    fund_ids = test_pred["fund_id"].to_list()
    for r in fund_pred.filter(pl.col("predicted") != pl.col("style_guess")).iter_rows(named=True):
        idx = [i for i, f in enumerate(fund_ids) if f == r["fund_id"]]
        push = sv[idx, :, classes.index(r["predicted"])].mean(0)
        top = np.argsort(push)[::-1][:3]
        rows.append({**r, "why": ", ".join(f"{FEATURES[k]} ({push[k]:+.2f})" for k in top)})
    return pl.DataFrame(rows) if rows else fund_pred.head(0).with_columns(pl.lit("").alias("why"))


def write_report(res: dict, imp: pl.DataFrame, dis: pl.DataFrame, path: Path) -> None:
    def table(df: pl.DataFrame) -> str:
        cols = df.columns
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for row in df.iter_rows():
            lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row) + " |")
        return "\n".join(lines)

    metrics = pl.DataFrame([{"evaluation": k, **v} for k, v in res.items()])
    path.write_text(f"""# Style classifier results

Generated by `uv run python -m src.models.style_classifier`. Label: seed-list
`style_guess` (weak, one per fund). Features: `fund_fingerprint`. Train
{TRAIN[0]}–{TRAIN[1]}, test {TEST_FROM}+; hyperparameters fixed, not tuned.

## Metrics

{table(metrics)}

"Time split" tests the same funds in later years; "funds held out" tests funds
the model never saw (5 folds over funds, still trained on 2018–2022 and scored
on {TEST_FROM}+). "Majority class" always predicts the most common label.

## Top features (mean |SHAP|, test rows)

{table(imp.head(15))}

![confusion](figures/06_style_confusion.png)
![shap](figures/07_style_shap.png)

## Funds whose predicted style disagrees with the label (time split, test quarters)

`share_of_quarters` = share of the fund's test quarters with that prediction;
`why` = the three features with the largest mean SHAP toward the predicted style.

{table(dis.select("fund_name", "style_guess", "predicted", "share_of_quarters", "quarters", "why")) if len(dis) else "None."}
""")


def main() -> None:
    con = db.connect()
    df = load(con)
    classes = sorted(df["style_guess"].unique().to_list())
    train, test = time_split(df)
    model = fit(train, classes)
    test_pred = predict(model, test, classes)
    all_pred = pl.concat([predict(model, train, classes).with_columns(pl.lit("train").alias("split")),
                          test_pred.with_columns(pl.lit("test").alias("split"))])
    con.execute("CREATE OR REPLACE TABLE style_predictions AS SELECT * FROM all_pred")

    held = funds_held_out(df, classes)
    majority = train["style_guess"].mode().sort()[0]
    fund_test, fund_held = by_fund(test_pred), by_fund(held)
    res = {
        "time split, fund-quarters": scores(test_pred["style_guess"], test_pred["predicted"]),
        "time split, funds (majority vote)": scores(fund_test["style_guess"], fund_test["predicted"]),
        "funds held out, fund-quarters": scores(held["style_guess"], held["predicted"]),
        "funds held out, funds (majority vote)": scores(fund_held["style_guess"], fund_held["predicted"]),
        "majority class, fund-quarters": scores(test["style_guess"], [majority] * len(test)),
    }
    sv = shap_values(model, test.select(FEATURES).to_pandas())
    FIG.mkdir(parents=True, exist_ok=True)
    plot_confusion(test_pred, classes, FIG / "06_style_confusion.png")
    imp = plot_shap(sv, FIG / "07_style_shap.png")
    dis = disagreements(fund_test, test_pred, sv, classes)
    write_report(res, imp, dis, REPORT)

    print(f"train {len(train):,} rows / {train['fund_id'].n_unique()} funds; test {len(test):,} rows / "
          f"{test['fund_id'].n_unique()} funds; classes {classes}")
    print(pl.DataFrame([{"evaluation": k, **{m: round(x, 3) for m, x in v.items()}} for k, v in res.items()]))
    print(imp.head(15))
    with pl.Config(tbl_rows=40, fmt_str_lengths=90, tbl_width_chars=250):
        print(dis.select("fund_name", "style_guess", "predicted", "share_of_quarters", "why") if len(dis) else "no disagreements")
    print(f"wrote style_predictions, {FIG}/06_style_confusion.png, {FIG}/07_style_shap.png, {REPORT}")


if __name__ == "__main__":
    main()
