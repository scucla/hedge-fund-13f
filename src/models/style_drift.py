"""Style drift (Milestone 4, Task A): each fund's predicted style over time, and
which funds changed.

Usage:
    uv run python -m src.models.style_drift      (after src.models.style_classifier)

Predictions must come from a model that never saw the fund. `style_predictions`
holds in-sample predictions for seed funds' training quarters (2018-2022), and
a model scored on rows it was fitted on just repeats the fund's label: no drift
would ever show. So every quarter here is scored **out of fund**:
  seed funds     5 folds over funds (same folds as the classifier's funds-held-out
                 evaluation); each fold's model is trained on the other funds'
                 2018-2022 rows and scores every quarter 2018+ of its held-out funds
  holdout funds  the model trained on all seed funds 2018-2022 (they were never in it)
The variant (labels, features) is the classifier's final model, read from
`style_predictions.variant`. This is a description of each fund's history, not a
forecast, so scoring a fund's 2019 rows with a model that saw other funds' 2022
rows is fine; no fund's own label or rows enter its predictions.

A fund **changed style** if its early style differs from its late style, each
the argmax of the mean class probability over its first and last WINDOW quarters
(2 years, to keep one-quarter flickers out), with at least MIN_QUARTERS quarters
in total so the windows don't overlap, **and** both winning mean probabilities
are at least CONFIDENCE. Without that last condition 33 of 93 funds "change",
mostly between two styles the model was never sure of (e.g. 0.38 vs 0.32); the
chart is meant to point at clear moves. `style_differs` keeps the looser flag,
and the report gives the count at several thresholds, since 0.6 is a reading
choice, not a tested one. `stability` is the share of quarters whose prediction
equals the fund's most common one.

Writes `style_drift` (fund-quarters) and `style_drift_funds` (one row per fund),
docs/figures/10_style_drift.png, 11_style_drift_changed.png and docs/style_drift.md.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Patch
from sklearn.model_selection import GroupKFold

from src.models.style_classifier import VARIANTS, fit, load, md_table, predict, time_split
from src.securities import db

WINDOW = 8
MIN_QUARTERS = 2 * WINDOW
CONFIDENCE = 0.6
FOLDS = 5
STYLE_ORDER = ["activist", "value_concentrated", "event_driven", "growth_tech",
               "healthcare", "multi_manager", "quant", "discretionary_other"]
# categorical slots 1-8 in fixed order (validated palette, light surface)
COLORS = dict(zip(STYLE_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                "#e87ba4", "#008300", "#6250d6", "#e34948"]))
FIG = Path("docs/figures")
REPORT = Path("docs/style_drift.md")
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"


def out_of_fund_predictions(df: pl.DataFrame, variant: str, classes: list[str]) -> pl.DataFrame:
    label, features = VARIANTS[variant]
    seed, hold = df.filter(pl.col("cohort") == "seed"), df.filter(pl.col("cohort") == "holdout")
    train_all = time_split(seed)[0]
    funds = train_all["fund_id"].unique().sort().to_list()
    out = []
    for _, te_idx in GroupKFold(n_splits=FOLDS).split(funds, groups=funds):
        held = [funds[i] for i in te_idx]
        model = fit(train_all.filter(~pl.col("fund_id").is_in(held)), label, features, classes)
        out.append(predict(model, seed.filter(pl.col("fund_id").is_in(held)), label, features, classes))
    out.append(predict(fit(train_all, label, features, classes), hold, label, features, classes))
    return pl.concat(out).sort("fund_id", "quarter")


def window_style(g: pl.DataFrame, classes: list[str]) -> tuple[str, float]:
    """argmax of the mean class probability, and that mean."""
    p = g.select([f"p_{c}" for c in classes]).mean().row(0)
    k = int(np.argmax(p))
    return classes[k], float(p[k])


def summarize(pred: pl.DataFrame, classes: list[str]) -> pl.DataFrame:
    rows = []
    for (fid,), g in pred.sort("quarter").group_by("fund_id", maintain_order=True):
        modal = g["predicted"].value_counts().sort(["count", "predicted"], descending=[True, False])["predicted"][0]
        early, p_early = window_style(g.head(WINDOW), classes)
        late, p_late = window_style(g.tail(WINDOW), classes)
        enough = len(g) >= MIN_QUARTERS
        rows.append({
            "fund_id": fid, "fund_name": g["fund_name"][0], "cohort": g["cohort"][0], "label": g["label"][0],
            "quarters": len(g), "first_quarter": g["quarter"][0], "last_quarter": g["quarter"][-1],
            "modal_style": modal, "stability": float((g["predicted"] == modal).mean()),
            "switches": int((g["predicted"] != g["predicted"].shift(1)).sum()),
            "early_style": early, "p_early": p_early, "late_style": late, "p_late": p_late,
            "style_differs": enough and early != late,
            "changed": enough and early != late and min(p_early, p_late) >= CONFIDENCE,
        })
    return pl.DataFrame(rows)


def plot_strip(pred: pl.DataFrame, funds: pl.DataFrame, path: Path) -> None:
    quarters = sorted(pred["quarter"].unique())
    qi = {q: i for i, q in enumerate(quarters)}
    changed = funds.filter(pl.col("changed")).sort("late_style", "fund_name")
    rest = funds.filter(~pl.col("changed")).with_columns(
        pl.col("label").replace_strict({s: i for i, s in enumerate(STYLE_ORDER)}).alias("o")
    ).sort("cohort", "o", "fund_name", descending=[True, False, False])
    order = changed["fund_id"].to_list() + rest["fund_id"].to_list()
    gap = 1.2   # vertical space between the changed block and the rest
    ypos = {f: i + (gap if i >= len(changed) else 0) for i, f in enumerate(order)}
    names = dict(funds.select("fund_id", "fund_name").iter_rows())
    labels_ = dict(funds.select("fund_id", "label").iter_rows())
    cohorts = dict(funds.select("fund_id", "cohort").iter_rows())

    fig, ax = plt.subplots(figsize=(13, 0.2 * len(order) + 0.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for r in pred.iter_rows(named=True):
        ax.add_patch(plt.Rectangle((qi[r["quarter"]], ypos[r["fund_id"]] - 0.5), 1, 1,
                                   facecolor=COLORS[r["predicted"]], edgecolor=SURFACE, linewidth=1.2))
    for f in order:
        y = ypos[f]
        bold = f in set(changed["fund_id"])
        tag = " (holdout)" if cohorts[f] == "holdout" else ""
        ax.text(-0.4, y, f"{names[f][:34]}{tag}", ha="right", va="center", fontsize=7,
                color=INK if bold else INK2, fontweight="bold" if bold else "normal")
        # the fund's label as a small swatch right of the strip
        ax.add_patch(plt.Rectangle((len(quarters) + 0.4, y - 0.35), 0.7, 0.7, facecolor=COLORS[labels_[f]],
                                   edgecolor=SURFACE, linewidth=1))
    ax.text(len(quarters) + 0.75, -1.3, "label", ha="center", fontsize=7, color=MUTED)
    ax.text(-0.4, -1.3, f"Changed style ({len(changed)} funds)", ha="right", fontsize=8.5, fontweight="bold", color=INK)
    ax.text(-0.4, len(changed) + gap - 1.0, "No change (sorted by label)", ha="right", fontsize=8.5,
            fontweight="bold", color=INK)
    ax.set_xlim(-0.2, len(quarters) + 1.4)
    ax.set_ylim(max(ypos.values()) + 0.8, -1.8)
    ticks = [i for i, q in enumerate(quarters) if q.endswith("Q1")]
    ax.set_xticks([t + 0.5 for t in ticks], [quarters[t][:4] for t in ticks], fontsize=8, color=INK2)
    ax.xaxis.tick_top()
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    handles = [Patch(facecolor=COLORS[s], label=s.replace("_", " ")) for s in STYLE_ORDER]
    ax.legend(handles=handles, ncol=8, loc="upper left", bbox_to_anchor=(0, -0.002), frameon=False, fontsize=8,
              handlelength=1.2, columnspacing=1.2)
    ax.set_title("Predicted style per quarter, scored by a model that never saw the fund", loc="left",
                 fontsize=12.5, fontweight="bold", color=INK, pad=22)
    ax.text(-0.2, max(ypos.values()) + 3.2, f"Changed = argmax of mean class probability over the first {WINDOW} quarters differs from "
             f"the last {WINDOW}, both with mean probability ≥ {CONFIDENCE} (funds with {MIN_QUARTERS}+ quarters). Seed funds: 5 folds over funds, trained on "
             f"the other funds' 2018–2022 rows. Holdout funds: model trained on all seed funds.",
            color=MUTED, fontsize=7.5, va="top")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_changed(pred: pl.DataFrame, funds: pl.DataFrame, classes: list[str], path: Path, cols: int = 3) -> None:
    changed = funds.filter(pl.col("changed")).sort("late_style", "fund_name")
    if not len(changed):
        return
    n = len(changed)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 2.5 * rows + 0.9), facecolor=SURFACE, squeeze=False)
    order = [s for s in STYLE_ORDER if s in classes]
    for ax, r in zip(axes.flat, changed.iter_rows(named=True)):
        g = pred.filter(pl.col("fund_id") == r["fund_id"]).sort("quarter")
        x = np.arange(len(g))
        ax.set_facecolor(SURFACE)
        ax.stackplot(x, *[g[f"p_{s}"].to_numpy() for s in order], colors=[COLORS[s] for s in order],
                     edgecolor=SURFACE, linewidth=0.6)
        ax.set_xlim(0, len(g) - 1)
        ax.set_ylim(0, 1)
        q = g["quarter"].to_list()
        ticks = [i for i, v in enumerate(q) if v.endswith("Q1")][::2]
        ax.set_xticks(ticks, [q[i][:4] for i in ticks], fontsize=7, color=INK2)
        ax.set_yticks([0, 0.5, 1], ["0", ".5", "1"], fontsize=7, color=INK2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{r['fund_name'][:30]}\n{r['early_style'].replace('_', ' ')} → {r['late_style'].replace('_', ' ')}"
                     f"  (label: {r['label'].replace('_', ' ')})", loc="left", fontsize=8.5, color=INK)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    handles = [Patch(facecolor=COLORS[s], label=s.replace("_", " ")) for s in order]
    fig.legend(handles=handles, ncol=8, loc="lower left", bbox_to_anchor=(0.01, -0.01), frameon=False, fontsize=8)
    fig.suptitle("Funds whose predicted style changed: class probabilities per quarter (out of fund)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=1.6)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    con = db.connect()
    variant = con.execute("SELECT any_value(variant) FROM style_predictions").fetchone()[0]
    df = load(con)
    classes = sorted(set(df["label_v1"]) | set(df["label_v2"]))
    pred = out_of_fund_predictions(df, variant, classes).with_columns(pl.lit(variant).alias("variant"))
    funds = summarize(pred, classes)
    con.execute("CREATE OR REPLACE TABLE style_drift AS SELECT * FROM pred")
    con.execute("CREATE OR REPLACE TABLE style_drift_funds AS SELECT * FROM funds")
    FIG.mkdir(parents=True, exist_ok=True)
    plot_strip(pred, funds, FIG / "10_style_drift.png")
    plot_changed(pred, funds, classes, FIG / "11_style_drift_changed.png")

    changed = funds.filter(pl.col("changed")).sort("late_style", "fund_name")
    differs = funds.filter(pl.col("style_differs"))
    thresholds = ", ".join(
        f"{t:.1f}: {differs.filter(pl.min_horizontal('p_early', 'p_late') >= t).height}" for t in (0.0, 0.5, 0.6, 0.7))
    cols = ["fund_name", "cohort", "label", "first_quarter", "early_style", "p_early", "late_style", "p_late",
            "stability", "switches"]
    REPORT.write_text(f"""# Style drift

Generated by `uv run python -m src.models.style_drift` (model variant: {variant}). Every quarter is scored by a
model that never saw the fund (method in the module docstring). Changed = the argmax of the mean class probability
over a fund's first {WINDOW} quarters differs from its last {WINDOW}, and both winning mean probabilities are at
least {CONFIDENCE}; funds need {MIN_QUARTERS}+ quarters.
`p_early` / `p_late`: the winning mean probability in each window. `stability`: share of quarters predicted as the
fund's most common style. `switches`: quarter-to-quarter changes of the top prediction.

{len(changed)} of {len(funds)} funds changed ({funds.filter(pl.col('quarters') >= MIN_QUARTERS).height} had enough
quarters). Median stability: {funds['stability'].median():.2f}.

Sensitivity to the confidence threshold (funds flagged): {thresholds}.

## Funds that changed

{md_table(changed.select(cols)) if len(changed) else "None."}

![drift](figures/10_style_drift.png)
![changed](figures/11_style_drift_changed.png)

## Least stable funds that did not change (top 10 by switches)

{md_table(funds.filter(~pl.col("changed")).sort("switches", descending=True).head(10).select(cols))}
""")
    with pl.Config(tbl_rows=40, tbl_width_chars=220, tbl_cols=12, fmt_str_lengths=40):
        print(f"variant {variant}; {len(pred):,} fund-quarters, {len(funds)} funds, {len(changed)} changed")
        print(changed.select(cols))
        print(f"median stability {funds['stability'].median():.2f}")


if __name__ == "__main__":
    main()
