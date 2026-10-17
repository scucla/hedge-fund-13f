"""Task C1: exit / trim / hold / add for every current long position at the next quarter.

Usage:
    uv run python -m src.models.task_c1    # -> data/processed/task_c1_*.parquet, figures 18-20

Variants (feature groups from src.features.task_c.FEATURE_GROUPS):
  logreg_base / logreg_full   multinomial logistic regression
  lgbm_base                   own + stock + prior + crowd
  lgbm_peer                   + what other seed funds did with the stock in their latest visible filing
  lgbm_full                   + the same within the fund's Milestone 5 cluster
The lift of peer and cluster features is lgbm_peer / lgbm_full minus lgbm_base,
with a paired bootstrap over fund-quarters on the test years.

**Selection rule, fixed before any run:** for each variant, hyperparameters
(logreg C; LightGBM num_leaves x min_child_samples, rounds by early stopping)
are chosen by validation (2022) log-loss with the model fitted on the
embargoed train frame. The final model is the variant with the lowest
validation log-loss. Then each variant is refitted once on train + val
(embargoed against 2023) with its choices frozen (same number of rounds) and
scored on the test years. Nothing is chosen on test numbers.
"""

import json

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.task_c import FEATURE_GROUPS
from src.models.task_c_common import CLASSES, c1_metrics, frames, load_c1, subsets
from src.securities import db

OUT = "data/processed/task_c1"
FIG = "docs/figures"
CATEGORICAL = ["ff12"]
VARIANTS = {
    "logreg_base": ("logreg", ["own", "stock", "prior", "crowd"]),
    "logreg_full": ("logreg", ["own", "stock", "prior", "crowd", "peer", "cluster"]),
    "lgbm_base": ("lgbm", ["own", "stock", "prior", "crowd"]),
    "lgbm_peer": ("lgbm", ["own", "stock", "prior", "crowd", "peer"]),
    "lgbm_full": ("lgbm", ["own", "stock", "prior", "crowd", "peer", "cluster"]),
}
LR_GRID = [0.01, 0.1, 1.0]
LGBM_GRID = [{"num_leaves": nl, "min_child_samples": mc} for nl in (31, 127) for mc in (100, 1000)]
LGBM_FIXED = {"objective": "multiclass", "num_class": 4, "learning_rate": 0.05, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": 0,
              "num_threads": 8}


def features(groups: list[str]) -> list[str]:
    return [c for g in groups for c in FEATURE_GROUPS[g]]


def X_of(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()


# --- logistic regression -------------------------------------------------------------

class Winsor(BaseEstimator, TransformerMixin):
    """Clip each column to its fit-period 1st-99th percentiles (heavy tails: share changes, gains)."""

    def fit(self, X, y=None):
        self.lo_, self.hi_ = np.nanpercentile(X, 1, axis=0), np.nanpercentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)


def logreg(cols: list[str], C: float) -> Pipeline:
    num = [i for i, c in enumerate(cols) if c not in CATEGORICAL]
    cat = [i for i, c in enumerate(cols) if c in CATEGORICAL]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("winsor", Winsor()), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=-1)),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=C, max_iter=500))])


# --- LightGBM ------------------------------------------------------------------------

def lgbm_fit(Xtr, ytr, cols, params, rounds=None, Xva=None, yva=None):
    cat = [i for i, c in enumerate(cols) if c in CATEGORICAL]
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat, free_raw_data=False)
    if rounds is None:
        dva = lgb.Dataset(Xva, yva, categorical_feature=cat, reference=dtr)
        return lgb.train(LGBM_FIXED | params, dtr, num_boost_round=3000, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.train(LGBM_FIXED | params, dtr, num_boost_round=rounds)


# --- run -----------------------------------------------------------------------------

def tune(kind, cols, fr):
    Xtr, ytr = X_of(fr["train"], cols), fr["train"]["y"].to_numpy()
    Xva, yva = X_of(fr["val"], cols), fr["val"]["y"].to_numpy()
    trials = []
    if kind == "logreg":
        for C in LR_GRID:
            m = logreg(cols, C).fit(Xtr, ytr)
            trials.append(({"C": C}, c1_metrics(yva, m.predict_proba(Xva))["log_loss"]))
    else:
        for p in LGBM_GRID:
            m = lgbm_fit(Xtr, ytr, cols, p, Xva=Xva, yva=yva)
            trials.append((p | {"rounds": m.best_iteration}, c1_metrics(yva, m.predict(Xva, num_iteration=m.best_iteration))["log_loss"]))
    for p, ll in trials:
        print(f"    {p}: val log-loss {ll:.4f}", flush=True)
    return min(trials, key=lambda t: t[1])


def fit_final(kind, cols, params, df):
    X, y = X_of(df, cols), df["y"].to_numpy()
    if kind == "logreg":
        return logreg(cols, params["C"]).fit(X, y)
    return lgbm_fit(X, y, cols, {k: v for k, v in params.items() if k != "rounds"}, rounds=params["rounds"])


def predict(kind, model, df, cols):
    X = X_of(df, cols)
    return model.predict_proba(X) if kind == "logreg" else model.predict(X)


def metrics_rows(name, split, df, proba):
    rows = []
    y = df["y"].to_numpy()
    for subset, mask in subsets(df).items():
        m = mask.to_numpy()
        rows.append({"variant": name, "split": split, "subset": subset} | c1_metrics(y[m], proba[m]))
    for style in sorted(df["style"].drop_nulls().unique()):
        m = (df["style"] == style).to_numpy()
        rows.append({"variant": name, "split": split, "subset": f"style:{style}"} | c1_metrics(y[m], proba[m]))
    return rows


def paired_bootstrap(df, pa, pb, n_boot=200, seed=0):
    """Difference b - a in log-loss and macro-F1, resampling fund-quarters (rows within one
    filing are not independent)."""
    y = df["y"].to_numpy()
    g = df.select(pl.concat_str("fund_id", "quarter").rank("dense")).to_series().to_numpy() - 1
    ng = g.max() + 1
    rng = np.random.default_rng(seed)
    lla = -np.log(np.clip(pa[np.arange(len(y)), y], 1e-6, 1))
    llb = -np.log(np.clip(pb[np.arange(len(y)), y], 1e-6, 1))
    preda, predb = pa.argmax(1), pb.argmax(1)

    def macro_f1(pred, w):
        cm = np.bincount(y * 4 + pred, weights=w, minlength=16).reshape(4, 4)
        tp = np.diag(cm)
        f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
        return f1.mean()

    out = []
    for _ in range(n_boot):
        w = rng.multinomial(ng, np.full(ng, 1 / ng))[g].astype(float)
        out.append(((llb * w).sum() / w.sum() - (lla * w).sum() / w.sum(), macro_f1(predb, w) - macro_f1(preda, w)))
    out = np.array(out)
    return {"d_log_loss": float((llb - lla).mean()), "d_log_loss_lo": float(np.percentile(out[:, 0], 2.5)),
            "d_log_loss_hi": float(np.percentile(out[:, 0], 97.5)),
            "d_macro_f1": float(macro_f1(predb, np.ones(len(y))) - macro_f1(preda, np.ones(len(y)))),
            "d_macro_f1_lo": float(np.percentile(out[:, 1], 2.5)), "d_macro_f1_hi": float(np.percentile(out[:, 1], 97.5))}


def calibration_figure(df, proba, path):
    y = df["y"].to_numpy()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    bins = np.linspace(0, 1, 11)
    for i, (ax, c) in enumerate(zip(axes, CLASSES)):
        p = proba[:, i]
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        mp = [p[idx == b].mean() for b in range(10) if (idx == b).sum() >= 200]
        fr = [(y[idx == b] == i).mean() for b in range(10) if (idx == b).sum() >= 200]
        ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
        ax.plot(mp, fr, marker="o", color="#2a6f97")
        ax.set_title(c)
        ax.set_xlabel("predicted probability")
    axes[0].set_ylabel("observed share")
    fig.suptitle("C1 calibration, final model, test years 2023-2026Q1 (seed funds)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def shap_figures(model, df, cols):
    import shap

    sample = df.sample(n=min(5000, len(df)), seed=0)
    X = X_of(sample, cols)
    sv = shap.TreeExplainer(model).shap_values(X)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv   # rows x features x classes
    imp = pl.DataFrame({"feature": cols, **{f"mean_abs_shap_{c}": np.abs(sv[:, :, i]).mean(0) for i, c in enumerate(CLASSES)}})
    imp = imp.with_columns(pl.mean_horizontal([f"mean_abs_shap_{c}" for c in CLASSES]).alias("mean_abs_shap")).sort(
        "mean_abs_shap", descending=True)
    for i, c in enumerate(CLASSES):
        if c not in ("exit", "add"):
            continue
        plt.figure()
        shap.summary_plot(sv[:, :, i], X, feature_names=cols, max_display=15, show=False)
        plt.title(f"SHAP, class '{c}' (test sample of 5,000 seed positions)")
        plt.tight_layout()
        plt.savefig(f"{FIG}/{19 if c == 'exit' else 20}_c1_shap_{c}.png", dpi=130)
        plt.close("all")
    return imp


def main() -> None:
    with db.connect(read_only=True) as con:
        df = load_c1(con)
    fr = frames(df)
    results, choices, test_proba = [], {}, {}
    for name, (kind, groups) in VARIANTS.items():
        cols = features(groups)
        print(f"{name}: {len(cols)} features", flush=True)
        params, val_ll = tune(kind, cols, fr)
        choices[name] = {"params": params, "val_log_loss": val_ll}
        # validation numbers from the train fit with the chosen params
        m_tr = fit_final(kind, cols, params, fr["train"])
        results += metrics_rows(name, "val", fr["val"], predict(kind, m_tr, fr["val"], cols))
        m = fit_final(kind, cols, params, fr["trainval"])
        test_proba[name] = predict(kind, m, fr["test"], cols)
        results += metrics_rows(name, "test", fr["test"], test_proba[name])
        results += metrics_rows(name, "test_holdout", fr["test_holdout"], predict(kind, m, fr["test_holdout"], cols))
        if kind == "lgbm":
            choices[name]["model"] = m
    final = min(choices, key=lambda n: choices[n]["val_log_loss"])
    print("final model (lowest val log-loss):", final)

    lifts = []
    for subset, mask in subsets(fr["test"]).items():
        d = fr["test"].filter(mask)
        m = mask.to_numpy()
        for b in ("lgbm_peer", "lgbm_full"):
            lifts.append({"subset": subset, "comparison": f"{b} - lgbm_base"}
                         | paired_bootstrap(d, test_proba["lgbm_base"][m], test_proba[b][m]))
        lifts.append({"subset": subset, "comparison": "lgbm_full - logreg_full"}
                     | paired_bootstrap(d, test_proba["logreg_full"][m], test_proba["lgbm_full"][m]))

    kind, groups = VARIANTS[final]
    cols = features(groups)
    calibration_figure(fr["test"], test_proba[final], f"{FIG}/18_c1_calibration.png")
    imp = shap_figures(choices[final]["model"], fr["test"], cols) if kind == "lgbm" else None

    pl.DataFrame(results).write_parquet(f"{OUT}_results.parquet")
    pl.DataFrame(lifts).write_parquet(f"{OUT}_lift.parquet")
    if imp is not None:
        imp.write_parquet(f"{OUT}_shap.parquet")
    with open(f"{OUT}_choices.json", "w") as f:
        json.dump({"final": final} | {n: {"params": c["params"], "val_log_loss": c["val_log_loss"]}
                                      for n, c in choices.items()}, f, indent=2)
    pred = fr["test"].select("fund_id", "cusip", "quarter", "pred_date", "action").with_columns(
        *[pl.Series(f"p_{c}", test_proba[final][:, i]) for i, c in enumerate(CLASSES)])
    pred.write_parquet(f"{OUT}_test_predictions.parquet")
    print(pl.DataFrame(results).filter(pl.col("subset") == "all").select(
        "variant", "split", "macro_f1", "log_loss", "exit_auc"))
    print(pl.DataFrame(lifts))


if __name__ == "__main__":
    main()
