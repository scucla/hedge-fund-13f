"""Fund clusters (Milestone 5, Task B): HDBSCAN per quarter on each embedding,
stability, purity against the v2 style labels, and the UMAP map.

Usage (after src.features.embeddings):
    uv run python -m src.models.clustering select   # pre-2023 only; writes the choice
    uv run python -m src.models.clustering          # everything, needs the choice

Two stages so the main method is fixed before any 2023+ number exists. `select`
computes metrics on SELECT_QUARTERS only (2016Q1-2022Q4; the economic check only
up to 2021Q4, so its 12-month return window ends by 2022Q4), ranks the methods
by the rule below and writes data/reference/embedding_choice.json. The full run
refuses to start without that file and never changes it.

Clustering is on seed funds only (holdout and background are drawn on the map
but never clustered). Distance = cosine distance between the fund vectors.
  hdbscan   min_cluster_size MIN_CLUSTER_SIZE, min_samples MIN_SAMPLES (small:
            77 funds and styles of 3-10 funds; fixed a priori, not tuned)
  kmeans    K_MEANS = 8 clusters (the number of v2 styles), comparison only

Metrics per method x algorithm x quarter:
  noise_share    share of funds HDBSCAN calls noise
  purity         over non-noise funds: sum over clusters of the largest style
                 count / funds; `purity_random` is the same with the labels
                 shuffled (mean of 50), since small clusters are pure by luck
  label_ari      ARI vs the v2 labels, every noise fund its own cluster (so
                 calling everything noise scores ~0, not perfect)
  stability_ari  ARI between q and q-1 on funds in both, noise as singletons
  econ_gap       mean pairwise correlation of the funds' 13F-clone monthly
                 returns over the next 12 months, within clusters minus across
                 clusters (non-noise funds; returns demeaned by the seed
                 cross-section each month so the market doesn't make every pair alike)

Selection rule (written before running): for HDBSCAN, average each method's
stability_ari, label_ari and econ_gap over the selection quarters; rank the
methods on each (1 = best); lowest rank sum wins, ties to lower mean noise_share.

Writes `fund_clusters`, `cluster_metrics`, figures 12-15, docs/embeddings_clustering.md.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import umap
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

from src.features import embeddings as emb
from src.models.style_classifier import labels as style_labels
from src.models.style_drift import COLORS, FIG, GRID, INK, INK2, MUTED, STYLE_ORDER, SURFACE
from src.securities import db

MIN_CLUSTER_SIZE, MIN_SAMPLES = 3, 2
K_MEANS = 8
SELECT_QUARTERS = [q for q in db.QUARTERS if "2016Q1" <= q <= "2022Q4"]
ECON_LAST_SELECT = "2021Q4"
FIRST_QUARTER = "2016Q1"      # fund2vec's 4-quarter corpus is full from 2015Q4
MAP_QUARTER = db.LAST_QUARTER
CHOICE_PATH = Path("data/reference/embedding_choice.json")
REPORT = Path("docs/embeddings_clustering.md")
N_SHUFFLE = 50


# --- vectors -------------------------------------------------------------------------

def quarter_vectors(con, books: pl.DataFrame | None, method: str, quarter: str,
                    cohorts: tuple[str, ...] = ("seed",)) -> tuple[list[str], np.ndarray]:
    """Unit-norm fund vectors for one quarter. Seed funds are only those in that
    quarter's fit (filed by asof). cosine is the sqrt-weight book itself."""
    if method == "cosine":
        bq = books.filter((pl.col("quarter") == quarter) & pl.col("cohort").is_in(cohorts))
        bq = bq.filter((pl.col("cohort") != "seed") | (pl.col("known_at") <= emb.asof(quarter)))
        ids = sorted(bq["fund_id"].unique())
        seed_keys = bq.filter(pl.col("cohort") == "seed")["issuer_key"].unique().sort()
        x, _ = emb.sqrt_matrix(bq, ids, {k: i for i, k in enumerate(seed_keys)})
        return ids, emb.unit_rows(x.toarray())
    df = con.execute("""
        SELECT fund_id, vec FROM fund_embeddings
        WHERE method = ? AND quarter = ? AND cohort IN (SELECT unnest(?))
          AND (cohort <> 'seed' OR known_at <= ?)
        ORDER BY fund_id
    """, [method, quarter, list(cohorts), emb.asof(quarter)]).pl()
    return df["fund_id"].to_list(), np.vstack(df["vec"].to_numpy()) if df.height else np.zeros((0, 1))


def cos_dist(x: np.ndarray) -> np.ndarray:
    return np.clip(1 - x @ x.T, 0, 2)


# --- clustering + metrics ------------------------------------------------------------

def hdbscan_labels(x: np.ndarray) -> np.ndarray:
    return HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES, copy=True,
                   metric="precomputed").fit_predict(cos_dist(x))


def kmeans_labels(x: np.ndarray) -> np.ndarray:
    return KMeans(n_clusters=K_MEANS, n_init=10, random_state=0).fit_predict(x)


def singletons(lab: np.ndarray) -> np.ndarray:
    """Noise (-1) -> its own cluster id each, for ARI."""
    out = lab.copy()
    noise = np.where(out < 0)[0]
    out[noise] = out.max(initial=-1) + 1 + np.arange(len(noise))
    return out


def purity(clusters: np.ndarray, labels: np.ndarray) -> float:
    keep = clusters >= 0
    if not keep.any():
        return float("nan")
    c, l = clusters[keep], labels[keep]
    return sum(np.unique(l[c == k], return_counts=True)[1].max() for k in np.unique(c)) / keep.sum()


def stability(prev: dict[str, int] | None, cur: dict[str, int]) -> float:
    if not prev:
        return float("nan")
    common = sorted(set(prev) & set(cur))
    a = singletons(np.array([prev[f] for f in common]))
    b = singletons(np.array([cur[f] for f in common]))
    return float(adjusted_rand_score(a, b))


def residual_returns(con) -> pl.DataFrame:
    """fund_id x month clone returns minus the seed cross-sectional mean."""
    r = con.execute("SELECT fund_id, month, ret FROM fund_clone_returns_monthly WHERE cohort = 'seed'").pl()
    return r.with_columns((pl.col("ret") - pl.col("ret").mean().over("month")).alias("res"))


def econ_gap(res: pl.DataFrame, quarter: str, ids: list[str], clusters: np.ndarray) -> float:
    start = emb.q_end(quarter)
    end = emb.q_end(db.QUARTERS[db.QUARTERS.index(quarter) + 4]) if db.QUARTERS.index(quarter) + 4 < len(db.QUARTERS) else None
    if end is None:
        return float("nan")
    keep = [f for f, c in zip(ids, clusters) if c >= 0]
    lab = {f: c for f, c in zip(ids, clusters)}
    w = (res.filter((pl.col("month") > start) & (pl.col("month") <= end) & pl.col("fund_id").is_in(keep))
         .pivot(on="fund_id", index="month", values="res"))
    cols = [c for c in w.columns if c != "month" and w[c].null_count() <= 2]
    if len(cols) < 4:
        return float("nan")
    corr = pl.DataFrame({c: w[c] for c in cols}).to_pandas().corr(min_periods=10).to_numpy()
    same = np.array([[lab[a] == lab[b] for b in cols] for a in cols])
    iu = np.triu_indices(len(cols), 1)
    v, s = corr[iu], same[iu]
    ok = ~np.isnan(v)
    if not (s & ok).any() or not (~s & ok).any():
        return float("nan")
    return float(v[s & ok].mean() - v[~s & ok].mean())


def cluster_quarter(x: np.ndarray, algo: str) -> np.ndarray:
    return hdbscan_labels(x) if algo == "hdbscan" else kmeans_labels(x)


def run(con, books: pl.DataFrame, quarters: list[str], econ_last: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    lab = style_labels().filter(pl.col("cohort") == "seed")
    style = dict(zip(lab["fund_id"], lab["label_v2"]))
    res = residual_returns(con)
    rng = np.random.default_rng(0)
    assign, metrics = [], []
    for method in emb.METHODS:
        for algo in ["hdbscan", "kmeans"]:
            prev = None
            for q in quarters:
                ids, x = quarter_vectors(con, books, method, q)
                ids_l = [f for f in ids if f in style]
                x = x[[ids.index(f) for f in ids_l]]
                c = cluster_quarter(x, algo)
                y = np.array([style[f] for f in ids_l])
                cur = dict(zip(ids_l, c.tolist()))
                pur = purity(c, y)
                metrics.append({
                    "method": method, "algo": algo, "quarter": q, "n_funds": len(ids_l),
                    "n_clusters": int(len(set(c[c >= 0]))), "noise_share": float((c < 0).mean()),
                    "purity": pur,
                    "purity_random": float(np.nanmean([purity(c, rng.permutation(y)) for _ in range(N_SHUFFLE)])),
                    "label_ari": float(adjusted_rand_score(y, singletons(c))),
                    "stability_ari": stability(prev, cur),
                    "econ_gap": econ_gap(res, q, ids_l, c) if q <= econ_last else float("nan"),
                })
                assign += [{"method": method, "algo": algo, "fund_id": f, "quarter": q, "cluster": int(k)}
                           for f, k in cur.items()]
                prev = cur
            m = pl.DataFrame([r for r in metrics if r["method"] == method and r["algo"] == algo]).fill_nan(None)
            print(f"{method:9s} {algo:8s} noise {m['noise_share'].mean():.2f} stab {m['stability_ari'].mean():.2f} "
                  f"label_ari {m['label_ari'].mean():.2f} econ {m['econ_gap'].mean():.3f}")
    # NaN (no previous quarter, window past the data) -> null, so means skip it
    return pl.DataFrame(assign), pl.DataFrame(metrics).fill_nan(None)


def summarise(metrics: pl.DataFrame) -> pl.DataFrame:
    return (metrics.group_by("method", "algo")
            .agg(pl.col("n_clusters").mean(), pl.col("noise_share").mean(), pl.col("purity").mean(),
                 pl.col("purity_random").mean(), pl.col("label_ari").mean(),
                 pl.col("stability_ari").mean(), pl.col("econ_gap").mean())
            .sort("algo", "method"))


def choose(summary: pl.DataFrame) -> pl.DataFrame:
    """The selection rule (module docstring), on HDBSCAN rows."""
    h = summary.filter(pl.col("algo") == "hdbscan")
    for m in ["stability_ari", "label_ari", "econ_gap"]:
        h = h.with_columns(pl.col(m).rank("min", descending=True).alias(f"rank_{m}"))
    h = h.with_columns((pl.col("rank_stability_ari") + pl.col("rank_label_ari") + pl.col("rank_econ_gap"))
                       .alias("rank_sum"))
    return h.sort("rank_sum", "noise_share")


# --- figures -------------------------------------------------------------------------

def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def umap_layout(x_seed: np.ndarray, x_other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit on seed only; holdout and background are transformed into it."""
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, metric="cosine", random_state=0).fit(x_seed)
    return reducer.embedding_, (reducer.transform(x_other) if len(x_other) else np.zeros((0, 2)))


def plot_map(con, books, method: str, path: Path) -> pl.DataFrame:
    """The 'map of the hedge fund industry' for MAP_QUARTER."""
    q = MAP_QUARTER
    lab = style_labels()
    style = dict(zip(lab["fund_id"], lab["label_v2"]))
    ids_s, x_s = quarter_vectors(con, books, method, q, ("seed",))
    ids_o, x_o = quarter_vectors(con, books, method, q, ("holdout", "background"))
    if method != "cosine":   # background row coverage: hide books mostly outside the seed vocabulary
        cov = dict(con.execute("SELECT fund_id, coverage FROM fund_embeddings WHERE method = ? AND quarter = ?",
                               [method, q]).fetchall())
        keep = [i for i, f in enumerate(ids_o) if cov.get(f, 0) >= 0.5]
        ids_o, x_o = [ids_o[i] for i in keep], x_o[keep]
    clusters = hdbscan_labels(x_s)
    xy_s, xy_o = umap_layout(x_s, x_o)
    names = dict(con.execute("SELECT fund_id, any_value(fund_name) FROM positions GROUP BY 1").fetchall())

    fig, ax = plt.subplots(figsize=(11, 8.5), facecolor=SURFACE)
    style_ax(ax)
    bg = np.array([f.startswith("bg_") for f in ids_o], dtype=bool)
    ax.scatter(xy_o[bg, 0], xy_o[bg, 1], s=7, color="#c9c8c3", lw=0, label="background filers (projected)", zorder=1)
    for st in STYLE_ORDER:
        m = np.array([style.get(f) == st for f in ids_s])
        ax.scatter(xy_s[m, 0], xy_s[m, 1], s=46, color=COLORS[st], edgecolor="white", lw=0.8, zorder=3, label=st)
        mh = np.array([style.get(f) == st for f in ids_o]) & ~bg
        ax.scatter(xy_o[mh, 0], xy_o[mh, 1], s=46, facecolor="none", edgecolor=COLORS[st], lw=1.4, zorder=3)
    # a ring per cluster member and the cluster number at its medoid
    for k in sorted(set(clusters[clusters >= 0])):
        idx = np.where(clusters == k)[0]
        cen = xy_s[idx].mean(0)
        med = idx[np.argmin(((xy_s[idx] - cen) ** 2).sum(1))]
        ax.annotate(f"C{k}", xy_s[med], xytext=(6, 6), textcoords="offset points", fontsize=9,
                    color=INK, fontweight="bold", zorder=5)
    for i, f in enumerate(ids_s):
        ax.annotate(names.get(f, f).replace(" Management", "").replace(" Capital", "")[:22], xy_s[i],
                    xytext=(3, -8), textcoords="offset points", fontsize=5.5, color=INK2, zorder=4)
    ax.scatter([], [], s=46, facecolor="none", edgecolor=MUTED, lw=1.4, label="holdout fund (projected)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0), labelcolor=INK2)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"Map of the hedge fund industry, {q} ({method}; UMAP fitted on 77 seed funds)",
                 loc="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, "Filled: seed funds, coloured by v2 style; Cn: HDBSCAN cluster (label at its medoid). "
             "Rings: holdout funds. Grey: 500 largest active, low-ETF 13F filers, projected, not fitted.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return pl.DataFrame({"fund_id": ids_s, "cluster": clusters.tolist(), "x": xy_s[:, 0], "y": xy_s[:, 1]})


def plot_methods(con, books, path: Path) -> None:
    lab = style_labels()
    style = dict(zip(lab["fund_id"], lab["label_v2"]))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), facecolor=SURFACE)
    for ax, m in zip(axes, emb.METHODS):
        style_ax(ax)
        ids, x = quarter_vectors(con, books, m, MAP_QUARTER)
        xy, _ = umap_layout(x, np.zeros((0, x.shape[1])))
        c = hdbscan_labels(x)
        for st in STYLE_ORDER:
            s = np.array([style.get(f) == st for f in ids])
            ax.scatter(xy[s, 0], xy[s, 1], s=18, color=COLORS[st], lw=0)
        ax.scatter(xy[c < 0, 0], xy[c < 0, 1], s=40, facecolor="none", edgecolor=MUTED, lw=0.6)
        ax.set_title(f"{m}: {len(set(c[c >= 0]))} clusters, {np.mean(c < 0):.0%} noise", fontsize=9, color=INK, loc="left")
        ax.set_xticks([]), ax.set_yticks([])
    fig.text(0.01, 0.01, f"{MAP_QUARTER}, seed funds coloured by v2 style (same colours as figure 12); "
             "grey rings = HDBSCAN noise.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


METHOD_COLORS = dict(zip(emb.METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]))


def plot_metrics(metrics: pl.DataFrame, path: Path) -> None:
    h = metrics.filter(pl.col("algo") == "hdbscan").sort("quarter")
    panels = [("stability_ari", "Stability (ARI vs previous quarter)"), ("noise_share", "Noise share"),
              ("label_ari", "Agreement with v2 styles (ARI)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), facecolor=SURFACE)
    qs = sorted(h["quarter"].unique())
    for ax, (col, title) in zip(axes, panels):
        style_ax(ax)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for m in emb.METHODS:
            g = h.filter(pl.col("method") == m)
            ax.plot([qs.index(q) for q in g["quarter"]], g[col].to_numpy(), color=METHOD_COLORS[m], lw=1.6, label=m)
        ax.axvline(qs.index("2023Q1") - 0.5, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks(range(0, len(qs), 8), [qs[i] for i in range(0, len(qs), 8)])
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "HDBSCAN per quarter on seed funds. Dashed line: the method was chosen on quarters "
             "left of it only.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_drift(con, path: Path) -> pl.DataFrame:
    d = con.execute("""SELECT method, quarter, median(move) AS med FROM embedding_drift
                       WHERE quarter >= ? GROUP BY ALL ORDER BY quarter""", [FIRST_QUARTER]).pl()
    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=SURFACE)
    style_ax(ax)
    ax.grid(axis="y", color=GRID, lw=0.6)
    qs = sorted(d["quarter"].unique())
    for m, col in [*METHOD_COLORS.items(), ("svd_unaligned", MUTED)]:
        g = d.filter(pl.col("method") == m)
        if g.height:
            ax.plot([qs.index(q) for q in g["quarter"]], g["med"], color=col, lw=1.6,
                    ls="--" if m == "svd_unaligned" else "-", label=m)
    ax.set_xticks(range(0, len(qs), 8), [qs[i] for i in range(0, len(qs), 8)])
    ax.set_ylabel("median cosine distance to own q-1 vector", fontsize=8, color=INK2)
    ax.set_title("Quarter-to-quarter fund movement in aligned embedding spaces", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return d.group_by("method").agg(pl.col("med").median()).sort("method")


# --- report --------------------------------------------------------------------------

def md(df: pl.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def cluster_table(con, books, clusters: pl.DataFrame) -> pl.DataFrame:
    """Members, styles and shared top names of each map cluster."""
    lab = style_labels()
    names = dict(con.execute("SELECT fund_id, any_value(fund_name) FROM positions GROUP BY 1").fetchall())
    iss = emb.issuer_names(con)
    bq = books.filter(pl.col("quarter") == MAP_QUARTER)
    rows = []
    for k in sorted(set(clusters["cluster"])):
        mem = clusters.filter(pl.col("cluster") == k)["fund_id"].to_list()
        sty = lab.filter(pl.col("fund_id").is_in(mem))["label_v2"].value_counts().sort("count", descending=True)
        top = (bq.filter(pl.col("fund_id").is_in(mem)).group_by("issuer_key")
               .agg(pl.col("fund_id").n_unique().alias("n"), pl.col("w").sum()).sort("n", "w", descending=True).head(5))
        rows.append({"cluster": "noise" if k < 0 else f"C{k}", "n": len(mem),
                     "styles": ", ".join(f"{s} {c}" for s, c in sty.iter_rows()),
                     "funds": ", ".join(sorted(names.get(f, f).replace(" Management", "") for f in mem)),
                     "most shared names (holders)": ", ".join(f"{iss.get(i, i)} ({n})" for i, n, _ in top.iter_rows())})
    return pl.DataFrame(rows)


def background_neighbours(con, books, method: str, clusters: pl.DataFrame) -> pl.DataFrame:
    """Each mapped background filer's nearest seed fund (cosine, MAP_QUARTER) and
    that fund's cluster: which part of the industry looks like which seed group."""
    ids_s, x_s = quarter_vectors(con, books, method, MAP_QUARTER, ("seed",))
    ids_b, x_b = quarter_vectors(con, books, method, MAP_QUARTER, ("background",))
    cov = dict(con.execute("SELECT fund_id, coverage FROM fund_embeddings WHERE method = ? AND quarter = ?",
                           [method, MAP_QUARTER]).fetchall())
    keep = [i for i, f in enumerate(ids_b) if cov.get(f, 0) >= 0.5]
    ids_b, x_b = [ids_b[i] for i in keep], x_b[keep]
    sim = x_b @ x_s.T
    near = sim.argmax(1)
    lab = dict(zip(clusters["fund_id"], clusters["cluster"]))
    mgr = dict(con.execute("""SELECT 'bg_' || cik::BIGINT::VARCHAR, any_value(manager_name)
                              FROM filings_db.filings WHERE quarter = ? GROUP BY 1""", [MAP_QUARTER]).fetchall())
    val = dict(con.execute("SELECT 'bg_' || cik::VARCHAR, any_value(long_value) FROM bg_w WHERE quarter = ? GROUP BY 1",
                           [MAP_QUARTER]).fetchall()) if con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'bg_w'").fetchone()[0] else {}
    df = pl.DataFrame({"fund_id": ids_b, "cluster": [lab[ids_s[j]] for j in near],
                       "sim": sim.max(1), "value": [val.get(f, 0.0) for f in ids_b]})
    return (df.sort("value", descending=True).group_by("cluster", maintain_order=True)
            .agg(pl.len().alias("n_background"), pl.col("sim").median().alias("median_cos"),
                 pl.col("fund_id").head(4).alias("largest"))
            .with_columns(pl.col("largest").map_elements(lambda s: ", ".join(mgr.get(f, f)[:28] for f in s),
                                                         return_dtype=pl.Utf8))
            .with_columns(pl.when(pl.col("cluster") < 0).then(pl.lit("noise"))
                          .otherwise(pl.lit("C") + pl.col("cluster").cast(pl.Utf8)).alias("cluster"))
            .sort("n_background", descending=True))


def main_select() -> None:
    con = db.connect()
    con.execute("SET enable_progress_bar = false")
    books = emb.load_books(con).filter(pl.col("cohort") == "seed")
    _, metrics = run(con, books, SELECT_QUARTERS, ECON_LAST_SELECT)
    ranked = choose(summarise(metrics))
    print(ranked)
    choice = {"method": ranked["method"][0], "chosen_at": datetime.now().isoformat(timespec="seconds"),
              "quarters": [SELECT_QUARTERS[0], SELECT_QUARTERS[-1]], "econ_last": ECON_LAST_SELECT,
              "rule": "rank sum of mean stability_ari, label_ari, econ_gap (HDBSCAN), ties to lower noise",
              "ranking": ranked.select("method", "noise_share", "purity", "purity_random", "label_ari",
                                       "stability_ari", "econ_gap", "rank_sum").to_dicts()}
    CHOICE_PATH.write_text(json.dumps(choice, indent=2) + "\n")
    print(f"chose {choice['method']}; wrote {CHOICE_PATH}")


def main() -> None:
    if not CHOICE_PATH.exists():
        sys.exit(f"{CHOICE_PATH} missing: run `python -m src.models.clustering select` first")
    choice = json.loads(CHOICE_PATH.read_text())
    method = choice["method"]
    con = db.connect()
    con.execute("SET enable_progress_bar = false")
    books_all = emb.load_books(con)
    books = books_all.filter(pl.col("cohort") == "seed")
    quarters = [q for q in db.QUARTERS if q >= FIRST_QUARTER]
    assign, metrics = run(con, books, quarters, db.LAST_QUARTER)
    known = con.execute("""SELECT method, fund_id, quarter, known_at FROM fund_embeddings WHERE cohort = 'seed'""").pl()
    # cosine has no model, but its known_at is the same as svd's: the fund's own
    # filing or the latest seed book in the quarter's fit, whichever is later
    known = pl.concat([known, known.filter(pl.col("method") == "svd").with_columns(pl.lit("cosine").alias("method"))])
    assign = assign.join(known, on=["method", "fund_id", "quarter"], how="left")
    for name, df in [("fund_clusters", assign), ("cluster_metrics", metrics)]:
        con.register("df", df.to_arrow())
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df")
        con.unregister("df")

    FIG.mkdir(parents=True, exist_ok=True)
    map_clusters = plot_map(con, books_all, method, FIG / "12_fund_map.png")
    plot_methods(con, books_all, FIG / "13_fund_map_methods.png")
    plot_metrics(metrics, FIG / "14_cluster_metrics.png")
    drift = plot_drift(con, FIG / "15_embedding_drift.png")

    pre = summarise(metrics.filter(pl.col("quarter") <= "2022Q4"))
    post = summarise(metrics.filter(pl.col("quarter") >= "2023Q1"))
    bg = con.execute(f"""SELECT quarter, count(*) AS n, median(coverage) AS median_coverage
                         FROM fund_embeddings WHERE cohort = 'background' AND method = '{method}'
                         GROUP BY 1 ORDER BY 1""").pl()
    themes = con.execute(f"SELECT theme, top_issuers FROM nmf_themes WHERE quarter = '{MAP_QUARTER}' ORDER BY theme").pl()
    ctab = cluster_table(con, books_all, map_clusters)
    emb.background_books(con)   # rebuilds the bg_w temp table (long values) for the names below
    bgn = background_neighbours(con, books_all, method, map_clusters)
    cols = ["method", "algo", "n_clusters", "noise_share", "purity", "purity_random", "label_ari", "stability_ari", "econ_gap"]
    REPORT.write_text(f"""# Fund embeddings and clusters (Milestone 5, Task B)

Generated by `src/features/embeddings.py` and `src/models/clustering.py`. Design
choices are in the module docstrings and `docs/decisions.md`; the reading is in
`docs/fund_map.md`.

**Main method: `{method}`**, chosen on {choice['quarters'][0]}-{choice['quarters'][1]} only
(economic check to {choice['econ_last']}) at {choice['chosen_at']}, before any 2023+
metric was computed (`{CHOICE_PATH}`). Rule: {choice['rule']}.

## Selection quarters ({SELECT_QUARTERS[0]}-2022Q4), all metrics are quarterly means

{md(pre.select(cols))}

## 2023Q1-{db.LAST_QUARTER} (after the choice)

{md(post.select(cols))}

`purity_random` = the same clusters with the style labels shuffled. `econ_gap` =
within-cluster minus across-cluster correlation of next-12-month clone returns
(market-demeaned); NaN where the window runs past the data.

## Alignment

Median quarter-to-quarter cosine move of a seed fund (2016Q1+), aligned vs not:

{md(drift)}

Without the stock-anchored Procrustes step an SVD fund "moves" ~{drift.filter(pl.col('method') == 'svd_unaligned')['med'][0]:.2f}
per quarter just because the basis rotates; aligned, the move is the fund's own.

## Map ({MAP_QUARTER}, `{method}`) — figure 12

![map](figures/12_fund_map.png)

{md(ctab)}

Background filers on the map (active share >= {emb.BG_MIN_ACTIVE_SHARE}, ETFs and funds <= {emb.BG_MAX_ETF_SHARE:.0%}
of the book, long book >= $100M, largest {emb.BACKGROUND_N} per quarter, projected; median share of their book inside the seed vocabulary):

{md(bg)}

Nearest seed fund's cluster for each mapped background filer (cosine in `{method}` space;
`largest` = the four largest filers by long 13F value):

{md(bgn)}

## Method comparison and metrics over time

![methods](figures/13_fund_map_methods.png)
![metrics](figures/14_cluster_metrics.png)
![drift](figures/15_embedding_drift.png)

## NMF themes, {MAP_QUARTER} (top loadings)

{md(themes)}
""")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main_select() if sys.argv[1:] == ["select"] else main()
