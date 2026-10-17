"""Crowding in stress episodes (Milestone 5, Task B): how much did funds in the
same cluster own the same names going in, how did those names do, and did the
cluster sell them together?

Usage (after src.models.clustering):
    uv run python -m src.models.crowding

Episodes (EPISODES): the book is the last quarter-end book before the episode,
clustered with the main method (data/reference/embedding_choice.json, HDBSCAN).
  covid    book 2019Q4 (filed by mid-February 2020, before the crash), return over 2020Q1
  growth   book 2021Q3 (filed by mid-November 2021, as the unwind began), return
           2021Q4-2022Q2 compounded
Returns are `stock_quarter.ret_next_q` (Yahoo, else delisting, else 13F-implied).
A stock delisted inside the episode earns its delisting return and then 0 (cash);
any other missing quarter drops the stock. Starting from the quarter end rather
than the filing date describes the funds' actual positions; it is not a
tradable strategy (that's Task D).

Per stock (issuer level, common stock and ADRs held long by a seed fund):
  n_holders        conviction holders: seed funds with the issuer at >= CROWD_MIN_WEIGHT
                   of their long book. Without this floor a cluster of 17
                   multi-managers and quants "crowds" ~1,700 names it holds at
                   0.05% each; crowding risk is about positions big enough to
                   hurt when everyone sells them together
  max_cluster_n    most holders inside one cluster; `top_cluster` is that cluster
  cluster_pairs    same-cluster co-holder pairs, sum over clusters of C(n_c, 2)
  crowded          some cluster with >= CROWD_MIN conviction holders and at least
                   CROWD_SHARE of that cluster's members (noise funds are their
                   own clusters, so they never make a stock crowded)
  bucket           crowded / widely held (>= CROWD_MIN conviction holders, not
                   crowded) / 0-2 conviction holders
  ret_adj          episode return minus the median return of its size quintile
                   (quintiles of the seed-held universe), so "crowded mega caps
                   fell" isn't just "mega caps fell"
Regression: ret ~ crowded + log(1 + n_holders) + log_me + beta + mom_12_1 (White
t-stats), so crowding within a cluster is separated from plain popularity.

Book overlap of two funds = sum over issuers of min(w_i, w_j). Within-cluster
overlap is high by construction (clusters are built from holdings), so it is
read as a time series (was it unusually high going in?), not as a level.

Exit together: position_moves at the book quarter (what each fund did by the
next filing) for positions in its own cluster's crowded names vs its other
positions of at least the same CROWD_MIN_WEIGHT (big positions behave differently
from small ones, so both groups are big).

Sensitivity: the episode results are recomputed with the conviction floor at
each of SENSITIVITY_FLOORS (the 1% floor stays the main one).

Writes `crowding_stocks`, `crowding_clusters`, `cluster_overlap`, `crowding_sensitivity`, figures 16-17,
docs/crowding.md.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.features import embeddings as emb
from src.models import clustering as cl
from src.models.factor_regressions import ols_newey_west
from src.models.style_classifier import labels as style_labels
from src.models.style_drift import FIG, GRID, INK, INK2, MUTED, SURFACE
from src.securities import db

EPISODES = {
    "covid": {"book": "2019Q4", "ret_quarters": ["2019Q4"], "title": "COVID crash, 2020Q1"},
    "growth": {"book": "2021Q3", "ret_quarters": ["2021Q3", "2021Q4", "2022Q1"],
               "title": "Growth unwind, 2021Q4-2022Q2"},
}
CROWD_MIN, CROWD_SHARE = 3, 0.5
CROWD_MIN_WEIGHT = 0.01
SENSITIVITY_FLOORS = [0.005, 0.01, 0.02]
REPORT = Path("docs/crowding.md")
CROWDED, WIDE, FEW = "crowded in a cluster", "widely held, not crowded", "0-2 conviction holders"


def chosen_method() -> str:
    return json.loads(cl.CHOICE_PATH.read_text())["method"]


def clusters_at(con, method: str, quarter: str) -> dict[str, int]:
    rows = con.execute("""SELECT fund_id, cluster FROM fund_clusters
                          WHERE method = ? AND algo = 'hdbscan' AND quarter = ?""", [method, quarter]).fetchall()
    return dict(rows)


def seed_books(books: pl.DataFrame, quarter: str, funds: list[str]) -> pl.DataFrame:
    return books.filter((pl.col("quarter") == quarter) & pl.col("fund_id").is_in(funds))


# --- per stock -----------------------------------------------------------------------

def crowding_measures(book: pl.DataFrame, clusters: dict[str, int],
                      min_weight: float = CROWD_MIN_WEIGHT) -> pl.DataFrame:
    """issuer_key -> n_holders, max_cluster_n, top_cluster, cluster_pairs, crowded."""
    size = pl.DataFrame({"cluster": list(clusters.values())}).group_by("cluster").len("cluster_size")
    b = (book.filter(pl.col("w") >= min_weight)
         .with_columns(pl.col("fund_id").replace_strict(clusters, return_dtype=pl.Int64).alias("cluster")))
    per = (b.filter(pl.col("cluster") >= 0).group_by("issuer_key", "cluster").agg(pl.col("fund_id").n_unique().alias("n_c"))
           .join(size, on="cluster")
           .with_columns(((pl.col("n_c") >= CROWD_MIN) & (pl.col("n_c") >= CROWD_SHARE * pl.col("cluster_size")))
                         .alias("crowded_c")))
    agg = per.group_by("issuer_key").agg(
        pl.col("n_c").max().alias("max_cluster_n"),
        pl.col("cluster").sort_by("n_c", descending=True).first().alias("top_cluster"),
        (pl.col("n_c") * (pl.col("n_c") - 1) // 2).sum().alias("cluster_pairs"),
        pl.col("crowded_c").any().alias("crowded"),
        pl.col("cluster").filter(pl.col("crowded_c")).alias("crowded_in"),
    )
    n = b.group_by("issuer_key").agg(pl.col("fund_id").n_unique().alias("n_holders"))
    out = (book.select("issuer_key").unique().join(n, on="issuer_key", how="left").join(agg, on="issuer_key", how="left")
           .with_columns(pl.col("n_holders").fill_null(0), pl.col("max_cluster_n").fill_null(0),
                         pl.col("cluster_pairs").fill_null(0), pl.col("crowded").fill_null(False)))
    return out.with_columns(
        pl.when(pl.col("crowded")).then(pl.lit(CROWDED))
        .when(pl.col("n_holders") >= CROWD_MIN).then(pl.lit(WIDE)).otherwise(pl.lit(FEW)).alias("bucket"))


def compound(rets: list[float | None], sources: list[str | None]) -> float | None:
    """Compound quarterly returns; after a delisting the stock is cash (0)."""
    total = 1.0
    for r, s in zip(rets, sources):
        if r is None:
            return None
        total *= 1 + r
        if s is not None and s.startswith("delist"):
            break
    return total - 1


def episode_returns(con, book_q: str, ret_quarters: list[str]) -> pl.DataFrame:
    """issuer_key, cusip, ret, log_me, beta, mom_12_1 for seed-held common/ADR issuers."""
    rep = con.execute(f"""
        SELECT {emb.ISSUER_KEY} AS issuer_key, arg_max(cusip, value_usd) AS cusip, any_value(symbol) AS symbol
        FROM positions
        WHERE cohort = 'seed' AND quarter = ? AND NOT is_option AND security_class IN ('common', 'adr')
          AND issuer_cik IS NOT NULL
        GROUP BY 1
    """, [book_q]).pl()
    sq = con.execute("""SELECT cusip, quarter, ret_next_q, ret_source, log_me, beta, mom_12_1
                        FROM stock_quarter WHERE quarter IN (SELECT unnest(?))""", [ret_quarters]).pl()
    chars = sq.filter(pl.col("quarter") == book_q).select("cusip", "log_me", "beta", "mom_12_1")
    rows = []
    by = {(c, q): (r, s) for c, q, r, s in sq.select("cusip", "quarter", "ret_next_q", "ret_source").iter_rows()}
    for k, c, sym in rep.iter_rows():
        rs = [by.get((c, q), (None, None)) for q in ret_quarters]
        rows.append({"issuer_key": k, "cusip": c, "symbol": sym,
                     "ret": compound([r for r, _ in rs], [s for _, s in rs])})
    out = pl.DataFrame(rows, schema={"issuer_key": pl.Utf8, "cusip": pl.Utf8, "symbol": pl.Utf8, "ret": pl.Float64})
    return out.join(chars, on="cusip", how="left")


def size_adjust(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col("log_me").qcut(5, labels=[f"q{i}" for i in range(1, 6)], allow_duplicates=True)
                         .alias("size_q"))
    return df.with_columns((pl.col("ret") - pl.col("ret").median().over("size_q")).alias("ret_adj"))


def regression(df: pl.DataFrame) -> pl.DataFrame:
    d = df.drop_nulls(["ret", "log_me", "beta", "mom_12_1"])
    names = ["crowded", "log_n_holders", "log_me", "beta", "mom_12_1"]
    X = np.column_stack([d["crowded"].cast(pl.Float64), np.log1p(d["n_holders"].to_numpy()),
                         d["log_me"], d["beta"], d["mom_12_1"]])
    b, t, r2 = ols_newey_west(d["ret"].to_numpy(), X, lags=0)
    return pl.DataFrame({"term": ["intercept", *names], "coef": b, "t": t}).with_columns(
        pl.lit(d.height).alias("n"), pl.lit(r2).alias("r2"))


def sensitivity_row(ep: str, floor: float, book: pl.DataFrame, clusters: dict[str, int],
                    rets: pl.DataFrame) -> dict:
    st = size_adjust(rets.join(crowding_measures(book, clusters, floor), on="issuer_key", how="inner").drop_nulls("ret"))
    crowded = st.filter(pl.col("crowded"))
    reg = regression(st)
    coef = reg.filter(pl.col("term") == "crowded")
    return {"episode": ep, "floor": f"{floor:.1%}", "n_crowded": crowded.height,
            "n_widely_held": st.filter(pl.col("bucket") == WIDE).height,
            "crowded_mean_ret": crowded["ret"].mean(), "crowded_mean_ret_adj": crowded["ret_adj"].mean(),
            "crowded_median_ret_adj": crowded["ret_adj"].median(),
            "reg_crowded": coef["coef"][0], "t_crowded": coef["t"][0]}


# --- overlap -------------------------------------------------------------------------

def overlap_matrix(book: pl.DataFrame, ids: list[str]) -> np.ndarray:
    keys = {k: i for i, k in enumerate(book["issuer_key"].unique().sort())}
    w = np.zeros((len(ids), len(keys)))
    row = {f: i for i, f in enumerate(ids)}
    for f, k, x in book.select("fund_id", "issuer_key", "w").iter_rows():
        if f in row:
            w[row[f], keys[k]] = x
    return np.vstack([np.minimum(w[i], w).sum(1) for i in range(len(ids))])


def overlap_by_quarter(con, books: pl.DataFrame, method: str) -> pl.DataFrame:
    out = []
    for q in [q for q in db.QUARTERS if q >= cl.FIRST_QUARTER]:
        c = clusters_at(con, method, q)
        ids = sorted(c)
        o = overlap_matrix(seed_books(books, q, ids), ids)
        lab = np.array([c[f] for f in ids])
        iu = np.triu_indices(len(ids), 1)
        a, b = lab[iu[0]], lab[iu[1]]
        both = (a >= 0) & (b >= 0)
        same = both & (a == b)
        out.append({"quarter": q, "within": float(o[iu][same].mean()), "across": float(o[iu][both & ~same].mean()),
                    "all_pairs": float(o[iu].mean())})
    return pl.DataFrame(out)


# --- per cluster ---------------------------------------------------------------------

def cluster_episode(book: pl.DataFrame, clusters: dict[str, int], stocks: pl.DataFrame,
                    names: dict[str, str]) -> pl.DataFrame:
    lab = style_labels()
    ret = dict(zip(stocks["issuer_key"], stocks["ret_adj"]))
    raw = dict(zip(stocks["issuer_key"], stocks["ret"]))
    sym = dict(zip(stocks["issuer_key"], stocks["symbol"]))
    rows = []
    for k in sorted(set(v for v in clusters.values() if v >= 0)):
        mem = [f for f, v in clusters.items() if v == k]
        o = overlap_matrix(seed_books(book, book["quarter"][0], mem), mem)
        iu = np.triu_indices(len(mem), 1)
        crowded = stocks.filter(pl.col("crowded_in").list.contains(k))
        ck = set(crowded["issuer_key"])
        b = book.filter(pl.col("fund_id").is_in(mem))
        share = b.filter(pl.col("issuer_key").is_in(list(ck)))["w"].sum() / len(mem)

        def book_ret(sub: pl.DataFrame, r: dict) -> float | None:
            sub = sub.with_columns(pl.col("issuer_key").replace_strict(r, default=None, return_dtype=pl.Float64).alias("r"))
            sub = sub.drop_nulls("r")
            return float((sub["w"] * sub["r"]).sum() / sub["w"].sum()) if sub.height else None

        top = crowded.sort("max_cluster_n", "ret", descending=[True, False]).head(6)
        sty = lab.filter(pl.col("fund_id").is_in(mem))["label_v2"].value_counts().sort("count", descending=True)
        rows.append({
            "cluster": f"C{k}", "n": len(mem),
            "styles": ", ".join(f"{s} {c}" for s, c in sty.iter_rows()),
            "within_overlap": float(o[iu].mean()) if len(mem) > 1 else None,
            "n_crowded": len(ck), "book_in_crowded": float(share),
            "crowded_ret_adj": book_ret(b.filter(pl.col("issuer_key").is_in(list(ck))), ret),
            "other_ret_adj": book_ret(b.filter(~pl.col("issuer_key").is_in(list(ck))), ret),
            "book_ret": book_ret(b, raw),
            "crowded names (episode return)": ", ".join(
                f"{sym.get(i) or names.get(i, i)} ({r:+.0%})" if r is not None else f"{sym.get(i) or i} (n/a)"
                for i, r in top.select("issuer_key", "ret").iter_rows()),
            "funds": ", ".join(sorted(names.get(f, f).replace(" Management", "").replace(" Capital", "") for f in mem)),
        })
    return pl.DataFrame(rows)


def exits_together(con, book_q: str, clusters: dict[str, int], stocks: pl.DataFrame) -> pl.DataFrame:
    """Next-filing actions on positions in the fund's own cluster's crowded names vs others."""
    crowded_in = {k: set(v) for k, v in zip(stocks["issuer_key"], stocks["crowded_in"]) if v is not None}
    pm = con.execute(f"""
        SELECT p.fund_id, {emb.ISSUER_KEY} AS issuer_key, m.action
        FROM positions p JOIN position_moves m USING (fund_id, cusip, quarter)
        WHERE p.quarter = ? AND p.cohort = 'seed' AND NOT p.is_option AND m.action IS NOT NULL
          AND p.weight_long >= {CROWD_MIN_WEIGHT}
          AND p.security_class IN ('common', 'adr')
    """, [book_q]).pl()
    pm = pm.filter(pl.col("fund_id").is_in(list(clusters)))
    own = [clusters[f] in crowded_in.get(k, set()) for f, k in pm.select("fund_id", "issuer_key").iter_rows()]
    pm = pm.with_columns(pl.Series("position", ["own cluster's crowded name" if x else "other position" for x in own]))
    return (pm.group_by("position").agg(
                pl.len().alias("n"),
                *[(pl.col("action") == a).mean().alias(a) for a in ["exit", "trim", "hold", "add"]])
            .sort("position"))


# --- figures -------------------------------------------------------------------------

def plot_episodes(per_cluster: dict[str, pl.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, len(per_cluster), figsize=(13, 4.6), facecolor=SURFACE)
    for ax, (ep, d) in zip(axes, per_cluster.items()):
        cl.style_ax(ax)
        ax.grid(axis="x", color=GRID, lw=0.6)
        d = d.filter(pl.col("n_crowded") > 0).sort("crowded_ret_adj")
        y = np.arange(d.height)
        ax.axvline(0, color=MUTED, lw=0.8)
        for i, (a, b) in enumerate(zip(d["crowded_ret_adj"], d["other_ret_adj"])):
            ax.plot([b, a], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter(d["other_ret_adj"], y, s=34, color="#8a8984", zorder=2, label="rest of the cluster's book")
        ax.scatter(d["crowded_ret_adj"], y, s=46, color="#e34948", zorder=3, label="cluster's crowded names")
        ax.set_yticks(y, [f"{c} ({s.split(',')[0]})" for c, s in zip(d["cluster"], d["styles"])], fontsize=8)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
        ax.set_title(EPISODES[ep]["title"], loc="left", fontsize=10, color=INK)
        ax.set_xlabel("weighted episode return minus size-quintile median", fontsize=8, color=INK2)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_overlap(ov: pl.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=SURFACE)
    cl.style_ax(ax)
    ax.grid(axis="y", color=GRID, lw=0.6)
    qs = ov["quarter"].to_list()
    for ep in EPISODES.values():
        i = qs.index(ep["book"])
        ax.axvspan(i + 0.5, i + 0.5 + len(ep["ret_quarters"]), color="#f3d9d6", lw=0)
    ax.plot(range(len(qs)), ov["within"], color="#e34948", lw=1.8, label="same cluster")
    ax.plot(range(len(qs)), ov["across"], color="#2a78d6", lw=1.8, label="different clusters")
    ax.set_xticks(range(0, len(qs), 8), [qs[i] for i in range(0, len(qs), 8)])
    ax.set_ylabel("mean pairwise book overlap", fontsize=8, color=INK2)
    ax.set_title("How much of their books do two seed funds share? (shaded: stress episodes)",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    method = chosen_method()
    con = db.connect()
    con.execute("SET enable_progress_bar = false")
    books = emb.load_books(con).filter(pl.col("cohort") == "seed")
    names = dict(con.execute("SELECT fund_id, any_value(fund_name) FROM positions GROUP BY 1").fetchall())
    names |= emb.issuer_names(con)

    stock_rows, cluster_rows, sections, per_cluster, sens = [], [], [], {}, []
    for ep, cfg in EPISODES.items():
        c = clusters_at(con, method, cfg["book"])
        book = seed_books(books, cfg["book"], list(c))
        rets = episode_returns(con, cfg["book"], cfg["ret_quarters"])
        sens += [sensitivity_row(ep, f, book, c, rets) for f in SENSITIVITY_FLOORS]
        cm = crowding_measures(book, c)
        st = size_adjust(rets.join(cm, on="issuer_key", how="inner").drop_nulls("ret"))
        pc = cluster_episode(book, c, st, names)
        per_cluster[ep] = pc
        buckets = (st.group_by("bucket").agg(pl.len().alias("n"), pl.col("ret").mean().alias("mean_ret"),
                                             pl.col("ret").median().alias("median_ret"),
                                             pl.col("ret_adj").mean().alias("mean_ret_adj"),
                                             pl.col("ret_adj").median().alias("median_ret_adj"))
                   .sort("bucket"))
        reg = regression(st)
        ex = exits_together(con, cfg["book"], c, st)
        n_dropped = rets.join(cm, on="issuer_key", how="inner")["ret"].null_count()
        stock_rows.append(st.with_columns(pl.lit(ep).alias("episode")).drop("crowded_in", "size_q"))
        cluster_rows.append(pc.with_columns(pl.lit(ep).alias("episode")))
        sections.append(f"""## {cfg['title']} (book {cfg['book']}, {method} clusters)

{sum(1 for v in c.values() if v >= 0)} of {len(c)} seed funds in {len(set(v for v in c.values() if v >= 0))} clusters
(the rest are noise). {st.height} common/ADR issuers with an episode return
({n_dropped} dropped for a missing return); {st.filter(pl.col('crowded')).height} crowded in a cluster.

### Returns by crowding bucket

{cl.md(buckets)}

### Regression: episode return on crowding and characteristics (White t-stats)

{cl.md(reg)}

### Clusters going in

{cl.md(pc.drop("funds"))}

Members: {"; ".join(f"**{r['cluster']}** {r['funds']}" for r in pc.iter_rows(named=True))}

### Did they leave together? Actions by the next filing

{cl.md(ex)}
""")
        print(ep, buckets, reg, ex, sep="\n")

    sens = pl.DataFrame(sens)
    print(sens)
    ov = overlap_by_quarter(con, books, method)
    stocks, clusters_df = pl.concat(stock_rows, how="diagonal"), pl.concat(cluster_rows)
    for name, df in [("crowding_stocks", stocks), ("crowding_clusters", clusters_df), ("cluster_overlap", ov),
                     ("crowding_sensitivity", sens)]:
        con.register("df", df.to_arrow())
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df")
        con.unregister("df")
    plot_episodes(per_cluster, FIG / "16_crowding_episodes.png")
    plot_overlap(ov, FIG / "17_cluster_overlap.png")
    pct = {ep: float((ov["within"] <= ov.filter(pl.col("quarter") == cfg["book"])["within"][0]).mean())
           for ep, cfg in EPISODES.items()}
    REPORT.write_text(f"""# Crowding in two stress episodes (Milestone 5, Task B)

Generated by `src/models/crowding.py` (definitions in its docstring). Reading:
`docs/fund_map.md`. A stock is **crowded in a cluster** if at least {CROWD_MIN} funds
of one cluster, and at least {CROWD_SHARE:.0%} of its members, hold it. Returns
start at the book's quarter end; `ret_adj` subtracts the median return of the
stock's size quintile.

![episodes](figures/16_crowding_episodes.png)

## Sensitivity to the conviction floor (main: {CROWD_MIN_WEIGHT:.0%})

{cl.md(sens)}

## Overlap through time

![overlap](figures/17_cluster_overlap.png)

Within-cluster overlap going in, as a percentile of all quarters 2016Q1-{db.LAST_QUARTER}:
{", ".join(f"{ep} ({EPISODES[ep]['book']}) {p:.0%}" for ep, p in pct.items())}.

{cl.md(ov.filter(pl.col("quarter").is_in([q for q in db.QUARTERS if "2019Q1" <= q <= "2022Q4"])))}

{chr(10).join(sections)}""")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
