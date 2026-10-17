"""Task C2: which stocks will a fund newly buy next quarter? A recommender.

Usage:
    uv run python -m src.models.task_c2 baselines   # popularity baselines -> docs/task_c_baselines.md (via task_c_baselines)
    uv run python -m src.models.task_c2 run         # tune on 2022, score 2023+ -> data/processed/task_c2_results.parquet

**Unit:** a fund-quarter (f, q), predicted on f's quarter-q filing date T. The
targets are the stock-universe securities f holds long at q+1 and did not hold
at q (`new_buys`); fund-quarters without any are not scored.

**Point in time:** every recommender sees only the seed funds' books whose
filing was public by T: each seed fund's latest filing for a quarter <= q with
known_at <= T (from `task_c_peer_latest`, the same rule as the C1 peer
features), plus, with `depth` > 1, its earlier filings. f's own quarter-q book
is the "user". `audit` rows record the latest known_at used per group, and
src.models.leakage_audit checks them against T.

Candidates: every security in a visible book, minus what f holds at q. A new
buy that no visible book holds can't be recommended and counts as a miss
(`reachable` reports the share that could be).

Recommenders:
  popularity          seed funds' new buys in their latest visible filing (count)
  cluster_popularity  the same within f's cluster (latest clustering known by T);
                      noise and holdout funds fall back to popularity
  reentry             f's own exits of the last 4 quarters, most recent first,
                      then popularity (an extra baseline: funds often come back)
  item_cf             item-item cosine over the visible books: score(j) =
                      sum_i u_i cos(i, j), u = f's book
  als                 implicit-feedback ALS (`implicit`), fitted at each T on the
                      visible books; f's factor recomputed from its own book

**Selection rule, fixed before any run:** each tuned recommender's grid is
scored on the 2022 seed fund-quarters by mean NDCG@20; the best setting is
then run once on 2023+. Train years (2018-2021) are not needed: nothing here
has parameters learned from labels.

Metrics per fund-quarter, then averaged with equal weight: recall@20 (hits /
new buys), NDCG@20 (binary relevance) and hit rate (any hit). Subsets: all,
`ret_clean` (new buys whose return over the quarter is a 0% fallback or missing
are removed from both the ranking and the targets), and without
multi-manager / quant funds.
"""

import itertools
import os
import sys
from dataclasses import dataclass

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # implicit warns (and is slower) with threaded BLAS

import numpy as np
import polars as pl
import scipy.sparse as sp

from src.features.task_c import CLUSTER_ALGO, CLUSTER_METHOD
from src.models.task_c_common import MULTI, SPLITS, split_of
from src.models import style_classifier
from src.securities import db

K = 20
ITEM_CLASSES = ("common", "adr", "etf", "fund")
RESULTS_PATH = "data/processed/task_c2_results.parquet"
GRIDS = {
    "item_cf": [{"depth": d, "binary": b} for d in (1, 4) for b in (True, False)],
    "als": [{"depth": d, "factors": f, "regularization": r, "alpha": a}
            for d, f, r, a in itertools.product((1, 4), (16, 64), (0.01, 0.1), (1.0, 10.0))],
}


@dataclass
class Data:
    books: pl.DataFrame        # fund_id, cohort, q_idx, cusip, w, is_new, known_at
    targets: pl.DataFrame      # fund_id, cohort, quarter, q_idx, pred_date, style, cusip, ret_clean
    peers: pl.DataFrame        # fund_id, quarter, pred_date, peer_id, peer_q_idx
    clusters: pl.DataFrame     # fund_id, quarter, peer_id (cluster peers as of T)


def load(con, quarters_from: str = "2022Q1") -> Data:
    books = con.execute(f"""
        SELECT p.fund_id, p.cohort, left(p.quarter, 4)::INT * 4 + right(p.quarter, 1)::INT AS q_idx,
               p.cusip, p.weight_long AS w, p.is_new, p.known_at
        FROM positions p JOIN security_master m USING (cusip)
        WHERE p.put_call = '' AND p.weight_long > 0 AND m.security_class IN {ITEM_CLASSES}
    """).pl()
    styles = style_classifier.labels().select("fund_id", pl.col("label_v2").alias("style"))
    targets = (con.execute(f"""
        SELECT fund_id, cohort, quarter, left(quarter, 4)::INT * 4 + right(quarter, 1)::INT AS q_idx,
               pred_date, label_known_at, cusip, ret_clean
        FROM new_buys WHERE quarter >= '{quarters_from}'
    """).pl().join(styles, on="fund_id", how="left")
               .with_columns(split_of(pl.col("quarter")).alias("split")).filter(pl.col("split").is_not_null()))
    peers = con.execute(f"""
        SELECT fund_id, quarter, pred_date, peer_id, peer_q_idx, peer_known_at
        FROM task_c_peer_latest WHERE quarter >= '{quarters_from}'
    """).pl()
    clusters = con.execute(f"""
        SELECT tc.fund_id, tc.quarter, c.fund_id AS peer_id
        FROM task_c_cluster tc JOIN fund_clusters c
          ON c.quarter = tc.cl_quarter AND c.cluster = tc.cluster AND tc.cluster >= 0 AND c.fund_id <> tc.fund_id
         AND c.method = '{CLUSTER_METHOD}' AND c.algo = '{CLUSTER_ALGO}'
        WHERE tc.quarter >= '{quarters_from}'
    """).pl()
    return Data(books, targets, peers, clusters)


def visible_rows(data: Data, quarter: str, pred_date, depth: int) -> pl.DataFrame:
    """Seed books public by pred_date: each seed fund's latest visible filing and up to depth-1 earlier ones."""
    group = data.targets.filter((pl.col("quarter") == quarter) & (pl.col("pred_date") == pred_date))
    latest = (data.peers.filter((pl.col("quarter") == quarter) & (pl.col("pred_date") == pred_date))
              .select(pl.col("peer_id").alias("fund_id"), pl.col("peer_q_idx").alias("latest")))
    own = group.filter(pl.col("cohort") == "seed").select("fund_id", pl.col("q_idx").alias("latest"))
    latest = pl.concat([latest, own]).unique()
    return (data.books.join(latest, on="fund_id")
            .filter((pl.col("q_idx") <= pl.col("latest")) & (pl.col("q_idx") > pl.col("latest") - depth)
                    & (pl.col("known_at") <= pred_date))
            .with_columns((pl.col("latest") - pl.col("q_idx")).alias("lag")))


def matrix(rows: pl.DataFrame, items: dict[str, int], binary: bool) -> sp.csr_matrix:
    keys = rows.select("fund_id", "q_idx").unique().sort("fund_id", "q_idx").with_row_index("r")
    r = rows.join(keys, on=["fund_id", "q_idx"])
    vals = np.ones(len(r)) if binary else np.sqrt(r["w"].to_numpy())
    cols = np.array([items[c] for c in r["cusip"]])
    return sp.csr_matrix((vals, (r["r"].to_numpy(), cols)), shape=(len(keys), len(items)))


def user_vector(book: pl.DataFrame, items: dict[str, int], binary: bool) -> sp.csr_matrix:
    b = book.filter(pl.col("cusip").is_in(list(items)))
    vals = np.ones(len(b)) if binary else np.sqrt(b["w"].to_numpy())
    cols = np.array([items[c] for c in b["cusip"]], dtype=int)
    return sp.csr_matrix((vals, (np.zeros(len(b), dtype=int), cols)), shape=(1, len(items)))


def popularity_scores(rows: pl.DataFrame, items: dict[str, int], funds: list[str] | None = None) -> np.ndarray:
    r = rows.filter((pl.col("lag") == 0) & pl.col("is_new"))
    if funds is not None:
        r = r.filter(pl.col("fund_id").is_in(funds))
    held = rows.filter(pl.col("lag") == 0)
    s = np.zeros(len(items))
    for c, n in r.group_by("cusip").len().iter_rows():
        s[items[c]] += n
    for c, n in held.group_by("cusip").len().iter_rows():
        s[items[c]] += 1e-3 * n   # tie-break by holders
    return s


def score_group(data: Data, quarter: str, pred_date, method: str, params: dict) -> tuple[list[dict], dict]:
    depth = params.get("depth", 1)
    rows = visible_rows(data, quarter, pred_date, max(depth, 1))
    items = {c: i for i, c in enumerate(sorted(set(rows["cusip"])))}
    inv = np.array(list(items))
    group = data.targets.filter((pl.col("quarter") == quarter) & (pl.col("pred_date") == pred_date))
    q_idx = group["q_idx"][0]
    audit = {"quarter": quarter, "pred_date": pred_date, "max_input_known_at": rows["known_at"].max(),
             "n_rows": rows.select("fund_id", "q_idx").n_unique(), "n_items": len(items)}

    if method in ("item_cf", "als"):
        M = matrix(rows, items, params.get("binary", True))
    if method == "item_cf":
        norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=0))).ravel()
        Mn = M @ sp.diags(1 / np.where(norms > 0, norms, 1))
    if method == "als":
        from implicit.cpu.als import AlternatingLeastSquares
        model = AlternatingLeastSquares(factors=params["factors"], regularization=params["regularization"],
                                        alpha=params["alpha"], iterations=15, random_state=0, num_threads=1)
        model.fit(M, show_progress=False)
    base_pop = popularity_scores(rows, items) if method != "als" else None

    out = []
    for f, cohort, style in group.select("fund_id", "cohort", "style").unique().iter_rows():
        book = data.books.filter((pl.col("fund_id") == f) & (pl.col("q_idx") == q_idx))
        held = set(book["cusip"])
        if method == "popularity":
            s = base_pop.copy()
        elif method == "cluster_popularity":
            cl = data.clusters.filter((pl.col("fund_id") == f) & (pl.col("quarter") == quarter))["peer_id"].to_list()
            s = popularity_scores(rows, items, cl) + 1e-3 * base_pop if cl else base_pop.copy()
        elif method == "reentry":
            past = data.books.filter((pl.col("fund_id") == f) & (pl.col("q_idx") < q_idx) & (pl.col("q_idx") >= q_idx - 4)
                                     & (pl.col("known_at") <= pred_date))
            s = base_pop.copy() / (base_pop.max() + 1)
            for c, qi in past.group_by("cusip").agg(pl.col("q_idx").max()).iter_rows():
                if c in items:
                    s[items[c]] += 1 + (qi - (q_idx - 4))   # most recent exit ranks first
        elif method == "item_cf":
            u = user_vector(book, items, params.get("binary", True))
            s = np.asarray((Mn.T @ (Mn @ u.T)).todense()).ravel()
            s += 1e-6 * base_pop   # tie-break among unscored items
        elif method == "als":
            u = user_vector(book, items, params.get("binary", True))
            ids, _ = model.recommend(0, u, N=min(len(items), K + len(held) + 200),
                                     filter_already_liked_items=True, recalculate_user=True)
            s = np.full(len(items), -np.inf)
            s[ids] = -np.arange(len(ids), dtype=float)
        if held:
            s[[items[c] for c in held if c in items]] = -np.inf
        order = np.argsort(-s, kind="stable")
        ranked = [inv[i] for i in order[:K + 500] if np.isfinite(s[i])]
        tg = group.filter(pl.col("fund_id") == f)
        out.append({"fund_id": f, "cohort": cohort, "style": style, "quarter": quarter, "pred_date": pred_date,
                    "ranked": ranked, "targets": tg["cusip"].to_list(), "clean": tg["ret_clean"].to_list(),
                    "reachable": float(np.mean([c in items for c in tg["cusip"]]))})
    return out, audit


def rank_metrics(ranked: list[str], targets: set[str]) -> dict[str, float]:
    top = ranked[:K]
    hits = [c in targets for c in top]
    dcg = sum(1 / np.log2(i + 2) for i, h in enumerate(hits) if h)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(targets), K)))
    return {"recall": sum(hits) / len(targets), "ndcg": dcg / idcg, "hit": float(any(hits))}


def evaluate(recs: list[dict]) -> pl.DataFrame:
    rows = []
    for r in recs:
        t_all = set(r["targets"])
        bad = {c for c, ok in zip(r["targets"], r["clean"]) if not ok}
        base = {k: r[k] for k in ("fund_id", "cohort", "style", "quarter")} | {"reachable": r["reachable"],
                                                                              "n_targets": len(t_all)}
        rows.append(base | {"subset": "all"} | rank_metrics(r["ranked"], t_all))
        if t_all - bad:
            rows.append(base | {"subset": "ret_clean"} | rank_metrics([c for c in r["ranked"] if c not in bad], t_all - bad))
    return pl.DataFrame(rows)


def summarize(ev: pl.DataFrame) -> pl.DataFrame:
    ev = pl.concat([ev, ev.filter((pl.col("subset") == "all") & ~pl.col("style").is_in(MULTI))
                   .with_columns(pl.lit("excl_multi_quant").alias("subset"))])
    return (ev.group_by("subset").agg(pl.len().alias("fund_quarters"), pl.col("recall").mean().alias("recall@20"),
                                      pl.col("ndcg").mean().alias("ndcg@20"), pl.col("hit").mean().alias("hit_rate"),
                                      pl.col("reachable").mean())
            .sort("subset"))


def run(data: Data, method: str, params: dict, split: str, cohorts=("seed",)) -> tuple[pl.DataFrame, pl.DataFrame]:
    t = data.targets.filter((pl.col("split") == split) & pl.col("cohort").is_in(cohorts))
    recs, audits = [], []
    for quarter, pred_date in t.select("quarter", "pred_date").unique().sort("quarter", "pred_date").iter_rows():
        d = Data(data.books, t, data.peers, data.clusters)
        r, a = score_group(d, quarter, pred_date, method, params)
        recs += r
        audits.append(a)
    return evaluate(recs), pl.DataFrame(audits)


BASELINES = ["popularity", "cluster_popularity", "reentry"]


def main(mode: str) -> None:
    with db.connect(read_only=True) as con:
        data = load(con)
    results, audits = [], []
    methods = [(m, {}) for m in BASELINES]
    if mode == "run":
        for method, grid in GRIDS.items():
            scores = []
            for params in grid:
                ev, _ = run(data, method, params, "val")
                ndcg = ev.filter(pl.col("subset") == "all")["ndcg"].mean()
                scores.append((ndcg, params))
                print(f"  val {method} {params}: ndcg@20 {ndcg:.4f}", flush=True)
                results.append(summarize(ev).with_columns(pl.lit(method).alias("method"), pl.lit(str(params)).alias("params"),
                                                          pl.lit("val_grid").alias("split"), pl.lit("seed").alias("cohort")))
            best = max(scores, key=lambda x: x[0])[1]
            print(f"chosen {method}: {best}")
            methods.append((method, best))
    for method, params in methods:
        for split in ("val", "test"):
            for cohort in ("seed", "holdout"):
                if split == "val" and cohort == "holdout":
                    continue
                ev, au = run(data, method, params, split, (cohort,))
                results.append(summarize(ev).with_columns(pl.lit(method).alias("method"), pl.lit(str(params)).alias("params"),
                                                          pl.lit(split).alias("split"), pl.lit(cohort).alias("cohort")))
                audits.append(au.with_columns(pl.lit(method).alias("method"), pl.lit(split).alias("split"),
                                              pl.lit(cohort).alias("cohort")))
                print(results[-1].filter(pl.col("subset") == "all").select("method", "split", "cohort", "recall@20",
                                                                           "ndcg@20", "hit_rate"), flush=True)
    res = pl.concat(results, how="diagonal")
    out = RESULTS_PATH if mode == "run" else RESULTS_PATH.replace(".parquet", "_baselines.parquet")
    res.write_parquet(out)
    pl.concat(audits).write_parquet(out.replace(".parquet", "_audit.parquet"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run")
