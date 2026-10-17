"""Fund embeddings (Milestone 5, Task B): one vector per fund-quarter from its
point-in-time long book.

Usage:
    uv run python -m src.features.embeddings

Input: `positions` (built from `seed_positions_pit`, so holdings are as of each
original filing), non-option rows, aggregated to issuers (issuer CIK for common
stock and ADRs, so share classes merge; the security key otherwise). Each book
is renormalised to sum to 1 and enters as **sqrt(weight)**: the row then has
unit L2 norm, the cosine of two rows is the Bhattacharyya overlap of the books,
and one 40% position doesn't swamp everything else.

Methods (every model is fitted per quarter on **seed** fund books only; holdout
funds and the background filers are projected into it, never fitted):
  cosine    the sqrt-weight row itself (similarity = cosine); no fitted space
  svd       truncated SVD, SVD_K dims, fund coordinates centred on the seed mean
  nmf       NMF, NMF_K themes; a fund is its (normalised) theme mix
  fund2vec  word2vec over sampled holdings, trained on the last W2V_WINDOW_Q
            quarters of seed books; a fund is the weight-average of its stocks'
            vectors, centred on the seed mean

Point in time: quarter q's models use only rows with known_at <= asof(q)
(quarter end + ASOF_DAYS; 99% of seed books are filed by day 48). A late filer
is projected into that quarter's model once it files. Each row's `known_at` is
the latest known_at of anything that went into it.

Alignment: per-quarter fits live in different bases, so consecutive quarters are
aligned before anything compares vectors across time. The anchors are the
**stocks**, not the funds: SVD loadings and fund2vec word vectors of issuers
present in both quarters are matched with an orthogonal Procrustes rotation
(aligning on funds would let the rotation soak up real fund drift). NMF themes
are matched by a permutation (Hungarian on theme cosine), which keeps them
non-negative and readable. Quarters are chained forward, so q is aligned using
only q-1's (already aligned) frame.

Background: for 2025Q3+ the SEC bulk data has every filer. The BACKGROUND_N
largest long books with active share >= BG_MIN_ACTIVE_SHARE (vs the seed
benchmark; the seed funds' 10th percentile is ~0.71) are projected for the map.
The active-share screen removes the banks, index and wealth managers that top a
pure size ranking.

Writes `fund_embeddings`, `nmf_themes`, `embedding_drift`.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import scipy.sparse as sp
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF, TruncatedSVD

from src.securities import db

METHODS = ["cosine", "svd", "nmf", "fund2vec"]
SVD_K, NMF_K, W2V_DIM = 16, 12, 32
W2V_WINDOW_Q = 4             # quarters of books in each fund2vec corpus
W2V_SENTENCES, W2V_LEN = 10, 40  # sampled "sentences" per book, tokens each
ASOF_DAYS = 60
BACKGROUND_N = 500
BG_MIN_ACTIVE_SHARE = 0.7
BG_MIN_LONG_VALUE = 1e8
BG_FIRST_QUARTER = "2025Q3"
SEED = 0

ISSUER_KEY = ("CASE WHEN security_class IN ('common', 'adr') AND issuer_cik IS NOT NULL "
              "THEN issuer_cik ELSE security_key END")


def q_end(quarter: str) -> date:
    y, q = int(quarter[:4]), int(quarter[-1])
    return date(y, 12, 31) if q == 4 else date(y, 3 * q + 1, 1) - timedelta(days=1)


def asof(quarter: str) -> date:
    return q_end(quarter) + timedelta(days=ASOF_DAYS)


def prev_quarters(quarter: str, n: int) -> list[str]:
    """quarter and the n-1 before it, oldest first."""
    i = db.QUARTERS.index(quarter)
    return db.QUARTERS[max(0, i - n + 1): i + 1]


# --- books ---------------------------------------------------------------------------

def load_books(con) -> pl.DataFrame:
    """fund_id, cohort, quarter, known_at, issuer_key, w (sums to 1 per fund-quarter)."""
    return con.execute(f"""
        WITH b AS (
            SELECT fund_id, cohort, quarter, max(known_at) AS known_at, {ISSUER_KEY} AS issuer_key,
                   sum(value_usd) AS v
            FROM positions WHERE NOT is_option AND value_usd > 0 GROUP BY ALL
        )
        SELECT fund_id, cohort, quarter, max(known_at) OVER (PARTITION BY fund_id, quarter) AS known_at,
               issuer_key, v / sum(v) OVER (PARTITION BY fund_id, quarter) AS w
        FROM b ORDER BY fund_id, quarter, issuer_key
    """).pl()


def background_books(con) -> pl.DataFrame:
    """The largest active non-seed filers per quarter, from the bulk 13F data
    (original 13F-HR only, earliest per CIK-quarter), same schema as load_books."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE bg_w AS
        WITH own AS (SELECT DISTINCT cik::BIGINT AS cik FROM fund_ciks),
        orig AS (
            SELECT cik::BIGINT AS cik, quarter, accession_number, known_at,
                   row_number() OVER (PARTITION BY cik, quarter ORDER BY known_at, accession_number) AS rk
            FROM filings_db.filings
            WHERE submission_type = '13F-HR' AND quarter BETWEEN '{BG_FIRST_QUARTER}' AND '{db.LAST_QUARTER}'
        ),
        keymap AS (SELECT cusip, any_value({ISSUER_KEY}) AS issuer_key FROM positions GROUP BY cusip),
        h AS (
            SELECT o.cik, any_value(o.quarter) AS quarter, any_value(o.known_at) AS known_at,
                   coalesce(k.issuer_key, h.cusip) AS issuer_key, sum(h.value_usd) AS v
            FROM orig o JOIN filings_db.holdings h USING (accession_number)
            LEFT JOIN keymap k ON k.cusip = h.cusip
            WHERE o.rk = 1 AND (h.put_call IS NULL OR h.put_call = '') AND h.value_usd > 0
              AND o.cik NOT IN (SELECT cik FROM own)
            GROUP BY o.cik, o.quarter, coalesce(k.issuer_key, h.cusip)
        )
        SELECT *, v / sum(v) OVER (PARTITION BY cik, quarter) AS w,
               sum(v) OVER (PARTITION BY cik, quarter) AS long_value
        FROM h
    """)
    return con.execute(f"""
        WITH both_sides AS (
            SELECT cik, quarter, issuer_key, w AS wf, 0.0 AS wb FROM bg_w
            UNION ALL
            SELECT f.cik, b.quarter, b.issuer_key, 0.0, b.w_bench
            FROM (SELECT DISTINCT cik, quarter FROM bg_w) f JOIN benchmark_weights b USING (quarter)
        ),
        act AS (
            SELECT cik, quarter, 0.5 * sum(abs(wf - wb)) AS active_share
            FROM (SELECT cik, quarter, issuer_key, sum(wf) AS wf, sum(wb) AS wb FROM both_sides GROUP BY ALL)
            GROUP BY ALL
        ),
        pick AS (
            SELECT a.cik, a.quarter, a.active_share,
                   row_number() OVER (PARTITION BY a.quarter ORDER BY v.long_value DESC) AS rk
            FROM act a JOIN (SELECT DISTINCT cik, quarter, long_value FROM bg_w) v USING (cik, quarter)
            WHERE a.active_share >= {BG_MIN_ACTIVE_SHARE} AND v.long_value >= {BG_MIN_LONG_VALUE}
        )
        SELECT 'bg_' || w.cik::VARCHAR AS fund_id, 'background' AS cohort, w.quarter, w.known_at,
               w.issuer_key, w.w
        FROM bg_w w JOIN pick p USING (cik, quarter)
        WHERE p.rk <= {BACKGROUND_N}
        ORDER BY fund_id, quarter, issuer_key
    """).pl()


def sqrt_matrix(books: pl.DataFrame, ids: list[str], vocab: dict[str, int]) -> tuple[sp.csr_matrix, np.ndarray]:
    """Rows = ids (fund_ids of one quarter), columns = vocab, values sqrt(w).
    Off-vocabulary weight is dropped; returns the in-vocabulary weight share."""
    row = {f: i for i, f in enumerate(ids)}
    b = books.filter(pl.col("fund_id").is_in(ids))
    keep = b.filter(pl.col("issuer_key").is_in(list(vocab)))
    r = np.array([row[f] for f in keep["fund_id"]], dtype=int)
    c = np.array([vocab[k] for k in keep["issuer_key"]], dtype=int)
    m = sp.csr_matrix((np.sqrt(keep["w"].to_numpy()), (r, c)), shape=(len(ids), len(vocab)))
    cov = keep.group_by("fund_id").agg(pl.col("w").sum())
    cov_map = dict(zip(cov["fund_id"], cov["w"]))
    return m, np.array([cov_map.get(f, 0.0) for f in ids])


def unit_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 0)


# --- alignment -----------------------------------------------------------------------

def procrustes_align(cur: dict[str, np.ndarray], prev: dict[str, np.ndarray] | None) -> np.ndarray:
    """Rotation R (d x d) so that cur[k] @ R ~ prev[k] over the shared keys
    (the anchor stocks). Identity for the first quarter."""
    d = next(iter(cur.values())).shape[0]
    if prev is None:
        return np.eye(d)
    common = [k for k in cur if k in prev]
    if len(common) < d:
        return np.eye(d)
    a = np.vstack([cur[k] for k in common])
    b = np.vstack([prev[k] for k in common])
    r, _ = orthogonal_procrustes(a, b)
    return r


def match_themes(h_cur: dict[str, np.ndarray], h_prev: dict[str, np.ndarray] | None, k: int) -> np.ndarray:
    """Permutation perm so that theme perm[j] of the current fit is theme j of
    the previous one (Hungarian on theme cosine over the shared stocks).
    h_* map issuer -> loading on each of the k themes."""
    if h_prev is None:
        return np.arange(k)
    common = [s for s in h_cur if s in h_prev]
    a = unit_rows(np.vstack([h_cur[s] for s in common]).T)
    b = unit_rows(np.vstack([h_prev[s] for s in common]).T)
    _, cols = linear_sum_assignment(-(b @ a.T))   # rows: previous themes, cols: current
    return cols


# --- per-quarter fits ----------------------------------------------------------------

def fit_svd(x_fit: sp.csr_matrix) -> TruncatedSVD:
    return TruncatedSVD(n_components=SVD_K, random_state=SEED).fit(x_fit)


def fit_nmf(x_fit: sp.csr_matrix) -> NMF:
    return NMF(n_components=NMF_K, init="nndsvda", max_iter=1000, random_state=SEED).fit(x_fit)


def w2v_corpus(books: pl.DataFrame, quarter: str, rng: np.random.Generator) -> tuple[list[list[str]], set[str]]:
    """Sentences sampled from seed books of the last W2V_WINDOW_Q quarters that
    were filed by asof(quarter). Tokens are drawn with probability = weight, so a
    big position appears often. Vocabulary: issuers in >= 2 distinct seed funds."""
    win = books.filter(pl.col("quarter").is_in(prev_quarters(quarter, W2V_WINDOW_Q))
                       & (pl.col("known_at") <= asof(quarter)) & (pl.col("cohort") == "seed"))
    vocab = set(win.group_by("issuer_key").agg(pl.col("fund_id").n_unique().alias("n"))
                .filter(pl.col("n") >= 2)["issuer_key"])
    win = win.filter(pl.col("issuer_key").is_in(list(vocab)))
    sentences = []
    for (_, _), g in win.group_by(["fund_id", "quarter"], maintain_order=True):
        keys, w = g["issuer_key"].to_list(), g["w"].to_numpy()
        p = w / w.sum()
        for _ in range(W2V_SENTENCES):
            sentences.append([keys[i] for i in rng.choice(len(keys), size=W2V_LEN, p=p)])
    return sentences, vocab


def fit_w2v(sentences: list[list[str]]) -> Word2Vec:
    # workers=1 so a fixed seed gives the same vectors every run
    return Word2Vec(sentences, vector_size=W2V_DIM, window=5, min_count=1, sg=1, negative=5,
                    epochs=10, seed=SEED, workers=1)


def w2v_fund_vectors(books: pl.DataFrame, ids: list[str], vec: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    out, cov = np.zeros((len(ids), W2V_DIM)), np.zeros(len(ids))
    for i, f in enumerate(ids):
        g = books.filter((pl.col("fund_id") == f) & pl.col("issuer_key").is_in(list(vec)))
        if g.height:
            w = g["w"].to_numpy()
            out[i] = (w[:, None] * np.vstack([vec[k] for k in g["issuer_key"]])).sum(0) / w.sum()
            cov[i] = w.sum()
    return out, cov


def embed_all(books: pl.DataFrame, issuer_names: dict[str, str]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows, themes, drift = [], [], []
    prev_svd = prev_w2v = prev_nmf = None
    prev_vecs: dict[str, dict[str, np.ndarray]] = {m: {} for m in METHODS[1:]}
    quarters = sorted(books.filter(pl.col("cohort") == "seed")["quarter"].unique())
    for q in quarters:
        bq = books.filter(pl.col("quarter") == q)
        meta = bq.group_by("fund_id").agg(pl.col("cohort").first(), pl.col("known_at").max()).sort("fund_id")
        fit_meta = meta.filter((pl.col("cohort") == "seed") & (pl.col("known_at") <= asof(q)))
        fit_ids = fit_meta["fund_id"].to_list()
        all_ids = meta["fund_id"].to_list()
        fit_known = fit_meta["known_at"].max()
        vocab_keys = sorted(bq.filter(pl.col("fund_id").is_in(fit_ids))["issuer_key"].unique())
        vocab = {k: i for i, k in enumerate(vocab_keys)}
        x_fit, _ = sqrt_matrix(bq, fit_ids, vocab)
        x_all, cov_all = sqrt_matrix(bq, all_ids, vocab)
        fit_mask = np.isin(all_ids, fit_ids)
        out: dict[str, tuple[np.ndarray, np.ndarray, date]] = {}

        # svd: rotate by the stock-anchored Procrustes, centre on the seed mean
        svd = fit_svd(x_fit)
        loads = {k: svd.components_[:, i] for k, i in vocab.items()}
        r = procrustes_align(loads, prev_svd)
        prev_svd = {k: v @ r for k, v in loads.items()}
        z = svd.transform(x_all) @ r
        out["svd"] = (unit_rows(z - z[fit_mask].mean(0)), cov_all, fit_known)

        # nmf: permute themes to match last quarter, normalise each theme's loadings
        nmf = fit_nmf(x_fit)
        norms = np.linalg.norm(nmf.components_, axis=1)
        h = nmf.components_ / norms[:, None]
        h_map = {k: h[:, i] for k, i in vocab.items()}
        perm = match_themes(h_map, prev_nmf, NMF_K)
        h, norms = h[perm], norms[perm]
        prev_nmf = {k: h[:, i] for k, i in vocab.items()}
        wmat = nmf.transform(x_all)[:, perm] * norms
        out["nmf"] = (unit_rows(wmat), cov_all, fit_known)
        for j in range(NMF_K):
            top = np.argsort(-h[j])[:8]
            themes.append({"quarter": q, "theme": j,
                           "top_issuers": "; ".join(issuer_names.get(vocab_keys[t], vocab_keys[t]) for t in top),
                           "top_loadings": [float(h[j, t]) for t in top]})

        # fund2vec: corpus up to q, word vectors rotated onto last quarter's
        sentences, w2v_vocab = w2v_corpus(books, q, rng)
        w2v = fit_w2v(sentences)
        # a tiny position may never be sampled: only trained tokens get a vector
        wv = {k: w2v.wv[k] for k in w2v.wv.index_to_key}
        r = procrustes_align(wv, prev_w2v)
        wv = {k: v @ r for k, v in wv.items()}
        prev_w2v = wv
        fv, fcov = w2v_fund_vectors(bq, all_ids, wv)
        corpus_known = books.filter(pl.col("quarter").is_in(prev_quarters(q, W2V_WINDOW_Q))
                                    & (pl.col("cohort") == "seed")
                                    & (pl.col("known_at") <= asof(q)))["known_at"].max()
        out["fund2vec"] = (unit_rows(fv - fv[fit_mask].mean(0)), fcov, corpus_known)

        known = dict(zip(meta["fund_id"], meta["known_at"]))
        cohort = dict(zip(meta["fund_id"], meta["cohort"]))
        for m, (vecs, cov, model_known) in out.items():
            for i, f in enumerate(all_ids):
                rows.append({"method": m, "fund_id": f, "cohort": cohort[f], "quarter": q,
                             "known_at": max(known[f], model_known), "coverage": float(cov[i]),
                             "vec": vecs[i].astype(float).tolist()})
                if cohort[f] == "seed" and f in prev_vecs[m]:
                    drift.append({"method": m, "fund_id": f, "quarter": q,
                                  "move": float(1 - vecs[i] @ prev_vecs[m][f])})
            prev_vecs[m] = {f: vecs[i] for i, f in enumerate(all_ids) if cohort[f] == "seed"}
        print(f"{q}: {len(fit_ids)} fit, {len(all_ids)} embedded, vocab {len(vocab)}, w2v vocab {len(wv)}/{len(w2v_vocab)}")
    return pl.DataFrame(rows), pl.DataFrame(themes), pl.DataFrame(drift)


def unaligned_drift(books: pl.DataFrame) -> pl.DataFrame:
    """The same quarter-to-quarter move for svd without the Procrustes step, to
    show what alignment removes (report only)."""
    out, prev = [], {}
    for q in sorted(books.filter(pl.col("cohort") == "seed")["quarter"].unique()):
        bq = books.filter((pl.col("quarter") == q) & (pl.col("cohort") == "seed") & (pl.col("known_at") <= asof(q)))
        ids = sorted(bq["fund_id"].unique())
        vocab = {k: i for i, k in enumerate(sorted(bq["issuer_key"].unique()))}
        x, _ = sqrt_matrix(bq, ids, vocab)
        z = fit_svd(x).transform(x)
        z = unit_rows(z - z.mean(0))
        cur = dict(zip(ids, z))
        out += [{"method": "svd_unaligned", "fund_id": f, "quarter": q, "move": float(1 - v @ prev[f])}
                for f, v in cur.items() if f in prev]
        prev = cur
    return pl.DataFrame(out)


def issuer_names(con) -> dict[str, str]:
    rows = con.execute(f"""
        SELECT {ISSUER_KEY} AS k, arg_max(coalesce(symbol, name_of_issuer), value_usd) AS name
        FROM positions GROUP BY 1
    """).fetchall()
    return dict(rows)


def main() -> None:
    con = db.connect()
    con.execute("SET enable_progress_bar = false")
    books = load_books(con)
    bg = background_books(con)
    print(f"background: {bg.select(pl.col('fund_id').n_unique()).item()} filers, "
          f"{bg['quarter'].n_unique()} quarters")
    emb, themes, drift = embed_all(pl.concat([books, bg]), issuer_names(con))
    drift = pl.concat([drift, unaligned_drift(books)])
    for name, df in [("fund_embeddings", emb), ("nmf_themes", themes), ("embedding_drift", drift)]:
        con.register("df", df.to_arrow())
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df")
        con.unregister("df")
        print(f"{name}: {df.height} rows")


if __name__ == "__main__":
    main()
