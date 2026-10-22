"""Crowded now: the Milestone 5 crowding measures on the latest point-in-time quarter.

Usage:
    uv run python -m src.analysis.crowding_now     # -> docs/crowding_now.md, app/data/crowded_now.parquet

Same definitions as src.models.crowding (applied unchanged to one quarter):
  conviction holder  a seed fund with the issuer at >= 1% of its long book
  crowded            >= 3 conviction holders from one fund2vec/HDBSCAN cluster, and
                     >= half of that cluster
Books are the seed funds' original 13F filings for the quarter (point in time);
clusters are that quarter's clustering. The as-of date is the latest filing or
clustering used: nothing published after it enters. Clusters are named by their
members' majority hand style label (the name only; membership comes from holdings).

Top N_TOP issuers by number of conviction holders, then by the largest count inside
one cluster, then by the holders' summed weight. ETFs and funds are left out of the
ranking (they are mostly hedges in a 13F, as in Milestone 7); weights stay on the
full long book.
"""

from collections import Counter
from pathlib import Path

import polars as pl

from src.features import embeddings as emb
from src.models import crowding, style_classifier
from src.models.task_c_common import md_table
from src.securities import db

N_TOP = 20
DOC = Path("docs/crowding_now.md")
APP_FILE = Path("app/data/crowded_now.parquet")


def cluster_names(clusters: dict[str, int]) -> dict[int, str]:
    """cluster id -> majority style of its members ('style', or 'style #2' if two clusters share one)."""
    styles = dict(style_classifier.labels().select("fund_id", "label_v2").iter_rows())
    members: dict[int, list[str]] = {}
    for f, c in clusters.items():
        if c >= 0:
            members.setdefault(c, []).append(f)
    votes = {c: sorted(Counter(styles.get(f) for f in fs if styles.get(f)).items(), key=lambda kv: (-kv[1], kv[0]))
             for c, fs in members.items()}
    raw = {c: v[0][0] if v else f"cluster {c}" for c, v in votes.items()}   # no labelled member: just the id
    seen: Counter = Counter()
    out = {}
    for c in sorted(raw):
        seen[raw[c]] += 1
        out[c] = raw[c] if seen[raw[c]] == 1 else f"{raw[c]} #{seen[raw[c]]}"
    return out | {-1: "no cluster"}


def snapshot(book: pl.DataFrame, clusters: dict[str, int], fund_names: dict[str, str],
             issuers: dict[str, str], exclude: set[str] = frozenset()) -> pl.DataFrame:
    """Top N_TOP crowded issuers with their conviction holders."""
    names = cluster_names(clusters)
    m = crowding.crowding_measures(book, clusters)
    conv = (book.filter(pl.col("w") >= crowding.CROWD_MIN_WEIGHT)
            .with_columns(pl.col("fund_id").replace_strict(clusters, default=-1, return_dtype=pl.Int64).alias("cluster")))
    holders = (conv.sort("w", descending=True).group_by("issuer_key")
               .agg(pl.col("w").sum().alias("holders_weight"),
                    pl.col("w").max().alias("max_weight"),
                    pl.struct("fund_id", "cluster", "w").alias("h")))
    top = (m.filter((pl.col("n_holders") > 0) & ~pl.col("issuer_key").is_in(list(exclude)))
           .join(holders, on="issuer_key", how="left")
           .sort(["n_holders", "max_cluster_n", "holders_weight", "issuer_key"], descending=[True, True, True, False])
           .head(N_TOP).with_row_index("rank", offset=1))
    return top.with_columns(
        pl.col("issuer_key").replace_strict(issuers, default=None).alias("stock"),
        pl.col("top_cluster").replace_strict(names, default="no cluster").alias("top_cluster_name"),
        pl.col("h").map_elements(lambda hs: "; ".join(f"{fund_names.get(x['fund_id'], x['fund_id'])} "
                                                      f"({names.get(x['cluster'], 'no cluster')}, {x['w']:.1%})" for x in hs),
                                 return_dtype=pl.Utf8).alias("holders"),
    ).with_columns(pl.col("holders").map_elements(_short, return_dtype=pl.Utf8).alias("top_holders")).select(
        "rank", "issuer_key", "stock", "n_holders", "max_cluster_n", "top_cluster_name", "crowded",
        "holders_weight", "max_weight", "top_holders", "holders")


def _short(holders: str, k: int = 8) -> str:
    parts = holders.split("; ")
    return "; ".join(parts[:k]) + (f"; and {len(parts) - k} more" if len(parts) > k else "")


def build(quarter: str = db.LAST_QUARTER) -> tuple[pl.DataFrame, dict]:
    with db.connect(read_only=True) as con:
        method = crowding.chosen_method()
        clusters = crowding.clusters_at(con, method, quarter)
        books = emb.load_books(con).filter(pl.col("cohort") == "seed")
        book = crowding.seed_books(books, quarter, list(clusters))
        cl_known = con.execute("""SELECT max(known_at) FROM fund_clusters WHERE method = ? AND algo = 'hdbscan'
                                  AND quarter = ?""", [method, quarter]).fetchone()[0]
        fund_names = dict(con.execute("SELECT DISTINCT fund_id, fund_name FROM positions").fetchall())
        etfs = {k for (k,) in con.execute(f"""SELECT DISTINCT {emb.ISSUER_KEY} FROM positions
                                              WHERE security_class IN ('etf', 'fund')""").fetchall()}
        issuers = emb.issuer_names(con)
        full = dict(con.execute(f"""SELECT {emb.ISSUER_KEY}, arg_max(name_of_issuer, value_usd) FROM positions
                                    WHERE quarter = ? GROUP BY 1""", [quarter]).fetchall())
    top = snapshot(book, clusters, fund_names, issuers, etfs).with_columns(
        pl.col("issuer_key").replace_strict(full, default=None).str.replace_all(r"\s+", " ").alias("issuer_name"))
    names = cluster_names(clusters)
    sizes = Counter(c for c in clusters.values())
    meta = {"quarter": quarter, "as_of": max(book["known_at"].max(), cl_known), "funds": len(clusters),
            "clusters": {names[c]: n for c, n in sorted(sizes.items()) if c >= 0}, "noise_funds": sizes.get(-1, 0)}
    return top, meta


def main() -> None:
    top, meta = build()
    out = top.with_columns(pl.lit(meta["quarter"]).alias("quarter"), pl.lit(meta["as_of"]).alias("as_of"))
    out.write_parquet(APP_FILE, compression="zstd", compression_level=19)
    show = top.select("rank", "stock", "issuer_name", "n_holders", "max_cluster_n", "top_cluster_name", "crowded",
                      pl.col("holders_weight").round(3), "top_holders")
    cl = pl.DataFrame({"cluster": list(meta["clusters"]), "funds": list(meta["clusters"].values())})
    DOC.write_text(f"""# Crowded now ({meta['quarter']} 13F filings, as of {meta['as_of']})

Generated by `src/analysis/crowding_now.py`: the Milestone 5 crowding measures (`src/models/crowding.py`,
`docs/crowding.md`) on the seed funds' latest point-in-time books. Nothing published after {meta['as_of']} is used.

- **Conviction holder:** a seed fund with the stock at 1% or more of its long book.
- **Crowded:** at least 3 conviction holders from one cluster, and at least half of that cluster.
- **Ranking:** by conviction holders, then the largest count inside one cluster, then the holders' summed weight.
  ETFs and funds are left out (mostly hedges in a 13F; SPY and IVV would otherwise rank 12th and 19th).
- {meta['funds']} seed funds filed for {meta['quarter']}; {meta['noise_funds']} are in no cluster.

## Top {N_TOP}

`max_cluster_n`: conviction holders inside the largest single cluster. `holders_weight`: sum of the holders' weights
(1.0 = one full book). The 8 largest holders by weight, with their cluster and the stock's weight in their book
(full lists in `app/data/crowded_now.parquet` and the app's "Crowded now" page).

{md_table(show)}

## Clusters this quarter (named by members' majority hand style)

{md_table(cl)}

**How to read it.** Crowding measured this way ranked names that fell hard when a cluster de-grossed in the 2021-22
growth unwind (−23% vs size-matched peers, t = −2.6) but not in the COVID crash (`docs/crowding.md`). A name on this
list is not a forecast; it is where the same funds would be selling at once if they had to.
""")
    print(DOC.read_text()[:3000])


if __name__ == "__main__":
    main()
