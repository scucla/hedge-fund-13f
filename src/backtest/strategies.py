"""Which stocks each strategy holds after each quarter's filings (Task D).

All portfolios are long-only, equal-weight, at most N_NAMES issuers, common
stock and ADRs only (ETFs and funds in a 13F are mostly hedges, not ideas).
Each uses only filings with known_at <= the quarter's cutoff
(src.backtest.engine.calendar). Share classes of one issuer are merged
(issuer CIK, else CUSIP) and held through the CUSIP most seed funds hold.
Parameters were fixed before any backtest ran. A stock whose delisting (SEC
Form 25/15) is dated on or before the trade date is skipped for the next-ranked
one: it can't be bought, and the delisting was public.

  consensus      best ideas across seed funds: issuers in the most funds' top-10
                 stock positions (ties: higher average weight). Funds with more
                 than MAX_POSITIONS positions in that filing don't vote: their top
                 10 is not conviction (a point-in-time rule, not the hindsight
                 style labels).
  cluster:<style> one portfolio per Milestone 5 style cluster: the latest
                 fund2vec/HDBSCAN clustering public by the cutoff; a cluster is
                 named by its members' majority style label (only the name uses
                 the hand labels; membership is point in time), clusters with the
                 same name are merged; issuers ranked by summed member weight.
                 A quarter without such a cluster holds cash.
  c1_adds        frozen C1 model: issuers ranked by the expected number of seed
                 funds adding, sum of p(add) over the funds holding it.
  c2_als, c2_reentry  frozen C2 lists: issuers ranked by how many seed funds have
                 them in their top 20 new-buy list (ties: sum of 1/rank).
  ew_seed        benchmark: every issuer any seed fund holds.
"""

from collections import Counter

import polars as pl

from src.features.task_c import CLUSTER_ALGO, CLUSTER_METHOD
from src.models import style_classifier

N_NAMES = 20
TOP_K_CONVICTION = 10
MAX_POSITIONS = 500
LIST_TOP = 20
STOCK_CLASSES = ("common", "adr")


def load_books(con) -> pl.DataFrame:
    return con.execute(f"""
        WITH n AS (SELECT fund_id, quarter, count(*) AS n_positions FROM positions WHERE put_call = '' GROUP BY ALL)
        SELECT p.fund_id, p.quarter, p.known_at, p.cusip, p.weight_long,
               coalesce(m.issuer_cik, p.cusip) AS issuer, n.n_positions
        FROM positions p JOIN security_master m USING (cusip) JOIN n USING (fund_id, quarter)
        WHERE p.cohort = 'seed' AND p.put_call = '' AND p.weight_long > 0 AND m.security_class IN {STOCK_CLASSES}
    """).pl()


def representative(books: pl.DataFrame) -> pl.DataFrame:
    """quarter, issuer -> the CUSIP most seed funds hold."""
    return (books.group_by("quarter", "issuer", "cusip").agg(pl.col("fund_id").n_unique().alias("n"))
            .sort(["quarter", "issuer", "n", "cusip"], descending=[False, False, True, False])
            .group_by("quarter", "issuer").first().select("quarter", "issuer", "cusip"))


def visible(books: pl.DataFrame, quarter: str, cutoff) -> pl.DataFrame:
    return books.filter((pl.col("quarter") == quarter) & (pl.col("known_at") <= cutoff))


def to_weights(ranked: list[str], rep: dict[str, str], gone: set[str] = frozenset()) -> dict[str, float]:
    """Equal weights on the first N_NAMES issuers; `gone` = CUSIPs delisted on or before the
    trade date (public then: a 13F can list a stock that was taken over before the fund filed)."""
    top = [rep[i] for i in ranked if i in rep and rep[i] not in gone][:N_NAMES]
    return {c: 1 / len(top) for c in top} if top else {}


def consensus(b: pl.DataFrame) -> list[str]:
    top = (b.filter(pl.col("n_positions") <= MAX_POSITIONS)
           .with_columns(pl.col("weight_long").rank("ordinal", descending=True).over("fund_id").alias("rk"))
           .filter(pl.col("rk") <= TOP_K_CONVICTION))
    s = (top.group_by("issuer").agg(pl.col("fund_id").n_unique().alias("n"), pl.col("weight_long").mean().alias("w"))
         .sort(["n", "w", "issuer"], descending=[True, True, False]))
    return s["issuer"].to_list()


def ew(b: pl.DataFrame) -> list[str]:
    return sorted(b["issuer"].unique().to_list())


def cluster_names(con, cutoff, quarter: str) -> dict[str, str]:
    """fund_id -> style name of its cluster, from the latest clustering public by the cutoff."""
    row = con.execute(f"""
        SELECT max(quarter) FROM fund_clusters
        WHERE method = '{CLUSTER_METHOD}' AND algo = '{CLUSTER_ALGO}' AND known_at <= ? AND quarter <= ?
    """, [cutoff, quarter]).fetchone()
    if row[0] is None:
        return {}
    members = con.execute(f"""
        SELECT fund_id, cluster FROM fund_clusters
        WHERE method = '{CLUSTER_METHOD}' AND algo = '{CLUSTER_ALGO}' AND quarter = ? AND cluster >= 0
    """, [row[0]]).pl()
    styles = dict(style_classifier.labels().select("fund_id", "label_v2").iter_rows())
    out = {}
    for _, grp in members.group_by("cluster"):
        votes = Counter(styles.get(f) for f in grp["fund_id"] if styles.get(f))
        name = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        out |= {f: name for f in grp["fund_id"]}
    return out


def cluster_portfolios(b: pl.DataFrame, names: dict[str, str]) -> dict[str, list[str]]:
    d = b.with_columns(pl.col("fund_id").replace_strict(names, default=None).alias("style")).drop_nulls("style")
    out = {}
    for (style,), g in d.group_by("style"):
        s = g.group_by("issuer").agg(pl.col("weight_long").sum()).sort(["weight_long", "issuer"], descending=[True, False])
        out[style] = s["issuer"].to_list()
    return out


def c1_adds(scores: pl.DataFrame, b: pl.DataFrame) -> list[str]:
    """scores: backtest_c1_scores rows for this quarter, public by the cutoff."""
    s = (scores.join(b.select("fund_id", "cusip", "issuer"), on=["fund_id", "cusip"])
         .group_by("issuer").agg(pl.col("p_add").sum()).sort(["p_add", "issuer"], descending=[True, False]))
    return s["issuer"].to_list()


def c2_buys(lists: pl.DataFrame, issuer_of: dict[str, str]) -> list[str]:
    """lists: backtest_c2_lists rows (one method, this quarter, public by the cutoff)."""
    l = (lists.with_columns(pl.col("cusip").replace_strict(issuer_of, default=None).alias("issuer"))
         .drop_nulls("issuer")
         .with_columns(pl.col("rank").rank("ordinal").over("fund_id").alias("rk")).filter(pl.col("rk") <= LIST_TOP))
    s = (l.group_by("issuer").agg(pl.col("fund_id").n_unique().alias("n"), (1 / pl.col("rk")).sum().alias("inv"))
         .sort(["n", "inv", "issuer"], descending=[True, True, False]))
    return s["issuer"].to_list()
