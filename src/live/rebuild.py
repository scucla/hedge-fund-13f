"""Extend the pipeline to a newly filed quarter: only the steps the forward test needs.

The steps run in order, each the same build function the milestones use.
`network` steps download (cached under data/raw/); `--offline` skips them, for a
rehearsal on a copy of the database. Not run: the style classifier, factor
regressions, crowding, the backtest, deal-document fetching, and every report
or figure (no file under docs/ changes).

Ingestion uses src.ingest.backfill, never src.ingest.load: load rebuilds
13f.duckdb from the raw data sets, which are deleted after loading, so it would
drop the four windows that database holds. backfill appends one data set's
seed (and holdout) rows to the history databases and is resumable.

**History is append-only for the clustering tables.** Embeddings and clusters
are recomputed for every quarter (each quarter's fit depends only on books filed
by its end + 60 days, so earlier quarters should come out the same); the rows of
quarters before the new one are then restored from the snapshot taken first, so a
forecast's inputs never change after the fact. The run reports how many earlier
cluster assignments differed in the recomputation.

Stale caches expired first (each is a superset when re-downloaded): XBRL
companyfacts and the Fama-French files cached before the new quarter ended, and
the SEC ticker file. The EDGAR index already re-fetches the current quarter, and
OpenFIGI only looks up new CUSIPs.
"""

import time
from datetime import date, datetime
from pathlib import Path

import polars as pl

from src.securities import db

APPEND_ONLY = ("fund_embeddings", "fund_clusters")


def quarter_end(q: str) -> date:
    from src.backtest.engine import quarter_end as qe
    return qe(q)


def expire_caches(q: str) -> None:
    from src.securities.cusip_map import TICKERS_PATH
    from src.securities.factors import RAW_DIR as FRENCH_DIR
    from src.securities.fundamentals import FACTS_DIR
    cutoff = datetime.combine(quarter_end(q), datetime.min.time()).timestamp()
    stale = [p for d in (FACTS_DIR, FRENCH_DIR) if d.exists() for p in d.iterdir() if p.stat().st_mtime < cutoff]
    if TICKERS_PATH.exists() and TICKERS_PATH.stat().st_mtime < cutoff:
        stale.append(TICKERS_PATH)
    for p in stale:
        p.unlink()
    print(f"  expired {len(stale):,} cached files older than {quarter_end(q)}")


def ingest(q: str) -> None:
    from src.ingest import backfill
    backfill.build(q, q)
    backfill.build(q, q, holdout=True)


def with_con(fn):
    def run(q: str) -> None:
        with db.connect() as con:
            fn(con)
    return run


def snapshot(con, q: str) -> dict[str, pl.DataFrame]:
    return {t: con.execute(f"SELECT * FROM {t} WHERE quarter < ?", [q]).pl() for t in APPEND_ONLY}


def embeddings_and_clusters(q: str) -> None:
    from src.features import embeddings
    from src.models import clustering
    with db.connect() as con:
        before = snapshot(con, q)
    embeddings.main()
    with db.connect() as con:
        con.execute("SET enable_progress_bar = false")
        clustering.build_tables(con)
        new = con.execute("SELECT method, algo, fund_id, quarter, cluster FROM fund_clusters WHERE quarter < ?", [q]).pl()
        old = before["fund_clusters"].select("method", "algo", "fund_id", "quarter", "cluster")
        changed = old.join(new, on=["method", "algo", "fund_id", "quarter"], how="full", suffix="_new").filter(
            pl.col("cluster").ne_missing(pl.col("cluster_new")))
        print(f"  {len(changed)} of {len(old)} earlier cluster assignments differ in the recomputation; "
              f"earlier quarters restored from the snapshot")
        for t, df in before.items():
            con.execute(f"DELETE FROM {t} WHERE quarter < ?", [q])
            con.register("snap", df.to_arrow())
            con.execute(f"INSERT INTO {t} BY NAME SELECT * FROM snap")
            con.unregister("snap")


def _steps():
    from src.features import books, fund_quarter, position_moves, positions, task_c
    from src.securities import (cusip_map, delistings, factors, fundamentals, implied_prices, labels, prices,
                                sec_events, stock_quarter, universe)
    return [
        ("ingest the quarter's 13F data set", True, ingest),
        ("expire stale caches", True, expire_caches),
        ("universe", False, lambda q: universe.build()),
        ("CUSIP map", True, lambda q: cusip_map.build()),
        ("prices (incremental)", True, lambda q: prices.build()),
        ("fundamentals", True, lambda q: fundamentals.build()),
        ("Fama-French factors", True, lambda q: factors.build()),
        ("stock characteristics", False, with_con(stock_quarter.build)),
        ("fund books", False, with_con(books.build)),
        ("positions (and detected splits)", False, with_con(positions.build)),
        ("fund-quarter features", False, with_con(fund_quarter.build)),
        # same order as the README's pipeline: implied prices use positions' splits;
        # labels need implied prices and delistings; position moves need both split tables and delistings
        ("13F-implied prices", False, with_con(implied_prices.build)),
        ("SEC events", True, with_con(sec_events.build)),
        ("delistings", False, with_con(lambda con: (delistings.exits(con), delistings.classify(con, fetch=False)))),
        ("labels", False, with_con(labels.fill)),
        ("position moves", False, with_con(position_moves.build)),
        ("embeddings and clusters (append-only)", False, embeddings_and_clusters),
        ("Task C panel", False, with_con(task_c.build)),
    ]


def run(q: str, offline: bool = False) -> None:
    if db.LAST_QUARTER != q:
        raise SystemExit(f"LAST_QUARTER is {db.LAST_QUARTER}, not {q}: run through src.live.quarter")
    for name, network, fn in _steps():
        if offline and network:
            print(f"- {name}: skipped (offline)")
            continue
        t0 = time.time()
        print(f"- {name} ...", flush=True)
        fn(q)
        print(f"  done in {time.time() - t0:.0f} s", flush=True)


def has_quarter(q: str) -> bool:
    """Is quarter q already through the pipeline (seed positions in the Task C panel)?"""
    with db.connect(read_only=True) as con:
        try:
            return con.execute("SELECT count(*) FROM c1_all WHERE cohort = 'seed' AND quarter = ?", [q]).fetchone()[0] > 0
        except Exception:
            return False
