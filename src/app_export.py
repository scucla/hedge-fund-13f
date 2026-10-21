"""Export the small parquet snapshot the Streamlit app reads (app/data/, committed).

Usage (after the Milestone 4-7 steps):
    uv run python -m src.app_export

data/ is gitignored, so the deployed app can't read the DuckDB database. This
writes only what the app shows, zstd-compressed, and fails if the total goes
over SIZE_BUDGET_MB:

  funds.parquet         fund_id, fund_name, cohort, hand style label, quarters filed
  fingerprint.parquet   fund_fingerprint (every fund-quarter) + out-of-fund style probabilities
  embeddings.parquet    fund2vec vectors (seed + holdout) for nearest neighbours
  map.parquet           UMAP coordinates per quarter (fitted on seed funds, the rest
                        projected, as in figure 12) and the HDBSCAN cluster
  books.parquet         from BOOKS_FIRST: each fund's top TOP_N long positions per
                        quarter by weight (weights are on the *full* book), share
                        change, realised next-quarter action, frozen C1 probabilities
  newbuys.parquet       frozen C2 top-20 lists (ALS, re-entry) and whether the fund
                        actually bought the name next quarter
  backtest_daily.parquet, backtest_results.parquet   Milestone 7 outputs
  meta.json             export date, the caps, data notes

No price series is exported (Yahoo data can't be redistributed); returns
appear only as derived backtest series.
"""

import json
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.backtest import signals
from src.features.task_c import CLUSTER_ALGO, CLUSTER_METHOD
from src.models import clustering, style_classifier
from src.models.task_c1 import OUT as C1_OUT, VARIANTS, X_of, features
from src.models.task_c_common import CLASSES
from src.securities import db

OUT = Path("app/data")
TOP_N = 200
BOOKS_FIRST = "2023Q1"      # the test years: predictions next to what happened
MAP_FIRST = "2016Q1"        # first quarter with clusters
SIZE_BUDGET_MB = 20
FINGERPRINT_COLS = ["fund_id", "quarter", "known_at", "n_positions", "top10_weight", "hhi", "active_share",
                    "turnover", "holding_period_q", "tilt_size", "tilt_value", "tilt_momentum",
                    "tilt_profitability", "tilt_investment", "tilt_beta"]


def write(df: pl.DataFrame, name: str) -> None:
    df.write_parquet(OUT / f"{name}.parquet", compression="zstd", compression_level=19)


def funds(con) -> pl.DataFrame:
    labels = style_classifier.labels().select("fund_id", pl.col("label_v2").alias("style"))
    return (con.execute("""SELECT fund_id, any_value(fund_name) AS fund_name, any_value(cohort) AS cohort,
                                  min(quarter) AS first_quarter, max(quarter) AS last_quarter, count(*) AS quarters
                           FROM fund_quarter GROUP BY fund_id""").pl()
            .join(labels, on="fund_id", how="left").sort("fund_name"))


def fingerprint(con) -> pl.DataFrame:
    # fund_fingerprint carries four tilts; investment and beta come from fund_quarter
    fp = con.execute("""SELECT f.*, q.tilt_investment, q.tilt_beta FROM fund_fingerprint f
                        JOIN fund_quarter q USING (fund_id, quarter)""").pl()
    cols = FINGERPRINT_COLS   # a missing column fails here rather than vanishing from the app
    probs = con.execute("SELECT fund_id, quarter, predicted, COLUMNS('^p_') FROM style_drift").pl()
    return fp.select(cols).join(probs, on=["fund_id", "quarter"], how="left").sort("fund_id", "quarter")


def embeddings(con) -> pl.DataFrame:
    return con.execute(f"""SELECT fund_id, quarter, known_at, vec FROM fund_embeddings
                           WHERE method = '{CLUSTER_METHOD}' AND cohort IN ('seed', 'holdout')
                             AND quarter >= '{MAP_FIRST}'""").pl().with_columns(pl.col("vec").cast(pl.List(pl.Float32)))


def umap_map(con) -> pl.DataFrame:
    """Per quarter: UMAP fitted on seed fund2vec vectors, holdout and background projected
    (figure 12's layout; UMAP is refitted per quarter, so axes are not comparable across quarters)."""
    # projected books mostly outside the seed vocabulary are hidden, as in figure 12
    emb = con.execute(f"""SELECT fund_id, cohort, quarter, vec FROM fund_embeddings
                          WHERE method = '{CLUSTER_METHOD}' AND quarter >= '{MAP_FIRST}'
                            AND (cohort = 'seed' OR coverage >= 0.5)""").pl()
    cl = con.execute(f"""SELECT fund_id, quarter, cluster FROM fund_clusters
                         WHERE method = '{CLUSTER_METHOD}' AND algo = '{CLUSTER_ALGO}'""").pl()
    out = []
    for (q,), g in emb.group_by("quarter"):
        seed, other = g.filter(pl.col("cohort") == "seed"), g.filter(pl.col("cohort") != "seed")
        if len(seed) < 15:
            continue
        xs, xo = np.vstack(seed["vec"].to_list()), (np.vstack(other["vec"].to_list()) if len(other) else np.zeros((0, len(seed["vec"][0]))))
        ys, yo = clustering.umap_layout(xs, xo)
        out.append(pl.concat([seed.select("fund_id", "cohort", "quarter"), other.select("fund_id", "cohort", "quarter")])
                   .with_columns(pl.Series("x", np.concatenate([ys[:, 0], yo[:, 0]]) if len(yo) else ys[:, 0]).cast(pl.Float32),
                                 pl.Series("y", np.concatenate([ys[:, 1], yo[:, 1]]) if len(yo) else ys[:, 1]).cast(pl.Float32)))
    return pl.concat(out).join(cl, on=["fund_id", "quarter"], how="left").sort("quarter", "fund_id")


def c1_probabilities(con) -> pl.DataFrame:
    """Seed: the backtest's frozen scores. Holdout: the same saved booster on c1_all."""
    seed = con.execute("SELECT fund_id, quarter, cusip, COLUMNS('^p_') FROM backtest_c1_scores").pl()
    model = lgb.Booster(model_file=f"{C1_OUT}_{signals.C1_VARIANT}.txt")
    cols = features(VARIANTS[signals.C1_VARIANT][1])
    h = con.execute(f"SELECT * FROM c1_all WHERE cohort = 'holdout' AND quarter >= '{BOOKS_FIRST}'").pl()
    p = model.predict(X_of(h, cols))
    hold = h.select("fund_id", "quarter", "cusip").with_columns(*[pl.Series(f"p_{c}", p[:, i]) for i, c in enumerate(CLASSES)])
    return pl.concat([seed, hold])


def books(con) -> pl.DataFrame:
    b = con.execute(f"""
        SELECT p.fund_id, p.quarter, p.cusip, coalesce(p.symbol, '') AS symbol, p.name_of_issuer AS name,
               m.security_class, p.value_usd, p.weight_long,
               row_number() OVER (PARTITION BY p.fund_id, p.quarter ORDER BY p.weight_long DESC, p.cusip) AS rank_long,
               p.share_change, p.is_new, p.age_q, p.known_at,
               mv.action AS next_action, mv.forced_exit AS next_forced_exit
        FROM positions p LEFT JOIN security_master m USING (cusip)
        LEFT JOIN position_moves mv ON mv.fund_id = p.fund_id AND mv.cusip = p.cusip AND mv.quarter = p.quarter
        WHERE p.put_call = '' AND p.quarter >= '{BOOKS_FIRST}' AND p.weight_long > 0
        QUALIFY rank_long <= {TOP_N}
    """).pl()
    return (b.join(c1_probabilities(con), on=["fund_id", "quarter", "cusip"], how="left")
            .with_columns(pl.col(pl.Float64).cast(pl.Float32)).sort("fund_id", "quarter", "rank_long"))


def newbuys(con) -> pl.DataFrame:
    lists = pl.concat([con.execute("SELECT * FROM backtest_c2_lists").pl(), signals.c2_lists(con, "holdout")])
    names = con.execute("SELECT cusip, coalesce(symbol, '') AS symbol, name_of_issuer AS name, security_class FROM security_master").pl()
    bought = con.execute(f"SELECT fund_id, quarter, cusip, true AS bought_next_q FROM new_buys WHERE quarter >= '{BOOKS_FIRST}'").pl()
    return (lists.filter(pl.col("rank") <= 20).join(names, on="cusip", how="left")
            .join(bought, on=["fund_id", "quarter", "cusip"], how="left")
            .with_columns(pl.col("bought_next_q").fill_null(False)).sort("method", "fund_id", "quarter", "rank"))


def size_mb() -> float:
    return sum(f.stat().st_size for f in OUT.glob("*.parquet")) / 2**20


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with db.connect(read_only=True) as con:
        for name, fn in (("funds", funds), ("fingerprint", fingerprint), ("embeddings", embeddings), ("map", umap_map),
                         ("books", books), ("newbuys", newbuys)):
            write(fn(con), name)
            print(f"{name}: {(OUT / f'{name}.parquet').stat().st_size / 2**20:.2f} MB", flush=True)
    for name in ("backtest_daily", "backtest_results"):
        write(pl.read_parquet(f"data/processed/{name}.parquet"), name)
    meta = {"exported": str(date.today()), "top_n_positions": TOP_N, "books_first_quarter": BOOKS_FIRST,
            "map_first_quarter": MAP_FIRST, "embedding_method": CLUSTER_METHOD,
            "notes": ["Positions capped at each fund's top 200 by weight per quarter; weights are on the full long book.",
                      "C1/C2 predictions are from the models frozen in Milestone 6 (trained on 2018-2022).",
                      "Holdout funds were never clustered, so their C1 cluster features are empty (weaker predictions)."]}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    total = size_mb()
    print(f"total {total:.1f} MB")
    if total > SIZE_BUDGET_MB:
        raise SystemExit(f"snapshot is {total:.1f} MB, over the {SIZE_BUDGET_MB} MB budget")


if __name__ == "__main__":
    main()
