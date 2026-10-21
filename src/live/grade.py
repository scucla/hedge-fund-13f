"""Grade a committed forecast against the filings that came after it.

A forecast for quarter q (predictions/{q}*) is graded when the q+1 filings are
in the database, with Milestone 6's definitions and metric code:
  C1  exit / trim / hold / add from position_moves (split-adjusted, 10% band);
      forced exits (acquisitions, performance delistings) and funds that didn't
      file q+1 are dropped. Model vs the pre-registered baselines saved in the
      forecast (fund's last-quarter and historical action shares; the train+val
      class shares as their fallback, from the frozen manifest).
  C2  new buys at q+1 (stock universe), recall@20, NDCG@20, hit rate per
      fund-quarter with at least one new buy; ALS vs the re-entry baseline.

**Eligibility:** the forecast's issue date is the day its manifest was first
committed to git. A fund whose q+1 filing is dated on or before that day is not
graded (its filing could have been seen), and the count is reported.

Rows are appended to predictions/scorecard.csv, which is append-only: a
forecast that already has rows is never graded again.
"""

import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from src.live import frozen, predict
from src.models import style_classifier
from src.models.task_c2 import rank_metrics
from src.models.task_c_baselines import baseline_proba
from src.models.task_c_common import CLASSES, MULTI, c1_metrics

SCORECARD = predict.PRED_DIR / "scorecard.csv"
C1_METRICS = ["macro_f1", "log_loss", "accuracy", "exit_auc", "R_exit", "R_trim", "R_hold", "R_add"]
C2_METRICS = ["recall_at_20", "ndcg_at_20", "hit_rate"]
BASELINES = {"fund_last_q": "baseline: fund's last-quarter action shares",
             "fund_prior": "baseline: fund's historical action shares",
             "class_prior": "baseline: 2018-2022 class shares"}


def issued_on(manifest_path: Path) -> date | None:
    """Date of the first commit that added the forecast's manifest (None if never committed)."""
    res = subprocess.run(["git", "log", "--diff-filter=A", "--format=%cs", "--", str(manifest_path)],
                         capture_output=True, text=True)
    out = res.stdout.split() if res.returncode == 0 else []   # outside the repo: not committed
    return date.fromisoformat(out[-1]) if out else None


def _styles() -> pl.DataFrame:
    return style_classifier.labels().select("fund_id", pl.col("label_v2").alias("style"))


def grade_c1(con, c1: pl.DataFrame, q: str, issued: date) -> tuple[list[dict], dict]:
    labels = con.execute("""SELECT fund_id, cusip, action, forced_exit, known_at AS next_filed
                            FROM position_moves WHERE cohort = 'seed' AND quarter = ?""", [q]).pl()
    d = c1.join(labels, on=["fund_id", "cusip"], how="left").join(_styles(), on="fund_id", how="left")
    early = d.filter(pl.col("next_filed") <= issued)["fund_id"].unique()
    info = {"funds_forecast": c1["fund_id"].n_unique(),
            "funds_not_filed_yet": d.filter(pl.col("next_filed").is_null())["fund_id"].n_unique(),
            "funds_filed_before_issue": len(early)}
    d = d.filter((pl.col("next_filed") > issued) & pl.col("action").is_not_null() & ~pl.col("forced_exit"))
    d = d.with_columns(pl.col("action").replace_strict({c: i for i, c in enumerate(CLASSES)}, return_dtype=pl.Int32).alias("y"))
    prior = np.array([frozen.manifest()["c1"]["class_prior_trainval"][c] for c in CLASSES])
    rows = []
    for subset, sub in (("all", d), ("excl_multi_quant", d.filter(~pl.col("style").is_in(MULTI)))):
        if sub.is_empty():
            continue
        y = sub["y"].to_numpy()
        models = {"lgbm_full": sub.select([f"p_{c}" for c in CLASSES]).to_numpy()}
        models |= {b: baseline_proba(sub, b, prior) for b in BASELINES}
        for model, proba in models.items():
            m = c1_metrics(y, proba)
            rows.append({"task": "c1", "model": model, "subset": subset, "funds": sub["fund_id"].n_unique(),
                         "rows": len(sub)} | {k: m[k] for k in C1_METRICS})
    return rows, info


def grade_c2(con, c2: pl.DataFrame, q: str, issued: date) -> list[dict]:
    buys = con.execute("""SELECT fund_id, cusip, label_known_at AS next_filed FROM new_buys
                          WHERE cohort = 'seed' AND quarter = ?""", [q]).pl()
    buys = buys.filter(pl.col("next_filed") > issued).join(_styles(), on="fund_id", how="left")
    rows = []
    for method, lists in c2.group_by("method"):
        per = []
        for (fund,), tg in buys.group_by("fund_id"):
            ranked = lists.filter(pl.col("fund_id") == fund).sort("rank")["cusip"].to_list()
            per.append({"fund_id": fund, "style": tg["style"][0]} | rank_metrics(ranked, set(tg["cusip"])))
        per = pl.DataFrame(per)
        for subset, sub in (("all", per), ("excl_multi_quant", per.filter(~pl.col("style").is_in(MULTI)))):
            if sub.is_empty():
                continue
            rows.append({"task": "c2", "model": method[0], "subset": subset, "funds": len(sub), "rows": len(sub),
                         "recall_at_20": sub["recall"].mean(), "ndcg_at_20": sub["ndcg"].mean(),
                         "hit_rate": sub["hit"].mean()})
    return rows


def grade(con, q: str, out_dir: Path = predict.PRED_DIR, issued: date | None = None) -> pl.DataFrame:
    """Scorecard rows for the forecast of quarter q, graded against the q+1 filings."""
    c1, c2, manifest = predict.read(q, out_dir)
    issued = issued or issued_on(predict.paths(q, out_dir)["manifest"])
    if issued is None:
        raise SystemExit(f"the {q} forecast was never committed to git: its issue date can't be proved, so it isn't graded")
    c1_rows, info = grade_c1(con, c1, q, issued)
    rows = c1_rows + grade_c2(con, c2, q, issued)
    meta = {"forecast_quarter": q, "graded_with": predict.next_quarter(q), "issued_on": str(issued),
            "graded_on": str(date.today()), "model_sha256": manifest["model_sha256"][:12]} | info
    cols = list(meta) + ["task", "model", "subset", "funds", "rows"] + C1_METRICS + C2_METRICS
    # C1 and C2 rows carry different metrics: every row gets every column
    return pl.DataFrame([{c: (meta | r).get(c) for c in cols} for r in rows],
                        schema_overrides={c: pl.Float64 for c in C1_METRICS + C2_METRICS})


def append(rows: pl.DataFrame, path: Path = SCORECARD) -> None:
    if path.exists():
        old = pl.read_csv(path, infer_schema_length=0)
        q = rows["forecast_quarter"][0]
        if q in old["forecast_quarter"].to_list():
            raise SystemExit(f"{path} already has grades for {q}: the scorecard is append-only")
        text = path.read_text().rstrip("\n") + "\n" + rows.write_csv(include_header=False)
    else:
        text = rows.write_csv()
    path.write_text(text)
