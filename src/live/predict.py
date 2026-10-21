"""Forward-test predictions for one quarter's filings, from the frozen models.

For quarter q (the seed funds' quarter-q 13F filings), writes to predictions/:
  {q}_c1.csv.gz   every seed fund's stock-universe long position at q: the frozen C1
                  model's p(exit / trim / hold / add) by the q+1 filing, plus the two
                  baselines' inputs as they stood on the fund's filing date (its
                  historical and last-quarter action shares), so the baselines are
                  pre-registered in the same file
  {q}_c2.csv.gz   each seed fund's top 20 new-buy candidates from ALS and from the
                  re-entry baseline
  {q}.json        what was predicted, from which data (latest filing used), with which
                  model (SHA-256) and code commit, and every file's SHA-256

Rows come from c1_all (every position, labelled or not) and every seed fund that
filed q: nothing about q+1 is known or used. Same feature code as Milestone 6;
the frozen booster is checked against its recorded hash before scoring.
"""

import gzip
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.live import frozen

PRED_DIR = Path("predictions")
C1_KEEP = ["quarter", "fund_id", "cusip", "symbol", "pred_date", "weight_long"]
BASELINE_COLS = ["prior_exit", "prior_trim", "prior_hold", "prior_add", "prior_n",
                 "last_exit", "last_trim", "last_hold", "last_add"]


def next_quarter(q: str) -> str:
    y, n = int(q[:4]), int(q[-1])
    return f"{y + 1}Q1" if n == 4 else f"{y}Q{n + 1}"


def prev_quarter(q: str) -> str:
    y, n = int(q[:4]), int(q[-1])
    return f"{y - 1}Q4" if n == 1 else f"{y}Q{n - 1}"


def paths(q: str, out_dir: Path = PRED_DIR) -> dict[str, Path]:
    return {"c1": out_dir / f"{q}_c1.csv.gz", "c2": out_dir / f"{q}_c2.csv.gz", "manifest": out_dir / f"{q}.json"}


def c1_predictions(con, q: str) -> pl.DataFrame:
    m = frozen.manifest()["c1"]
    booster = frozen.load_c1()
    df = con.execute("SELECT * FROM c1_all WHERE cohort = 'seed' AND quarter = ?", [q]).pl()
    if df.is_empty():
        raise SystemExit(f"no seed positions for {q} in c1_all: rebuild first")
    X = df.select([pl.col(c).cast(pl.Float64) for c in m["features"]]).to_numpy()
    p = booster.predict(X)
    names = con.execute("SELECT cusip, name_of_issuer AS name FROM security_master").pl()
    return (df.select(C1_KEEP + BASELINE_COLS)
            .with_columns(*[pl.Series(f"p_{c}", p[:, i]) for i, c in enumerate(m["classes"])])
            .join(names, on="cusip", how="left")
            .select(C1_KEEP[:4] + ["name"] + C1_KEEP[4:] + [f"p_{c}" for c in m["classes"]] + BASELINE_COLS)
            .sort("fund_id", "weight_long", descending=[False, True]))


def c2_predictions(con, q: str) -> pl.DataFrame:
    from src.backtest import signals
    c2 = frozen.manifest()["c2"]
    lists = signals.c2_lists(con, "seed", quarter=q, methods=c2["methods"], depth=c2["top_k"])
    names = con.execute("SELECT cusip, symbol, name_of_issuer AS name FROM security_master").pl()
    return (lists.join(names, on="cusip", how="left")
            .select("quarter", "method", "fund_id", "pred_date", "rank", "cusip", "symbol", "name")
            .sort("method", "fund_id", "rank"))


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def write(con, q: str, out_dir: Path = PRED_DIR) -> dict:
    p = paths(q, out_dir)
    if any(x.exists() for x in p.values()):
        raise SystemExit(f"a forecast for {q} already exists in {out_dir}: forecasts are never overwritten")
    c1, c2 = c1_predictions(con, q), c2_predictions(con, q)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, df in (("c1", c1), ("c2", c2)):
        # gzip with no timestamp in the header, so the file's hash depends only on its contents
        p[key].write_bytes(gzip.compress(df.write_csv().encode(), compresslevel=9, mtime=0))
    funds = con.execute("""SELECT count(DISTINCT fund_id), max(known_at) FROM fund_holdings
                           WHERE cohort = 'seed' AND quarter = ?""", [q]).fetchone()
    manifest = {
        "forecast_quarter": q,
        "predicts": f"moves revealed by the seed funds' {next_quarter(q)} 13F filings",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": git_head(),
        "model_sha256": frozen.manifest()["c1"]["sha256"],
        "c2_settings": frozen.manifest()["c2"]["methods"],
        "data": {"seed_funds_filed": funds[0], "latest_filing_used": str(funds[1]),
                 "c1_positions": len(c1), "c1_funds": c1["fund_id"].n_unique(),
                 "c2_funds": c2["fund_id"].n_unique()},
        "files": {x.name: file_sha(x) for x in (p["c1"], p["c2"])},
        "grading_rule": ("a fund is graded only if its next filing is dated strictly after the day this file was "
                         "first committed to git; forced exits (acquisitions, performance delistings) are dropped"),
    }
    p["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read(q: str, out_dir: Path = PRED_DIR) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """A saved forecast, after checking its files against the manifest's hashes."""
    p = paths(q, out_dir)
    manifest = json.loads(p["manifest"].read_text())
    for key in ("c1", "c2"):
        if file_sha(p[key]) != manifest["files"][p[key].name]:
            raise SystemExit(f"{p[key]} changed since the forecast was written: refusing to grade it")
    c1 = pl.read_csv(p["c1"], schema_overrides={"cusip": pl.Utf8, "symbol": pl.Utf8}, try_parse_dates=True)
    c2 = pl.read_csv(p["c2"], schema_overrides={"cusip": pl.Utf8, "symbol": pl.Utf8}, try_parse_dates=True)
    return c1, c2, manifest
