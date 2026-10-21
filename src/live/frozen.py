"""The frozen Task C models used by the forward test, committed under models/frozen/.

Usage (once; refuses to overwrite):
    uv run python -m src.live.frozen

Writes
  models/frozen/c1_lgbm_full.txt.gz   the LightGBM booster refitted on the embargoed
                                      2018-2022 frame with the 2022-chosen settings
                                      (Milestone 6/7; the same booster the backtest used)
  models/frozen/manifest.json         its SHA-256 (of the uncompressed text), the feature
                                      list, the train+val class shares (the baselines'
                                      fallback), the ALS settings chosen on 2022, and the
                                      commits the models come from

**Never retuned.** The forward test loads these and checks the hash before every
run; nothing in src/live can fit or select a model. ALS is refitted at each
prediction date on the books public then, with these fixed settings and seed:
that is the frozen procedure, not retuning (as in the backtest).
"""

import ast
import gzip
import hashlib
import json
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

FROZEN = Path("models/frozen")
BOOSTER = FROZEN / "c1_lgbm_full.txt.gz"
MANIFEST = FROZEN / "manifest.json"
SOURCE_COMMITS = {"milestone_6_models": "5e3637f", "milestone_7_booster": "dd831d1"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def load_c1() -> lgb.Booster:
    """The frozen booster, after checking it is byte-for-byte the one recorded."""
    text = gzip.decompress(BOOSTER.read_bytes())
    expected = manifest()["c1"]["sha256"]
    if sha256(text) != expected:
        raise SystemExit(f"{BOOSTER} does not match its recorded SHA-256 ({expected[:12]}…): refusing to score")
    return lgb.Booster(model_str=text.decode())


def freeze() -> None:
    from src.backtest.signals import C1_VARIANT
    from src.models import task_c2
    from src.models.task_c1 import OUT as C1_OUT, VARIANTS, features
    from src.models.task_c_common import CLASSES, frames, load_c1 as load_panel
    from src.securities import db

    if MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} exists: frozen models are never replaced")
    text = Path(f"{C1_OUT}_{C1_VARIANT}.txt").read_bytes()
    choices = json.loads(Path(f"{C1_OUT}_choices.json").read_text())
    res = pl.read_parquet(task_c2.RESULTS_PATH)
    als = ast.literal_eval(res.filter((pl.col("method") == "als") & (pl.col("split") == "test"))["params"][0])
    with db.connect(read_only=True) as con:
        trainval = frames(load_panel(con))["trainval"]
    prior = np.bincount(trainval["y"].to_numpy(), minlength=len(CLASSES)) / len(trainval)
    booster = lgb.Booster(model_str=text.decode())
    cols = features(VARIANTS[C1_VARIANT][1])
    assert booster.num_feature() == len(cols), "booster and feature list disagree"

    FROZEN.mkdir(parents=True, exist_ok=True)
    BOOSTER.write_bytes(gzip.compress(text, compresslevel=9, mtime=0))
    MANIFEST.write_text(json.dumps({
        "frozen_on": str(date.today()),
        "source_commits": SOURCE_COMMITS,
        "c1": {"variant": C1_VARIANT, "sha256": sha256(text), "params": choices[C1_VARIANT]["params"],
               "features": cols, "classes": CLASSES,
               "class_prior_trainval": dict(zip(CLASSES, prior.round(6).tolist())),
               "fit_frame": "seed funds, 2018Q1-2022Q4 rows, embargoed against the first 2023 prediction date"},
        "c2": {"methods": {"als": als, "reentry": {}}, "top_k": task_c2.K, "random_state": 0, "iterations": 15,
               "tuned_on": "2022 seed fund-quarters, mean NDCG@20"},
    }, indent=2) + "\n")
    print(f"wrote {BOOSTER} ({BOOSTER.stat().st_size / 2**20:.1f} MB) and {MANIFEST}")


if __name__ == "__main__":
    freeze()
