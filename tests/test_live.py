"""Forward test (src/live): frozen models, forecast files, grading rules, scorecard."""

import gzip
import json
import re
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest

from src.live import frozen, grade, predict

D = date.fromisoformat


# --- frozen models -------------------------------------------------------------------------

def test_frozen_booster_matches_its_hash():
    m = frozen.manifest()
    booster = frozen.load_c1()
    assert booster.num_feature() == len(m["c1"]["features"]) == 52
    assert abs(sum(m["c1"]["class_prior_trainval"].values()) - 1) < 1e-4
    assert set(m["c2"]["methods"]) == {"als", "reentry"}


def test_tampered_booster_is_refused(tmp_path, monkeypatch):
    text = gzip.decompress(frozen.BOOSTER.read_bytes()).replace(b"num_leaves=", b"num_leaves =", 1)
    fake = tmp_path / "c1.txt.gz"
    fake.write_bytes(gzip.compress(text))
    monkeypatch.setattr(frozen, "BOOSTER", fake)
    with pytest.raises(SystemExit, match="SHA-256"):
        frozen.load_c1()


def test_live_code_never_fits_or_selects_a_model():
    banned = ["fit_final", "lgbm_fit", "lgb.train", "tune(", "GRIDS", "LogisticRegression", "main_select",
              "task_c1.main", "task_c2.main", "frozen.freeze"]
    for f in Path("src/live").glob("*.py"):
        if f.name == "frozen.py":   # freeze() only copies the already-fitted booster, once
            continue
        src = f.read_text()
        assert not [b for b in banned if b in src], f


# --- quarters and files --------------------------------------------------------------------

def test_quarter_arithmetic():
    assert predict.next_quarter("2026Q4") == "2027Q1" and predict.prev_quarter("2027Q1") == "2026Q4"
    assert predict.paths("2026Q3")["c1"].name == "2026Q3_c1.csv.gz"


def _write_forecast(out: Path, q: str = "2026Q1") -> None:
    c1 = pl.DataFrame({"quarter": [q] * 4, "fund_id": ["f", "f", "g", "h"], "cusip": ["A", "B", "A", "A"],
                       "symbol": ["A", "B", "A", "A"], "name": ["a", "b", "a", "a"],
                       "pred_date": [D("2026-05-10")] * 4, "weight_long": [0.5, 0.5, 1.0, 1.0],
                       "p_exit": [0.7, 0.1, 0.2, 0.25], "p_trim": [0.1, 0.1, 0.2, 0.25],
                       "p_hold": [0.1, 0.2, 0.4, 0.25], "p_add": [0.1, 0.6, 0.2, 0.25]}
                      | {c: [0.25] * 4 for c in ["prior_exit", "prior_trim", "prior_hold", "prior_add",
                                                 "last_exit", "last_trim", "last_hold", "last_add"]}
                      | {"prior_n": [10] * 4})
    c2 = pl.DataFrame({"quarter": [q] * 4, "method": ["als", "als", "reentry", "reentry"], "fund_id": ["f"] * 4,
                       "pred_date": [D("2026-05-10")] * 4, "rank": [1, 2, 1, 2], "cusip": ["X", "Y", "Y", "Z"],
                       "symbol": ["X", "Y", "Y", "Z"], "name": ["x", "y", "y", "z"]})
    p = predict.paths(q, out)
    out.mkdir(parents=True, exist_ok=True)
    for key, df in (("c1", c1), ("c2", c2)):
        p[key].write_bytes(gzip.compress(df.write_csv().encode(), mtime=0))
    p["manifest"].write_text(json.dumps({"forecast_quarter": q, "model_sha256": "x" * 64,
                                         "files": {p[k].name: predict.file_sha(p[k]) for k in ("c1", "c2")}}))


def test_forecast_is_never_overwritten(tmp_path):
    _write_forecast(tmp_path)
    with pytest.raises(SystemExit, match="never overwritten"):
        predict.write(None, "2026Q1", tmp_path)


def test_edited_forecast_is_refused(tmp_path):
    _write_forecast(tmp_path)
    p = predict.paths("2026Q1", tmp_path)["c1"]
    p.write_bytes(gzip.compress(gzip.decompress(p.read_bytes()).replace(b"0.7", b"0.9"), mtime=0))
    with pytest.raises(SystemExit, match="changed since"):
        predict.read("2026Q1", tmp_path)


# --- grading rules -------------------------------------------------------------------------

@pytest.fixture
def labels_db():
    con = duckdb.connect()
    con.execute("""CREATE TABLE position_moves AS SELECT * FROM (VALUES
        ('f', 'seed', 'A', '2026Q1', DATE '2026-08-14', 'exit', false),
        ('f', 'seed', 'B', '2026Q1', DATE '2026-08-14', 'add', false),
        ('g', 'seed', 'A', '2026Q1', DATE '2026-07-02', 'hold', false),   -- filed before the issue date
        ('h', 'seed', 'A', '2026Q1', DATE '2026-08-14', 'exit', true)     -- forced exit (deal)
        ) t(fund_id, cohort, cusip, quarter, known_at, action, forced_exit)""")
    con.execute("""CREATE TABLE new_buys AS SELECT * FROM (VALUES
        ('f', 'seed', '2026Q1', 'Y', DATE '2026-08-14')) t(fund_id, cohort, quarter, cusip, label_known_at)""")
    yield con
    con.close()


def test_grading_skips_early_filers_and_forced_exits(tmp_path, labels_db):
    _write_forecast(tmp_path)
    rows = grade.grade(labels_db, "2026Q1", tmp_path, issued=D("2026-07-15"))
    c1 = rows.filter((pl.col("task") == "c1") & (pl.col("model") == "lgbm_full") & (pl.col("subset") == "all")).row(0, named=True)
    assert c1["rows"] == 2 and c1["funds"] == 1                  # only f: g filed early, h's exit was forced
    assert c1["funds_filed_before_issue"] == 1
    assert c1["accuracy"] == 1.0                                 # f: exit (p 0.7) and add (p 0.6) both right
    c2 = rows.filter((pl.col("task") == "c2") & (pl.col("subset") == "all"))
    by = dict(zip(c2["model"], c2["recall_at_20"]))
    assert by == {"als": 1.0, "reentry": 1.0}                    # Y is on both lists
    assert set(rows["model"]) == {"lgbm_full", "fund_last_q", "fund_prior", "class_prior", "als", "reentry"}


def test_uncommitted_forecast_is_not_graded(tmp_path, labels_db):
    _write_forecast(tmp_path)
    with pytest.raises(SystemExit, match="never committed"):
        grade.grade(labels_db, "2026Q1", tmp_path)              # tmp_path is outside git: no issue date


def test_scorecard_is_append_only(tmp_path, labels_db):
    _write_forecast(tmp_path)
    rows = grade.grade(labels_db, "2026Q1", tmp_path, issued=D("2026-07-15"))
    path = tmp_path / "scorecard.csv"
    grade.append(rows, path)
    with pytest.raises(SystemExit, match="append-only"):
        grade.append(rows, path)
    assert len(pl.read_csv(path)) == len(rows)


# --- what is committed ---------------------------------------------------------------------

def test_committed_forecasts_match_their_hashes():
    for m in Path("predictions").glob("*Q?.json"):
        q = json.loads(m.read_text())["forecast_quarter"]
        c1, c2, manifest = predict.read(q)                       # raises if a file changed
        assert manifest["model_sha256"] == frozen.manifest()["c1"]["sha256"]
        assert c1["fund_id"].n_unique() == manifest["data"]["c1_funds"]
        probs = c1.select([f"p_{c}" for c in ("exit", "trim", "hold", "add")]).to_numpy()
        assert np.allclose(probs.sum(axis=1), 1, atol=1e-6)
        assert c2.group_by("method", "fund_id").agg(pl.col("rank").max())["rank"].max() <= 20


def test_scorecard_rows_unique():
    path = Path("predictions/scorecard.csv")
    if not path.exists():
        pytest.skip("nothing graded yet")
    sc = pl.read_csv(path)
    assert sc.select("forecast_quarter", "task", "model", "subset").is_duplicated().sum() == 0


def test_live_page_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string("import sys; sys.path.insert(0, 'app'); import streamlit_app as a; a.live_page()",
                             default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Live track record"


def test_filing_deadline_rolls_weekends():
    import sys
    sys.path.insert(0, "app")
    import streamlit_app as a
    assert a.filing_deadline("2026Q3") == "2026-11-16"   # 45 days after Sep 30 is Saturday Nov 14
    assert a.filing_deadline("2026Q2") == "2026-08-14"
    assert a.filing_deadline("2026Q4") == "2027-02-15"   # Feb 14, 2027 is a Sunday
