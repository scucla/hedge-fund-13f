"""Milestone 8: the committed app snapshot (app/data) and the Streamlit app."""

import json
from pathlib import Path

import polars as pl
import pytest

from src import app_export

DATA = Path("app/data")
pytestmark = pytest.mark.skipif(not (DATA / "meta.json").exists(), reason="run src.app_export first")


def load(name: str) -> pl.DataFrame:
    return pl.read_parquet(DATA / f"{name}.parquet")


def test_snapshot_is_under_budget():
    assert app_export.size_mb() <= app_export.SIZE_BUDGET_MB


@pytest.mark.parametrize("name, keys", [
    ("funds", ["fund_id"]),
    ("fingerprint", ["fund_id", "quarter"]),
    ("embeddings", ["fund_id", "quarter"]),
    ("map", ["fund_id", "quarter"]),
    ("books", ["fund_id", "quarter", "cusip"]),
    ("newbuys", ["method", "fund_id", "quarter", "cusip"]),
    ("backtest_daily", ["strategy", "date"]),
])
def test_unique_keys_and_no_nulls(name, keys):
    df = load(name)
    assert len(df) > 0
    assert df.select(keys).is_duplicated().sum() == 0
    assert df.select(pl.any_horizontal(pl.col(keys).is_null())).to_series().sum() == 0


def test_books_capped_and_weights_from_the_full_book():
    b = load("books")
    per = b.group_by("fund_id", "quarter").agg(pl.len().alias("n"), pl.col("weight_long").sum().alias("w"))
    assert per["n"].max() <= app_export.TOP_N
    assert per["w"].max() <= 1 + 1e-4
    # books under the cap are complete: their weights add up to the whole long book
    assert (per.filter(pl.col("n") < app_export.TOP_N)["w"] - 1).abs().max() < 1e-3
    # capped books are a share of it
    assert per.filter(pl.col("n") == app_export.TOP_N)["w"].min() < 1
    assert b["quarter"].min() >= app_export.BOOKS_FIRST


def test_predictions_are_probabilities_and_lists_have_20_names():
    b = load("books").drop_nulls("p_add")
    assert ((b["p_exit"] + b["p_trim"] + b["p_hold"] + b["p_add"]) - 1).abs().max() < 1e-4
    n = load("newbuys").group_by("method", "fund_id", "quarter").agg(pl.col("rank").max())
    assert n["rank"].max() == 20


def test_every_fund_is_in_the_snapshot_and_meta_matches():
    funds = set(load("funds")["fund_id"])
    assert set(load("fingerprint")["fund_id"]) <= funds and len(funds) == 93
    meta = json.loads((DATA / "meta.json").read_text())
    assert meta["top_n_positions"] == app_export.TOP_N


def test_no_price_series_exported():
    cols = {c for f in DATA.glob("*.parquet") for c in pl.read_parquet_schema(f)}
    assert not cols & {"close", "price", "px0", "px1", "adj_close"}


# --- the app renders against the snapshot ------------------------------------------------------

@pytest.mark.parametrize("page", ["fund_page", "backtest_page", "about_page"])
def test_app_page_renders(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(f"import sys; sys.path.insert(0, 'app'); import streamlit_app as a; a.{page}()",
                             default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title


def test_fund_page_works_for_a_holdout_fund_and_an_old_quarter():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string("import sys; sys.path.insert(0, 'app'); import streamlit_app as a; a.fund_page()",
                             default_timeout=120).run()
    holdout = next(o for o in at.sidebar.selectbox[0].options if "holdout" in o)
    at.sidebar.selectbox[0].select(holdout).run()
    at.sidebar.selectbox[1].select(at.sidebar.selectbox[1].options[-1]).run()   # its first quarter
    assert not at.exception, [e.value for e in at.exception]
