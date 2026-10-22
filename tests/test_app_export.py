"""Milestone 8: the committed app snapshot (app/data) and the Streamlit app."""

import json
from pathlib import Path

import numpy as np
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
    ("crowded_now", ["issuer_key"]),
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

@pytest.mark.parametrize("page", ["fund_page", "backtest_page", "crowded_page", "about_page"])
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


# --- formatting and chart layout ---------------------------------------------------------------

def _app():
    import sys
    sys.path.insert(0, "app")
    import streamlit_app
    return streamlit_app


def test_formatters():
    a = _app()
    assert a.fmt_pct(0.1234) == "12.3%" and a.fmt_pct(None) == "—" and a.fmt_pct(float("nan")) == "—"
    assert a.fmt_pct(0.0003) == "<0.1%" and a.fmt_pct(0.0) == "0.0%" and a.fmt_pct(-0.468) == "-46.8%"
    assert a.fmt_num(1.0234) == "1.02" and a.fmt_num(None) == "—" and a.fmt_int(103.0) == "103"
    assert a.strategy_label("cluster:growth_tech") == "Clone: growth tech cluster"
    assert a.clean_name().meta.output_name() == "name"


def test_backtest_table_is_readable():
    a = _app()
    res = load("backtest_results").filter(pl.col("missing_sold_out") == 0)
    t = a.table(res, a.BACKTEST_COLUMNS)
    assert t.columns[:8] == ["Strategy", "Annual return", "Volatility", "Sharpe", "Max drawdown",
                             "Turnover (per year)", "FF5+Mom alpha", "t-stat"]
    cells = [v for col in t.iter_columns() for v in col.to_list()]
    assert not any(v in ("None", "nan", "") or v is None for v in cells)
    assert "—" in t["Alpha vs S&P 500"].to_list()   # SPY against itself
    assert all(v.endswith("%") or v == "—" for v in t["Annual return"].to_list())


def test_equity_curve_autoranges_with_visible_legend():
    a = _app()
    d = (load("backtest_daily").filter(pl.col("strategy").is_in(["spy", "cluster:activist"])).sort("strategy", "date")
         .with_columns((1 + pl.col("r")).cum_prod().over("strategy").alias("Growth of $1"),
                       pl.col("strategy").alias("Strategy")))
    fig = a.equity_figure(d)
    # autorange with no fixed range: plotly fits every line (checking the rendered range needs kaleido)
    assert fig.layout.yaxis.autorange is True and fig.layout.yaxis.range is None and fig.layout.yaxis.type == "log"
    assert fig.layout.showlegend and fig.layout.legend.orientation == "h"
    assert {tr.name for tr in fig.data} == {"spy", "cluster:activist"}


def test_radar_and_map_ranges_contain_every_point():
    a = _app()
    fp = load("fingerprint").filter(pl.col("quarter") == "2026Q2")
    for fund in fp["fund_id"].to_list():
        fig = a.radar(fp, fund)
        lo, hi = fig.layout.polar.radialaxis.range
        vals = [v for tr in fig.data for v in tr.r if v is not None]
        assert lo <= min(vals) and max(vals) <= hi
    fig = a.map_figure(load("map"), load("funds"), "pershing_square_capital_management", "2026Q2")
    xs = [v for tr in fig.data for v in tr.x]
    assert fig.layout.xaxis.range[0] < min(xs) and max(xs) < fig.layout.xaxis.range[1]


def test_bought_shows_dash_when_next_filing_is_not_out():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string("import sys; sys.path.insert(0, 'app'); import streamlit_app as a; a.fund_page()",
                             default_timeout=120).run()   # default: latest quarter, next filing not out
    bought = [df.value for df in at.dataframe if "Bought" in df.value.columns]
    assert bought and all(set(df["Bought"]) == {"—"} for df in bought)
    for df in at.dataframe:
        assert not df.value.isin(["None", "nan"]).any().any()
