"""Reverse-engineer a hedge fund: explore 13F fingerprints, neighbours, predictions and the backtest.

Run locally:   uv run streamlit run app/streamlit_app.py
Deploy:        Streamlit Community Cloud, entry point app/streamlit_app.py (deps in app/requirements.txt)

Reads only the parquet snapshot in app/data/ (written by src/app_export.py), so it
runs without the DuckDB database or any of the modelling libraries.
"""

import json
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

DATA = Path(__file__).parent / "data"
TILTS = {"tilt_size": "Size", "tilt_value": "Value", "tilt_momentum": "Momentum",
         "tilt_profitability": "Profitability", "tilt_investment": "Investment", "tilt_beta": "Beta"}
STYLE_COLS = ["p_activist", "p_discretionary_other", "p_event_driven", "p_growth_tech", "p_healthcare",
              "p_multi_manager", "p_quant", "p_value_concentrated"]
REPO = "https://github.com/scucla/hedge-fund-13f"

st.set_page_config(page_title="Reverse-engineer a hedge fund", layout="wide")


@st.cache_data
def load(name: str) -> pl.DataFrame:
    return pl.read_parquet(DATA / f"{name}.parquet")


@st.cache_data
def meta() -> dict:
    return json.loads((DATA / "meta.json").read_text())


def fund_label(row: dict) -> str:
    style = (row["style"] or "unlabelled").replace("_", " ")
    return f"{row['fund_name']} ({style}{', holdout' if row['cohort'] == 'holdout' else ''})"


def pct(x) -> str:
    return "–" if x is None else f"{x:.1%}"


# --- fund page -------------------------------------------------------------------------------

def radar(fp_q: pl.DataFrame, fund: str) -> go.Figure:
    seed_median = fp_q.select([pl.col(c).median() for c in TILTS]).row(0)
    mine = fp_q.filter(pl.col("fund_id") == fund).select(list(TILTS)).row(0)
    cats = list(TILTS.values())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(seed_median) + [seed_median[0]], theta=cats + [cats[0]], name="median fund",
                                  line={"color": "#9a9a9a", "dash": "dash"}))
    fig.add_trace(go.Scatterpolar(r=list(mine) + [mine[0]], theta=cats + [cats[0]], name="this fund", fill="toself",
                                  line={"color": "#2a6f97"}))
    lim = max(1.0, float(np.nanmax(np.abs([v for v in mine if v is not None] + [1.0]))))
    fig.update_layout(polar={"radialaxis": {"range": [-lim, lim]}}, height=380, margin={"t": 30, "b": 20},
                      legend={"orientation": "h"})
    return fig


def neighbours(emb: pl.DataFrame, funds: pl.DataFrame, fund: str, quarter: str, k: int = 10) -> pl.DataFrame:
    e = emb.filter(pl.col("quarter") == quarter)
    if fund not in e["fund_id"].to_list():
        return pl.DataFrame()
    X = np.vstack(e["vec"].to_list()).astype(float)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    i = e["fund_id"].to_list().index(fund)
    sim = X @ X[i]
    out = e.select("fund_id").with_columns(pl.Series("similarity", sim)).filter(pl.col("fund_id") != fund)
    return (out.sort("similarity", descending=True).head(k).join(funds, on="fund_id", how="left")
            .select(pl.col("fund_name").alias("fund"), pl.col("style"), pl.col("cohort"), pl.col("similarity").round(3)))


def map_figure(mp: pl.DataFrame, funds: pl.DataFrame, fund: str, quarter: str) -> go.Figure | None:
    m = mp.filter(pl.col("quarter") == quarter).join(funds.select("fund_id", "fund_name", "style"), on="fund_id", how="left")
    if m.is_empty():
        return None
    m = m.with_columns(pl.when(pl.col("cohort") == "background").then(pl.lit("background filer"))
                       .when(pl.col("cohort") == "holdout").then(pl.lit("holdout (projected)"))
                       .when(pl.col("cluster") < 0).then(pl.lit("noise"))
                       .otherwise(pl.concat_str(pl.lit("cluster "), pl.col("cluster").cast(pl.Utf8))).alias("group"),
                       pl.col("fund_name").fill_null(pl.col("fund_id")).alias("name"))
    fig = px.scatter(m.to_pandas(), x="x", y="y", color="group", hover_name="name", hover_data={"style": True, "x": False, "y": False},
                     height=460, color_discrete_map={"background filer": "#d6d5d0", "noise": "#8c8c8c"})
    me = m.filter(pl.col("fund_id") == fund)
    if not me.is_empty():
        fig.add_trace(go.Scatter(x=me["x"], y=me["y"], mode="markers+text", text=me["name"], textposition="top center",
                                 marker={"size": 18, "symbol": "star", "color": "#d62728", "line": {"width": 1, "color": "black"}},
                                 name="this fund"))
    fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, margin={"t": 10, "b": 10},
                      legend={"font": {"size": 10}})
    return fig


def fund_page() -> None:
    funds = load("funds")
    fp, emb, mp = load("fingerprint"), load("embeddings"), load("map")
    books, nb = load("books"), load("newbuys")

    options = {fund_label(r): r["fund_id"] for r in funds.iter_rows(named=True)}
    default = next((i for i, k in enumerate(options) if k.startswith("Pershing Square")), 0)
    label = st.sidebar.selectbox("Search a fund", list(options), index=default)
    fund = options[label]
    info = funds.filter(pl.col("fund_id") == fund).row(0, named=True)
    quarters = fp.filter(pl.col("fund_id") == fund)["quarter"].sort(descending=True).to_list()
    quarter = st.sidebar.selectbox("Quarter", quarters, index=0)

    fq = fp.filter((pl.col("fund_id") == fund) & (pl.col("quarter") == quarter)).row(0, named=True)
    st.title(info["fund_name"])
    pred = fq.get("predicted")
    st.caption(f"Hand style label: **{(info['style'] or 'none').replace('_', ' ')}** · model's out-of-fund prediction "
               f"this quarter: **{(pred or 'n/a').replace('_', ' ')}** · {info['cohort']} fund · filed {fq['known_at']} · "
               f"history {info['first_quarter']}–{info['last_quarter']}")

    c = st.columns(6)
    c[0].metric("Positions", f"{fq['n_positions']:,}")
    c[1].metric("Top-10 weight", pct(fq["top10_weight"]))
    c[2].metric("HHI", f"{fq['hhi']:.3f}")
    c[3].metric("Turnover", pct(fq["turnover"]))
    c[4].metric("Active share", pct(fq["active_share"]))
    c[5].metric("Avg holding (quarters)", "–" if fq["holding_period_q"] is None else f"{fq['holding_period_q']:.1f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Fingerprint")
        st.caption("Holdings-weighted factor tilts, z-scores across the stocks seed funds hold, vs the median seed fund.")
        st.plotly_chart(radar(fp.filter(pl.col("quarter") == quarter), fund), width="stretch")
    with right:
        st.subheader("Nearest funds")
        st.caption("Cosine similarity of fund2vec embeddings this quarter (Milestone 5).")
        nn = neighbours(emb, funds, fund, quarter)
        if nn.is_empty():
            st.info("No embedding for this fund-quarter.")
        else:
            st.dataframe(nn, hide_index=True, width="stretch")

    st.subheader("Map of the hedge fund industry")
    fig = map_figure(mp, funds, fund, quarter)
    if fig is None:
        st.info("No map for this quarter.")
    else:
        st.caption("UMAP of fund2vec embeddings, fitted on the 77 seed funds each quarter (holdout and background filers "
                   "projected), colored by HDBSCAN cluster. Axes have no meaning and change between quarters.")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Predicted next moves")
    b = books.filter((pl.col("fund_id") == fund) & (pl.col("quarter") == quarter))
    if b.is_empty():
        st.info(f"Predictions cover {meta()['books_first_quarter']} onward (the test years of the frozen models).")
    else:
        if info["cohort"] == "holdout":
            st.warning("Holdout funds were never clustered, so their cluster features are empty; C1 predictions are weaker for them.")
        known = b["next_action"].is_not_null().any()
        st.caption("C1: probability of each move by next quarter's filing, from the model frozen on 2018–2022. "
                   + ("'Actual' is what the next filing showed." if known else "The next filing isn't out yet."))
        cols = ["name", "symbol", "weight_long", "p_exit", "p_trim", "p_hold", "p_add", "next_action"]
        show = lambda df: (df.select(cols).rename({"weight_long": "weight", "next_action": "actual"})
                           .with_columns(pl.col("weight").round(4), *[pl.col(f"p_{x}").round(2) for x in ("exit", "trim", "hold", "add")]))
        l2, r2 = st.columns(2)
        l2.markdown("**Most likely exits**")
        l2.dataframe(show(b.drop_nulls("p_exit").sort("p_exit", descending=True).head(10)), hide_index=True, width="stretch")
        r2.markdown("**Most likely adds**")
        r2.dataframe(show(b.drop_nulls("p_add").sort("p_add", descending=True).head(10)), hide_index=True, width="stretch")
        n = nb.filter((pl.col("fund_id") == fund) & (pl.col("quarter") == quarter))
        st.markdown("**New buys it might make** (C2 top 20)")
        st.caption("ALS over the books public on the filing date; the re-entry baseline (names it sold in the last year) "
                   "predicted better in Milestone 6. 'bought' = in the next filing, not held now.")
        lc, rc = st.columns(2)
        for col, method, title in ((lc, "als", "ALS"), (rc, "reentry", "Re-entry baseline")):
            t = n.filter(pl.col("method") == method).select("rank", "name", "symbol", pl.col("bought_next_q").alias("bought"))
            col.markdown(title)
            col.dataframe(t, hide_index=True, width="stretch", height=300)

    st.subheader("Recent history")
    h = books.filter(pl.col("fund_id") == fund)
    if not h.is_empty():
        last8 = sorted(h["quarter"].unique().to_list())[-8:]
        top = (h.filter(pl.col("quarter") == last8[-1]).sort("rank_long").head(15)["cusip"].to_list())
        grid = (h.filter(pl.col("quarter").is_in(last8) & pl.col("cusip").is_in(top))
                .pivot(on="quarter", index="name", values="weight_long", aggregate_function="sum").fill_null(0))
        grid = grid.select(["name"] + [q for q in last8 if q in grid.columns])
        st.caption("Weight of today's top 15 holdings over the last 8 quarters (share of the full long book).")
        fig = px.imshow(grid.drop("name").to_numpy(), x=grid.columns[1:], y=grid["name"].to_list(), aspect="auto",
                        color_continuous_scale="Blues", height=420, labels={"color": "weight"})
        st.plotly_chart(fig, width="stretch")
    ts = fp.filter(pl.col("fund_id") == fund).sort("quarter")
    a, bcol = st.columns(2)
    a.caption("Concentration and turnover")
    a.line_chart(ts.select("quarter", "top10_weight", "turnover").to_pandas().set_index("quarter"))
    if ts.select(pl.col("p_growth_tech").is_not_null().any()).item():
        bcol.caption("Style classifier, out-of-fund probabilities by quarter")
        bcol.area_chart(ts.select(["quarter"] + STYLE_COLS).rename({c: c[2:] for c in STYLE_COLS}).to_pandas().set_index("quarter"))


# --- backtest page ---------------------------------------------------------------------------

def backtest_page() -> None:
    st.title("Backtest: does copying 13Fs make money?")
    st.caption("Quarterly rebalance at the close the day after the 45-day filing deadline, using only filings public by then. "
               "Long-only, equal-weight, 20 names. Net of 10 bp per side unless chosen otherwise. Model strategies run only "
               "over the 2023+ test years, with the models frozen.")
    daily, res = load("backtest_daily"), load("backtest_results")
    period = st.radio("Period", ["2018+", "2021+", "2023+"], horizontal=True)
    cost = st.radio("Cost per side (bp)", [10, 20], horizontal=True)
    avail = res.filter((pl.col("period") == period) & (pl.col("cost_bp") == cost) & (pl.col("missing_sold_out") == 0))
    names = avail["strategy"].to_list()
    default = [s for s in ("consensus", "cluster:growth_tech", "c2_als", "c1_adds", "spy", "ew_seed") if s in names]
    chosen = st.multiselect("Strategies", names, default=default)
    start = avail["start"].min()
    d = daily.filter(pl.col("strategy").is_in(chosen) & (pl.col("date") > pl.lit(start).str.to_date()))
    if cost == 20:
        st.caption("Curves are drawn net of 10 bp; the table below uses 20 bp.")
    d = d.sort("strategy", "date").with_columns(((1 + pl.col("r")).cum_prod().over("strategy")).alias("growth of $1"))
    fig = px.line(d.to_pandas(), x="date", y="growth of $1", color="strategy", log_y=True, height=480)
    st.plotly_chart(fig, width="stretch")
    cols = ["strategy", "ann_return", "ann_vol", "sharpe", "max_dd", "turnover", "ff6_alpha", "ff6_t", "alpha_spy", "t_spy",
            "beta_spy", "alpha_ew_seed", "t_ew_seed", "months"]
    st.dataframe(avail.filter(pl.col("strategy").is_in(chosen)).select(cols)
                 .with_columns(pl.col(pl.Float64).round(3)), hide_index=True, width="stretch")
    st.markdown("Alphas: monthly, Newey-West t-stats (3 lags), FF5 + momentum, through 2026-08. "
                f"Full tables and the conclusion: [docs/backtest.md]({REPO}/blob/main/docs/backtest.md).")


# --- about -----------------------------------------------------------------------------------

def about_page() -> None:
    m = meta()
    st.title("About")
    st.markdown(f"""
This app explores **{len(load('funds'))} hedge funds** (77 seed funds plus 16 hand-labelled holdout funds) through
their SEC 13F filings, 2015–2026: what each one is doing, who it resembles, what it may do next, and whether
copying them would have paid. Code, method and every design decision: [{REPO}]({REPO}).

**What this snapshot contains**
- Positions are capped at each fund's **top {m['top_n_positions']} holdings per quarter** by weight; weights are
  computed on the **full** long book, so for multi-manager funds with thousands of positions the shown weights add up
  to well under 100%.
- Position-level predictions and holdings history cover **{m['books_first_quarter']} onward**, the test years of
  models frozen on 2018–2022 data. Fingerprints, neighbours and the map go back to {m['map_first_quarter']}.
- No price series are included; backtest returns are derived series only. Snapshot exported {m['exported']}.

**What 13F data can't tell you**
- Only long positions in US-listed stocks and listed options, as quarter-end snapshots published up to 45 days later.
  No shorts, bonds, futures, FX, non-US stocks or intra-quarter trades, so a long/short fund's net exposure is unknown.
- Funds can ask to delay disclosing positions (confidential treatment); the app shows what was public at the time,
  from each quarter's original filing, not later amendments.
- Style labels are hand-assigned for 93 funds and are a judgement call; the "predicted style" is a model trained on them.
- Stocks without daily prices (about a fifth to a third of seed weight before 2023) earn returns inferred from the
  funds' own reported values, with approximate timing; see the backtest notes.
""")


if __name__ == "__main__":   # streamlit runs the script as __main__; tests import the page functions
    pg = st.navigation([st.Page(fund_page, title="Fund explorer", icon="🔎", default=True),
                        st.Page(backtest_page, title="Backtest", icon="📈"),
                        st.Page(about_page, title="About", icon="ℹ️")])
    pg.run()
