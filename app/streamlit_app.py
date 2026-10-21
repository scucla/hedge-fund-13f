"""Reverse-engineer a hedge fund: explore 13F fingerprints, neighbours, predictions and the backtest.

Run locally:   uv run streamlit run app/streamlit_app.py
Deploy:        Streamlit Community Cloud, entry point app/streamlit_app.py (deps in app/requirements.txt)

Reads only the parquet snapshot in app/data/ (written by src/app_export.py), so it
runs without the DuckDB database or any of the modelling libraries.

Tables are formatted as text for a non-technical reader: readable headers,
percentages with 1 decimal, ratios and t-stats with 2, and "—" for a missing value.
Charts set their axis ranges explicitly and put legends below the plot.
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
STRATEGIES = {"consensus": "Consensus best ideas", "c1_adds": "Model: predicted adds (C1)",
              "c2_als": "Model: predicted new buys (C2, ALS)", "c2_reentry": "Baseline: re-entry new buys (C2)",
              "spy": "S&P 500 (SPY)", "ew_seed": "Equal-weight, all seed holdings"}
MISSING = "—"
REPO = "https://github.com/scucla/hedge-fund-13f"
LEGEND_BELOW = {"orientation": "h", "yanchor": "top", "y": -0.12, "xanchor": "left", "x": 0, "title": None}

st.set_page_config(page_title="Reverse-engineer a hedge fund", layout="wide")


@st.cache_data
def load(name: str) -> pl.DataFrame:
    return pl.read_parquet(DATA / f"{name}.parquet")


@st.cache_data
def meta() -> dict:
    return json.loads((DATA / "meta.json").read_text())


# --- formatting --------------------------------------------------------------------------------

def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def fmt_pct(x, decimals: int = 1) -> str:
    """0.1234 -> '12.3%'; tiny non-zero shares -> '<0.1%' rather than a misleading '0.0%'."""
    if _missing(x):
        return MISSING
    if 0 < abs(x) < 0.5 * 10 ** -(decimals + 2):
        return f"<{10 ** -decimals:.{decimals}f}%"
    return f"{x * 100:.{decimals}f}%"


def fmt_num(x, decimals: int = 2) -> str:
    return MISSING if _missing(x) else f"{x:,.{decimals}f}"


def fmt_int(x) -> str:
    return MISSING if _missing(x) else f"{int(x):,}"


def readable(s: str | None) -> str:
    return MISSING if s is None else s.replace("_", " ")


def strategy_label(s: str) -> str:
    if s.startswith("cluster:"):
        return f"Clone: {readable(s.split(':', 1)[1])} cluster"
    return STRATEGIES.get(s, s)


def table(df: pl.DataFrame, formats: dict[str, tuple[str, callable]]) -> pl.DataFrame:
    """Rename and format the given columns, in that order, as text."""
    return pl.DataFrame({header: [fn(v) for v in df[col].to_list()] for col, (header, fn) in formats.items()})


def clean_name(col: str = "name") -> pl.Expr:
    """13F issuer names come with runs of spaces ('GAMESTOP      CORP NEW')."""
    return pl.col(col).str.replace_all(r"\s+", " ").str.strip_chars()


def fund_label(row: dict) -> str:
    style = (row["style"] or "unlabelled").replace("_", " ")
    return f"{row['fund_name']} ({style}{', holdout' if row['cohort'] == 'holdout' else ''})"


# --- fund page -------------------------------------------------------------------------------

def radar(fp_q: pl.DataFrame, fund: str) -> go.Figure:
    seed_median = fp_q.select([pl.col(c).median() for c in TILTS]).row(0)
    mine = fp_q.filter(pl.col("fund_id") == fund).select(list(TILTS)).row(0)
    cats = list(TILTS.values())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(seed_median) + [seed_median[0]], theta=cats + [cats[0]], name="Median seed fund",
                                  line={"color": "#9a9a9a", "dash": "dash"}))
    fig.add_trace(go.Scatterpolar(r=list(mine) + [mine[0]], theta=cats + [cats[0]], name="This fund", fill="toself",
                                  line={"color": "#2a6f97"}))
    # the range covers both traces, so neither is clipped
    values = [abs(v) for v in list(mine) + list(seed_median) if not _missing(v)]
    lim = 1.1 * max(values + [1.0])
    fig.update_layout(polar={"radialaxis": {"range": [-lim, lim]}}, height=400, margin={"t": 30, "b": 60},
                      legend=LEGEND_BELOW | {"y": -0.05}, showlegend=True)
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
    return out.sort("similarity", descending=True).head(k).join(funds, on="fund_id", how="left")


def map_figure(mp: pl.DataFrame, funds: pl.DataFrame, fund: str, quarter: str) -> go.Figure | None:
    m = mp.filter(pl.col("quarter") == quarter).join(funds.select("fund_id", "fund_name", "style"), on="fund_id", how="left")
    if m.is_empty():
        return None
    m = m.with_columns(pl.when(pl.col("cohort") == "background").then(pl.lit("Background filer"))
                       .when(pl.col("cohort") == "holdout").then(pl.lit("Holdout fund (projected)"))
                       .when(pl.col("cluster") < 0).then(pl.lit("No cluster"))
                       .otherwise(pl.concat_str(pl.lit("Cluster "), pl.col("cluster").cast(pl.Utf8))).alias("Group"),
                       pl.col("fund_name").fill_null(pl.col("fund_id")).alias("name"),
                       pl.col("style").str.replace_all("_", " ").fill_null(MISSING).alias("Style"))
    fig = px.scatter(m.sort("Group").to_pandas(), x="x", y="y", color="Group", hover_name="name",
                     hover_data={"Style": True, "x": False, "y": False}, height=520,
                     color_discrete_map={"Background filer": "#d6d5d0", "No cluster": "#8c8c8c"})
    me = m.filter(pl.col("fund_id") == fund)
    if not me.is_empty():
        fig.add_trace(go.Scatter(x=me["x"], y=me["y"], mode="markers+text", text=me["name"], textposition="top center",
                                 marker={"size": 18, "symbol": "star", "color": "#d62728", "line": {"width": 1, "color": "black"}},
                                 name="This fund", cliponaxis=False))
    # padded explicit ranges: points on the edge and the fund's label stay inside the plot
    pad = lambda lo, hi: [lo - 0.08 * (hi - lo), hi + 0.12 * (hi - lo)]
    fig.update_layout(xaxis={"visible": False, "range": pad(m["x"].min(), m["x"].max())},
                      yaxis={"visible": False, "range": pad(m["y"].min(), m["y"].max())},
                      margin={"t": 10, "b": 10}, legend=LEGEND_BELOW | {"y": -0.02, "font": {"size": 11}},
                      showlegend=True)
    return fig


def history_figures(h: pl.DataFrame, ts: pl.DataFrame) -> tuple[go.Figure | None, go.Figure, go.Figure | None]:
    heat = None
    if not h.is_empty():
        last8 = sorted(h["quarter"].unique().to_list())[-8:]
        top = h.filter(pl.col("quarter") == last8[-1]).sort("rank_long").head(15)["cusip"].to_list()
        grid = (h.filter(pl.col("quarter").is_in(last8) & pl.col("cusip").is_in(top)).with_columns(clean_name())
                .pivot(on="quarter", index="name", values="weight_long", aggregate_function="sum").fill_null(0))
        grid = grid.select(["name"] + [q for q in last8 if q in grid.columns])
        z = grid.drop("name").to_numpy()
        heat = go.Figure(go.Heatmap(z=z, x=grid.columns[1:], y=grid["name"].to_list(), colorscale="Blues",
                                    text=[[fmt_pct(v) for v in row] for row in z], texttemplate="%{text}",
                                    hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
                                    colorbar={"title": "Weight", "tickformat": ".0%"}))
        heat.update_layout(height=460, margin={"t": 10, "b": 10}, yaxis={"autorange": "reversed"})
    conc = go.Figure()
    for col, name in (("top10_weight", "Top-10 weight"), ("turnover", "Turnover")):
        conc.add_trace(go.Scatter(x=ts["quarter"], y=ts[col], mode="lines", name=name))
    conc.update_layout(height=340, yaxis={"tickformat": ".0%", "rangemode": "tozero", "autorange": True},
                       margin={"t": 10, "b": 10}, legend=LEGEND_BELOW | {"y": -0.18}, showlegend=True)
    style = None
    if ts.select(pl.col("p_growth_tech").is_not_null().any()).item():
        s = ts.drop_nulls("p_growth_tech")
        style = go.Figure()
        for c in STYLE_COLS:
            style.add_trace(go.Scatter(x=s["quarter"], y=s[c], mode="lines", stackgroup="one", name=readable(c[2:])))
        style.update_layout(height=340, yaxis={"tickformat": ".0%", "range": [0, 1]}, margin={"t": 10, "b": 10},
                            legend=LEGEND_BELOW | {"y": -0.18, "font": {"size": 10}}, showlegend=True)
    return heat, conc, style


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
    st.caption(f"Hand style label: **{readable(info['style'])}** · model's out-of-fund prediction this quarter: "
               f"**{readable(fq.get('predicted'))}** · {info['cohort']} fund · filed {fq['known_at']} · "
               f"history {info['first_quarter']}–{info['last_quarter']}")

    c = st.columns(6)
    c[0].metric("Positions", fmt_int(fq["n_positions"]))
    c[1].metric("Top-10 weight", fmt_pct(fq["top10_weight"]))
    c[2].metric("Concentration (HHI)", fmt_num(fq["hhi"], 3))
    c[3].metric("Turnover", fmt_pct(fq["turnover"]))
    c[4].metric("Active share", fmt_pct(fq["active_share"]))
    c[5].metric("Avg holding (quarters)", fmt_num(fq["holding_period_q"], 1))

    left, right = st.columns(2)
    with left:
        st.subheader("Fingerprint")
        st.caption("Holdings-weighted factor tilts (z-scores across the stocks seed funds hold), vs the median seed fund.")
        st.plotly_chart(radar(fp.filter(pl.col("quarter") == quarter), fund), width="stretch")
    with right:
        st.subheader("Nearest funds")
        st.caption("Most similar books this quarter (cosine similarity of fund2vec embeddings; 1 = identical).")
        nn = neighbours(emb, funds, fund, quarter)
        if nn.is_empty():
            st.info("No embedding for this fund-quarter.")
        else:
            st.dataframe(table(nn, {"fund_name": ("Fund", str), "style": ("Style", readable),
                                    "cohort": ("Group", readable), "similarity": ("Similarity", fmt_num)}),
                         hide_index=True, width="stretch")

    st.subheader("Map of the hedge fund industry")
    fig = map_figure(mp, funds, fund, quarter)
    if fig is None:
        st.info("No map for this quarter.")
    else:
        st.caption("UMAP of fund2vec embeddings, fitted on the 77 seed funds each quarter (holdout and background filers "
                   "projected), colored by HDBSCAN cluster. Axes have no meaning and change between quarters.")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Predicted next moves")
    b = books.filter((pl.col("fund_id") == fund) & (pl.col("quarter") == quarter)).with_columns(clean_name())
    if b.is_empty():
        st.info(f"Predictions cover {meta()['books_first_quarter']} onward (the test years of the frozen models).")
    else:
        if info["cohort"] == "holdout":
            st.warning("Holdout funds were never clustered, so their cluster features are empty; predictions are weaker for them.")
        known = b["next_action"].is_not_null().any()
        st.caption("Probability of each move by next quarter's filing, from the model frozen on 2018–2022 data. "
                   + ("'Actual' is what the next filing showed." if known else "The next filing isn't out yet."))
        moves = {"name": ("Stock", str), "symbol": ("Ticker", lambda s: s or MISSING),
                 "weight_long": ("Weight", fmt_pct), "p_exit": ("Exit", fmt_pct), "p_trim": ("Trim", fmt_pct),
                 "p_hold": ("Hold", fmt_pct), "p_add": ("Add", fmt_pct), "next_action": ("Actual", readable)}
        l2, r2 = st.columns(2)
        l2.markdown("**Most likely exits**")
        l2.dataframe(table(b.drop_nulls("p_exit").sort("p_exit", descending=True).head(10), moves),
                     hide_index=True, width="stretch")
        r2.markdown("**Most likely adds**")
        r2.dataframe(table(b.drop_nulls("p_add").sort("p_add", descending=True).head(10), moves),
                     hide_index=True, width="stretch")
        n = nb.filter((pl.col("fund_id") == fund) & (pl.col("quarter") == quarter)).with_columns(clean_name())
        st.markdown("**New buys it might make** (top 20)")
        st.caption("ALS recommender over the books public on the filing date; the re-entry baseline (names it sold in the "
                   "last year) predicted better in testing. 'Bought' = in the next filing and not held now.")
        bought = (lambda v: "Yes" if v else "No") if known else (lambda v: MISSING)
        lc, rc = st.columns(2)
        for col, method, title in ((lc, "als", "ALS recommender"), (rc, "reentry", "Re-entry baseline")):
            t = n.filter(pl.col("method") == method)
            col.markdown(title)
            col.dataframe(table(t, {"rank": ("Rank", fmt_int), "name": ("Stock", str),
                                    "symbol": ("Ticker", lambda s: s or MISSING), "bought_next_q": ("Bought", bought)}),
                          hide_index=True, width="stretch", height=300)

    st.subheader("Recent history")
    heat, conc, style = history_figures(books.filter(pl.col("fund_id") == fund), fp.filter(pl.col("fund_id") == fund).sort("quarter"))
    if heat is not None:
        st.caption("Weight of today's top 15 holdings over the last 8 quarters (share of the full long book).")
        st.plotly_chart(heat, width="stretch")
    a, bcol = st.columns(2)
    a.caption("Concentration and turnover")
    a.plotly_chart(conc, width="stretch")
    if style is not None:
        bcol.caption("Style classifier, out-of-fund probabilities by quarter")
        bcol.plotly_chart(style, width="stretch")


# --- backtest page ---------------------------------------------------------------------------

BACKTEST_COLUMNS = {
    "strategy": ("Strategy", strategy_label),
    "ann_return": ("Annual return", fmt_pct),
    "ann_vol": ("Volatility", fmt_pct),
    "sharpe": ("Sharpe", fmt_num),
    "max_dd": ("Max drawdown", fmt_pct),
    "turnover": ("Turnover (per year)", fmt_pct),
    "ff6_alpha": ("FF5+Mom alpha", fmt_pct),
    "ff6_t": ("t-stat", fmt_num),
    "alpha_spy": ("Alpha vs S&P 500", fmt_pct),
    "t_spy": ("t-stat vs S&P 500", fmt_num),
    "beta_spy": ("Beta to S&P 500", fmt_num),
    "alpha_ew_seed": ("Alpha vs equal-weight", fmt_pct),
    "t_ew_seed": ("t-stat vs equal-weight", fmt_num),
    "months": ("Months", fmt_int),
}


def equity_figure(d: pl.DataFrame) -> go.Figure:
    fig = px.line(d.to_pandas(), x="date", y="Growth of $1", color="Strategy", log_y=True, height=560)
    fig.update_yaxes(autorange=True, title="Growth of $1 (log scale)")
    fig.update_xaxes(title=None)
    fig.update_layout(margin={"t": 10, "b": 10}, legend=LEGEND_BELOW, showlegend=True, hovermode="x unified")
    return fig


def backtest_page() -> None:
    st.title("Backtest: does copying 13Fs make money?")
    st.caption("Quarterly rebalance at the close the day after the 45-day filing deadline, using only filings public by then. "
               "Long-only, equal-weight, 20 names. Model strategies run only over the 2023+ test years, with the models frozen.")
    daily, res = load("backtest_daily"), load("backtest_results")
    period = st.radio("Period", ["2018+", "2021+", "2023+"], horizontal=True)
    cost = st.radio("Trading cost per side", [10, 20], horizontal=True, format_func=lambda c: f"{c} bp")
    avail = res.filter((pl.col("period") == period) & (pl.col("cost_bp") == cost) & (pl.col("missing_sold_out") == 0))
    names = sorted(avail["strategy"].to_list(), key=strategy_label)
    default = [s for s in ("consensus", "cluster:growth_tech", "c2_als", "c1_adds", "spy", "ew_seed") if s in names]
    chosen = st.multiselect("Strategies", names, default=default, format_func=strategy_label)
    if not chosen:
        st.info("Pick at least one strategy.")
        return
    start = avail["start"].min()
    d = (daily.filter(pl.col("strategy").is_in(chosen) & (pl.col("date") > pl.lit(start).str.to_date()))
         .sort("strategy", "date")
         .with_columns((1 + pl.col("r")).cum_prod().over("strategy").alias("Growth of $1"),
                       pl.col("strategy").map_elements(strategy_label, return_dtype=pl.Utf8).alias("Strategy")))
    st.caption("Curves are net of 10 bp per side" + ("; the table below uses 20 bp." if cost == 20 else "."))
    st.plotly_chart(equity_figure(d), width="stretch")
    rows = avail.filter(pl.col("strategy").is_in(chosen)).sort(pl.col("strategy").map_elements(strategy_label, return_dtype=pl.Utf8))
    st.dataframe(table(rows, BACKTEST_COLUMNS), hide_index=True, width="stretch")
    st.markdown(f"""
- **Annual return, volatility:** annualised, net of costs. **Sharpe:** excess return over the risk-free rate per unit of volatility.
- **Max drawdown:** the largest fall from a peak. **Turnover:** share of the portfolio traded per year (one way).
- **FF5+Mom alpha:** annual return left after the five Fama-French factors and momentum; the **t-stat** says how
  distinguishable it is from zero (about 2 or more is the usual bar). Monthly regressions, Newey-West, through 2026-08.
- {MISSING} = not applicable (a benchmark against itself). Full tables and the conclusion:
  [docs/backtest.md]({REPO}/blob/main/docs/backtest.md).
""")


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
