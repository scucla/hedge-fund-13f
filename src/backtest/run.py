"""Task D backtest: clone and model strategies traded after 13F filings are public.

Usage (after src.backtest.signals):
    uv run python -m src.backtest.run    # -> docs/backtest.md, docs/figures/21_backtest_equity.png

Periods:
  clones   2018+ (first portfolio from the 2017Q4 filings, entered Feb 2018),
           also reported 2021+, where daily-price coverage is high
  models   2023+ only (test years; first portfolio from the 2023Q1 filings), with
           the C1 / C2 models frozen as in Milestone 6
Costs: 10 and 20 bp per side on traded weight. Benchmarks: SPY (adjusted close,
dividends in) and `ew_seed`, every issuer any seed fund holds, on the same
calendar and cost rule.

Metrics on daily net returns: annualised return and volatility, Sharpe (excess
of the Fama-French daily risk-free rate), max drawdown, turnover (one-way,
sum |dw| / 2, per year). Alphas: monthly regressions with Newey-West t-stats
(3 lags) of excess returns on FF5 + momentum, on SPY, and on ew_seed. Monthly
returns compound the same trading days for the strategy, factors and
benchmarks, so a partial first month is compared like for like. Regressions
stop at the last month of the factor data (2026-08).
"""

import os
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.backtest import engine, strategies as st
from src.models.factor_regressions import ols_newey_west
from src.models.task_c_common import md_table
from src.securities import db

DOC = "docs/backtest.md"
CONCLUSION = "docs/backtest_conclusion.md"   # hand-written; included in DOC
FIG = "docs/figures/21_backtest_equity.png"
OUT = "data/processed/backtest"
CLONE_FIRST, MODEL_FIRST = "2017Q4", "2023Q1"
LAST_QUARTER = "2026Q2"
COSTS = (10, 20)
SOLD_OUT_SENSITIVITY = -0.30
SPARSE_CASH_SHARE = 0.25
FACTORS = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]
# returns are counted from the day after these dates; set in main() from the calendar (first clone and model entries)
PERIODS: dict[str, date] = {}


def quarters(first: str, last: str) -> list[str]:
    out, q = [], first
    while q <= last:
        out.append(q)
        q = engine._next_q(q)
    return out


def build_weights(con, cal: pl.DataFrame, delist: pl.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    books = st.load_books(con)
    rep_df = st.representative(books)
    c1 = con.execute("SELECT * FROM backtest_c1_scores").pl()
    c2 = con.execute("SELECT * FROM backtest_c2_lists").pl()
    issuer_of = dict(books.select("cusip", "issuer").unique("cusip").iter_rows())
    w: dict[str, dict[str, dict[str, float]]] = {}
    for q, cutoff, entry in cal.select("quarter", "cutoff", "entry").iter_rows():
        b = st.visible(books, q, cutoff)
        rep = dict(rep_df.filter(pl.col("quarter") == q).select("issuer", "cusip").iter_rows())
        gone = set(delist.filter(pl.col("delisted_on") <= entry)["cusip"])
        w.setdefault("consensus", {})[q] = st.to_weights(st.consensus(b), rep, gone)
        alive = [rep[i] for i in st.ew(b) if rep[i] not in gone]
        w.setdefault("ew_seed", {})[q] = {c: 1 / len(alive) for c in alive}
        for style, ranked in st.cluster_portfolios(b, st.cluster_names(con, cutoff, q)).items():
            w.setdefault(f"cluster:{style}", {})[q] = st.to_weights(ranked, rep, gone)
        if q >= MODEL_FIRST:
            s1 = c1.filter((pl.col("quarter") == q) & (pl.col("pred_date") <= cutoff))
            w.setdefault("c1_adds", {})[q] = st.to_weights(st.c1_adds(s1, b), rep, gone)
            for m in ("als", "reentry"):
                s2 = c2.filter((pl.col("method") == m) & (pl.col("quarter") == q) & (pl.col("pred_date") <= cutoff))
                w.setdefault(f"c2_{m}", {})[q] = st.to_weights(st.c2_buys(s2, issuer_of), rep, gone)
    return w


def max_drawdown(r: np.ndarray) -> float:
    v = np.cumprod(1 + r)
    return float((v / np.maximum.accumulate(v) - 1).min())


def monthly(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return (df.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
            .group_by("month").agg([((1 + pl.col(c)).product() - 1).alias(c) for c in cols]).sort("month"))


def alpha(y: np.ndarray, X: np.ndarray) -> tuple[float, float, np.ndarray]:
    b, t, _ = ols_newey_west(y, X)
    return b[0] * 12, t[0], b[1:]


def metrics(name: str, d: pl.DataFrame, reb: pl.DataFrame, start: date, ff: pl.DataFrame, bench: pl.DataFrame) -> dict:
    d = d.filter(pl.col("date") > start)
    reb = reb.filter(pl.col("entry") >= start)
    j = d.join(ff, on="date", how="left").join(bench, on="date", how="left").with_columns(pl.col("RF").fill_null(strategy="forward"))
    r, rf = j["r"].to_numpy(), j["RF"].to_numpy()
    years = len(r) / 252
    ex = r - rf
    out = {"strategy": name, "start": str(d["date"].min()), "ann_return": float(np.prod(1 + r) ** (1 / years) - 1),
           "ann_vol": float(r.std() * np.sqrt(252)), "sharpe": float(ex.mean() / ex.std() * np.sqrt(252)),
           "max_dd": max_drawdown(r), "turnover": float(reb["traded"].sum() / 2 / years)}
    m = monthly(j.filter(pl.col("Mkt_RF").is_not_null()), ["r", "RF", "spy", "ew_seed"] + FACTORS)
    y = (m["r"] - m["RF"]).to_numpy()
    a, t, _ = alpha(y, m.select(FACTORS).to_numpy())
    out |= {"ff6_alpha": a, "ff6_t": t}
    for b in ("spy", "ew_seed"):
        if b == name:   # a benchmark regressed on itself says nothing
            out |= {f"alpha_{b}": None, f"t_{b}": None, f"beta_{b}": None}
            continue
        a, t, beta = alpha(y, (m[b] - m["RF"]).to_numpy()[:, None])
        out |= {f"alpha_{b}": a, f"t_{b}": t, f"beta_{b}": float(beta[0])}
    out["months"] = len(m)
    return out


def coverage(rebals: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows = []
    for name, reb in rebals.items():
        for y, g in reb.filter(pl.col("n_names") > 0).group_by(pl.col("quarter").str.slice(0, 4).alias("year")):
            rows.append({"strategy": name, "year": y[0]} | {s: g[f"w_{s}"].mean() for s in engine.SOURCE_RANK})
    return pl.DataFrame(rows).sort("strategy", "year")


def figure(curves: dict[str, pl.DataFrame], path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    groups = [("Clone strategies, 2018+", PERIODS["2018+"], lambda n: not n.startswith(("c1_", "c2_"))),
              ("Model strategies (frozen), 2023+", PERIODS["2023+"], lambda n: n.startswith(("c1_", "c2_", "consensus")) or n in ("spy", "ew_seed"))]
    for ax, (title, start, keep) in zip(axes, groups):
        for n, d in curves.items():
            if not keep(n):
                continue
            d = d.filter(pl.col("date") > start)
            v = np.cumprod(1 + d["r"].to_numpy())
            style = {"lw": 2.2, "color": "black"} if n == "spy" else {"lw": 2.2, "color": "grey", "ls": "--"} if n == "ew_seed" else {"lw": 1.2}
            ax.plot(d["date"].to_list(), v, label=n, **style)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    fig.suptitle("Growth of $1, net of 10 bp per side, log scale. Quarterly rebalance the day after the 13F deadline.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    with db.connect(read_only=True) as con:
        rd = engine.load_returns(con, date(2017, 10, 1))
        cal = engine.calendar(rd.days, quarters(CLONE_FIRST, LAST_QUARTER))
        entry = dict(cal.select("quarter", "entry").iter_rows())
        PERIODS.update({"2018+": entry[CLONE_FIRST], "2021+": date(2020, 12, 31), "2023+": entry[MODEL_FIRST]})
        weights = build_weights(con, cal, rd.delist)
        ff = con.execute("SELECT date, RF, " + ", ".join(FACTORS) + " FROM ff_factors_daily WHERE date >= '2017-10-01'").pl()
        spy = con.execute("""SELECT date, close / lag(close) OVER (ORDER BY date) - 1 AS spy FROM prices_daily
                             WHERE symbol = 'SPY' AND date >= '2017-10-01'""").pl()
    windows = cal.to_dicts()
    cache: dict = {}
    runs: dict[tuple[str, int, float], tuple[pl.DataFrame, pl.DataFrame]] = {}
    for name, wq in weights.items():
        first = MODEL_FIRST if name.startswith(("c1_", "c2_")) else CLONE_FIRST
        win = [w for w in windows if w["quarter"] >= first]
        for c in COSTS:
            runs[(name, c, 0.0)] = engine.simulate(win, wq, rd, c, cache=cache)
        if not name.startswith(("c1_", "c2_")):
            runs[(name, 10, SOLD_OUT_SENSITIVITY)] = engine.simulate(win, wq, rd, 10, SOLD_OUT_SENSITIVITY, cache=cache)
        print("simulated", name, flush=True)

    ew_daily = runs[("ew_seed", 10, 0.0)][0].select("date", pl.col("r").alias("ew_seed"))
    bench = spy.join(ew_daily, on="date", how="inner")
    spy_run = (spy.filter(pl.col("date") > cal["entry"][0]).rename({"spy": "r"}).with_columns(pl.col("r").alias("r_gross")),
               pl.DataFrame({"quarter": [], "entry": [], "traded": [], "n_names": []},
                            schema={"quarter": pl.Utf8, "entry": pl.Date, "traded": pl.Float64, "n_names": pl.Int64}))
    runs[("spy", 10, 0.0)] = runs[("spy", 20, 0.0)] = spy_run

    rows = []
    for (name, c, sens), (d, reb) in runs.items():
        for period, start in PERIODS.items():
            is_model = name.startswith(("c1_", "c2_"))
            if is_model and period != "2023+":
                continue
            rows.append({"period": period, "cost_bp": c, "missing_sold_out": sens} | metrics(name, d, reb, start, ff, bench))
    res = pl.DataFrame(rows)
    res.write_parquet(f"{OUT}_results.parquet")
    rebals = {n: reb for (n, c, s), (d, reb) in runs.items() if c == 10 and s == 0.0 and n != "spy"}
    pl.concat([reb.with_columns(pl.lit(n).alias("strategy")) for n, reb in rebals.items()], how="diagonal").write_parquet(
        f"{OUT}_rebalances.parquet")
    curves = {n: d for (n, c, s), (d, _) in runs.items() if c == 10 and s == 0.0}
    pl.concat([d.select("date", "r").with_columns(pl.lit(n).alias("strategy")) for n, d in curves.items()]).write_parquet(
        f"{OUT}_daily.parquet")
    figure(curves, FIG)
    write_doc(res, rebals, cal["exit"].max())


def write_doc(res: pl.DataFrame, rebals: dict[str, pl.DataFrame], last_day: date) -> None:
    cov = coverage(rebals)
    conclusion = open(CONCLUSION).read().strip() if os.path.exists(CONCLUSION) else "(not written yet)"
    cols = ["strategy", "ann_return", "ann_vol", "sharpe", "max_dd", "turnover", "ff6_alpha", "ff6_t",
            "alpha_spy", "t_spy", "beta_spy", "alpha_ew_seed", "t_ew_seed", "months"]

    names = (pl.concat([reb.with_columns(pl.lit(n).alias("strategy")) for n, reb in rebals.items()], how="diagonal")
             .group_by("strategy").agg(pl.col("n_names").mean().alias("avg_names"), (pl.col("n_names") == 0).sum().alias("cash_quarters"),
                                       pl.len().alias("rebalances")).sort("strategy"))
    # a style cluster that exists in under 3 of 4 quarters is mostly cash: its ratios aren't comparable
    sparse = names.filter(pl.col("cash_quarters") > SPARSE_CASH_SHARE * pl.col("rebalances"))["strategy"].to_list()

    def tab(period, cost, only=None):
        t = res.filter((pl.col("period") == period) & (pl.col("cost_bp") == cost) & (pl.col("missing_sold_out") == 0.0))
        t = t.filter(pl.col("strategy").is_in(only)) if only else t.filter(~pl.col("strategy").is_in(sparse))
        t = t.with_columns(pl.col("strategy").is_in(["spy", "ew_seed"]).alias("bench")).sort("bench", "strategy")
        return md_table(t.select(cols))

    sens = (res.filter((pl.col("period") == "2018+") & (pl.col("cost_bp") == 10))
            .pivot(on="missing_sold_out", index="strategy", values="ann_return")
            .rename({"0.0": "ann_return", str(SOLD_OUT_SENSITIVITY): "ann_return_sold_out_-30%"})
            .drop_nulls())
    cov_avg = (cov.with_columns(pl.when(pl.col("year") < "2021").then(pl.lit("2018-2020")).otherwise(pl.lit("2021+")).alias("span"))
               .group_by("strategy", "span").agg(pl.col(s).mean() for s in engine.SOURCE_RANK).sort("strategy", "span"))
    text = f"""# Task D: backtest

Generated by `src/backtest/run.py` (selection rules in `src/backtest/strategies.py`, engine in
`src/backtest/engine.py`, frozen model signals in `src/backtest/signals.py`). Figure:
`docs/figures/21_backtest_equity.png`.

**Rules.** One rebalance per quarter. A portfolio uses only 13F filings public by the 45-day deadline and
trades at the close of the next trading day, so every filing it uses was public before the trade; late
filers wait a quarter. Long-only, equal-weight, 20 issuers, common stock and ADRs. Costs 10 and 20 bp per
side on traded weight. Clone strategies run from the 2017Q4 filings (Feb 2018); model strategies only
over the 2023+ test years with the Milestone 6 models frozen. Alphas: monthly, Newey-West t-stats (3 lags),
through 2026-08 where the factor data ends. Return series run to {last_day}.

## Conclusion

{conclusion}

**Returns of unpriced names.** Stocks without Yahoo daily prices earn the filled quarterly return
(13F-implied or delisting return) spread evenly over the quarter's trading days. **Those returns run
quarter end to quarter end, while positions are held filing date to filing date, so the timing of the
unpriced names is approximate**: a position entered in mid-May earns two-thirds of the Q2 return and
a third of the Q3 return, not its actual May-to-August move. Spreading also smooths those names, which
biases volatility and betas down. Names with no return at all earn 0% (sensitivity below at -30% for
names every seed fund sold).

## Results, 2023+ (all strategies, net of 10 bp per side)

{tab("2023+", 10)}

## Clone strategies, 2018+ (net of 10 bp per side)

{tab("2018+", 10)}

## Clone strategies, 2021+ (high daily-price coverage; net of 10 bp per side)

{tab("2021+", 10)}

## Net of 20 bp per side

2018+:

{tab("2018+", 20)}

2021+:

{tab("2021+", 20)}

2023+:

{tab("2023+", 20)}

## Share of entry weight by return source

`daily` = Yahoo daily closes; `filled` = filled quarterly return spread over the quarter (13F-implied or
delisting); `missing` = no return (0%). A name counts by its worst source over its holding window.

Average by span:

{md_table(cov_avg)}

By year of the filing quarter:

{md_table(cov)}

## Sensitivity: names with no return because every seed fund sold them, at -30% instead of 0% (2018+, 10 bp)

{md_table(sens)}

## Style clusters that are mostly cash

{", ".join(sparse)}: no cluster with that majority style existed in more than a quarter of the
rebalances, so the portfolio held cash then and its Sharpe and alphas are not comparable with the rest.
Shown for completeness (2018+, 10 bp):

{tab("2018+", 10, only=sparse)}

## Portfolio sizes

`cash_quarters`: quarters with no qualifying cluster (the style portfolio holds cash).

{md_table(names)}
"""
    with open(DOC, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["doc"]:   # rewrite the report from the saved results
        reb = pl.read_parquet(f"{OUT}_rebalances.parquet")
        write_doc(pl.read_parquet(f"{OUT}_results.parquet"),
                  {n: g.drop("strategy") for (n,), g in reb.group_by("strategy")},
                  pl.read_parquet(f"{OUT}_daily.parquet")["date"].max())
    elif sys.argv[1:] == ["figure"]:   # redraw from the saved daily net returns
        daily = pl.read_parquet(f"{OUT}_daily.parquet")
        with db.connect(read_only=True) as con:
            days = [d for (d,) in con.execute("SELECT date FROM prices_daily WHERE symbol = 'SPY' AND date >= '2017-10-01' ORDER BY 1").fetchall()]
        entry = dict(engine.calendar(days, quarters(CLONE_FIRST, LAST_QUARTER)).select("quarter", "entry").iter_rows())
        PERIODS.update({"2018+": entry[CLONE_FIRST], "2021+": date(2020, 12, 31), "2023+": entry[MODEL_FIRST]})
        figure({n: g.drop("strategy").sort("date") for (n,), g in daily.group_by("strategy")}, FIG)
    else:
        main()
