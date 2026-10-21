"""Copycat event study: is a fund's new position priced in by the time its 13F is public?

Usage (after src.analysis.edgar_times):
    uv run python -m src.analysis.event_study    # -> docs/event_study.md, docs/figures/22_event_study_car.png

**Events.** Every new long position (common stock or ADR, absent from the fund's
previous quarter) in a seed fund's point-in-time 13F, 2015Q2-2026Q2. An event is
dated by the *first* public filing that lists the stock: the earliest original
13F-HR among the fund's CIKs reporting it.

**Day 0.** From the filing's EDGAR header acceptance time (Eastern): accepted
before 16:00 on a trading day -> that day; otherwise the next trading day. A
filing whose header couldn't be had uses its filing date with day 0 = the next
trading day, and its events are flagged (`timing` = 'fallback').

**Abnormal return.** Daily stock return (Yahoo adjusted close, dividends in)
minus SPY's, summed over each window: [-5,-1], [0,+1], [+2,+10], [+11,+60]
trading days. A window counts only if the stock's price history spans it (from
the close before the window to its last day). If the series ends inside a window
and the stock's delisting (SEC Form 25/15) follows within 10 days of its last
close, the delisting return is booked the next day (acquisition: deal value /
last close - 1 when the terms parsed; performance delisting: -30%; otherwise 0)
and the abnormal return is 0 after it. Otherwise the event is dropped from that
window and counted.

**Winsorizing.** Each window's abnormal returns are clipped at the 0.5th and 99.5th
percentiles across all events (WINSOR), and so is each day of the cumulative
path. Thinly traded OTC tickers carry Yahoo price glitches (one stock "returns"
+90,000% in [+11,+60]) that otherwise decide the long-window means. This rule was
added after seeing those values; the raw means are reported next to it.

**Statistics.** Event-weighted: every event counts once. Fund-weighted: each
fund's events in a window share a total weight of 1, so the 7 multi-managers'
thousands of small new positions don't swamp the 70 other funds. For both: the
mean, with a t-stat clustered by event date (many events share a filing day),
and the median (fund-weighted: median of the funds' own means).

**Splits.** Fund cluster: the latest Milestone 5 clustering (fund2vec + HDBSCAN)
public by day 0, named by its members' majority hand style; noise funds and
2015 (before clusters) separately. Fund size: terciles of the long book's value
among seed funds that quarter. Multi-manager/quant vs other funds. Timing:
accepted during market hours vs after the close.

**Survivorship.** Yahoo has no prices at all for most stocks that later
delisted, so they can't enter; the report counts the dropped events and how
many of their stocks later left the exchange.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.analysis.edgar_times import OUT as FILING_TIMES
from src.features.task_c import CLUSTER_ALGO, CLUSTER_METHOD
from src.models import style_classifier
from src.models.task_c_common import MULTI, md_table
from src.securities import db

DOC = "docs/event_study.md"
CONCLUSION = "docs/event_study_conclusion.md"   # hand-written; included in DOC
FIG = "docs/figures/22_event_study_car.png"
OUT = "data/processed/event_study_cars.parquet"
WINDOWS = {"[-5,-1]": (-5, -1), "[0,+1]": (0, 1), "[+2,+10]": (2, 10), "[+11,+60]": (11, 60)}
PATH_DAYS = (-5, 60)
MARKET_CLOSE_HOUR = 16
DELIST_GAP_DAYS = 10
PERFORMANCE_DELIST_RET = -0.30
WINSOR = (0.005, 0.995)


# --- timing ------------------------------------------------------------------------------------

def day0_index(accepted, filing_date, trading_days: np.ndarray) -> tuple[int, str]:
    """Index into trading_days (sorted datetime64[D]) of day 0, and the timing flag."""
    if accepted is None:
        d = np.datetime64(filing_date, "D")
        return int(np.searchsorted(trading_days, d, side="right")), "fallback"
    d = np.datetime64(accepted.date(), "D")
    i = int(np.searchsorted(trading_days, d, side="left"))
    on_trading_day = i < len(trading_days) and trading_days[i] == d
    if on_trading_day and accepted.hour < MARKET_CLOSE_HOUR:
        return i, "intraday"
    return (i + 1 if on_trading_day else i), "after_close"


# --- statistics --------------------------------------------------------------------------------

def clustered_mean(x: np.ndarray, cluster: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
    """Weighted mean and its t-stat with standard errors clustered by `cluster`."""
    w = np.ones_like(x) if w is None else w
    m = float((w * x).sum() / w.sum())
    _, inv = np.unique(cluster, return_inverse=True)
    g = np.bincount(inv, weights=w * (x - m))
    n_g = len(g)
    if n_g < 2:
        return m, float("nan")
    se = np.sqrt((g ** 2).sum() * n_g / (n_g - 1)) / w.sum()
    return m, float(m / se) if se > 0 else float("nan")


def summarise(df: pl.DataFrame) -> dict:
    """Event- and fund-weighted statistics of `car` (one row per event, one window)."""
    x, cl = df["car"].to_numpy(), df["day0"].cast(pl.Utf8).to_numpy()
    n_f = df.group_by("fund_id").len()
    w = df.join(n_f, on="fund_id", how="left")["len"].to_numpy()
    ev_mean, ev_t = clustered_mean(x, cl)
    fw_mean, fw_t = clustered_mean(x, cl, 1 / w)
    fund_means = df.group_by("fund_id").agg(pl.col("car").mean())["car"]
    return {"events": len(df), "funds": df["fund_id"].n_unique(), "dates": df["day0"].n_unique(),
            "mean": ev_mean, "t": ev_t, "median": float(np.median(x)),
            "fund_wtd_mean": fw_mean, "fund_wtd_t": fw_t, "fund_median": float(fund_means.median())}


def winsorize(cars: pl.DataFrame) -> pl.DataFrame:
    """`car_raw` keeps the original; `car` is clipped at WINSOR quantiles within each window."""
    lo, hi = WINSOR
    return (cars.rename({"car": "car_raw"})
            .with_columns(pl.col("car_raw").clip(pl.col("car_raw").quantile(lo).over("window"),
                                                 pl.col("car_raw").quantile(hi).over("window")).alias("car")))


# --- data ---------------------------------------------------------------------------------------

def load_events(con, trading_days: list) -> pl.DataFrame:
    """New positions with the first filing that disclosed them, and day 0."""
    con.execute(f"CREATE OR REPLACE TEMP TABLE ft AS SELECT * FROM read_parquet('{FILING_TIMES}')")
    ev = con.execute("""
        WITH new AS (
            SELECT p.fund_id, p.fund_name, p.quarter, p.cusip, p.symbol, p.weight_long, p.value_usd
            FROM positions p JOIN security_master m USING (cusip)
            WHERE p.cohort = 'seed' AND p.put_call = '' AND p.is_new AND m.security_class IN ('common', 'adr')
        ),
        disclosed AS (   -- every seed filing of that fund-quarter listing the stock
            SELECT n.fund_id, n.quarter, n.cusip, s.cik, s.known_at
            FROM new n JOIN seed_positions_pit s
              ON s.fund_name = n.fund_name AND s.quarter = n.quarter AND s.cusip = n.cusip AND s.put_call = ''
        ),
        timed AS (
            SELECT d.*, ft.accepted_et, ft.source,
                   row_number() OVER (PARTITION BY d.fund_id, d.quarter, d.cusip
                                      ORDER BY coalesce(ft.accepted_et, d.known_at::TIMESTAMP + INTERVAL 1 DAY), d.cik) AS rn
            FROM disclosed d LEFT JOIN ft ON ft.cik = d.cik AND ft.quarter = d.quarter AND ft.filing_date = d.known_at
        )
        SELECT n.*, t.cik, t.known_at AS filing_date, t.accepted_et, coalesce(t.source, 'fallback') AS source
        FROM new n JOIN timed t USING (fund_id, quarter, cusip) WHERE t.rn = 1
    """).pl()
    days = np.array(trading_days, dtype="datetime64[D]")
    idx = [day0_index(a, f, days) for a, f in ev.select("accepted_et", "filing_date").iter_rows()]
    return ev.with_columns(pl.Series("day0_i", [i for i, _ in idx]), pl.Series("timing", [t for _, t in idx]))


def daily_abnormal(con, symbols: list[str], cal: pl.DataFrame) -> pl.DataFrame:
    """Creates the DuckDB temp table `ar` (symbol, i, ar = stock minus SPY) on each symbol's own
    span of history (a day the stock didn't trade returns 0: forward-filled close), and returns
    each symbol's span: symbol, first_i, last_i, last_close."""
    con.register("cal_df", cal.to_arrow())
    con.register("sym_df", pl.DataFrame({"symbol": symbols}).to_arrow())
    con.execute("CREATE OR REPLACE TEMP TABLE cal AS SELECT * FROM cal_df")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE ar AS
        WITH p AS (SELECT symbol, date, close FROM prices_daily WHERE symbol IN (SELECT symbol FROM sym_df) AND close > 0),
        span AS (SELECT symbol, min(date) AS d0, max(date) AS d1 FROM p GROUP BY 1),
        grid AS (SELECT s.symbol, c.i, c.date FROM span s JOIN cal c ON c.date BETWEEN s.d0 AND s.d1),
        filled AS (SELECT g.symbol, g.i, p.close FROM grid g ASOF JOIN p ON p.symbol = g.symbol AND p.date <= g.date),
        spy AS (SELECT c.i, p.close / lag(p.close) OVER (ORDER BY c.i) - 1 AS r_spy
                FROM cal c JOIN prices_daily p ON p.symbol = 'SPY' AND p.date = c.date)
        SELECT f.symbol, f.i, f.close, f.close / lag(f.close) OVER (PARTITION BY f.symbol ORDER BY f.i) - 1 - spy.r_spy AS ar
        FROM filled f JOIN spy USING (i)
    """)
    con.unregister("cal_df")
    con.unregister("sym_df")
    return con.execute("""SELECT symbol, min(i) AS first_i, max(i) AS last_i, arg_max(close, i) AS last_close
                          FROM ar GROUP BY symbol""").pl()


def sum_ar(con, rows: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Sum of `ar` over days lo..hi for each row (keys, symbol, lo, hi), via the DuckDB table `ar`."""
    con.register("rows_df", rows.select(keys + ["symbol", "lo", "hi"]).to_arrow())
    k = ", ".join(f"r.{c}" for c in keys)
    out = con.execute(f"""
        SELECT {k}, sum(a.ar) AS car_px
        FROM rows_df r, generate_series(r.lo, r.hi) g(i)
        JOIN ar a ON a.symbol = r.symbol AND a.i = g.i
        GROUP BY ALL
    """).pl()
    con.unregister("rows_df")
    return out


def delisting_terminal(con) -> pl.DataFrame:
    """cusip, delist_date, the return from the last close to the delisting value (needs the close)."""
    return con.execute(f"""
        SELECT cusip, delist_date, exit_class,
               CASE WHEN exit_class = 'performance' THEN 'performance'
                    WHEN terms_ok AND exit_class IN ('cash_acquisition', 'stock_acquisition') THEN 'deal'
                    ELSE 'cash' END AS kind,
               coalesce(cash, 0) + coalesce(ratio, 0) * coalesce(acq_px_delist, 0) AS deal_value
        FROM delisting_events WHERE delist_date IS NOT NULL
    """).pl().unique("cusip", keep="first")


def window_cars(con, events: pl.DataFrame, span: pl.DataFrame, delist: pl.DataFrame, cal: pl.DataFrame) -> pl.DataFrame:
    """One row per (event, window) with car, or the reason it was left out (`status`)."""
    e = (events.join(span, on="symbol", how="left").join(delist, on="cusip", how="left")
         .join(cal.rename({"date": "last_date", "i": "last_i"}), on="last_i", how="left"))
    n_days = cal["i"].max()
    delisted = (pl.col("delist_date").is_not_null()
                & ((pl.col("delist_date") - pl.col("last_date")).dt.total_days().is_between(0, DELIST_GAP_DAYS)))
    w = pl.concat([
        e.with_columns(pl.lit(name).alias("window"), (pl.col("day0_i") + a).alias("lo"), (pl.col("day0_i") + b).alias("hi"))
        for name, (a, b) in WINDOWS.items()])
    w = w.with_columns(
        pl.when(pl.col("first_i").is_null()).then(pl.lit("no daily prices"))
        .when(pl.col("hi") > n_days).then(pl.lit("window not over yet"))
        .when(pl.col("first_i") > pl.col("lo") - 1).then(pl.lit("price history starts too late"))
        .when(pl.col("last_i") >= pl.col("hi")).then(pl.lit("ok"))
        .when(delisted & (pl.col("last_i") >= pl.col("lo") - 1)).then(pl.lit("ok, delisting-adjusted"))
        .otherwise(pl.lit("price history ends, no delisting record")).alias("status"))
    keys = ["fund_id", "quarter", "cusip", "window"]
    ok = w.filter(pl.col("status").str.starts_with("ok"))
    term = (pl.when(pl.col("kind") == "performance").then(PERFORMANCE_DELIST_RET)
            .when(pl.col("kind") == "deal").then(pl.col("deal_value") / pl.col("last_close") - 1)
            .otherwise(0.0))
    ok = (ok.join(sum_ar(con, ok, keys), on=keys, how="left")
          .with_columns(pl.col("car_px").fill_null(0.0))
          .with_columns(pl.when(pl.col("status") == "ok, delisting-adjusted")
                        .then(pl.col("car_px") + term).otherwise(pl.col("car_px")).alias("car"))
          .drop("car_px"))
    left_out = w.filter(~pl.col("status").str.starts_with("ok")).with_columns(pl.lit(None, dtype=pl.Float64).alias("car"))
    return pl.concat([ok, left_out], how="diagonal")


def car_path(con, events: pl.DataFrame, span: pl.DataFrame) -> pl.DataFrame:
    """Average cumulative abnormal return from day -5 to +60, for events with full history over it."""
    a, b = PATH_DAYS
    e = (events.join(span, on="symbol").filter((pl.col("first_i") <= pl.col("day0_i") + a - 1)
                                               & (pl.col("last_i") >= pl.col("day0_i") + b)))
    con.register("ev_df", e.select("fund_id", "quarter", "cusip", "symbol", "day0_i").to_arrow())
    out = con.execute(f"""
        WITH d AS (
            SELECT e.fund_id, e.quarter, e.cusip, g.t, a.ar
            FROM ev_df e, generate_series({a}, {b}) g(t) JOIN ar a ON a.symbol = e.symbol AND a.i = e.day0_i + g.t
        ),
        c0 AS (SELECT fund_id, t, sum(ar) OVER (PARTITION BY fund_id, quarter, cusip ORDER BY t) AS cum FROM d),
        q AS (SELECT t, quantile_cont(cum, {WINSOR[0]}) AS lo, quantile_cont(cum, {WINSOR[1]}) AS hi FROM c0 GROUP BY t),
        c AS (SELECT c0.fund_id, c0.t, greatest(least(c0.cum, q.hi), q.lo) AS cum FROM c0 JOIN q USING (t)),
        n AS (SELECT fund_id, count(*) AS n FROM ev_df GROUP BY 1)
        SELECT t, avg(cum) AS event_weighted, sum(cum / n) / sum(1.0 / n) AS fund_weighted
        FROM c JOIN n USING (fund_id) GROUP BY t ORDER BY t
    """).pl()
    con.unregister("ev_df")
    return out


def fund_attributes(con, events: pl.DataFrame, cal: pl.DataFrame) -> pl.DataFrame:
    """Per (fund, quarter): cluster name known by day 0, size tercile, multi-manager/quant flag."""
    styles = dict(style_classifier.labels().select("fund_id", "label_v2").iter_rows())
    cl = con.execute(f"""SELECT fund_id, quarter AS cl_quarter, cluster, known_at FROM fund_clusters
                         WHERE method = '{CLUSTER_METHOD}' AND algo = '{CLUSTER_ALGO}'""").pl()
    names = (cl.filter(pl.col("cluster") >= 0)
             .with_columns(pl.col("fund_id").replace_strict(styles, default=None).alias("style"))
             .drop_nulls("style").group_by("cl_quarter", "cluster", "style").len()
             .sort(["cl_quarter", "cluster", "len", "style"], descending=[False, False, True, False])
             .group_by("cl_quarter", "cluster").first().select("cl_quarter", "cluster", pl.col("style").alias("cl_name")))
    fq = (events.select("fund_id", "quarter", "day0_i").group_by("fund_id", "quarter").agg(pl.col("day0_i").min())
          .join(cal.rename({"i": "day0_i", "date": "day0"}), on="day0_i", how="left"))
    best = (fq.join(cl, on="fund_id").filter((pl.col("known_at") <= pl.col("day0")) & (pl.col("cl_quarter") <= pl.col("quarter")))
            .sort("cl_quarter").group_by("fund_id", "quarter").last()
            .join(names, on=["cl_quarter", "cluster"], how="left")
            .with_columns(pl.when(pl.col("cluster") < 0).then(pl.lit("no cluster (noise)"))
                          .otherwise(pl.col("cl_name")).alias("cluster_name"))
            .select("fund_id", "quarter", "cluster_name"))
    size = (con.execute("SELECT fund_id, quarter, long_value FROM fund_quarter WHERE cohort = 'seed'").pl()
            .with_columns(pl.col("long_value").rank().over("quarter").alias("rk"), pl.len().over("quarter").alias("n"))
            .with_columns(pl.when(pl.col("rk") <= pl.col("n") / 3).then(pl.lit("small"))
                          .when(pl.col("rk") <= 2 * pl.col("n") / 3).then(pl.lit("mid")).otherwise(pl.lit("large"))
                          .alias("size_tercile"))
            .select("fund_id", "quarter", "size_tercile"))
    return (fq.select("fund_id", "quarter").join(best, on=["fund_id", "quarter"], how="left")
            .join(size, on=["fund_id", "quarter"], how="left")
            .with_columns(pl.col("cluster_name").fill_null("none yet (2015)"),
                          pl.col("fund_id").replace_strict(styles, default=None).is_in(MULTI).alias("multi_quant")))


# --- report ------------------------------------------------------------------------------------

def table(cars: pl.DataFrame, by: str | None = None) -> pl.DataFrame:
    rows = []
    ok = cars.filter(pl.col("car").is_not_null())
    groups = [(None, ok)] if by is None else sorted(ok.group_by(by), key=lambda kv: str(kv[0][0]))
    for key, g in groups:
        for name in WINDOWS:
            sub = g.filter(pl.col("window") == name)
            if len(sub) < 30:
                continue
            rows.append(({by: str(key[0])} if by else {}) | {"window": name} | summarise(sub))
    return pl.DataFrame(rows)


def figure(path_df: pl.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(path_df["t"], path_df["event_weighted"] * 100, label="event-weighted", color="#2a6f97", lw=2)
    ax.plot(path_df["t"], path_df["fund_weighted"] * 100, label="fund-weighted", color="#c1121f", lw=2, ls="--")
    ax.axvline(0, color="grey", lw=1)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("trading days relative to the 13F becoming public (day 0)")
    ax.set_ylabel("cumulative abnormal return vs SPY (%)")
    ax.set_title("New 13F positions of seed funds, 2015Q2-2026Q2: return after disclosure")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    with db.connect(read_only=True) as con:
        days = [d for (d,) in con.execute("SELECT date FROM prices_daily WHERE symbol = 'SPY' ORDER BY 1").fetchall()]
        cal = pl.DataFrame({"i": range(len(days)), "date": days})
        events = load_events(con, days).join(cal.rename({"i": "day0_i", "date": "day0"}), on="day0_i", how="left")
        span = daily_abnormal(con, events["symbol"].drop_nulls().unique().to_list(), cal)
        events = events.with_columns(pl.col("symbol").is_in(span["symbol"].implode()).fill_null(False).alias("priced"))
        cars = window_cars(con, events, span, delisting_terminal(con), cal)
        path_df = car_path(con, events, span)
        attrs = fund_attributes(con, events, cal)
        delisted_later = set(con.execute("SELECT DISTINCT cusip FROM delisting_events WHERE exit_class IN "
                                         "('cash_acquisition', 'stock_acquisition', 'performance', 'other')").pl()["cusip"])
    cars = winsorize(cars.join(attrs, on=["fund_id", "quarter"], how="left"))
    cars.write_parquet(OUT)
    figure(path_df, FIG)
    write_doc(events, cars, path_df, delisted_later)


def write_doc(events: pl.DataFrame, cars: pl.DataFrame, path_df: pl.DataFrame, delisted_later: set) -> None:
    main_tab = table(cars)
    cols = ["window", "events", "funds", "mean", "t", "median", "fund_wtd_mean", "fund_wtd_t", "fund_median"]
    timing = events.group_by("timing").len().sort("timing")
    status = (cars.group_by("window", "status").len().pivot(on="window", index="status", values="len")
              .select(["status"] + list(WINDOWS)).sort("status"))
    later = events.with_columns(pl.col("cusip").is_in(list(delisted_later)).alias("later_delisted"))
    no_px = later.filter(~pl.col("priced"))
    cov = (later.with_columns(pl.col("quarter").str.slice(0, 4).alias("year"))
           .group_by("year").agg(pl.len().alias("events"), pl.col("priced").mean().alias("priced_share"),
                                 (pl.col("timing") == "fallback").sum().alias("fallback_timing"))
           .sort("year"))
    surv = later.group_by("priced").agg(pl.len().alias("events"), pl.col("later_delisted").mean().alias("later_delisted_share"))
    at = lambda t: path_df.filter(pl.col("t") == t).row(0, named=True)
    text = f"""# Copycat event study: is 13F news priced in when it becomes public?

Generated by `src/analysis/event_study.py` (definitions in its docstring). Figure:
`docs/figures/22_event_study_car.png`. Acceptance times: `src/analysis/edgar_times.py` (EDGAR filing headers).

## Answer

{Path(CONCLUSION).read_text().strip() if Path(CONCLUSION).exists() else "(not written yet)"}

**Events:** {len(events):,} new positions (common stock and ADRs) in {events['fund_id'].n_unique()} seed funds' 13F filings,
2015Q2-2026Q2. Abnormal return = stock minus SPY, summed over each window of trading days around day 0, the first
trading day the filing could be traded on.

## Results

Abnormal returns winsorized at the 0.5th/99.5th percentiles within each window (see the docstring; raw below).
Event-weighted (`mean`, `t`, `median`) and fund-weighted (`fund_wtd_*`: each fund's events in a window share a
weight of 1; `fund_median` = median of the funds' own means). t-stats clustered by event date. Returns are
fractions (0.010 = 1.0%).

{md_table(main_tab.select(cols), 4)}

Cumulative path (figure 22): day +1 {at(1)['event_weighted']:.4f} event-weighted / {at(1)['fund_weighted']:.4f} fund-weighted;
day +10 {at(10)['event_weighted']:.4f} / {at(10)['fund_weighted']:.4f}; day +60 {at(60)['event_weighted']:.4f} / {at(60)['fund_weighted']:.4f}
(events with full price history over -5..+60).

Raw, not winsorized (the same events):

{md_table(table(cars.drop("car").rename({"car_raw": "car"})).select(cols), 4)}

## By fund cluster (as known on day 0)

{md_table(table(cars, 'cluster_name').select(['cluster_name'] + cols), 4)}

## By fund size (tercile of long-book value among seed funds that quarter)

{md_table(table(cars, 'size_tercile').select(['size_tercile'] + cols), 4)}

## Multi-manager / quant funds vs the rest

{md_table(table(cars, 'multi_quant').select(['multi_quant'] + cols), 4)}

## By filing time

`intraday`: accepted during market hours, so day 0 is the filing day itself and its close-to-close return starts
before the filing. `after_close`: accepted at or after 16:00 (or on a non-trading day).

{md_table(table(cars, 'timing').select(['timing'] + cols), 4)}

## Coverage and survivorship

Events by timing source (`fallback` = no EDGAR header, filing date used, day 0 = next trading day):

{md_table(timing)}

Events kept and left out, per window:

{md_table(status)}

Events with daily prices, by year of the filing quarter:

{md_table(cov, 3)}

**Survivorship:** {len(no_px):,} events ({len(no_px) / len(events):.1%}) have no daily prices at all. Share of events whose
stock later left the exchange (acquisition, performance or other delisting recorded in `delisting_events`), by
whether it is priced:

{md_table(surv, 3)}

`delisting_events` only covers stocks without Yahoo prices (that is how the delisting work was scoped), so the
delisting adjustment applies to very few kept events; the bias runs through the dropped ones.
"""
    Path(DOC).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
