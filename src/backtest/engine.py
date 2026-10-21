"""Backtest engine: rebalance calendar, daily returns per stock, portfolio simulation.

**Calendar.** One rebalance per quarter q. A 13F is due 45 days after quarter
end (the next business day if that is a holiday or weekend). A portfolio for q
uses only filings with known_at on or before that deadline (`cutoff`), and
trades at the close of the next trading day (`entry`), so every filing it uses
was public before the trade. Late filers wait until the next quarter. The
position is held, weights drifting, until the next quarter's entry.

**Daily returns, per stock and calendar quarter Q** (`source`):
  daily    Yahoo adjusted closes (dividends included) when the symbol has a
           close on at least half of Q's trading days; a missing day returns 0
  filled   otherwise the filled quarterly return for Q (stock_quarter row of
           Q-1: 13F-implied or delisting return, src.securities.labels) spread
           evenly over Q's trading days; a delisting return is spread up to the
           delisting date, then the position sits in cash (0)
           After a stock's delisting date it is cash for the rest of the window
           (counted as `filled`: its return is known, the delisting return then 0)
  missing  no return at all: 0% (cash) by default; `missing_sold_out` sets the
           return of `missing:sold_out` names (every seed fund sold them, so
           likely losers) for a sensitivity run
The filled returns run quarter end to quarter end while positions are held
filing date to filing date, so the timing of unpriced names is approximate.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

DEADLINE_DAYS = 45
DAILY_MIN_SHARE = 0.5
SOURCE_RANK = {"daily": 0, "filled": 1, "missing": 2}


def quarter_of(d: pl.Expr) -> pl.Expr:
    return pl.concat_str(d.dt.year().cast(pl.Utf8), pl.lit("Q"), ((d.dt.month() - 1) // 3 + 1).cast(pl.Utf8))


def prev_quarter(q: str) -> str:
    y, n = int(q[:4]), int(q[-1])
    return f"{y - 1}Q4" if n == 1 else f"{y}Q{n - 1}"


def quarter_end(q: str) -> date:
    y, n = int(q[:4]), int(q[-1])
    return date(y + (n == 4), 1 if n == 4 else 3 * n + 1, 1) - timedelta(days=1)


def calendar(days: list[date], quarters: list[str]) -> pl.DataFrame:
    """quarter, cutoff (last filing date used), entry (trade date), exit (next entry or the last day)."""
    arr = np.array(days, dtype="datetime64[D]")
    rows = []
    for q in quarters:
        deadline = np.datetime64(quarter_end(q) + timedelta(days=DEADLINE_DAYS))
        i = int(np.searchsorted(arr, deadline))           # first trading day on or after the deadline
        if i + 1 >= len(arr):
            continue
        rows.append({"quarter": q, "cutoff": days[i], "entry": days[i + 1]})
    cal = pl.DataFrame(rows)
    return cal.with_columns(pl.col("entry").shift(-1).fill_null(days[-1]).alias("exit"))


@dataclass
class ReturnData:
    days: list[date]
    prices: pl.DataFrame      # symbol, date, r
    counts: pl.DataFrame      # symbol, Q, n_days
    q_days: pl.DataFrame      # Q, n_trading (trading days in Q)
    filled: pl.DataFrame      # cusip, Q, R, ret_source, delist_date
    symbols: pl.DataFrame     # cusip, symbol
    delist: pl.DataFrame      # cusip, delisted_on (SEC Form 25/15 date; public that day)


def load_returns(con, start: date) -> ReturnData:
    days = [d for (d,) in con.execute("SELECT date FROM prices_daily WHERE symbol = 'SPY' AND date >= ? ORDER BY 1",
                                      [start]).fetchall()]
    symbols = con.execute("SELECT DISTINCT cusip, symbol FROM security_master WHERE symbol IS NOT NULL").pl()
    prices = con.execute("""
        SELECT symbol, date, close / lag(close) OVER (PARTITION BY symbol ORDER BY date) - 1 AS r
        FROM prices_daily WHERE date >= ? - INTERVAL 10 DAY
          AND symbol IN (SELECT symbol FROM security_master WHERE security_class IN ('common', 'adr'))
    """, [start]).pl().drop_nulls("r")
    counts = prices.group_by("symbol", quarter_of(pl.col("date")).alias("Q")).agg(pl.len().alias("n_days"))
    q_days = (pl.DataFrame({"date": days}).group_by(quarter_of(pl.col("date")).alias("Q"))
              .agg(pl.len().alias("n_trading")))
    filled = con.execute("""
        SELECT s.cusip, s.quarter AS q_prev, s.ret_next_q AS R, s.ret_source, d.delist_date
        FROM stock_quarter s LEFT JOIN delisting_events d ON d.cusip = s.cusip AND d.quarter = s.quarter
    """).pl()
    filled = filled.with_columns(pl.col("q_prev").map_elements(_next_q, return_dtype=pl.Utf8).alias("Q"))
    delist = con.execute("""SELECT cusip, min(delist_date) AS delisted_on FROM delisting_events
                             WHERE delist_date IS NOT NULL GROUP BY 1""").pl()
    return ReturnData(days, prices, counts, q_days, filled.drop("q_prev"), symbols, delist)


def _next_q(q: str) -> str:
    y, n = int(q[:4]), int(q[-1])
    return f"{y + 1}Q1" if n == 4 else f"{y}Q{n + 1}"


def window_returns(rd: ReturnData, cusips: list[str], entry: date, exit_: date,
                   missing_sold_out: float = 0.0) -> tuple[np.ndarray, pl.DataFrame]:
    """Days x stocks matrix of daily returns over (entry, exit], and each stock's worst source."""
    days = pl.DataFrame({"date": [d for d in rd.days if entry < d <= exit_]}).with_columns(quarter_of(pl.col("date")).alias("Q"))
    stocks = pl.DataFrame({"cusip": cusips}).join(rd.symbols, on="cusip", how="left").unique("cusip")
    q_start = pl.DataFrame({"date": rd.days}).group_by(quarter_of(pl.col("date")).alias("Q")).agg(pl.col("date").min().alias("q_first"))
    seg = (stocks.join(days.select("Q").unique(), how="cross")
           .join(rd.counts, on=["symbol", "Q"], how="left").join(rd.q_days, on="Q", how="left")
           .with_columns((pl.col("n_days").fill_null(0) >= DAILY_MIN_SHARE * pl.col("n_trading")).alias("is_daily"))
           .join(rd.filled, on=["cusip", "Q"], how="left")
           .join(rd.delist, on="cusip", how="left")
           .join(q_start, on="Q", how="left")
           # a quarter that starts after the delisting: the position is cash
           .with_columns((pl.col("delisted_on") < pl.col("q_first")).fill_null(False).alias("gone"))
           .with_columns(pl.when(pl.col("gone")).then(None).otherwise(pl.col("R")).alias("R"),
                         (pl.col("is_daily") & ~pl.col("gone")).alias("is_daily"))
           .with_columns(pl.when(pl.col("is_daily")).then(pl.lit("daily"))
                         .when(pl.col("gone") | pl.col("R").is_not_null()).then(pl.lit("filled"))
                         .otherwise(pl.lit("missing")).alias("source")))
    seg = seg.with_columns(
        pl.when(pl.col("gone")).then(0.0)
        .when(pl.col("source") == "missing")
        .then(pl.when(pl.col("ret_source") == "missing:sold_out").then(pl.lit(missing_sold_out)).otherwise(0.0))
        .otherwise(pl.col("R")).alias("R_used"))
    # filled and missing: spread over the quarter's trading days (up to the delisting date for delistings)
    all_days = pl.DataFrame({"date": rd.days}).with_columns(quarter_of(pl.col("date")).alias("Q"))
    spread = seg.filter(~pl.col("is_daily")).join(all_days, on="Q").filter(
        pl.col("delist_date").is_null() | (pl.col("date") <= pl.col("delist_date")))
    n_spread = spread.group_by("cusip", "Q").agg(pl.len().alias("n"))
    spread = (spread.join(n_spread, on=["cusip", "Q"])
              .with_columns(((1 + pl.col("R_used")) ** (1 / pl.col("n")) - 1).alias("r"))
              .join(days.select("date"), on="date").select("date", "cusip", "r"))
    daily = (seg.filter(pl.col("is_daily")).select("cusip", "symbol", "Q")
             .join(days, on="Q").join(rd.prices, on=["symbol", "date"], how="left")
             .select("date", "cusip", pl.col("r").fill_null(0.0)))
    long = pl.concat([daily, spread])
    grid = (days.select("date").join(pl.DataFrame({"cusip": cusips}), how="cross")
            .join(long, on=["date", "cusip"], how="left").with_columns(pl.col("r").fill_null(0.0)))
    mat = (grid.pivot(on="cusip", index="date", values="r").sort("date").select(cusips).to_numpy())
    src = (seg.with_columns(pl.col("source").replace_strict(SOURCE_RANK).alias("rk"))
           .group_by("cusip").agg(pl.col("rk").max())
           .with_columns(pl.col("rk").replace_strict({v: k for k, v in SOURCE_RANK.items()}).alias("source")))
    return mat, src.select("cusip", "source")


def simulate(windows: list[dict], weights: dict[str, dict[str, float]], rd: ReturnData, cost_bps: float,
             missing_sold_out: float = 0.0, cache: dict | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Daily net returns of a quarterly-rebalanced, buy-and-hold-within-quarter portfolio.

    windows: [{quarter, entry, exit}]; weights: quarter -> {cusip: weight} (missing quarter = cash).
    Costs: cost_bps per side on sum |w_new - w_drifted|, charged on the entry day.
    Returns (daily: date, r, r_gross) and (per-rebalance: quarter, turnover, source shares)."""
    drift: dict[str, float] = {}
    daily, rebal = [], []
    for w in windows:
        target = weights.get(w["quarter"], {})
        names = sorted(target)
        traded = sum(abs(target.get(c, 0.0) - drift.get(c, 0.0)) for c in set(target) | set(drift))
        key = (w["quarter"], tuple(names), missing_sold_out)
        if names:
            if cache is not None and key in cache:
                mat, src = cache[key]
            else:
                mat, src = window_returns(rd, names, w["entry"], w["exit"], missing_sold_out)
                if cache is not None:
                    cache[key] = (mat, src)
            wv = np.array([target[c] for c in names])
            growth = np.cumprod(1 + mat, axis=0)
            value = growth @ wv
            r = np.diff(np.concatenate([[1.0], value])) / np.concatenate([[1.0], value[:-1]])
            drift = dict(zip(names, wv * growth[-1] / value[-1])) if len(value) else dict(zip(names, wv))
            shares = (src.join(pl.DataFrame({"cusip": names, "w": wv}), on="cusip")
                      .group_by("source").agg(pl.col("w").sum()))
            share = {s: shares.filter(pl.col("source") == s)["w"].sum() for s in SOURCE_RANK}
        else:
            n = sum(1 for d in rd.days if w["entry"] < d <= w["exit"])
            r, drift, share = np.zeros(n), {}, {s: 0.0 for s in SOURCE_RANK}
        gross = r.copy()
        if len(r):
            r[0] = (1 + r[0]) * (1 - cost_bps / 1e4 * traded) - 1
        dates = [d for d in rd.days if w["entry"] < d <= w["exit"]]
        daily.append(pl.DataFrame({"date": dates, "r": r, "r_gross": gross}))
        rebal.append({"quarter": w["quarter"], "entry": w["entry"], "n_names": len(names), "traded": traded} |
                     {f"w_{s}": share[s] for s in SOURCE_RANK})
    return pl.concat(daily), pl.DataFrame(rebal)
