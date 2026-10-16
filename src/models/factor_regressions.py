"""Returns-based factor regressions per fund (Milestone 4, Task A): FF5 + momentum
loadings of each fund's 13F long book, compared with its holdings-based tilts.

Usage:
    uv run python -m src.models.factor_regressions

Hedge funds don't publish returns, so the return series is the **13F clone**:
the long equity book reported for quarter q (non-option rows, `weight_long`)
bought at the quarter-end close and held, untouched, through the three months
of quarter q+1. Weights drift with prices inside the quarter. A stock whose
monthly return is missing (no price; Yahoo drops most delisted stocks) leaves
the book from that month on and the rest is renormalised; `coverage` is the
starting weight still priced, and months below MIN_COVERAGE are dropped.

Monthly excess clone returns are regressed on Mkt-RF, SMB, HML, RMW, CMA and
Mom (French's monthly factors), 2018Q2 on (the first month after the 2018Q1
book, matching the modelling period), with Newey-West t-statistics (3 lags).
Funds need MIN_MONTHS months.

Why this is a fair comparison with the holdings-based tilts: both describe the
same portfolio (the long 13F book), one from the characteristics of the stocks
(z-scores of size, B/M, 12-1 momentum, gross profitability) and one from how
its returns co-move with long-short factor portfolios. Expected signs: tilt_size
vs SMB negative (tilt_size is big = large caps), value vs HML, momentum vs Mom,
profitability vs RMW positive. Where they disagree, the clone's return also
reflects things the quarter-end characteristics miss (trading inside the
quarter is invisible; the factors are rebalanced once a year for HML/RMW).

Delisted stocks: a stock with no Yahoo prices (acquired or delisted since)
enters with its filled quarterly return (13F-implied, or the delisting
return; src.securities.labels) spread evenly over the three months. Without
this, 12 seed funds (merger arbitrage, event-driven, private biotech) fell
below MIN_COVERAGE. The Yahoo-only loadings are kept in
`fund_factor_loadings_yahoo_only`.

Not point-in-time, on purpose: a full-period regression uses every month of
returns, so it is a **description** of a fund, never a model feature. Nothing
in src/models/style_classifier.py reads it. The clone also ignores shorts,
leverage, options and non-US assets, so the market beta is the long book's,
not the fund's.

Writes `fund_factor_loadings` (one row per fund), `fund_clone_returns_monthly`,
docs/figures/09_returns_vs_holdings_tilts.png and docs/factor_regressions.md.
"""

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.models.style_classifier import labels
from src.securities import db

FACTORS = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]
FIRST_BOOK = "2018Q1"
MIN_COVERAGE = 0.6      # share of the starting book still priced in the month
MIN_MONTHS = 36
NW_LAGS = 3
# holdings tilt -> factor it should line up with, and the expected sign
PAIRS = [("tilt_size", "SMB", -1), ("tilt_value", "HML", 1), ("tilt_momentum", "Mom", 1),
         ("tilt_profitability", "RMW", 1)]
FIG = Path("docs/figures")
REPORT = Path("docs/factor_regressions.md")
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE = "#2a78d6"


def monthly_stock_returns(con) -> pl.DataFrame:
    """symbol, month (first of month), ret: month-end to month-end adjusted close,
    only when the previous month-end exists (no returns spanning a gap)."""
    return con.execute(f"""
        WITH held AS (SELECT DISTINCT symbol FROM positions WHERE symbol IS NOT NULL AND quarter >= '{FIRST_BOOK}'),
        me AS (
            SELECT p.symbol, date_trunc('month', p.date)::DATE AS month, arg_max(p.close, p.date) AS close
            FROM prices_daily p JOIN held USING (symbol)
            WHERE p.date >= DATE '{FIRST_BOOK[:4]}-01-01' - INTERVAL 1 MONTH AND p.close > 0
            GROUP BY ALL
        )
        SELECT symbol, month,
               close / lag(close) OVER w - 1 AS ret,
               lag(month) OVER w = month - INTERVAL 1 MONTH AS consecutive
        FROM me WINDOW w AS (PARTITION BY symbol ORDER BY month)
        QUALIFY consecutive AND ret IS NOT NULL
    """).pl().drop("consecutive")


def filled_monthly_returns(con, yahoo: pl.DataFrame) -> pl.DataFrame:
    """Monthly returns for stocks Yahoo doesn't price, from the filled quarterly label
    (13F-implied or delisting return; src.securities.labels): each of the three months
    of q+1 gets (1 + R)^(1/3) - 1. The monthly path inside the quarter is unknown, so
    spreading it evenly smooths these stocks (biases their betas slightly toward 0).
    `big_move_few_funds` rows stay out: one unverified 20x return can swing a fund.
    Only (symbol, month) pairs with no Yahoo return are added."""
    if "ret_source" not in {r[0] for r in con.execute("DESCRIBE stock_quarter").fetchall()}:
        return yahoo.clear()    # labels not filled yet (src.securities.labels): Yahoo only
    q = con.execute("""
        SELECT symbol, quarter, avg(ret_next_q) AS ret_q
        FROM stock_quarter
        WHERE ret_source <> 'yahoo' AND ret_source NOT LIKE 'missing:%' AND symbol IS NOT NULL
          AND coalesce(label_flag, '') <> 'big_move_few_funds'
        GROUP BY ALL
    """).pl()
    rows = [(sym, m, (1 + r) ** (1 / 3) - 1) for sym, quarter, r in q.iter_rows() for m in holding_months(quarter)]
    filled = pl.DataFrame(rows, schema={"symbol": pl.Utf8, "month": pl.Date, "ret": pl.Float64}, orient="row")
    return filled.join(yahoo.select("symbol", "month"), on=["symbol", "month"], how="anti")


def books(con) -> pl.DataFrame:
    """fund_id, cohort, quarter, symbol, w: the long book, weights summing to 1 per fund-quarter
    over all non-option rows (unpriced and unmapped rows included, so coverage is honest)."""
    return con.execute(f"""
        SELECT fund_id, cohort, quarter, symbol, sum(weight_long) AS w
        FROM positions WHERE NOT is_option AND quarter >= '{FIRST_BOOK}' AND weight_long > 0
        GROUP BY ALL
    """).pl().with_columns((pl.col("w") / pl.col("w").sum().over("fund_id", "quarter")).alias("w"))


def holding_months(quarter: str) -> list[date]:
    """The three months (first of month) of the quarter after `quarter`."""
    y, q = int(quarter[:4]), int(quarter[-1])
    y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return [date(y, 3 * (q - 1) + k, 1) for k in (1, 2, 3)]


def clone_returns(book: pl.DataFrame, rets: pl.DataFrame) -> pl.DataFrame:
    """Buy-and-hold monthly return of each fund-quarter book over the next quarter.

    For stock i in holding month k: value v_ik = w_i * prod_{j<k}(1 + r_ij), defined
    only if r_i1..r_ik are all present (a missing month removes the stock from then
    on). Portfolio return_k = sum v_ik r_ik / sum v_ik; coverage_k = sum of w_i of
    the stocks still in."""
    months = pl.DataFrame(
        [(q, m, k) for q in book["quarter"].unique() for k, m in enumerate(holding_months(q), 1)],
        schema={"quarter": pl.Utf8, "month": pl.Date, "k": pl.Int64}, orient="row")
    grid = book.join(months, on="quarter").join(rets, on=["symbol", "month"], how="left").sort(
        "fund_id", "quarter", "symbol", "k")
    grid = grid.with_columns(
        # the stock is in month k only if months 1..k all have a return
        (pl.col("ret").is_null().cast(pl.Int32).cum_sum().over("fund_id", "quarter", "symbol") == 0).alias("alive"),
        (1 + pl.col("ret").fill_null(0)).cum_prod().shift(1, fill_value=1.0)
        .over("fund_id", "quarter", "symbol").alias("growth"),
    ).filter(pl.col("alive")).with_columns((pl.col("w") * pl.col("growth")).alias("v"))
    return (grid.group_by("fund_id", "cohort", "quarter", "month")
            .agg(((pl.col("v") * pl.col("ret")).sum() / pl.col("v").sum()).alias("ret"),
                 pl.col("w").sum().alias("coverage"), pl.len().alias("n_stocks"))
            .sort("fund_id", "month"))


def ols_newey_west(y: np.ndarray, X: np.ndarray, lags: int = NW_LAGS) -> tuple[np.ndarray, np.ndarray, float]:
    """Coefficients (intercept first), Newey-West (Bartlett) t-stats, R^2."""
    X = np.column_stack([np.ones(len(y)), X])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    e = y - X @ beta
    u = X * e[:, None]
    s = u.T @ u
    for lag in range(1, lags + 1):
        g = u[lag:].T @ u[:-lag]
        s += (1 - lag / (lags + 1)) * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return beta, beta / np.sqrt(np.diag(cov)), r2


def regress(clone: pl.DataFrame, factors: pl.DataFrame) -> pl.DataFrame:
    df = clone.filter(pl.col("coverage") >= MIN_COVERAGE).join(
        factors.select(pl.col("date").alias("month"), *FACTORS, "RF"), on="month", how="inner")
    rows = []
    for (fid, cohort), g in df.group_by("fund_id", "cohort"):
        if len(g) < MIN_MONTHS:
            continue
        y = (g["ret"] - g["RF"]).to_numpy()
        beta, t, r2 = ols_newey_west(y, g.select(FACTORS).to_numpy())
        rows.append({"fund_id": fid, "cohort": cohort, "n_months": len(g),
                     "first_month": g["month"].min(), "last_month": g["month"].max(),
                     "alpha_ann": beta[0] * 12, "t_alpha": t[0],
                     **{f"b_{f}": b for f, b in zip(FACTORS, beta[1:])},
                     **{f"t_{f}": v for f, v in zip(FACTORS, t[1:])},
                     "r2": r2, "avg_coverage": g["coverage"].mean()})
    return pl.DataFrame(rows).sort("fund_id")


def mean_tilts(con, loadings: pl.DataFrame) -> pl.DataFrame:
    """Average holdings tilt per fund over the books whose clone months entered the regression."""
    tilts = con.execute(f"""
        SELECT fund_id, fund_name, quarter, {', '.join(t for t, _, _ in PAIRS)}
        FROM fund_fingerprint WHERE quarter >= '{FIRST_BOOK}'
    """).pl()
    return loadings.join(tilts, on="fund_id").filter(
        pl.col("quarter").map_elements(lambda q: holding_months(q)[0], return_dtype=pl.Date)
        .is_between(pl.col("first_month"), pl.col("last_month"))
    ).group_by("fund_id").agg(pl.col("fund_name").first(), *[pl.col(t).mean() for t, _, _ in PAIRS])


def spearman(a: pl.Series, b: pl.Series) -> float:
    return float(np.corrcoef(a.rank().to_numpy(), b.rank().to_numpy())[0, 1])


def comparison(both: pl.DataFrame) -> pl.DataFrame:
    """Cross-fund Spearman correlation between each holdings tilt and its factor loading."""
    rows = []
    for tilt, factor, sign in PAIRS:
        d = both.drop_nulls([tilt, f"b_{factor}"])
        rho = spearman(d[tilt], d[f"b_{factor}"])
        agree = ((np.sign(d[tilt]) * sign) == np.sign(d[f"b_{factor}"])).mean()
        rows.append({"holdings tilt": tilt, "factor loading": f"b_{factor}", "expected sign": "+" if sign > 0 else "−",
                     "spearman": round(rho, 2), "same sign share": round(float(agree), 2), "funds": len(d)})
    return pl.DataFrame(rows)


def plot(both: pl.DataFrame, comp: pl.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), facecolor=SURFACE)
    for ax, (tilt, factor, _), row in zip(axes, PAIRS, comp.iter_rows(named=True)):
        d = both.drop_nulls([tilt, f"b_{factor}"])
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=GRID, linewidth=1, zorder=1)
        ax.axvline(0, color=GRID, linewidth=1, zorder=1)
        ax.scatter(d[tilt], d[f"b_{factor}"], s=34, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        # label the two funds furthest out on each axis
        for col in (tilt, f"b_{factor}"):
            for r in d.sort(pl.col(col).abs(), descending=True).head(2).iter_rows(named=True):
                ax.annotate(r["fund_name"].split()[0], (r[tilt], r[f"b_{factor}"]), fontsize=7, color=INK2,
                            xytext=(4, 3), textcoords="offset points")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_xlabel(f"holdings {tilt.removeprefix('tilt_')} tilt (z)", color=INK2)
        ax.set_ylabel(f"{factor} loading", color=INK2)
        ax.set_title(f"{tilt.removeprefix('tilt_')} vs {factor}: ρ = {row['spearman']:+.2f}",
                     loc="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.suptitle("Holdings-based tilts vs returns-based FF5+Mom loadings, same 13F long book (seed funds)",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.01, -0.04, f"One dot per fund. x: mean value-weighted characteristic z-score of the book, 2018+. "
             f"y: loading from monthly clone returns (quarter-end book held for the next quarter), "
             f"Newey-West OLS on Mkt-RF, SMB, HML, RMW, CMA, Mom. ρ: Spearman across funds. "
             f"tilt_size is large-cap positive, so it should run against SMB.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def md_table(df: pl.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def recovered_funds(con, clone_yahoo, clone, loadings_yahoo, loadings) -> pl.DataFrame:
    """Funds with too few covered months on Yahoo prices alone, before and after the fill."""
    names = con.execute("SELECT DISTINCT fund_id, fund_name FROM positions").pl()
    lost = (clone_yahoo.select("fund_id", "cohort").unique()
            .join(loadings_yahoo.select("fund_id"), on="fund_id", how="anti"))

    def months(c, col):
        return (c.filter(pl.col("coverage") >= MIN_COVERAGE).group_by("fund_id").len().rename({"len": col}))

    def cov(c, col):
        return c.group_by("fund_id").agg(pl.col("coverage").median().alias(col))

    return (lost.join(names, on="fund_id", how="left")
            .join(months(clone_yahoo, "months_yahoo"), on="fund_id", how="left")
            .join(months(clone, "months_filled"), on="fund_id", how="left")
            .join(cov(clone_yahoo, "median_cov_yahoo"), on="fund_id", how="left")
            .join(cov(clone, "median_cov_filled"), on="fund_id", how="left")
            .join(loadings.select("fund_id", "alpha_ann", *[f"b_{f}" for f in FACTORS], "r2"), on="fund_id", how="left")
            .with_columns(pl.col("months_yahoo", "months_filled").fill_null(0),
                          (pl.col("r2").is_not_null()).alias("regressed_now"))
            .sort("cohort", "fund_name"))


def main() -> None:
    con = db.connect()
    yahoo = monthly_stock_returns(con)
    book = books(con)
    factors = con.execute("SELECT * FROM ff_factors_monthly").pl()
    # Yahoo-only clone (the Milestone 4 result), kept to show what the delisting fill changed
    clone_yahoo = clone_returns(book, yahoo)
    loadings_yahoo = regress(clone_yahoo, factors)
    clone = clone_returns(book, pl.concat([yahoo, filled_monthly_returns(con, yahoo)]))
    loadings = regress(clone, factors)
    recovered = recovered_funds(con, clone_yahoo, clone, loadings_yahoo, loadings)
    con.execute("CREATE OR REPLACE TABLE fund_factor_loadings_yahoo_only AS SELECT * FROM loadings_yahoo")
    con.execute("CREATE OR REPLACE TABLE fund_clone_returns_monthly AS SELECT * FROM clone")
    tilts = mean_tilts(con, loadings)
    loaded = loadings.join(tilts, on="fund_id").join(
        labels().select("fund_id", pl.col("label_v2").alias("style")), on="fund_id", how="left")
    con.execute("CREATE OR REPLACE TABLE fund_factor_loadings AS SELECT * FROM loaded")
    seed = loaded.filter(pl.col("cohort") == "seed")
    comp = comparison(seed)
    comp_all = comparison(loaded)
    FIG.mkdir(parents=True, exist_ok=True)
    plot(seed, comp, FIG / "09_returns_vs_holdings_tilts.png")

    by_style = (seed.group_by("style").agg(pl.len().alias("funds"), pl.col("alpha_ann").median(),
                                           *[pl.col(f"b_{f}").median() for f in FACTORS], pl.col("r2").median())
                .sort("style"))
    per_fund = loaded.sort("cohort", "style", "fund_name").select(
        "fund_name", "cohort", "style", "n_months", pl.col("alpha_ann").round(3), pl.col("t_alpha").round(1),
        *[pl.col(f"b_{f}").round(2) for f in FACTORS], pl.col("r2").round(2),
        *[pl.col(t).round(2) for t, _, _ in PAIRS])
    used = clone.filter(pl.col("coverage") >= MIN_COVERAGE)
    used_yahoo = clone_yahoo.filter(pl.col("coverage") >= MIN_COVERAGE)
    rec = recovered.select("fund_name", "cohort", "months_yahoo", "months_filled",
                           pl.col("median_cov_yahoo").round(2), pl.col("median_cov_filled").round(2),
                           pl.col("alpha_ann").round(3), *[pl.col(f"b_{f}").round(2) for f in FACTORS],
                           pl.col("r2").round(2))
    REPORT.write_text(f"""# Returns-based factor regressions vs holdings-based tilts

Generated by `uv run python -m src.models.factor_regressions`. Method and caveats in the module
docstring and `docs/decisions.md`. Return series: the 13F long book of quarter q held from the quarter-end
close through quarter q+1 (the "clone"); monthly excess returns regressed on FF5 + momentum with Newey-West
t-stats ({NW_LAGS} lags). Months with less than {MIN_COVERAGE:.0%} of the starting book priced are dropped
({len(clone) - len(used):,} of {len(clone):,} fund-months); funds need {MIN_MONTHS}+ months.

Stocks Yahoo doesn't price (acquired or delisted since) enter with the filled quarterly label
(13F-implied or delisting return, `src.securities.labels`), spread evenly over the quarter's three
months. On Yahoo prices alone {len(used_yahoo):,} fund-months and {len(loadings_yahoo)} funds pass; with the fill,
{len(used):,} fund-months and {len(loadings)} funds.

## Funds that failed coverage on Yahoo prices alone

{md_table(rec)}

## Do the two methods agree?

Seed funds ({len(seed)}):

{md_table(comp)}

Seed and holdout funds ({len(loaded)}):

{md_table(comp_all)}

`same sign share`: funds where the tilt and the loading point the same way (after flipping size).

![tilts vs loadings](figures/09_returns_vs_holdings_tilts.png)

## Median loadings by style (seed funds, v2 labels)

{md_table(by_style)}

## Per fund

`alpha_ann`: annualised intercept of the clone (not the fund's alpha: no shorts, fees, or intra-quarter
trading). Tilt columns: mean holdings z-scores over the same books.

{md_table(per_fund)}
""")
    with pl.Config(tbl_rows=20, tbl_width_chars=220, tbl_cols=20):
        print(f"clone fund-months {len(clone):,}, used {len(used):,} (Yahoo only {len(used_yahoo):,}); "
              f"funds regressed {len(loaded)} (seed {len(seed)}; Yahoo only {len(loadings_yahoo)})")
        print(rec)
        print(comp)
        print(comp_all)
        print(by_style)


if __name__ == "__main__":
    main()
