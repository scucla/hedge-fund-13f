"""Clone-the-best-funds strategy: copy the recent winners' top positions as their 13Fs come out.

Usage:
    uv run python -m src.backtest.clone_best dev     # 2018-2022 only: develop, inspect, fix the rules
    uv run python -m src.backtest.clone_best test    # the one 2023+ run -> docs/clone_strategy.md

**Rules (fixed on 2018-2022 before the 2023+ run; see docs/decisions.md):**
  selection  At each quarter end S_q, rank the seed funds by trailing 3-year
             holdings-implied return: the compounded return of each quarter-end book
             over the following quarter, for the 12 quarters k = q-11 .. q (book at k-1
             held through quarter k). Only data known at S_q: books with known_at <= S_q
             and stock returns with label_known_at <= S_q (13F-implied returns for
             quarter q itself only exist after S_q, so the window is the 12 quarters
             ending at q-1: k = q-12 .. q-1). Weights are the long book's stock-universe
             positions, renormalised over stocks with a known return. A fund needs all 12
             books and >= 50% of each book's weight with a known return. Top N_FUNDS = 10.
  sleeve     Each selected fund's TOP_K = 5 largest common-stock/ADR positions in its
             quarter-q filing, equal-weighted, bought at the close of the first trading
             day after that filing's date. Names delisted on or before that day are
             skipped for the next ones (as in Milestone 7).
  slots      10 slots. A fund selected again keeps its slot (and rebalances it to its new
             top 5 at its new filing); new funds take the slots of dropped ones in rank
             order. A slot's old sleeve is held, drifting, until its new fund's filing
             is bought; a fund that never files before the next selection leaves its slot
             to the next cycle. Slots compound independently: a switch moves the slot's
             whole value, with no rebalancing across slots.
  cash       Slots not yet invested (the start) earn the Fama-French daily risk-free
             rate (owner's rule), forward-filled past the end of the factor data.
  costs      10 and 20 bp per side on the traded value at every switch or rebalance.
  returns    Milestone 7 engine: Yahoo daily returns, else the filled quarterly return
             spread over the quarter; delisted names go to cash after the delisting.
  periods    dev: returns through 2022-12-31 only. test: the same simulation, reported
             from the Milestone 7 "2023+" start (the 2023Q1 entry date, 2023-05-16), so it
             compares one-for-one with the existing strategies; alphas through 2026-08.
"""

import sys
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import polars as pl

from src.backtest import engine
from src.backtest.run import quarters
from src.securities import db

N_FUNDS, TOP_K, TRAIL_Q, MIN_KNOWN_WEIGHT = 10, 5, 12, 0.5
FIRST_CYCLE, LAST_CYCLE = "2018Q2", "2026Q2"   # first quarter with 12 trailing quarters from 2015Q1 books
DEV_END = date(2022, 12, 31)
COSTS = (10, 20)
STOCK_CLASSES = ("common", "adr")
OUT = "data/processed/clone_best"
DOC = "docs/clone_strategy.md"
CONCLUSION = "docs/clone_strategy_conclusion.md"   # hand-written after the run; included in DOC
FIG = "docs/figures/23_clone_best.png"


def q_idx(q: str) -> int:
    return int(q[:4]) * 4 + int(q[-1])


# --- selection --------------------------------------------------------------------------------

def load_inputs(con) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    books = con.execute("""
        SELECT p.fund_id, p.quarter, p.known_at, p.cusip, p.weight_long
        FROM positions p JOIN stock_quarter s ON s.cusip = p.cusip AND s.quarter = p.quarter
        WHERE p.cohort = 'seed' AND p.put_call = '' AND p.weight_long > 0
    """).pl().with_columns(pl.col("quarter").map_elements(q_idx, return_dtype=pl.Int64).alias("qi"))
    rets = con.execute("""SELECT cusip, quarter, ret_next_q, label_known_at FROM stock_quarter
                          WHERE ret_next_q IS NOT NULL""").pl()
    top = con.execute(f"""
        SELECT p.fund_id, p.quarter, p.known_at, p.cusip, p.weight_long
        FROM positions p JOIN security_master m USING (cusip)
        WHERE p.cohort = 'seed' AND p.put_call = '' AND p.weight_long > 0 AND m.security_class IN {STOCK_CLASSES}
    """).pl()
    return books, rets, top


def trailing_returns(books: pl.DataFrame, rets: pl.DataFrame, q: str, asof: date) -> pl.DataFrame:
    """fund_id, trailing (compounded 12-quarter return), n_q, min_known; only data known at `asof`."""
    qi = q_idx(q)
    b = books.filter(pl.col("qi").is_between(qi - TRAIL_Q - 1, qi - 2) & (pl.col("known_at") <= asof))
    r = rets.filter(pl.col("label_known_at") <= asof)
    j = b.join(r, on=["cusip", "quarter"], how="left")   # ret_next_q of the book quarter = return over the next one
    per_q = (j.group_by("fund_id", "qi")
             .agg((pl.col("weight_long") * pl.col("ret_next_q")).sum().alias("wr"),
                  pl.col("weight_long").filter(pl.col("ret_next_q").is_not_null()).sum().alias("known_w"),
                  pl.col("weight_long").sum().alias("w"))
             .with_columns((pl.col("wr") / pl.col("known_w")).alias("r_q"), (pl.col("known_w") / pl.col("w")).alias("known")))
    return (per_q.group_by("fund_id").agg(pl.len().alias("n_q"), pl.col("known").min().alias("min_known"),
                                          ((1 + pl.col("r_q")).product() - 1).alias("trailing"))
            .with_columns(pl.lit(q).alias("cycle")))


def select(books: pl.DataFrame, rets: pl.DataFrame, q: str) -> pl.DataFrame:
    s = engine.quarter_end(q)
    t = trailing_returns(books, rets, q, s)
    ok = t.filter((pl.col("n_q") == TRAIL_Q) & (pl.col("min_known") >= MIN_KNOWN_WEIGHT))
    return (ok.sort(["trailing", "fund_id"], descending=[True, False]).head(N_FUNDS)
            .with_row_index("rank", offset=1).with_columns(pl.lit(s).alias("selected_on")))


def sleeve(top: pl.DataFrame, fund: str, q: str, entry: date, gone: set[str]) -> dict[str, float]:
    b = top.filter((pl.col("fund_id") == fund) & (pl.col("quarter") == q) & ~pl.col("cusip").is_in(list(gone)))
    names = b.sort(["weight_long", "cusip"], descending=[True, False])["cusip"].head(TOP_K).to_list()
    return {c: 1 / len(names) for c in names} if names else {}


# --- slots ------------------------------------------------------------------------------------

@dataclass
class Switch:
    slot: int
    fund: str
    cycle: str
    filing: date      # the fund's quarter-q filing date
    entry: date       # trade date: first trading day after the filing
    weights: dict = field(default_factory=dict)


def assign_slots(selections: dict[str, pl.DataFrame]) -> dict[str, dict[str, int]]:
    """cycle -> {fund: slot}: kept funds keep their slot; new funds fill freed slots in rank order."""
    out, prev = {}, {}
    for q in sorted(selections):
        funds = selections[q]["fund_id"].to_list()
        keep = {f: s for f, s in prev.items() if f in funds}
        free = [s for s in range(N_FUNDS) if s not in keep.values()]
        new = [f for f in funds if f not in keep]
        cur = keep | dict(zip(new, free))
        out[q], prev = cur, cur
    return out


def schedule(selections, slots, filings: pl.DataFrame, days: list[date], top: pl.DataFrame,
             delist: pl.DataFrame, end: date) -> list[Switch]:
    """Every slot switch / rebalance: the selected fund's quarter-q filing, bought the next trading day."""
    arr = np.array(days, dtype="datetime64[D]")
    cycles = sorted(selections)
    out = []
    for n, q in enumerate(cycles):
        nxt_sel = engine.quarter_end(cycles[n + 1]) if n + 1 < len(cycles) else end
        for f, slot in slots[q].items():
            row = filings.filter((pl.col("fund_id") == f) & (pl.col("quarter") == q))
            if row.is_empty():
                continue
            filed = row["known_at"][0]
            i = int(np.searchsorted(arr, np.datetime64(filed, "D"), side="right"))
            if i >= len(days) or days[i] > min(nxt_sel, end):
                continue   # no filing in time: the slot waits for the next cycle
            gone = set(delist.filter(pl.col("delisted_on") <= days[i])["cusip"])
            out.append(Switch(slot, f, q, filed, days[i], sleeve(top, f, q, days[i], gone)))
    return sorted(out, key=lambda s: (s.entry, s.slot))


def simulate(switches: list[Switch], rd: engine.ReturnData, rf: dict[date, float], start: date, end: date,
             cost_bps: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Daily portfolio return of independently compounding slots; cash at the risk-free rate."""
    days = [d for d in rd.days if start < d <= end]
    value = np.full(N_FUNDS, 1.0 / N_FUNDS)
    holding: list[dict[str, float]] = [{} for _ in range(N_FUNDS)]   # current drifted weights within each slot
    by_day: dict[date, list[Switch]] = {}
    for s in switches:
        if start < s.entry <= end:
            by_day.setdefault(s.entry, []).append(s)
    # each segment's daily return matrix, computed once per (slot, entry)
    seg_ret: dict[tuple[int, date], tuple[list[str], dict[date, np.ndarray]]] = {}
    per_slot = {k: sorted([s for s in switches if s.slot == k and start < s.entry <= end], key=lambda s: s.entry)
                for k in range(N_FUNDS)}
    for k, ss in per_slot.items():
        for a, b in zip(ss, ss[1:] + [None]):
            stop = b.entry if b else end
            names = sorted(a.weights)
            if names:
                mat, _ = engine.window_returns(rd, names, a.entry, stop)
                seg_ret[(k, a.entry)] = (names, dict(zip([d for d in rd.days if a.entry < d <= stop], mat)))
    current_seg = {k: None for k in range(N_FUNDS)}
    out, trades = [], []
    last_rf = 0.0
    for d in days:
        total_before = value.sum()
        # 1) daily returns of what each slot held coming into the day
        for k in range(N_FUNDS):
            seg = current_seg[k]
            if seg is None:
                last_rf = rf.get(d, last_rf)
                value[k] *= 1 + last_rf
            else:
                names, rmap = seg_ret[seg]
                r = rmap.get(d)
                if r is not None and holding[k]:
                    w = np.array([holding[k].get(c, 0.0) for c in names])
                    g = w * (1 + r)
                    value[k] *= g.sum() / w.sum() if w.sum() > 0 else 1.0
                    holding[k] = dict(zip(names, g / g.sum())) if g.sum() > 0 else {}
        # 2) switches at today's close
        for s in by_day.get(d, []):
            new = s.weights
            traded = sum(abs(new.get(c, 0.0) - holding[s.slot].get(c, 0.0)) for c in set(new) | set(holding[s.slot]))
            cost = value[s.slot] * traded * cost_bps / 1e4
            value[s.slot] -= cost
            trades.append({"date": d, "slot": s.slot, "fund_id": s.fund, "cycle": s.cycle, "filing": s.filing,
                           "traded": traded, "slot_value": value[s.slot], "port_value": value.sum(),
                           "names": ",".join(sorted(new))})
            holding[s.slot] = dict(new)
            current_seg[s.slot] = (s.slot, s.entry) if new else None
        out.append({"date": d, "r": value.sum() / total_before - 1})
    return pl.DataFrame(out), pl.DataFrame(trades)


def build(end: date) -> dict:
    with db.connect(read_only=True) as con:
        books, rets, top = load_inputs(con)
        filings = con.execute("""SELECT fund_id, quarter, max(known_at) AS known_at FROM positions
                                 WHERE cohort = 'seed' GROUP BY ALL""").pl()
        rd = engine.load_returns(con, date(2018, 1, 1))
        rf = dict(con.execute("SELECT date, RF FROM ff_factors_daily WHERE date >= '2018-01-01'").fetchall())
    cycles = [q for q in quarters(FIRST_CYCLE, LAST_CYCLE) if engine.quarter_end(q) < end]
    selections = {q: select(books, rets, q) for q in cycles}
    slots = assign_slots(selections)
    switches = schedule(selections, slots, filings, rd.days, top, rd.delist, end)
    start = engine.quarter_end(FIRST_CYCLE)
    runs = {c: simulate(switches, rd, rf, start, end, c) for c in COSTS}
    return {"selections": selections, "slots": slots, "switches": switches, "runs": runs, "books": books,
            "rets": rets, "filings": filings, "rd": rd}


# --- leakage audit ---------------------------------------------------------------------------

def audit(res: dict) -> pl.DataFrame:
    """Every check counts violations; all must be 0."""
    rows = []
    # poison test: make everything not yet known on the selection date wildly attractive;
    # if any of it leaked into the ranking, the selected funds would change
    changed = 0
    for q, chosen in res["selections"].items():
        s = engine.quarter_end(q)
        rets = res["rets"].with_columns(pl.when(pl.col("label_known_at") > s).then(10.0)
                                        .otherwise(pl.col("ret_next_q")).alias("ret_next_q"))
        books = res["books"].with_columns(pl.when(pl.col("known_at") > s).then(pl.col("weight_long") * 100)
                                          .otherwise(pl.col("weight_long")).alias("weight_long"))
        changed += select(books, rets, q)["fund_id"].to_list() != chosen["fund_id"].to_list()
    rows.append({"check": "selection ignores data published after the selection date", "checked": len(res["selections"]),
                 "violations": changed, "note": "unknown returns set to +1000% and late books x100: same funds chosen"})
    sw = res["switches"]
    rows.append({"check": "entry after the fund's filing", "checked": len(sw),
                 "violations": sum(not (x.entry > x.filing) for x in sw), "note": "next trading day's close"})
    rows.append({"check": "entry after the selection date", "checked": len(sw),
                 "violations": sum(not (x.entry > engine.quarter_end(x.cycle)) for x in sw), "note": ""})
    f = res["filings"]
    bad_src = 0
    top = res["books"]
    for x in sw:
        filed = f.filter((pl.col("fund_id") == x.fund) & (pl.col("quarter") == x.cycle))["known_at"][0]
        book = set(top.filter((pl.col("fund_id") == x.fund) & (pl.col("quarter") == x.cycle))["cusip"])
        bad_src += (filed != x.filing) or not set(x.weights) <= book
    rows.append({"check": "holdings come from that filing", "checked": len(sw), "violations": bad_src,
                 "note": "sleeve names are in the fund's quarter-q book, filed on the recorded date"})
    dl = dict(res["rd"].delist.iter_rows())
    rows.append({"check": "no name delisted by the trade date", "checked": sum(len(x.weights) for x in sw),
                 "violations": sum(1 for x in sw for c in x.weights if c in dl and dl[c] <= x.entry), "note": ""})
    return pl.DataFrame(rows)


# --- test-period report ------------------------------------------------------------------------

def report(res: dict) -> None:
    from src.backtest.run import FACTORS, metrics
    from src.models.task_c_common import md_table
    m7 = pl.read_parquet("data/processed/backtest_results.parquet")
    start = date.fromisoformat(m7.filter(pl.col("period") == "2023+")["start"].min())
    with db.connect(read_only=True) as con:
        ff = con.execute("SELECT date, RF, " + ", ".join(FACTORS) + " FROM ff_factors_daily WHERE date >= '2022-01-01'").pl()
    daily = pl.read_parquet("data/processed/backtest_daily.parquet")
    bench = (daily.filter(pl.col("strategy") == "spy").select("date", pl.col("r").alias("spy"))
             .join(daily.filter(pl.col("strategy") == "ew_seed").select("date", pl.col("r").alias("ew_seed")), on="date"))
    # Milestone 7's `start` is the first return date; metrics count returns strictly after the date given
    entry_start = start - timedelta(days=1)
    rows = []
    for c, (d, tr) in res["runs"].items():
        reb = tr.select(pl.col("date").alias("entry"), (pl.col("traded") * pl.col("slot_value") / pl.col("port_value")).alias("traded"))
        rows.append({"cost_bp": c} | metrics("clone_best", d, reb, entry_start, ff, bench))
        d.write_parquet(f"{OUT}_daily_{c}bp.parquet")
        tr.write_parquet(f"{OUT}_trades_{c}bp.parquet")
    mine = pl.DataFrame(rows)
    others = m7.filter((pl.col("period") == "2023+") & (pl.col("missing_sold_out") == 0)
                       & pl.col("strategy").is_in(["consensus", "cluster:growth_tech", "cluster:multi_manager",
                                                   "c1_adds", "c2_als", "spy", "ew_seed"]))
    cols = ["strategy", "cost_bp", "start", "ann_return", "ann_vol", "sharpe", "max_dd", "turnover", "ff6_alpha", "ff6_t",
            "alpha_spy", "t_spy", "beta_spy", "alpha_ew_seed", "t_ew_seed", "months"]
    tab = pl.concat([mine.select(cols), others.select(cols)], how="diagonal").sort("cost_bp", "strategy")
    aud = audit(res)
    _figure(res, entry_start, daily)
    from pathlib import Path
    conclusion = Path(CONCLUSION).read_text().strip() if Path(CONCLUSION).exists() else "(not written yet)"
    Path(DOC).write_text(f"""# Clone the best funds (2023+ test)

Generated by `src/backtest/clone_best.py`; rules in its docstring and `docs/decisions.md`, fixed on 2018-2022 and
committed before this run. Figure: `docs/figures/23_clone_best.png`.

## Conclusion

{conclusion}

## 2023+ results, against the existing strategies

Same window as Milestone 7's 2023+ results (returns after {entry_start}); alphas monthly, Newey-West t-stats (3 lags),
through 2026-08. `turnover`: one-way, share of the portfolio per year.

{md_table(tab, 3)}

## Leakage audit

{md_table(aud)}

## Funds selected in the test period

{md_table(_selected_table(res, entry_start))}
""")
    print(tab)
    print(aud)


def _selected_table(res: dict, start: date) -> pl.DataFrame:
    rows = [{"cycle": q, "selected_on": str(engine.quarter_end(q)),
             "funds": ", ".join(d["fund_id"].to_list())} for q, d in sorted(res["selections"].items())
            if engine.quarter_end(q) >= date(start.year - 1, 10, 1)]
    return pl.DataFrame(rows)


def _figure(res: dict, start: date, daily: pl.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    d = res["runs"][10][0].filter(pl.col("date") > start)
    ax.plot(d["date"].to_list(), np.cumprod(1 + d["r"].to_numpy()), lw=2.2, color="#c1121f", label="clone the best 10 funds")
    for name, style in (("spy", {"color": "black", "lw": 2}), ("ew_seed", {"color": "grey", "ls": "--"}),
                        ("consensus", {"color": "#2a6f97"}), ("c2_als", {"color": "#6a994e"})):
        s = daily.filter((pl.col("strategy") == name) & (pl.col("date") > start)).sort("date")
        ax.plot(s["date"].to_list(), np.cumprod(1 + s["r"].to_numpy()), label=name, **style)
    ax.set_yscale("log")
    ax.set_title("Clone the best funds vs benchmarks, 2023+ (net of 10 bp per side; growth of $1)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if mode == "dev":
        res = build(DEV_END)
        print(audit(res))
        for c, (d, tr) in res["runs"].items():
            print(f"{c} bp: {len(d)} days, growth {np.cumprod(1 + d['r'].to_numpy())[-1]:.3f}, switches {len(tr)}")
    elif mode == "test":
        report(build(date(2026, 10, 14)))
