"""Merger-target weak label (plan §5 Task A: "merger arb / event-driven: more than
X% of the portfolio in announced M&A targets"). One row per fund x quarter.

Usage:
    uv run python -m src.features.merger_targets     # after src.securities.delistings

Targets come from the acquisition list in `delisting_events` (exit_class cash or
stock acquisition): a stock is an **announced target** from its first
target-side deal filing (merger proxy, SC 14D9 / SC TO-T / SC 13E3, or a 425 /
S-4 naming it) until its delisting. The first filing lags the press release by
days (425, tender offer) to weeks (preliminary proxy), so the share is a slight
undercount early in a deal.

`target_share` = long weight (non-option `weight_long`) in announced targets at
the quarter end; `merger_arb` = target_share > MERGER_ARB_SHARE.

**Weak label and validation measure only, never a classifier feature.** The
list holds deals that *completed* (the stock was delisted), which is not known
at the quarter end: a deal that later broke is not on it. As a feature that is
look-ahead; as a description of what a fund held it is fine. Also missed:
broken and still-pending deals, and targets no seed fund held at their exit
(the acquisition list is built from seed holdings).

MERGER_ARB_SHARE is fixed at 10% before looking at results; 5% and 20% are
reported as sensitivity (docs/merger_targets.md).

Writes `merger_targets` and `merger_target_share`.
"""

from pathlib import Path

import polars as pl
from sklearn.metrics import roc_auc_score

from src.models.style_classifier import labels
from src.securities import db

MERGER_ARB_SHARE = 0.10
SENSITIVITY = (0.05, 0.10, 0.20)
FIRST_QUARTER = "2018Q1"   # the modelling period, as for the other weak labels
REPORT = Path("docs/merger_targets.md")
QEND = "last_day(make_date(left({q}, 4)::INT, right({q}, 1)::INT * 3, 1))"


def build(con) -> None:
    con.execute("""
        CREATE OR REPLACE TABLE merger_targets AS
        SELECT cusip, issuer_cik, name, exit_class, exit_subclass,
               first_deal_filing AS announce_date, delist_date AS close_date, quarter AS last_held_quarter
        FROM delisting_events
        WHERE exit_class IN ('cash_acquisition', 'stock_acquisition') AND first_deal_filing IS NOT NULL
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE merger_target_share AS
        WITH p AS (
            SELECT fund_id, cohort, quarter, cusip, weight_long, {QEND.format(q='quarter')} AS q_end
            FROM positions WHERE NOT is_option AND weight_long > 0
        ),
        fq AS (SELECT DISTINCT fund_id, cohort, quarter, known_at FROM fund_fingerprint)
        SELECT fq.fund_id, fq.cohort, fq.quarter, fq.known_at,
               coalesce(sum(p.weight_long) FILTER (WHERE t.cusip IS NOT NULL), 0) AS target_share,
               count(DISTINCT t.cusip) AS n_targets,
               coalesce(sum(p.weight_long) FILTER (WHERE t.cusip IS NOT NULL), 0) > {MERGER_ARB_SHARE} AS merger_arb
        FROM fq
        LEFT JOIN p USING (fund_id, cohort, quarter)
        LEFT JOIN merger_targets t ON t.cusip = p.cusip
             AND t.announce_date <= p.q_end AND t.close_date > p.q_end      -- announced, not yet closed
        GROUP BY ALL
    """)


def validation(con) -> tuple[pl.DataFrame, pl.DataFrame, float, pl.DataFrame]:
    """Fund level (seed funds, 2018+): mean target_share vs the v2 style label."""
    fq = con.execute(f"""
        SELECT fund_id, cohort, quarter, target_share FROM merger_target_share WHERE quarter >= '{FIRST_QUARTER}'
    """).pl()
    lab = labels().select("fund_id", "cohort", pl.col("label_v2").alias("style"))
    names = con.execute("SELECT DISTINCT fund_id, fund_name FROM fund_fingerprint").pl()
    funds = (fq.join(lab, on=["fund_id", "cohort"])
             .group_by("fund_id", "cohort", "style")
             .agg(pl.col("target_share").mean().alias("mean_share"),
                  *[(pl.col("target_share") > x).mean().alias(f"q_over_{int(x * 100)}") for x in SENSITIVITY])
             .join(names, on="fund_id", how="left"))
    seed = funds.filter(pl.col("cohort") == "seed")
    auc = roc_auc_score((seed["style"] == "event_driven").to_numpy(), seed["mean_share"].to_numpy())
    # a fund is flagged when the rule fires in at least half its quarters
    rows = []
    for x in SENSITIVITY:
        flag = seed[f"q_over_{int(x * 100)}"] >= 0.5
        ed = seed["style"] == "event_driven"
        tp = int((flag & ed).sum())
        rows.append({"threshold": x, "funds_flagged": int(flag.sum()), "event_driven_flagged": tp,
                     "event_driven_total": int(ed.sum()),
                     "precision": round(tp / max(int(flag.sum()), 1), 2), "recall": round(tp / max(int(ed.sum()), 1), 2)})
    by_style = (seed.group_by("style").agg(pl.len().alias("funds"), pl.col("mean_share").median().round(3),
                                           pl.col("mean_share").max().round(3).alias("max_share"))
                .sort("mean_share", descending=True))
    top = funds.sort("mean_share", descending=True).head(15).select(
        "fund_name", "cohort", "style", pl.col("mean_share").round(3), pl.col("q_over_10").round(2))
    return pl.DataFrame(rows), by_style, auc, top


def _md(df: pl.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in df.iter_rows())


def write_report(con) -> None:
    thresholds, by_style, auc, top = validation(con)
    n = con.execute("SELECT count(*), sum((exit_class = 'cash_acquisition')::INT) FROM merger_targets").fetchone()
    REPORT.write_text(f"""# Merger-target weak label

Generated by `uv run python -m src.features.merger_targets`. Method in the module docstring and
docs/decisions.md. {n[0]:,} completed acquisitions of seed-held stocks ({n[1]:,} cash), 2015–2026.
Seed funds, {FIRST_QUARTER} on. Weak label only, never a feature (the list knows which deals completed).

**AUC of a fund's mean target share for the event_driven label (v2): {auc:.2f}**

## Median target share by style

{_md(by_style)}

## Threshold (fixed at {MERGER_ARB_SHARE:.0%} beforehand; fund flagged if over it in >= half its quarters)

{_md(thresholds)}

## Highest mean target share (seed and holdout)

`q_over_10`: share of the fund's quarters above 10%.

{_md(top)}
""")


if __name__ == "__main__":
    with db.connect() as con:
        build(con)
        write_report(con)
        print(REPORT.read_text())
