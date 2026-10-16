# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Milestone 4: the strategy fingerprint per style
#
# Figures only; the fingerprint is built in `src/features/fingerprint.py` and
# the 13D counts in `src/features/activist_13d.py`. Seed funds, quarters
# 2018Q1 on (the modelling period), styles from the v2 labels
# (`data/reference/style_labels_v2.csv`, high and medium confidence).

# %%
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
import os  # noqa: E402
import sys  # noqa: E402

os.chdir(ROOT)                 # src/ uses paths relative to the repo root
sys.path.insert(0, str(ROOT))
from src.models.style_classifier import labels  # noqa: E402

FIG = ROOT / "docs" / "figures"
con = duckdb.connect(str(ROOT / "data/processed/securities.duckdb"), read_only=True)

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, OTHER = "#2a78d6", "#b8b6b0"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "font.size": 9})

# %% [markdown]
# ## Radar chart of the fingerprint, one panel per style
#
# Each axis is a fingerprint feature turned into a **percentile rank** across
# all seed fund-quarters 2018+ (0 = lowest, 1 = highest), so features on very
# different scales share one radius. The blue shape is the style's median
# fund-quarter; the gray ring at 0.5 is the median of all funds. Percentiles,
# not z-scores, because position counts and 13D counts are heavily skewed.

# %%
AXES = {  # column -> axis label
    "log_n_positions": "positions",
    "top10_weight": "top-10 weight",
    "active_share": "active share",
    "gross_turnover": "turnover",
    "holding_period_q": "holding period",
    "tilt_size": "size",
    "tilt_value": "value",
    "tilt_momentum": "momentum",
    "tilt_profitability": "profitability",
    "sector_BusEq": "tech weight",
    "sector_Hlth": "healthcare weight",
    "n_13d_2y": "13D filings",
}
fp = con.execute(f"""
    SELECT f.fund_id, f.quarter, {", ".join("f." + c for c in AXES if c != "n_13d_2y")}, a.n_13d_2y
    FROM fund_fingerprint f LEFT JOIN activist_13d a USING (fund_id, quarter)
    WHERE f.cohort = 'seed' AND f.quarter >= '2018Q1'
""").pl().join(labels().filter(pl.col("cohort") == "seed").select("fund_id", "label_v2"), on="fund_id")
pct = fp.with_columns([(pl.col(c).rank("average") / pl.col(c).count()).alias(c) for c in AXES])
med = pct.group_by("label_v2").agg(*[pl.col(c).median() for c in AXES], pl.col("fund_id").n_unique().alias("funds"))
order = ["activist", "value_concentrated", "event_driven", "growth_tech",
         "healthcare", "multi_manager", "quant", "discretionary_other"]

angles = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False)
closed = np.r_[angles, angles[:1]]
fig, axes = plt.subplots(2, 4, figsize=(15, 9.6), subplot_kw={"polar": True})
for ax, style in zip(axes.flat, order):
    row = med.filter(pl.col("label_v2") == style).row(0, named=True)
    v = np.array([row[c] for c in AXES])
    ax.plot(closed, np.full(len(closed), 0.5), color=OTHER, linewidth=1, linestyle=(0, (3, 3)))
    ax.fill(closed, np.r_[v, v[:1]], color=BLUE, alpha=0.25)
    ax.plot(closed, np.r_[v, v[:1]], color=BLUE, linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75], [])
    ax.set_xticks(angles, list(AXES.values()), fontsize=7.5, color=INK2)
    ax.grid(color=GRID)
    ax.spines["polar"].set_color(GRID)
    ax.set_title(f"{style.replace('_', ' ')} ({row['funds']} funds)", fontsize=10.5, fontweight="bold", pad=16)
fig.suptitle("Strategy fingerprints by style: median fund-quarter, as percentiles of all seed funds (2018–2026)",
             x=0.06, ha="left", fontsize=13, fontweight="bold")
fig.text(0.06, 0.005, "Blue: style median percentile per feature; dashed gray: 0.5 (median fund). Styles: v2 labels. "
         "13D filings: original and amended 13Ds filed in the 2 years before the 13F filing date.",
         color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.02, 1, 0.95), h_pad=3.5)
fig.savefig(FIG / "08_style_radars.png", dpi=160, bbox_inches="tight")
plt.show()
with pl.Config(tbl_cols=16, tbl_width_chars=220):
    print(med.with_columns([pl.col(c).round(2) for c in AXES]).sort("label_v2"))

# %% [markdown]
# **Reading.** The styles have distinct shapes, which is why a classifier on
# these features works at all. Activists sit at the 83rd percentile on top-10
# weight and on 13D filings with few positions; concentrated value funds look
# similar on concentration but have the highest value tilt (0.86), the longest
# holding periods after quants and far fewer 13Ds, which is the activist/value
# confusion the classifier still makes. Event-driven funds are the "low
# everything" shape: small, cheap, low-momentum, low-profitability holdings
# turned over fast with the shortest holding period (0.16), as deal targets
# are. Healthcare is one spike (healthcare weight 0.95) plus many 13Ds
# (biotech stakes over 5%). Growth/tech funds lean large, profitable,
# high-momentum and tech-heavy. Multi-managers and quants share the
# positions spike and low concentration; turnover separates them (0.88 vs
# 0.54), and quants hold positions longer. Discretionary/other has only 3
# funds after the v2 review (Bridgewater, Soros, Duquesne) and no strong
# shape: the leftover class, not a style.
