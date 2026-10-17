# Reverse-engineer a hedge fund: project plan

*An ML project built on free SEC 13F filings. It infers each fund's hidden strategy, groups similar funds, and predicts what they will buy and sell next quarter.*

---

## 1. The pitch, in one paragraph

Every quarter, any investment manager with more than $100M in US-listed stocks has to tell the SEC exactly what it owns, on a form called **13F**. Hedge funds hate this, but the data is public and free going back to 1999 (machine-readable since 2013). You'll build a pipeline that turns ~10 years of these filings into a dataset, then answer three questions with ML: **(1) What is each fund actually doing?** (its style and factor tilts, whatever its marketing says). **(2) Which funds are secretly alike?** (clusters, crowding, "Tiger cubs" that trade like their parent). **(3) What will a fund do next quarter?** (which positions it will add, trim or exit, and which new names it will buy). You finish with a backtest that asks whether any of this makes money after the 45-day filing delay, and a small dashboard to demo it.

This is a strong fit for the roles you named. It reads like buy-side research (alternative asset management, equity research), it's about flows and positioning (sales & trading cares a lot about crowding and who owns what), and it shows real ML and data engineering rather than a Kaggle notebook.

---

## 2. What 13F data is (and isn't)

Understanding the limits of the data is half of what makes this project credible in an interview. Lead with them.

**What's in a 13F-HR filing**
- Filed by institutional investment managers with ≥$100M in "13(f) securities". That includes hedge funds, but also mutual fund families, banks, pensions, RIAs and insurers.
- Due within **45 days after each quarter end**, so the Q1 (March 31) snapshot can arrive as late as mid-May.
- One row per position: issuer name, **CUSIP** (a 9-character security ID), share count, market value, put/call flag, investment discretion and voting authority.
- Values were reported in thousands of dollars until the end of 2022 and in whole dollars since January 2023. Your parser has to handle both, or every post-2022 fund will look 1,000× larger.

**What's not in it (the "hidden" part)**
- **No short positions.** A long/short fund looks long-only. This is the biggest blind spot.
- **No non-US stocks, bonds, futures, FX, cash, or most derivatives.** A macro fund's 13F may be close to meaningless.
- **Quarter-end snapshots only.** Anything bought and sold within a quarter is invisible, and funds may "window dress" around quarter end.
- **Confidential treatment.** Funds can ask the SEC to delay disclosing positions they're still building, which then appear later in amendments (13F-HR/A).
- Options show up as the underlying's value with a put/call flag, not as option exposure.

You should state these up front in the README and design around them. For example, you can predict well for long-biased equity funds and say plainly that you don't model macro or quant HFT shops.

**Prior work worth knowing** (cite these in your write-up; it shows you read the literature)
- Cremers & Petajisto (2009), *How Active Is Your Fund Manager?* introduces **Active Share**, a holdings-based measure of how different a portfolio is from its benchmark.
- Kacperczyk, Sialm & Zheng (2008), *Unobserved Actions of Mutual Funds*: the "return gap" between reported and holdings-implied returns. This is the same thing as your blind spot, measured.
- Cohen, Polk & Silli (2010), *Best Ideas*: managers' highest-conviction positions outperform. That motivates weighting by conviction.
- Agarwal, Jiang, Tang & Yang (2013), *Uncovering Hedge Fund Skill from the Portfolio Holdings They Hide*: confidential 13F positions tend to outperform.
- Existing products: WhaleWisdom, Dataroma, HedgeFollow and the 13F "clone" ETFs. Your edge over them is the ML layer and honest evaluation, not the data.

---

## 3. How it works: the system end to end

```
SEC EDGAR (13F XML / quarterly data sets)
        │  ingest + parse
        ▼
Raw holdings table  (filer, quarter, CUSIP, shares, value, put/call)
        │  clean: dedupe amendments, fix units, map CUSIP → ticker/permno
        ▼
Securities master + prices + fundamentals + factor returns
        │  join point-in-time
        ▼
Panel: fund × stock × quarter  (+ fund × quarter summary)
        │
        ├──► A. Strategy fingerprint   (factor tilts, style label, SHAP)
        ├──► B. Fund embeddings + clusters (UMAP map, crowding)
        ├──► C. Next-quarter move prediction (add / trim / exit / new buys)
        │
        ▼
D. Backtest: does following the predictions or the clusters earn anything after the 45-day lag?
        ▼
Dashboard (Streamlit): pick a fund → fingerprint, peers, predicted moves
```

### 3.1 Data sources (all free)

| Need | Source | Notes |
|---|---|---|
| Holdings | **SEC "Form 13F Data Sets"** (quarterly zipped TSVs on sec.gov) | Easiest start: already flattened into SUBMISSION, COVERPAGE, INFOTABLE tables. |
| Holdings (alternative) | EDGAR full-text filings, `information table` XML | Gives you control; use if the data sets have gaps. EDGAR asks for a `User-Agent` header with your email and ≤10 requests/second. |
| Security IDs | SEC **Official List of Section 13(f) Securities**, plus the **OpenFIGI** API (free key) for CUSIP → ticker | CUSIP mapping is the most tedious part of the project. Budget time for it. |
| Prices / returns | `yfinance` to start; **CRSP via WRDS** if your university has access | yfinance lacks delisted stocks (survivorship bias). Say so, or use CRSP. |
| Fundamentals | SEC **XBRL companyfacts API** (free); Compustat via WRDS if available | Needed for book-to-market, profitability, investment. |
| Factor returns | **Ken French Data Library** (Fama-French 5 factors + momentum) | Monthly and daily, free CSVs. |
| Activist labels | **Schedule 13D** filings on EDGAR | A fund that files 13Ds is an activist. That's a free label. |
| Deal labels | Announced M&A (Wikipedia lists, news, or SEC merger filings such as DEFM14A) | Funds that hold many announced targets are doing merger arbitrage. |

If you get WRDS through school, use it. CRSP + Compustat + Thomson/Refinitiv 13F (the `s34` tables) saves you weeks and is what academics use. If not, the free stack above is fine and is itself a good story about data engineering.

### 3.2 Choosing the universe

13F filers include ~7,000+ managers, most of which aren't hedge funds. There's no "hedge fund" flag, so you build one:
1. **Seed list:** hand-pick 100–200 well-known hedge funds (Pershing Square, Third Point, Tiger Global, Coatue, Viking, Lone Pine, Appaloosa, Baupost, Elliott, Icahn, Renaissance, Citadel, Millennium, etc.) and find their CIKs (SEC filer IDs).
2. **Heuristic filter for the rest:** exclude names containing "Bank", "Trust", "Pension", "Retirement", "Insurance", and the giant index families. Keep filers with roughly 10–400 positions, high turnover and high active share vs. the S&P 500.
3. Start with the **top ~300–500 hedge funds by reported 13F value**. That's big enough for ML and small enough to sanity-check by hand.

---

## 4. Features: turning holdings into numbers

You compute features at two levels. Everything must be **point-in-time**: only use data that existed on the date the filing became public.

### 4.1 Position level (fund × stock × quarter)
- Portfolio weight, rank of the position within the portfolio, and whether it's in the top 10 ("best idea" flag).
- Change in shares vs. last quarter (% and sign); number of consecutive quarters held (position age).
- Approximate **unrealized gain**: current price vs. the volume-weighted price over quarters when shares were added. Funds tend to trim winners and hold losers (the disposition effect), which is a real, testable signal.
- Stock characteristics: market cap, book-to-market, 12-1 momentum, 1-month reversal, volatility, beta, profitability, asset growth, sector, recent earnings surprise.
- Crowding: number of other hedge funds holding it, and change in that count.
- Put/call flag.

### 4.2 Fund level (fund × quarter)
- **Holdings-based factor exposures:** the weight-averaged characteristic of the portfolio, standardized cross-sectionally (z-scores). This is what "factor tilts" means here: "this fund sits at +1.2σ on momentum and −0.8σ on value".
- **Returns-based exposures:** compute each fund's implied quarterly return assuming it held its reported portfolio through the quarter, then regress on Fama-French 5 + momentum. Comparing the two is a nice robustness check, and the gap between implied and actual returns (if you know actual returns for some funds) relates to the return gap.
- Sector weights (11 GICS sectors).
- Concentration: number of positions, top-10 weight, Herfindahl index.
- Turnover: share of the portfolio that changed vs. last quarter.
- **Active share** vs. the S&P 500 and vs. the Russell 2000.
- Options usage: share of value in put/call rows.
- Average holding period.

---

## 5. The three ML tasks

### Task A: Strategy fingerprint ("what is this fund actually doing?")

**The problem:** there are no ground-truth strategy labels in 13F. You handle that in two layers.

**Layer 1, unsupervised and interpretable.** The factor-tilt vector from §4.2 *is* the fingerprint. Show it as a radar chart per fund. This needs no labels and is already useful.

**Layer 2, weakly supervised style classification.** Build labels for a few hundred funds from cheap sources:
- *Activist*: filed at least one 13D in the past 2 years.
- *Merger arb / event-driven*: more than X% of the portfolio in announced M&A targets.
- *Quant / high-turnover*: hundreds or thousands of positions, very high turnover, low concentration.
- *Growth / tech long-biased* (Tiger cubs): high momentum, low value, heavy tech and communication services.
- *Value / concentrated long-term*: high value tilt, low turnover, few positions.
- *Other / multi-strategy*.

Hand-check 50–100 funds yourself (fund websites, news, investor letters) to build a clean test set. Then train a **gradient-boosted classifier (LightGBM)** on the fund-level features, explain it with **SHAP** ("Pershing Square is called *concentrated value* mostly because of 8 positions, a top-10 weight of 100% and low turnover"), and look at where the classifier disagrees with your labels. Those disagreements are often the most interesting findings, such as a "value" fund that has quietly drifted into growth (*style drift*).

**Evaluation:** macro-F1 on your hand-labeled test set, a confusion matrix, and a **style drift** chart over time for a few famous funds.

### Task B: Fund embeddings and clustering ("who's secretly alike?")

**Representation options, from simplest to fanciest:**
1. **Weight vectors:** each fund is a sparse vector of weights over ~5,000 stocks. Use cosine similarity. It's a solid baseline and very interpretable ("these two funds share 40% of their book").
2. **Low-rank embeddings:** apply truncated SVD or NMF to the fund × stock weight matrix. NMF components often come out as readable themes ("mega-cap tech", "biotech", "energy").
3. **fund2vec:** treat each fund-quarter as a "document" and the stocks as "words", then train word2vec/doc2vec so that stocks held by similar funds land near each other, and funds become the average of their holdings. It's a nice story and cheap to train.
4. **Graph approach (stretch):** build a bipartite fund–stock graph weighted by portfolio weight. Learn node embeddings with node2vec or a small **GNN** (PyTorch Geometric). It's the most impressive option, but only worth it after 1–3 work.

**Clustering:** HDBSCAN (it finds a variable number of clusters and labels outliers) or k-means/spectral as a comparison. Visualize with **UMAP** as a 2-D "map of the hedge fund industry", colored by style label from Task A.

**Evaluation (unsupervised, so be explicit):**
- **Stability:** the adjusted Rand index between clusterings in consecutive quarters. Real styles should persist.
- **Agreement with known structure:** do Tiger cubs cluster together? Do activists? Measure cluster purity against your Task A labels.
- **Economic validity:** do funds in the same cluster have correlated implied returns *next* quarter? If clusters mean something, they should.

**Crowding insight (great for S&T interviews):** for each stock, count how many funds in the same cluster hold it. Then test whether highly crowded names have worse drawdowns when the cluster de-grosses (e.g., early 2021 Archegos/meme stocks, 2022 growth sell-off).

### Task C: Predicting next quarter's moves

This is the core supervised ML task. Frame it two ways.

**C1: What happens to existing positions?**
- **Unit:** (fund, stock, quarter t) for every current holding.
- **Target:** action at t+1 ∈ {**exit**, **trim** (>10% share cut), **hold**, **add** (>10% share increase)}. Tune the 10% thresholds and also try a regression target (Δlog shares).
- **Baselines (essential; your model is only interesting if it beats these):**
  - *Persistence:* predict "hold" for everything (likely 50–60% accuracy, which is why accuracy is a bad metric).
  - *Fund-level prior:* predict each class using the fund's own historical turnover rates.
- **Models:** multinomial logistic regression, then **LightGBM** (likely your best), then optionally a **sequence model** (GRU or a small transformer over the last 8 quarters of a position's weights and a fund's actions).
- **Features:** everything in §4.1, plus fund-level features, plus **peer signals**: what other funds in the same cluster did with this stock last quarter. Herding is a real effect and makes Task B feed directly into Task C, which is a nice design point.

**C2: Which new stocks will the fund buy?**
- This is a **recommender-system** problem with implicit feedback, like "users who bought X also bought Y".
- **Approaches:** item-item collaborative filtering on the fund × stock matrix; matrix factorization (ALS via the `implicit` library); then a **two-tower model** (a fund tower from fund features and history, a stock tower from stock characteristics) trained with negative sampling.
- **Metrics:** recall@20, NDCG@20 and hit rate. "Of the stocks the fund actually initiated, how many were in our top 20?"
- **Baseline:** recommend the stocks most bought by all hedge funds last quarter (popularity), and the stocks most bought by the fund's cluster.

**Splitting and leakage (where most student projects quietly fail):**
- **Split by time, never randomly.** For example: train 2014–2020, validate 2021–2022, test 2023–2025. Use a **walk-forward** (expanding window) evaluation for the final numbers.
- **Point-in-time features:** the quarter-t filing becomes public up to 45 days after quarter end. Any price or fundamental feature must be dated on or before the filing date.
- **Amendments:** use the version of the filing that was public at the time, not a later 13F-HR/A, or your model will "know" confidential positions early.
- **Survivorship:** funds that shut down stop filing; keep them in training rather than only using funds alive today.

**Metrics for C1:** macro-F1 and per-class precision/recall (especially for *exit* and *add*, the rare and useful classes), log-loss, a calibration plot, and AUC for the binary "exit vs. not" version.

### Task D: Backtest ("does it make money?")

Keep this honest and simple. It's the part finance interviewers will push hardest on.
- **Strategy 1, clone:** on each filing date, buy the top-N highest-conviction positions of a fund or cluster and hold until the next filing.
- **Strategy 2, predicted adds:** buy the stocks your C2 model ranks highest across funds; optionally short the predicted exits.
- **Benchmarks:** SPY and an equal-weighted universe. Report annualized return, volatility, Sharpe, max drawdown, turnover, and **Fama-French 5 + momentum alpha** with t-stats.
- **Realism:** trade on the filing date (not quarter end), include a simple transaction-cost assumption (e.g., 10 bps per side), and don't tune on the test period.

A null result is fine and still resume-worthy if it's rigorous: "After the 45-day lag and costs, cloning captures X% of the funds' implied alpha" is a real finding. Don't oversell.

---

## 6. Tech stack and repo layout

- **Python 3.11+**, **polars** or pandas, **DuckDB + Parquet** for storage (fast and laptop-friendly; the full dataset fits comfortably).
- **scikit-learn, LightGBM, SHAP, umap-learn, hdbscan, gensim** (fund2vec), **implicit** (ALS), **PyTorch** (sequence model, two-tower, optional GNN).
- **Streamlit** for the dashboard, **matplotlib/plotly** for charts.
- **pytest** for data-quality tests (e.g., "portfolio weights sum to 1", "no duplicate (fund, CUSIP, quarter)").

```
hedge-fund-13f/
├── README.md               # pitch, data limits, results table, demo GIF
├── data/                   # gitignored; raw/, interim/, processed/ (parquet)
├── src/
│   ├── ingest/             # download SEC data sets, parse XML, dedupe amendments
│   ├── securities/         # CUSIP → ticker mapping, prices, fundamentals
│   ├── features/           # position- and fund-level features (point-in-time)
│   ├── models/
│   │   ├── fingerprint.py  # Task A
│   │   ├── embeddings.py   # Task B
│   │   ├── moves.py        # Task C1
│   │   └── recommender.py  # Task C2
│   └── backtest/           # Task D
├── notebooks/              # EDA and figures only; logic lives in src/
├── app/                    # Streamlit dashboard
└── tests/
```

---

## 7. How to build it: step-by-step milestones

The plan assumes about 8–10 hours a week over roughly 12 weeks. Each milestone ends with something you could show someone. If time runs short, **milestones 1–5 plus the README are already a complete project**; the rest makes it excellent.

### Milestone 0: Setup and reading (week 1)
- [ ] Create the GitHub repo, environment (`uv` or conda) and folder layout above.
- [ ] Skim the SEC's 13F FAQ and the 13F data set README, and read the Cremers & Petajisto and Cohen-Polk-Silli papers (abstract, intro, main tables).
- [ ] Pick your seed list of ~100 funds and record their CIKs in `seed_funds.csv`.
- **Done when:** the repo exists and you can explain 13F limits in two minutes.

### Milestone 1: Ingest holdings (weeks 1–2)
- [ ] Download the SEC 13F quarterly data sets from 2014 onward.
- [ ] Load INFOTABLE + SUBMISSION + COVERPAGE into DuckDB; keep 13F-HR, handle 13F-HR/A (restatement vs. new holdings).
- [ ] Fix the thousands-vs-dollars unit change at 2023.
- [ ] Write data-quality tests.
- **Done when:** you can query "Pershing Square's holdings in Q2 2024" and the numbers match what WhaleWisdom or the filing shows.

### Milestone 2: Securities master, prices and fundamentals (weeks 2–3)
- [ ] Map CUSIPs to tickers (13(f) list + OpenFIGI, cached locally). Track your match rate by portfolio value; aim for >95%.
- [ ] Pull daily prices and compute quarterly returns; pull fundamentals from XBRL; download Fama-French factors.
- [ ] Build point-in-time stock characteristics (size, value, momentum, profitability, investment, beta, volatility, sector).
- **Done when:** you have a `stock_quarter` table with characteristics and next-quarter returns.

### Milestone 3: Feature panel and EDA (weeks 3–4)
- [ ] Build the fund × stock × quarter and fund × quarter feature tables (§4).
- [ ] Define the hedge fund universe (seed list + heuristics).
- [ ] Do EDA: distribution of position counts, turnover, the most-crowded stocks over time, and a fund's portfolio over 10 years.
- **Done when:** you have 3–4 clean charts you'd put in the README.

### Milestone 4: Strategy fingerprint, Task A (week 5)
- [x] Holdings-based factor tilts and a radar chart per fund.
- [x] Returns-based FF5+Mom regressions, compared with the holdings-based tilts.
- [x] Build weak labels (13D, merger targets, heuristics) and hand-label a test set.
- [x] Train LightGBM + SHAP; produce style-drift charts.
- **Done when:** you can show "here's what Fund X really does, and here's why the model thinks so."

### Milestone 5: Embeddings and clustering, Task B (week 6)
- [x] Cosine baseline → SVD/NMF → fund2vec.
- [x] HDBSCAN clusters + UMAP map; stability and purity metrics.
- [x] Crowding analysis for one or two stress episodes.
- **Done when:** you have the "map of the hedge fund industry" figure and a short written interpretation.

### Milestone 6: Next-quarter moves, Task C (weeks 7–9)
- [x] Build targets, time-based splits and baselines first. Write down the baseline numbers before training anything.
- [x] C1: logistic regression → LightGBM → (optional) GRU. Add peer/cluster features and measure the lift.
- [x] C2: popularity → item-item CF → ALS → (optional) two-tower.
- [x] Do a leakage audit: re-check every feature's timestamp against the filing date.
- **Done when:** you have a results table with baselines vs. models on the held-out test years, plus SHAP for C1.

### Milestone 7: Backtest, Task D (week 10)
- [ ] Clone and predicted-add strategies, traded on filing dates, with costs.
- [ ] Sharpe, drawdown and factor alpha with t-stats vs. SPY.
- **Done when:** you have an equity curve and an honest one-paragraph conclusion.

### Milestone 8: Dashboard and write-up (weeks 11–12)
- [ ] Streamlit app: search a fund → fingerprint radar, nearest-neighbor funds, cluster on the UMAP map, predicted next moves, and recent history.
- [ ] README with the pitch, data limitations, a results table, figures and a 30-second demo GIF.
- [ ] A 1,500-word blog post or PDF write-up on your decisions and what didn't work.
- **Done when:** a recruiter can understand it in 60 seconds and a quant can poke at it for 30 minutes.

---

## 8. Pitfalls to plan for

1. **CUSIP mapping eats time.** Start it early, cache everything, and measure coverage by value, not by row count.
2. **Look-ahead bias.** It's the #1 thing finance interviewers will probe. Keep a `known_at` date on every table.
3. **Accuracy is misleading.** Most positions are "hold". Report macro-F1, per-class metrics and baselines.
4. **Units and splits.** Thousands vs. dollars, stock splits changing share counts (use split-adjusted shares when computing adds and trims, or you'll "detect" a huge add on every split), and duplicate rows for the same CUSIP across sub-managers.
5. **Shorts are invisible.** Don't claim to know a long/short fund's net exposure.
6. **Big multi-manager platforms** (Citadel, Millennium) hold thousands of positions with huge turnover and behave very differently. Model them separately or exclude them from C1.
7. **Scope creep.** The GNN, transformer and two-tower model are stretch goals. A rigorous LightGBM with good baselines beats a sloppy transformer.

---

## 9. How to present it

**Resume bullets** (fill in your real numbers once you have them):
- *Built an end-to-end pipeline ingesting 10 years of SEC 13F filings (~N million positions across M hedge funds) into a point-in-time DuckDB/Parquet dataset with automated data-quality tests.*
- *Inferred fund strategies from holdings using factor-tilt fingerprints and a SHAP-explained LightGBM style classifier (macro-F1 X on hand-labeled funds); clustered funds with embeddings and HDBSCAN to measure crowding.*
- *Predicted next-quarter position changes and new buys (macro-F1 X vs. Y baseline; recall@20 Z), and backtested a clone strategy with filing-date execution and costs (Sharpe S, FF5 alpha A).*

**Talking points by role**
- **Alternative asset management / hedge funds:** holdings-based style analysis, style drift, crowding risk, why confidential positions matter, and what 13F can't tell you.
- **Equity research:** which stocks smart money is accumulating, the "best ideas" effect, and how ownership changes relate to subsequent returns.
- **Sales & trading:** positioning and crowding (who's long what, and what happens when a cluster de-grosses), flows around filing dates.
- **GCM (capital markets):** institutional ownership and investor targeting. Which funds are likely buyers for an IPO or follow-on given their fingerprint is a real use of the C2 recommender.

**Interview questions you should be ready for**
- "How do you know there's no look-ahead bias?" Point to your `known_at` dates and walk-forward splits.
- "What does 13F miss?" Shorts, non-US positions, derivatives, intra-quarter trades and confidential positions.
- "Does it make money?" Give the honest backtest numbers, the effect of the 45-day lag, and costs.
- "Why LightGBM over a neural net?" Tabular data, interpretability, and you checked that the sequence model didn't add much (or did, by how much).

---

## 10. What to do this week

1. Create the repo and environment.
2. Download one quarter of the SEC 13F data set and load INFOTABLE into DuckDB.
3. Find 5 funds you find interesting, look up their CIKs, and print their top 10 holdings for that quarter.
4. Compare against the actual filing on EDGAR to make sure your units and parsing are right.

A minimal EDGAR request, for when you need individual filings:

```python
import requests

HEADERS = {"User-Agent": "Your Name your.email@example.com"}  # the SEC requires this
cik = "0001336528"  # example CIK; look up each fund on EDGAR company search
url = f"https://data.sec.gov/submissions/CIK{cik}.json"
filings = requests.get(url, headers=HEADERS, timeout=30).json()["filings"]["recent"]

thirteen_f = [
    (form, date, acc)
    for form, date, acc in zip(filings["form"], filings["filingDate"], filings["accessionNumber"])
    if form.startswith("13F-HR")
]
print(thirteen_f[:8])
```

Check each fund's CIK on EDGAR's company search rather than trusting the example above, and keep requests under 10 per second.

Once milestone 1 is done, that's a good point to come back here. I can help you review the schema, write the CUSIP mapping, or set up the feature pipeline.
