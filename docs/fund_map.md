# Who's secretly alike? Reading the fund map and the crowding episodes

Milestone 5 (Task B). Numbers and tables: `docs/embeddings_clustering.md`, `docs/crowding.md`.
Design choices: `docs/decisions.md` (2026-10-16 entries).

## The map (figure 12)

![map](figures/12_fund_map.png)

The fund2vec map for 2026Q2 recovers the industry's main structure from holdings alone:

- **Specialists form the tightest groups.** Healthcare (C0: Baker Bros., RA, Deerfield,
  OrbiMed, Perceptive and four others; 8 of 9 labelled healthcare) and merger arbitrage
  (C3: Glazer, HBK, Pentwater, Water Island, Westchester; every one of the five holds EA,
  WBD, ROKU and TECH, all pending deals) are the clearest clusters in every quarter.
  Their nearest outsiders among the background filers also make sense: RTW near
  healthcare, and Tenor, Whitebox and Polar near merger arb.
- **The Tiger family splits in two.** C6 (Coatue, Tiger Global, Altimeter, Whale Rock,
  Light Street, Maverick, plus Appaloosa) all hold TSM, AMZN, NVDA and GOOGL. C7 (D1,
  Dragoneer, Atreides) is set apart by private-to-public stakes. Baillie Gifford and
  Situational Awareness are their nearest outside neighbours.
- **Multi-managers and quants are one cluster (C2, 17 funds).** Their books are broad and
  close to the market, so holdings can't tell Millennium from Two Sigma. That is a limit
  of 13F data, not of the method, and it is where ~390 of the 500 background filers land.
- **Activists and concentrated value funds are mostly noise** (22 of 73 funds in 2026Q2).
  This is correct rather than a failure: a 10-stock activist book shares almost nothing
  with any other fund. HDBSCAN says so, where k-means would force each one into a group.

## How good are the clusters?

The main method was **chosen on 2016–2022 only** and logged and committed before any 2023+
metric existed. fund2vec tied with the plain cosine baseline on the pre-set rule and won
the tie-break on noise (26% vs 46% of funds). The 2023–2026 numbers are therefore a
genuine out-of-sample check:

| fund2vec, HDBSCAN | 2016–2022 | 2023–2026 |
|---|---|---|
| noise share | 0.26 | 0.28 |
| purity vs v2 styles (shuffled labels) | 0.73 (0.33) | 0.74 (0.35) |
| ARI vs v2 styles | 0.40 | 0.35 |
| stability, ARI vs previous quarter | 0.74 | 0.76 |
| within − across correlation of next-year clone returns | 0.43 | 0.44 |

- **Clusters are about twice as pure as chance and persist quarter to quarter.**
- **They mean something economically.** Funds in the same cluster have clone returns over
  the next 12 months that are much more correlated than funds in different clusters,
  after removing the common market move.
- **SVD and NMF are less stable** (ARI 0.53–0.61). NMF's themes are still the most readable
  output, e.g. a "pending deals" theme and a "small-cap biotech" theme.
- **Why alignment was needed:** without the Procrustes alignment, an SVD fund appears to
  move 0.45 (cosine distance) per quarter purely because the basis rotates. Aligned, the
  median move is 0.09.

## Crowding in two stress episodes

**Definition.** A name is **crowded** if at least 3 funds of one cluster, and at least half
of the cluster's members, each hold it at ≥1% of their book. The 1% floor matters: without
it, the multi-manager cluster "crowds" 1,700 names it holds at 0.05% each.

**COVID crash (2019Q4 books, 2020Q1 returns): crowding didn't hurt.**
- Only 13 names were crowded. They beat stocks of the same size by about +20% (mean);
  in a regression with controls the effect is +11%, t = 1.5.
- The growth cluster's consensus names were mega-cap platforms (MSFT 0%, AMZN +6%,
  NFLX +16%), which were exactly what held up in that crash.
- The one cluster hurt by its crowded name was a small activist/event group whose
  consensus name was MGM (−64%).
- The crash was a market-wide liquidity event, so what a stock was exposed to mattered
  more than who owned it.

**Growth unwind (2021Q3 books, returns 2021Q4–2022Q2): crowding hurt.**
- The 18 crowded names fell 38% (median −44%), which is 20–28 points worse than stocks of
  the same size.
- In the regression, crowded is **−23%, t = −2.6**, on top of beta and momentum. More
  conviction holders overall was also negative (t = −2.9).
- The 12-fund Tiger cluster had 29% of its books in 10 shared names: META −52%, SHOP −77%,
  SNOW −54%, PTON −89%. Its crowded names did about 9 points worse than the rest of its
  already falling book (−47% buy-and-hold).
- Healthcare's one crowded name (MRTX, −62%) shows the same pattern on a small scale.

**Crowding wasn't visible beforehand.** Within-cluster book overlap was unremarkable going
into both episodes: the 62nd and 76th percentiles of 2016–2026. The 13F snapshot shows
*who* is crowded together, not *when* the crowding will matter. The highest overlap in the
sample is now, in 2025–2026, driven by AI mega-caps.

**The funds did not visibly run for the exit together.**
- By the next filing, funds had exited crowded positions less often than their other ≥1%
  positions after COVID (11% vs 20%), and only slightly more often in the unwind (21% vs 16%).
- Quarterly 13F snapshots can't show intra-quarter de-grossing, so this is weak evidence.
  The most it supports is that the losses came from holding, not from a coordinated exit.

## Caveats

- **Small sample.** 73–77 seed funds and 13–18 crowded names per episode; the t-statistics
  are cross-sectional and ignore industry clustering of returns.
- **Long book only.** Short positions are invisible in a 13F, so the multi-manager and quant
  clusters are underdescribed and nobody's hedges show.
- **Returns start at quarter end,** not the filing date. This is a description of the funds'
  actual exposure, not a tradable signal; that test is Task D.
- **The background screen is imperfect.** Selecting by active share still admits wealth
  platforms whose ETF holdings count as "active" against a stock benchmark (e.g. LPL,
  Envestnet). They only affect the grey points.

## What this feeds next

`fund_clusters` (with `known_at`) supplies the Task C peer signal: what a fund's cluster did
with a stock last quarter. `crowding_stocks` is a candidate risk feature for Task D.
