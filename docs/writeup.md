# What 11 years of 13F filings can and can't tell you about hedge funds

Every quarter, US investment managers with more than $100 million in listed equities file a Form 13F: a list
of their long positions as of quarter end, due 45 days later. It is the only systematic, free window into what
hedge funds hold. I wanted to see how far it goes. Can you tell a fund's strategy from its book? Can you find
funds that are secretly alike? Can you predict next quarter's moves? And can you make money copying them? This
write-up covers the decisions that mattered, the leaks I caught (several of them in my own code), and the
things that didn't work.

## The setup

The data is 77 "seed" funds (activists, Tiger cubs, multi-managers, quants, healthcare specialists, merger
arbitrage, concentrated value) from 2015 to 2026, plus 16 holdout funds, two per style. I labelled the holdout
funds by hand and kept them out of every training set. Everything is built from SEC data sets and EDGAR,
with Yahoo prices, SEC XBRL fundamentals and Kenneth French's factors.

The first rule of the project was that **every table carries a `known_at` date**: the date the information became
public. A feature can only use rows whose `known_at` is on or before the prediction date. That rule sounds
obvious. Most of the interesting bugs were places where it was quietly broken.

## Decisions that mattered

**Original filings, not restated ones.** Funds file amendments (13F-HR/A), sometimes years later. 11% of
filer-quarters have one, a median of six months after the original. The first version of the holdings table was "as
of today", so a 2019 quarter already included positions disclosed in 2021. The fix was to keep two views: the
latest restated truth, and each quarter as of its original filing. Every feature uses the second. This hit
event-driven funds hardest: amendments add 8.6% of their value, because merger-arbitrage positions are often
filed under confidential treatment and disclosed after the deal. So a merger-arb fund looks *less* like a
merger-arb fund on the day an observer could have seen it. That is the right answer, not a bug.

**Filling the price gap honestly.** Yahoo has no price at all for a stock that stopped trading before today. In 2015,
a third of the seed funds' value had no return. Dropping those rows would bias every result, because the missing
stocks are disproportionately acquisition targets and failures. I filled them in three ways:
- **Returns implied by the funds themselves:** the median of value/shares across funds, quarter to quarter.
- **Delisting returns:** deal terms parsed from merger proxies, and a −30% convention for performance delistings.
- **An explicit `ret_source` on every label,** so any model can be rerun with or without the filled rows.

Coverage went from 66% to 99% of value. One finding shaped the rest: rows that every fund
sold are likely losers, so "has a label" must never be a feature.

**Labels you can defend.** There are no ground-truth strategy labels in 13F. I used a weak rule-based first pass,
reviewed the disagreements, and corrected the labels (v2). Because the correction looked at where the model
disagreed, the seed-fund scores are optimistic by construction. The only clean evaluation is the 16 holdout funds,
scored once. There the classifier reaches macro-F1 0.58 across eight styles (majority class: 0.05). Its mistakes
fall on real boundaries: multi-manager vs quant, activist vs concentrated value.

**Choosing the embedding before looking at the test years.** For "who's alike" I compared four representations:
cosine on raw weights, SVD, NMF, and fund2vec (word2vec over funds' books as "sentences" of stocks). The
selection rule was written down first: rank each method on cluster stability, agreement with the style labels
and next-year return correlation, using pre-2023 quarters only. fund2vec tied with the cosine baseline and won
the tie-break by calling fewer funds noise. The map it produces is the project's best figure:
- Healthcare specialists and merger-arb funds form tight islands.
- Tiger cubs sit together.
- The multi-managers and quants form one big diversified blob.

## The leaks I caught

1. **Amendments**, above.
2. **Same-quarter peers.** "How many other funds hold this stock" sounds harmless. But funds file quarter q on
   different days, so on the day fund A files, fund B's quarter-q filing often doesn't exist yet. The naive
   crowding count used a not-yet-public filing on 27% of rows. The fix: each peer is seen through its latest
   filing public on the prediction date.
3. **Cluster timing.** A quarter's clusters were fitted on filings up to 60 days after quarter end. A fund filing
   on day 40 can't know them. 79% of rows end up using the previous quarter's clusters.
4. **The embargo.** A training row's label is public at the fund's *next* filing. So the 2021Q4 labels arrive in
   mid-2022, inside the validation year. Training on them uses outcomes nobody had when the first validation
   prediction was made. That drops 40,000 training rows at the 2022 boundary, and the same rule applies at the 2023 boundary.
5. **Label-selected backtest rows.** My saved test predictions covered only positions with a next-quarter label.
   But whether a fund files next quarter, or whether a stock is acquired, is future information. For the backtest
   the frozen model scores every position.
6. **Buying the dead.** A 13F can list a stock that was taken over between quarter end and the trade date. The
   backtest was "buying" CSRA a month after its last trading day. The delisting date is public, so those names are
   now skipped.

An automated audit now checks every feature's timestamp row by row:
- 16 checks, all at zero violations.
- An independent recount of peer features from raw positions.
- Synthetic tests that inject each leak and confirm the audit catches it.

## What the models found

**Next-quarter moves (C1)** are more predictable than I expected, but not for the reason I expected. On the
2023–2026 test years, LightGBM reaches macro-F1 0.49 against 0.38 for the best baseline (the fund's own action
rates last quarter). Two surprises:
- **"Hold" is the rarest class,** not the majority the plan guessed. With 10% bands, active books are always
  moving, so "predict hold" is a useless baseline.
- **SHAP is dominated by the position itself:** its size, its last change, its age. The fund's own habits come next.

**Herding was the hypothesis that didn't survive.** Peer and cluster features ("what did similar funds just do
with this stock?") improve log-loss by 0.0035. That is statistically real across fund-quarters, and economically
nothing. Once you know a position's own history, the crowd tells you little.

**New buys (C2)** were humbling. Item-item collaborative filtering and ALS both lost to a one-line baseline:
recommend the stocks the fund sold in the past year (NDCG@20 0.164 vs 0.156 for ALS). Most predictable new buys are
*returns* to old names. The recommenders only did well when given four quarters of history, which is them
rediscovering re-entry. Cluster popularity did beat plain popularity by about 40%, the clearest sign that the
clusters carry trading information.

## Does it make money?

No. The backtest trades the day after each quarter's filing deadline, using only filings public by then, at
10–20 bp per side. The consensus best-ideas portfolio returned 9.3% a year over 2018–2026, against 14.5% for
the S&P 500, with a Fama-French five-factor plus momentum alpha of −1.8% (t = −0.7). The best style clone
(growth/tech) matched the market with more risk: its return is beta and growth exposure. The frozen model
portfolios on the test years split: the C2 new-buy list beat SPY by about two points a year (t = 0.9), and the
C1 predicted adds lagged it. The re-entry list, the best *predictor* of what funds buy, was the worst
*portfolio*: predicting what funds do is not the same as predicting what goes up. With about ten strategies and
40 test months, none of it is distinguishable from noise. That is the standard finding after the 45-day lag,
and I'd rather report it than tune until something wins.

## What I'd do differently

- **Tuning on a single year.** All hyperparameters were chosen on 2022, a rate-hike, growth-unwind year with
  unusually high turnover. Walk-forward re-tuning would be more robust. The LightGBM grid was nearly flat,
  which limits the damage.
- **Hand labels.** Ninety-three hand-labelled funds is enough for a classifier, not enough to settle where
  "activist" ends and "concentrated value" begins.
- **The merger-arb blind spot.** Confidential treatment hides exactly the positions that define these funds. A
  point-in-time 13F will always understate them. The honest fix is to say so, not to peek at the amendments.
- **Spending time on leaks earlier.** Every leak above was found by asking "when could I have known this?" about
  a feature that already worked. The audit should have come first, not in Milestone 6.

The code, every decision and every number above are in the repository. The decision log
(`docs/decisions.md`) records each choice with its reason, including the ones that turned out wrong.
