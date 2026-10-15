# Design decisions

Short records of non-obvious choices, newest at the bottom. Each one says what was decided and why.

## 2026-10-14: Python and tooling

- **Python 3.11 via `uv`**, pinned in `.python-version`. The system Python on the dev machine is 3.9; `uv` installs its own interpreter so the project doesn't depend on it.
- **The SEC User-Agent lives in `.env`** (gitignored) and is read through `SEC_USER_AGENT`. `.env.example` shows the format without the real email.

## 2026-10-14: Downloading the SEC data sets (`src/ingest/download.py`)

- **Since March 2024 the SEC publishes 13F data sets by filing-date window** (e.g. `01jun2026-31aug2026_form13f.zip`), not by calendar quarter. Before that, files were `YYYYqN_form13f.zip` and held the filings made *during* that quarter. The downloader takes *holdings* quarters and picks the file whose window contains that quarter's 45-day filing deadline, so both schemes work with one rule.
- **URLs come from the SEC listing page, not a template.** The newest file was under `/files/datastandardsinnovation/...` while older ones are under `/files/structureddata/...`; a hard-coded pattern would have broken.
- **Downloads write to a `.part` file and rename when complete**, so an interrupted download is never mistaken for a cached one.
- **Requests are spaced 0.2 s apart** (≤5/s, half the SEC's 10/s limit) and send `SEC_USER_AGENT`.

## 2026-10-14: Loading into DuckDB (`src/ingest/load.py`)

- **Values are normalized to dollars by *filing date*, not report period.** The SEC readme says market value is "rounded to the nearest dollar" starting January 3, 2023, and in thousands before. The data confirms it: a 2001 holdings report filed in 2026 is already in dollars.
- **`known_at` = filing date.** That is when the public could first see the holdings. `quarter` comes from the report period, never from the file the filing arrived in: the Jun–Aug 2026 file also contains late filings for older quarters (one goes back to 2001).
- **Three tables:**
  - `filings`: one row per 13F-HR / 13F-HR/A. Notices (13F-NT, and amendments whose report type is NOTICE) are dropped because they carry no holdings. `amendment_type` (RESTATEMENT / NEW HOLDINGS) is kept as filed.
  - `infotable_raw`: every row, keyed on the SEC's own primary key `(accession_number, infotable_sk)`. We first planned a wider key `(accession, CUSIP, put/call, discretion, other manager)`, but it isn't unique: ~222k of 3.8M rows in 2026Q2 repeat it because some filers split one position into lots.
  - `holdings`: `infotable_raw` combined per `(accession, CUSIP, put/call)`. This is the table with the strict uniqueness test. Options stay separate from the stock (`put_call` = '' for the stock itself so it can be in a key).
- **Shares and bond principal are summed into separate columns.** ~1,090 holdings report one CUSIP partly in shares (SH) and partly in principal (PRN); adding those would be meaningless.
- **$0 values are kept.** ~0.3% of rows are reported at exactly $0 (mostly tiny or worthless positions, median 8 shares). The test checks "no negative or missing values" instead of "strictly positive", plus a check that zeros stay under 1%.
- **The build writes to a temp file and swaps it in only on success**, and builds `holdings` one source file at a time. Grouping all 14M rows at once spilled several GB of temp data and failed on a nearly full disk.

## 2026-10-14: Which filings count for a fund-quarter (`src/ingest/positions.py`)

Point-in-time rule, as of a date:
1. Only filings with `known_at <= as_of`.
2. Base report = the latest 13F-HR original or RESTATEMENT amendment (a restatement replaces the whole report). 4 fund-quarters have two originals; the later one wins.
3. NEW HOLDINGS amendments filed after the base are added on top (often positions that had confidential treatment).
4. Amendments with no amendment type (9 in the current data) are ambiguous, so they're skipped with a warning. We go by `submission_type`, not `amendment_type`, to decide original vs. amendment, because 5 originals carry an amendment type by mistake.

## 2026-10-14: One firm, several filer IDs

Pershing Square files under two CIKs. For 2025Q3–2026Q1 both *Pershing Square Capital Management* (1336528) and *Pershing Square Holdco / Inc.* (2026053) filed holdings reports; for 2026Q2 Capital Management filed only a 13F-NT notice and Pershing Square Inc. reported the holdings. The seed list in Milestone 0 needs to map each fund to *all* of its CIKs, and avoid double-counting quarters where both entities report the same book.

## 2026-10-14: Seed list → filer IDs (`src/ingest/seed_ciks.py`, `data/reference/`)

- **`data/reference/` is committed; the rest of `data/` is not.** The seed list and the reviewed CIK map are small, hand-made and can't be re-downloaded.
- **Candidate CIKs come from the EDGAR quarterly full-index (`src/ingest/edgar_index.py`)**, 1999–2026: every 13F filing with the filer's name *at the time*. That finds old entity names a current-name search misses (SAC Capital → Point72, OZ Management → Sculptor, JGD Management → York, CNH Partners → AQR Arbitrage, Duquesne Capital → Duquesne Family Office).
- **The fund → CIK mapping is reviewed by hand (`seed_cik_map.csv`), not automatic.** Name matching alone is wrong too often: "York" matches New York Life, "Millennium" matches Marque Millennium, "Elliott & Associates" is now an Indianapolis wealth manager. Doubtful matches were settled by comparing business addresses in the EDGAR profile; ones that still can't be confirmed are kept with `role = uncertain` and left out of the overlap flag.
- **Only CIKs that filed at least one 13F-HR are included.** Coatue alone has ~40 fund entities that only file 13F-NT notices pointing to the manager; those carry no holdings.
- **Roles:** `primary` (the main filer today), `predecessor` (an earlier ID that stopped when another took over), `successor` (DME Capital for Greenlight), `affiliate` (another entity of the same firm filing in parallel, including founders who file personally), `uncertain`.
- **`multi_cik_same_quarter`** = two or more non-uncertain CIKs filed an original 13F-HR for the same report period (from `reportDate` in each EDGAR profile, not the filing date). It mixes two different situations that later milestones must treat differently:
  - *Hand-overs* (1–4 quarters, e.g. Elliott 2020, Brave Warrior 2012): one book changing filer, possibly reported twice in the switch quarter.
  - *Parallel filers* (e.g. Soros, Two Sigma, AQR, Millennium, Point72, Steadfast, Icahn): either separate books (Point72's regional entities) or the same book reported twice (a founder filing personally). Which one needs a holdings comparison; that's for the feature milestone, so no fund-quarter gets double-counted.
- **`style_guess` is a weak label**, copied as given. It will be checked against hand labels in Milestone 4.

## 2026-10-14: Seed list review decisions (owner)

- **Abrams: the intended fund is Abrams Capital Management** (David Abrams, Boston; CIK 1358706, with 1426355 as its 2008–2010 predecessor), not Abrams Bison Investments (Bethesda, MD), which the original list named by mistake.
- **Unconfirmed IDs are dropped from the mapping.** Only Gotham Asset Management LLC (1510387, Greenblatt's current firm) counts for Gotham. Removed: Gotham Asset Management LP (1279148), Gotham Select Fund Management LP (1352839), Gotham Select Fund GP (1352421), and York GP, Ltd. (1289994). If any is confirmed later, add it back to `seed_cik_map.csv`.
- **Funds that stopped filing stay in.** Scion (last 13F-HR for 2025Q3) and Tybourne (2025Q3) are kept: dropping funds that closed would bias every historical result toward survivors.
- **DME Capital Management (1489933) is Greenlight's successor filer.** Greenlight Capital Inc (1079114) filed its last 13F-HR in February 2024; DME (David M. Einhorn's firm) shares Greenlight's address, 140 East 45th Street, and files from then on. Both IDs map to "Greenlight Capital".

## 2026-10-14: Milestone 2 scope and storage

- **Scope: securities the seed funds held in 2025Q3–2026Q2** (`seed_positions`, built with the point-in-time filing rules from `positions.py`). All mapped CIKs are included, affiliates too; de-duplicating parallel filers is for the feature milestone.
- **Milestone 2 tables live in `data/processed/securities.duckdb`**, separate from `data/interim/13f.duckdb`, because `src.ingest.load` rebuilds the 13F database from scratch. Long downloads open the database only at the start and end so other steps can run meanwhile.

## 2026-10-14: CUSIP → ticker (`src/securities/cusip_map.py`)

- **The SEC 13(f) list has no tickers.** It gives issuer name, a short description and an options-listed flag; tickers come from OpenFIGI.
- **CUSIPs that start with a letter are CINS codes** (non-US domicile: G = Cayman/Bermuda/Ireland, H = Switzerland, N = Netherlands, L = Luxembourg…). OpenFIGI only maps them as `ID_CINS`. Retrying them that way took unmapped value from 4.7% to 0.13% (Chubb, ASML, Seagate, Spotify, Linde, Accenture, Medtronic…). Last resort: the FIGI some filers put in their own 13F rows.
- **Listing choice:** the US composite listing (`exchCode = US`), then any US exchange, then anything.
- **Security class from OpenFIGI's `securityType` before `securityType2`**: ETFs come back as `ETP` / `Mutual Fund`, and reading the second field first labelled every ETF a mutual fund.
- **Coverage is measured by value, not rows:** 99.8–99.9% of seed-fund value maps to a ticker in every quarter. The unmapped rest is mostly SPACs and recently delisted names.
- **The mapping is current, not point-in-time.** OpenFIGI returns today's ticker for a CUSIP. A company that changed ticker maps to its new one (fine for prices, since Yahoo history follows the security), and a CUSIP whose security no longer trades may not map at all. `security_master.known_at` is NULL to make that explicit.

## 2026-10-14: Prices (`src/securities/prices.py`)

- **yfinance adjusted closes (`auto_adjust=True`)**, so returns include dividends and splits. Adjusted history is restated by Yahoo after every dividend; that changes price *levels* but not returns, and only returns are used.
- **Survivorship bias.** Yahoo drops most delisted tickers. Stocks that were acquired or delisted between the holdings date and today have no prices, so their features and labels are missing, and missing labels skew toward names that left the market (often takeovers, which tend to have positive returns, or failures, negative). This is a known limitation of the free stack; CRSP fixes it. Price coverage by value is reported every run, and `prices_missing` lists every symbol with no data.
- **Only US listings in the priced classes** (common, ADR, ETF, closed-end fund) are downloaded. Warrants, rights, units and preferreds (0.2% of value) are out of scope; their OpenFIGI symbols rarely match Yahoo's.
- **Rate limits:** Yahoo throttles bulk downloads. Only successful symbols and genuine "no data" answers are cached; everything else is retried with exponential backoff. A symbol counts as "no data" only when yfinance's error message says so ("possibly delisted", "no price data found"); yfinance 1.7 keeps no public per-symbol error record, so the messages are read from its log. Two earlier versions got this wrong (one cached throttled chunks as empty, one read an error dict yfinance no longer fills) and made AAPL and SPY look delisted. Final run: 9,219 of 9,370 symbols priced, 97.0–98.8% of seed-fund value per quarter; the 151 confirmed misses are mostly names acquired or delisted since (CyberArk, Comerica, Verona Pharma, Frontier, TEGNA, Civitas).

## 2026-10-14: Fundamentals (`src/securities/fundamentals.py`)

- **SEC companyfacts per issuer, every version kept, `known_at` = filing date.** A restatement is a new row with a later `known_at`, never an overwrite. Features as of date d use, for each concept and period, the latest version filed on or before d.
- **USD only.** Foreign filers reporting in other currencies get no fundamentals rather than mixing currencies.
- **Fallback concepts in priority order** (e.g. `StockholdersEquity`, then including non-controlling interest; IFRS equivalents). For shares outstanding: the cover-page `dei:EntityCommonStockSharesOutstanding`, then balance-sheet shares, then weighted-average basic shares. Companies with several share classes (Alphabet) tag the cover-page count per class, which companyfacts leaves out, hence the fallbacks.
- **Sector from the SIC code in each issuer's EDGAR profile.** That is the company's current SIC, not a history; SIC codes rarely change.

## 2026-10-14: Factors (`src/securities/factors.py`)

- **Fama-French 5 factors + momentum, daily and monthly, from Ken French's library**, converted from percent to decimals. French revises past values when he updates the files, so this is a fixed-after-the-fact series. It's used for beta and later risk adjustment, not as a feature anyone could have known at the time.
- **FF12 industries come from French's own `Siccodes12` file**, not a hand-typed table. SIC codes outside every listed range are industry 12 ("Other"), as French defines it.

## 2026-10-14: `stock_quarter` (`src/securities/stock_quarter.py`)

- **As of the calendar quarter end** (owner decision): 13F holdings are reported as of that date, so it's the natural snapshot. The label `ret_next_q` is the next calendar quarter's total return, with its own `label_known_at`. Lining up the 45-day filing lag is for the prediction and backtest milestones.
- **Book-to-market is NULL when book equity is negative, zero or missing; the stock stays in the table** (owner decision). Dropping them would remove exactly the distressed and buyback-heavy firms some funds specialise in.
- **Gross profitability (Novy-Marx 2013) instead of Fama-French operating profitability** (owner decision): FF's definition needs interest expense and SG&A, which XBRL filers often don't tag, so coverage would be poor. Annual gross profit / total assets at the same fiscal year end; revenue − cost of revenue when gross profit isn't tagged.
- **No market cap (so no size or book-to-market) for ADRs.** Their XBRL share count is in ordinary shares, not ADRs (one TSMC ADR = 5 shares), and the ratio isn't tagged. ADRs still get book equity, profitability, asset growth and the price-based features. ETFs and closed-end funds get price-based features only.
- **Staleness guards:** a "quarter-end price" must be from the last 10 days; book equity older than ~18 months and share counts older than ~13 months are ignored.
- **Beta and volatility** use daily returns over the past 365 calendar days (at least 120 days), regressed on the FF market excess return. Daily data gives a usable estimate within one year, which a 60-month monthly beta can't for recent listings.
- **Momentum (12-1)** is the return from the month-end 12 months before the quarter end to the month-end 1 month before it, skipping the last month (short-term reversal).
- **Fixes found by checking the real table:**
  - *Linked CIKs (`data/reference/issuer_cik_links.csv`, hand-reviewed).* ExxonMobil reorganized under a new holding company in 2026 (CIK 2115436, first XBRL filing 2026-08-03), and the SEC ticker file now points XOM there. The history is under the old Exxon Mobil Corp CIK (34088), so linked CIKs' facts are loaded under the current issuer. Other issuers whose filings start after they were first held are almost all 2025 IPOs and spin-offs (Figma, Chime, Circle, Klarna, Ralliant, Amrize); having no fundamentals before their first filing is correct point-in-time behaviour, not a gap.
  - *Share counts ≤ 0 are ignored* (36 rows had market cap 0 from a mis-tagged zero).
  - *A fact's period end must also be on or before the quarter end*, not just its filing date: one filing carried a balance sheet dated after its own filing date.
- **Known gaps, not fixed:** banks and insurers have no gross profit, so `gp_at` is NULL for them (expected); Canadian issuers filing on Form 40-F (Teck, Rogers, Manulife, CGI) have little or no XBRL data; Yahoo has no prices for names delisted since (survivorship, above).
- **Finding, not a bug:** defensive and energy stocks (KO, PEP, PG, JNJ, WMT, XOM, CVX) have small *negative* betas in 2026Q2 (12 months to June 2026), after positive betas in earlier quarters. The pipeline was checked: SPY, priced and aligned the same way, has 0.992 correlation and 0.96 beta with the FF market factor. It is a market regime, and worth remembering when beta is used as a feature.

## 2026-10-15: Pre-commit checks on `stock_quarter` (owner-requested)

- **Negative book equity: no tag problem.** Of common-stock rows that have a book-equity value, 10–12% are negative per quarter (8.6–11.1% of all common-stock rows). The 25 largest by market cap are the known buyback- and debt-funded names (AbbVie, Philip Morris, Dell, McDonald's, Booking, Lowe's, Altria, Starbucks, Marriott, McKesson, HCA, Hilton, TransDigm, AutoZone, O'Reilly, Yum, Verisk…), all from the primary `StockholdersEquity` tag. Five were checked against the balance sheet in the 10-Q itself (EDGAR's rendered R pages, cached under `data/raw/edgar/spotcheck/`), and all match to the reported precision: ABBV −6,656, LOW −9,270, SBUX −8,465.1, MCD −1,286, VRSK −1,167.7 ($M). Verisk confirms the parent-only line is used, not total equity including non-controlling interest.
  - **Why the rate is above market-wide levels:** the universe is what hedge funds hold. SPACs (SIC 6770) are 181 of 527 negatives in 2026Q2; 83% of SPACs show negative stockholders' equity because their redeemable Class A shares sit in temporary equity, outside stockholders' equity. Without SPACs the rate is ~8%, the rest being cash-burning small caps (biotech, recent IPOs with accumulated deficits) and de-SPACs whose SIC code is no longer 6770. `bm` is already NULL for all of them; anyone filtering out SPACs can use `sic = 6770`, though that misses de-SPACs.
- **Negative betas are real; no alignment bug.** For the year to 2026-06-30, betas against SPY daily returns are within 0.03 of the Fama-French ones for every name checked (KO −0.28, PEP −0.17, PG −0.06, JNJ −0.10, WMT −0.04, XOM −0.38, CVX −0.34; AAPL 0.87; SPY on itself 1.00). Against SPY *weekly* returns (52 points, noisier): CVX −0.43, JNJ −0.39, XOM −0.84 stay negative; KO −0.01 and WMT 0.00 are flat; PEP +0.19 and PG +0.41 turn slightly positive. Kept as computed (daily, FF market). Defensive and energy stocks had roughly zero-to-negative market sensitivity that year.
- **Beta after French's data ends: fall back to SPY** (my call). For days after the last date in the FF daily file, the market excess return is SPY's daily return minus the last published RF. `beta_market` = `'ff'` or `'ff+spy'` and `n_proxy_days` count the SPY days, so the switch is visible. Chosen over marking betas incomplete because SPY tracks the FF market closely (daily correlation 0.992, SPY beta 0.96 on it), and dropping September would also have dropped those days from `vol`. No row uses the fallback yet (the latest quarter ends 2026-06-30); 2026Q3 holdings will.

## Open issues

- **ADRs have no size or book-to-market.** Their XBRL share count is ordinary shares, not ADRs (one TSMC ADR = 5 shares), and the ratio isn't tagged. Fix later with an ADR-ratio source (depositary bank ratio lists or the 20-F cover page) or a market cap from another provider. ADRs are ~3% of seed-fund value.
- **Survivorship bias in prices** (yfinance drops delisted names): 151 symbols with no data, mostly companies acquired since the holdings date. CRSP via WRDS would fix it.
- **Canadian 40-F filers** (Teck, Rogers, Manulife, CGI…) have little or no XBRL data, so no fundamentals.
- **Banks and insurers have no gross profit**, so `gp_at` is NULL for them by construction.
- **De-SPACs keep SPAC-style balance sheets for a while** and aren't flagged; SIC 6770 only catches SPACs that haven't completed a merger.
- **The CUSIP → ticker mapping is current, not point-in-time** (OpenFIGI returns today's ticker).
- **Parallel filers aren't de-duplicated yet** (e.g. Soros Fund Management and George Soros personally), so `seed_positions` can count the same book twice. Due in the feature milestone.

## 2026-10-15: Milestone 3 Phase A — fund books (`src/features/books.py`)

- **One `fund_id` per seed fund per quarter.** For every fund-quarter with more than one filing CIK, the portfolios are compared: overlap = Σ min(w_a, w_b) over common positions. **≥ 0.9 → the same book reported twice: keep one** (primary > successor > affiliate > predecessor, then the larger book). **Otherwise the books are separate and added up.** Every decision is in `fund_book_decisions` with its reason and overlap.
- **Decisions on the current four quarters:** no pair is a duplicate; the highest overlap is 0.57 (Two Sigma Investments vs Two Sigma Advisers). So all parallel filers are added: AQR + AQR Arbitrage (overlap 1%), Millennium + WorldQuant Millennium (18%), Soros Fund Management + George Soros personally (7–9%), Pershing Square Capital Management + Pershing Square Holdco/Inc. (9–11%), Point72 + its regional entities (5–27%), Two Sigma Investments + Advisers. Icahn, Trian and Steadfast didn't co-file in these quarters. The duplicate rule is there for the hand-over quarters the backfill will bring.
- **Two Sigma Securities is excluded** (`data/reference/fund_book_exclusions.csv`): it's a broker-dealer, and its 13F is market-making inventory, not an investment decision.
- **Empty books are kept as facts.** A 13F-HR whose only row is a placeholder (CUSIP 000000000, "N/A", no value or shares) means the fund held no 13F securities that quarter. Mantle Ridge did this in 2026Q2 after selling Dollar Tree. Placeholder rows are dropped from positions; the fund-quarter is kept in `empty_books` and in `fund_quarter` with `n_positions = 0`. Rows with CUSIP 000000000 but a real issuer and value (filer errors, $85M in total) are kept as positions without stock features.

## 2026-10-15: Positions (`src/features/positions.py`)

- **Two weights.** `weight` = value / total reported value, all rows, so it sums to 1 per fund-quarter as specified. `weight_long` = value / total of non-option rows. 13F reports options at the *underlying's* value, so they would swamp the book (Citadel: 80% of reported value is options); concentration, active share and tilts use `weight_long`.
- **Splits are detected from the data, not a corporate-actions feed.** For each stock and quarter, each fund that held it in both quarters gives (reported shares t / t−1) ÷ (value-implied shares t / t−1, using Yahoo's split-adjusted close). Funds that just held on show the split ratio exactly; the median across funds is snapped to a standard ratio (1.25, 1.5, 2…50, 100 or reciprocals) **only if within 1.2%**. Real splits land within 0.5% (NFLX 10.000, NOW 5.000, BKNG 25.06, KLAC 10.01, TPL 3.006, BN 1.502). Spin-offs and special dividends also move Yahoo's adjusted prices without changing share counts, and land further off (BDX 1.28 after its 2026 spin-off, LILAK 1.47, AIV 1.40, ELME 6.5); applying them would invent 20%+ position cuts, so they're rejected. 59 splits accepted, 37 candidates rejected (`stock_splits`). Missing a spin-off is harmless, because share changes compare reported counts directly.
- **Position age is left-censored** in the first quarter of data (`age_censored`); the backfill fixes that for later quarters.
- **Crowding** = number of *other* seed funds holding the same security long that quarter (by ticker; GOOG and GOOGL count separately).
- **`known_at` = the fund-quarter's filing date** (latest among its included CIKs). Holdings-based features can't be known before the filing, which is after the quarter end; stock characteristics stay as of the quarter end. The tests check each against its own date.

## 2026-10-15: Fund-quarter features (`src/features/fund_quarter.py`)

- **Active share uses a proxy benchmark.** There are no free S&P 500 constituent weights, so the benchmark is the 500 largest *issuers* by market cap in `stock_quarter`, cap-weighted. Issuer level, so Alphabet's two share classes aren't counted twice. Limits: `stock_quarter` only covers stocks some seed fund holds (in practice every large cap), and ADRs have no market cap, so they can't be in the benchmark.
- **Turnover = min(buys, sells) / average long book value** (the mutual-fund definition). Buys and sells are split-adjusted quantity changes at this quarter's price; a new position is a buy of its value, an exit a sale of last quarter's value grown by the stock's return. NULL when the previous quarter isn't filed. Known property: a fund that only sells (e.g. Mantle Ridge selling Air Products) shows 0, because the measure ignores one-sided flows. Values above 1 (possible through price moves) are clipped and flagged (`turnover_clipped`); none so far.
- **Factor tilts:** each characteristic (log ME, log B/M, 12-1 momentum, GP/A, asset growth, beta) is winsorized at 1%/99% and z-scored across that quarter's common stocks (equal-weighted), then averaged with `weight_long` over the fund's stocks that have a value; `cov_*` is that covered weight. The cross-section is *stocks held by seed funds*, not the whole market, so a z-score of 0 means "average hedge-fund holding". Every fund tilts large because funds value-weight large caps against an equal-weighted cross-section.
- **A DuckDB trap:** `greatest(NULL, 0)` returns 0, not NULL. Clipping turnover that way turned "no previous quarter" into "no trading"; a synthetic test caught it.
- **`gross_turnover` = (buys + sells) / 2 / average long book** (owner request), next to the standard min(buys, sells) measure. The min ignores one-sided trading; gross doesn't: Mantle Ridge selling Air Products in 2026Q1 is 0 vs 0.31, and its full exit in 2026Q2 (an empty book) is 0 vs 1.0. For an empty book after a filed quarter, turnover = 0 and gross_turnover = 1 by definition. Median across fund-quarters: 0.19 standard, 0.25 gross.
- **Flows are bounded by what's possible.** Bond rows (principal, no shares) use value changes, not quantity × price: filers aren't consistent about principal units, and one Sculptor convertible (JD.com) implied a $123B trade on a $3–19M position, pushing gross turnover above 1. And no sale can exceed last quarter's holding grown by the stock's return. Since that fix, no fund-quarter needs clipping.
