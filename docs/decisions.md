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
