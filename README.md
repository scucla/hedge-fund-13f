# Reverse-engineer a hedge fund

Infers hedge fund strategies from SEC 13F filings, clusters similar funds, and predicts next-quarter position changes. See [`docs/plan.md`](docs/plan.md) for the full plan and [`docs/decisions.md`](docs/decisions.md) for design choices.

**Status:** Milestones 0–2 done; Milestone 3 features built for 2015Q1–2026Q2 (fund books, positions, fund-quarter features, EDA in `notebooks/`).

## What 13F data can't tell you

13F filings show only long positions in US-listed stocks (plus listed options), as quarter-end snapshots, published up to 45 days late. They don't show shorts, non-US stocks, bonds, futures, FX, cash or intra-quarter trades, and funds can delay disclosing some positions. Any conclusion here is about the visible long book only.

## Setup

```bash
uv sync
cp .env.example .env   # then add your name and email; the SEC requires it
```

## Usage

```bash
uv run python -m src.ingest.download --start 2025Q3 --end 2026Q2   # SEC data sets → data/raw/
uv run python -m src.ingest.load                                    # → data/interim/13f.duckdb
uv run python -m src.ingest.show_fund --cik 2026053 --quarter 2026Q2   # Pershing Square Inc.
uv run pytest
```

### Milestone 2: securities

```bash
uv run python -m src.securities.universe       # seed funds' point-in-time positions
uv run python -m src.securities.cusip_map      # CUSIP -> ticker (needs OPENFIGI_API_KEY for speed)
uv run python -m src.securities.prices         # yfinance daily prices (slow: Yahoo rate-limits)
uv run python -m src.securities.fundamentals   # SEC XBRL companyfacts + SIC codes
uv run python -m src.securities.factors        # Fama-French 5 + momentum, FF12 industries
uv run python -m src.securities.stock_quarter  # point-in-time characteristics -> data/processed/securities.duckdb
```

### Milestone 3: history and features

```bash
uv run python -m src.ingest.backfill --start 2015Q1 --end 2025Q2          # seed-fund 13F history, one data set at a time
uv run python -m src.ingest.check_units --cik 1336528 --quarter 2016Q4 --db data/interim/13f_history.duckdb
uv run python -m src.securities.prices --start 2014-01-01 --slow          # one symbol per request; respects Yahoo limits
uv run python -m src.features.books                                        # one book per fund-quarter
uv run python -m src.features.positions                                    # fund x stock x quarter features
uv run python -m src.features.fund_quarter                                 # fund x quarter features
```
