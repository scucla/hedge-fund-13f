# Hedge fund 13F project

ML project that infers hedge fund strategies from SEC 13F filings, clusters similar funds, and predicts next-quarter position changes. The full plan is in `docs/plan.md`; work milestone by milestone and don't start a later milestone until the current one's "done when" check passes.

## Stack
- Python 3.11+, managed with `uv`. polars for dataframes, DuckDB + Parquet for storage.
- scikit-learn, LightGBM, SHAP for modeling. pytest for tests.
- Logic lives in `src/`; notebooks are for exploration and figures only.

## Layout
- `data/raw/`, `data/interim/`, `data/processed/` (gitignored).
- `src/ingest/`, `src/securities/`, `src/features/`, `src/models/`, `src/backtest/`.
- `tests/` holds data-quality and unit tests.

## Rules that matter
- **Point-in-time only.** Every table carries a `known_at` date (the filing date for holdings, the publish date for fundamentals). Features may only use rows with `known_at` on or before the prediction date.
- **13F values:** reported in thousands of dollars before 2023, whole dollars from 2023 on. Normalize to dollars at ingest.
- **Amendments:** keep 13F-HR and 13F-HR/A separate; record whether an amendment restates or adds holdings. Never let a later amendment leak into an earlier quarter's features.
- **Splits:** compare share counts on a split-adjusted basis when computing adds and trims.
- **Time-based splits only** for any model evaluation. Never split randomly. Never tune on the test years.
- **SEC requests:** send a `User-Agent` header with name and email from the `SEC_USER_AGENT` env var, stay under 10 requests/second, and cache every download under `data/raw/`.
- Every new table gets at least one test (unique keys, no null CUSIPs, weights sum to ~1 per fund-quarter).

## Working style
- Explain non-obvious design choices in a short comment or in `docs/decisions.md`, because the owner needs to defend them in interviews.
- Prefer small, reviewable changes. Run `uv run pytest` before saying something is done.
