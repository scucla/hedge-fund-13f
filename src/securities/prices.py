"""Daily prices from yfinance and quarterly returns.

Usage:
    uv run python -m src.securities.prices

Prices: split- and dividend-adjusted closes (auto_adjust), so returns are
total returns. Downloaded in chunks, each cached under data/raw/prices/ so a
rerun only fetches what's missing. Writes `prices_daily`, `prices_missing`
and `returns_quarterly` to the securities database.

Survivorship bias: Yahoo drops most delisted tickers, so stocks acquired or
delisted before today are missing (see docs/decisions.md).
"""

import hashlib
import json
import logging
import re
import time
from datetime import date
from pathlib import Path

import polars as pl
import yfinance as yf

from src.securities import db

RAW_DIR = Path("data/raw/prices")
START = "2023-01-01"
CHUNK = 100
PRICED_CLASSES = ("common", "adr", "etf", "fund")


def symbols_to_price(con) -> list[str]:
    return [r[0] for r in con.execute(f"""
        SELECT DISTINCT symbol FROM security_master
        WHERE symbol IS NOT NULL AND security_class IN {PRICED_CLASSES}
          AND (exch_code = 'US' OR exch_code LIKE 'U_')
        ORDER BY symbol
    """).fetchall()]


NO_DATA_PATH = RAW_DIR / "no_data.json"
PAUSE_SECONDS = 4
EMPTY = {"symbol": pl.Utf8, "date": pl.Date, "close": pl.Float64, "volume": pl.Float64}


def _cache_dir(end: str) -> Path:
    return RAW_DIR / f"{START}_{end}"


def load_cache(end: str) -> tuple[pl.DataFrame, set[str]]:
    """Cached prices plus symbols Yahoo confirmed it has no data for. Only
    successful downloads and genuine 'no data' answers are cached; rate-limit
    failures never are, so a rerun retries them."""
    files = sorted(_cache_dir(end).glob("*.parquet"))
    prices = pl.concat([pl.read_parquet(f) for f in files]) if files else pl.DataFrame(schema=EMPTY)
    no_data = set(json.loads(NO_DATA_PATH.read_text())) if NO_DATA_PATH.exists() else set()
    return prices, no_data


class _ErrorLog(logging.Handler):
    """yfinance logs failures as "['AAA', 'BBB']: <error message>" and keeps no
    public per-symbol error record, so read them from its log."""

    def __init__(self):
        super().__init__(logging.ERROR)
        self.by_symbol: dict[str, str] = {}

    def emit(self, record):
        msg = record.getMessage()
        if m := re.match(r"^\[(.*?)\]: (.*)$", msg, re.S):
            for sym in re.findall(r"'([^']+)'", m.group(1)):
                self.by_symbol[sym] = m.group(2)


# Only these messages mean Yahoo really has nothing; anything else is retried.
NO_DATA_MESSAGES = ("possibly delisted", "no price data found", "no data found", "symbol may be delisted")


def download_chunk(symbols: list[str], end: str) -> tuple[pl.DataFrame, set[str], set[str]]:
    """Returns (prices, symbols with no data, symbols that failed for other reasons)."""
    log = _ErrorLog()
    yf_logger = logging.getLogger("yfinance")
    yf_logger.addHandler(log)
    try:
        raw = yf.download(symbols, start=START, end=end, auto_adjust=True, group_by="ticker",
                          threads=True, progress=False)
    except Exception as e:  # e.g. YFRateLimitError raised outright: retry the whole chunk
        print(f"  download raised {type(e).__name__}; will retry")
        return pl.DataFrame(schema=EMPTY), set(), set(symbols)
    finally:
        yf_logger.removeHandler(log)
    errors = log.by_symbol
    frames, got = [], set()
    present = set(raw.columns.get_level_values(0)) if len(raw.columns) else set()
    for sym in symbols:
        if sym not in present or sym in errors:
            continue
        sub = raw[sym][["Close", "Volume"]].dropna(subset=["Close"])
        if sub.empty:
            continue
        got.add(sym)
        frames.append(pl.DataFrame({
            "symbol": sym,
            "date": [d.date() for d in sub.index],
            "close": sub["Close"].astype(float).to_list(),
            "volume": sub["Volume"].astype(float).to_list(),
        }))
    no_data = {s for s, e in errors.items() if any(m in e.lower() for m in NO_DATA_MESSAGES)} - got
    # Missing without any message, or with any other error (rate limits, timeouts): retry.
    failed = set(symbols) - got - no_data
    out = pl.concat(frames) if frames else pl.DataFrame(schema=EMPTY)
    if len(out):
        d = _cache_dir(end)
        d.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(",".join(sorted(got)).encode()).hexdigest()[:12]
        out.write_parquet(d / f"{digest}.parquet")
    return out, no_data, failed


def quarterly_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Return over each calendar quarter, from the last close of the previous
    quarter to the last close of this one. known_at = the last trading day used."""
    q_end = (
        prices.with_columns(
            (pl.col("date").dt.year().cast(pl.Utf8) + "Q" + pl.col("date").dt.quarter().cast(pl.Utf8)).alias("quarter"))
        .sort("symbol", "date")
        .group_by("symbol", "quarter", maintain_order=True)
        .agg(pl.col("date").last().alias("last_date"), pl.col("close").last().alias("close_q"))
        .sort("symbol", "quarter")
    )
    return (
        q_end.with_columns(
            pl.col("close_q").shift(1).over("symbol").alias("close_prev"),
            pl.col("quarter").shift(1).over("symbol").alias("prev_quarter"),
        )
        .with_columns(
            pl.when(pl.col("prev_quarter") == _previous_quarter(pl.col("quarter")))
            .then(pl.col("close_q") / pl.col("close_prev") - 1)
            .alias("ret_q"),
            pl.col("last_date").alias("known_at"),
        )
        .select("symbol", "quarter", "last_date", "close_q", "ret_q", "known_at")
    )


def _previous_quarter(q: pl.Expr) -> pl.Expr:
    year = q.str.slice(0, 4).cast(pl.Int32)
    n = q.str.slice(5, 1).cast(pl.Int32)
    return (pl.when(n == 1).then(year - 1).otherwise(year).cast(pl.Utf8) + "Q"
            + pl.when(n == 1).then(4).otherwise(n - 1).cast(pl.Utf8))


def build(end: str | None = None) -> None:
    end = end or date.today().isoformat()
    # Hold the database only briefly: the download takes a long time and other
    # steps need the database meanwhile.
    with db.connect(read_only=True) as con:
        symbols = symbols_to_price(con)
    cached, no_data = load_cache(end)
    todo = [s for s in symbols if s not in set(cached["symbol"].unique()) and s not in no_data]
    print(f"  {len(symbols):,} symbols: {len(symbols) - len(todo):,} cached, {len(todo):,} to fetch")
    frames, backoff = [cached], 60
    while todo:
        chunk, todo = todo[:CHUNK], todo[CHUNK:]
        out, none, failed = download_chunk(chunk, end)
        frames.append(out)
        no_data |= none
        NO_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        NO_DATA_PATH.write_text(json.dumps(sorted(no_data)))
        if failed:  # rate limited: put them back and wait longer each time
            todo = sorted(failed) + todo
            print(f"  rate limited on {len(failed)} symbols; waiting {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)
        else:
            backoff = 60
            time.sleep(PAUSE_SECONDS)
        print(f"  {len(todo):,} left")
    prices = pl.concat(frames).filter(pl.col("symbol").is_in(symbols)).unique(["symbol", "date"]).sort("symbol", "date")
    missing = pl.DataFrame({"symbol": sorted(set(symbols) - set(prices["symbol"].unique()))}, schema={"symbol": pl.Utf8})
    returns = quarterly_returns(prices)
    with db.connect() as con:
        con.execute("CREATE OR REPLACE TABLE prices_daily AS SELECT * FROM prices")
        con.execute("CREATE OR REPLACE TABLE prices_missing AS SELECT * FROM missing")
        con.execute("CREATE OR REPLACE TABLE returns_quarterly AS SELECT * FROM returns")
    print(f"prices_daily: {len(prices):,} rows, {prices['symbol'].n_unique():,} symbols; "
          f"{len(missing):,} symbols with no data")


def coverage_report(con) -> None:
    print("\nPrice coverage by share of seed-fund value (priced classes only):")
    print(con.execute(f"""
        WITH priced AS (SELECT DISTINCT symbol FROM prices_daily)
        SELECT p.quarter,
               sum(p.value_usd) FILTER (WHERE m.symbol IN (SELECT symbol FROM priced)) / sum(p.value_usd) AS priced
        FROM seed_positions p JOIN security_master m USING (cusip)
        WHERE m.security_class IN {PRICED_CLASSES}
        GROUP BY 1 ORDER BY 1
    """).pl())
    print("\nLargest positions with no price data:")
    with pl.Config(tbl_rows=15, fmt_str_lengths=36):
        print(con.execute("""
            SELECT m.symbol, any_value(m.figi_name) AS name, any_value(m.security_class) AS class,
                   round(sum(p.value_usd) / 1e6, 1) AS value_all_q_musd, max(p.quarter) AS last_quarter
            FROM seed_positions p JOIN security_master m USING (cusip)
            WHERE m.symbol IN (SELECT symbol FROM prices_missing)
            GROUP BY 1 ORDER BY 4 DESC LIMIT 15
        """).pl())


if __name__ == "__main__":
    build()
    coverage_report(db.connect(read_only=True))
