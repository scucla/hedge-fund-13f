"""Daily prices from yfinance and quarterly returns.

Usage:
    uv run python -m src.securities.prices [--start 2023-01-01]

Prices: split- and dividend-adjusted closes (auto_adjust), so returns are
total returns. Writes `prices_daily`, `prices_missing` and
`returns_quarterly` to the securities database.

Cache: one parquet file per symbol in data/raw/prices/store/, plus
manifest.json recording the date window fetched for each symbol (or that
Yahoo has no data for it). A run downloads only what is missing: a newer
tail or an older head. Yahoo restates adjusted history after every dividend
or split by one constant factor, so each extra piece overlaps the stored data
by a few days and the stored history is rescaled to the new piece's basis
(median price ratio on the overlap). Without that, levels would jump at the
seam; with no usable overlap the symbol is re-downloaded whole.

Survivorship bias: Yahoo drops most delisted tickers, so stocks acquired or
delisted before today are missing (see docs/decisions.md).
"""

import argparse
import json
import logging
import re
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import yfinance as yf

from src.securities import db

RAW_DIR = Path("data/raw/prices")
STORE_DIR = RAW_DIR / "store"
MANIFEST_PATH = RAW_DIR / "manifest.json"
RATE_LIMIT_PATH = RAW_DIR / "last_rate_limit.txt"   # ISO timestamp of the latest rate-limit hit
START = "2014-01-01"   # 12-1 momentum and 1-year beta for 2015Q1 need prices from early 2014
CHUNK = 100
PAUSE_SECONDS = 4
OVERLAP_DAYS = 10            # calendar days of overlap used to stitch pieces
MAX_RATIO_SPREAD = 0.002     # overlap ratios must agree within 0.2%, else re-download
PRICED_CLASSES = ("common", "adr", "etf", "fund")
EMPTY = {"symbol": pl.Utf8, "date": pl.Date, "close": pl.Float64, "volume": pl.Float64}


def symbols_to_price(con) -> list[str]:
    return [r[0] for r in con.execute(f"""
        SELECT DISTINCT symbol FROM security_master
        WHERE symbol IS NOT NULL AND security_class IN {PRICED_CLASSES}
          AND (exch_code = 'US' OR exch_code LIKE 'U_' OR map_source IN ('openfigi_foreign_listing', 'override'))
          -- retired CUSIPs: the old ticker may now trade as a different company
          AND coalesce(map_source, '') <> 'openfigi_retired'
        ORDER BY symbol
    """).fetchall()]


# --- store ---------------------------------------------------------------------

def _path(sym: str) -> Path:
    return STORE_DIR / f"{sym.replace('/', '_')}.parquet"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}


def save_manifest(m: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=0, sort_keys=True))
    tmp.replace(MANIFEST_PATH)


def stitch(old: pl.DataFrame, new: pl.DataFrame) -> pl.DataFrame | None:
    """Combine stored prices with a newly fetched piece. The stored part is
    rescaled to the new piece's adjustment basis using the overlap; None if
    there is no overlap or the ratio isn't constant (adjustments are
    multiplicative and apply to all earlier dates, so it should be)."""
    if old.is_empty():
        return new
    ov = old.join(new, on="date", suffix="_new").filter((pl.col("close") > 0) & (pl.col("close_new") > 0))
    if ov.is_empty():
        return None
    ratios = ov["close_new"] / ov["close"]
    r = ratios.median()
    if (ratios / r - 1).abs().max() > MAX_RATIO_SPREAD:
        return None
    kept = old.filter(~pl.col("date").is_in(new["date"].implode())).with_columns(pl.col("close") * r)
    return pl.concat([kept, new.select(old.columns)]).sort("date")


def missing_pieces(entry: dict | None, start: str, end: str) -> list[tuple[str, str]]:
    """Windows [from, to) still to fetch for one symbol, with overlap."""
    if entry is None:
        return [(start, end)]
    if entry.get("no_data"):
        # a "no data" answer only covers the window it was given for
        return [] if entry["from"] <= start and entry["to"] >= end else [(start, end)]
    pieces = []
    ov = timedelta(days=OVERLAP_DAYS)
    if start < entry["from"]:
        pieces.append((start, (date.fromisoformat(entry["from"]) + ov).isoformat()))
    if end > entry["to"]:
        pieces.append(((date.fromisoformat(entry["to"]) - ov).isoformat(), end))
    return pieces


# --- download -------------------------------------------------------------------

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
NO_DATA_MESSAGES = ("possibly delisted", "no price data found", "no data found", "symbol may be delisted",
                    "data doesn't exist for startdate")   # e.g. asking for 2014-2022 for a 2023 IPO
REQUEST_TIMEOUT = 30   # seconds; a dropped connection must fail, not hang


def download_chunk(symbols: list[str], start: str, end: str) -> tuple[dict[str, pl.DataFrame], set[str], set[str]]:
    """Returns ({symbol: prices}, symbols with no data, symbols that failed for other reasons).
    The last call's error messages are kept in LAST_ERRORS."""
    global LAST_ERRORS
    log = _ErrorLog()
    yf_logger = logging.getLogger("yfinance")
    yf_logger.addHandler(log)
    try:
        raw = yf.download(symbols, start=start, end=end, auto_adjust=True, group_by="ticker",
                          threads=len(symbols) > 1, progress=False, timeout=REQUEST_TIMEOUT)
    except Exception as e:  # e.g. YFRateLimitError raised outright: retry the whole chunk
        print(f"  download raised {type(e).__name__}; will retry")
        LAST_ERRORS = {s: f"{type(e).__name__}: {e}" for s in symbols}
        return {}, set(), set(symbols)
    finally:
        yf_logger.removeHandler(log)
    errors = log.by_symbol
    LAST_ERRORS = dict(errors)
    got = {}
    multi = getattr(raw.columns, "nlevels", 1) > 1
    present = set(raw.columns.get_level_values(0)) if multi and len(raw.columns) else (set(symbols) if len(raw) else set())
    for sym in symbols:
        if sym not in present or sym in errors:
            continue
        frame = raw[sym] if multi else raw
        sub = frame[["Close", "Volume"]].dropna(subset=["Close"])
        if sub.empty:
            continue
        got[sym] = pl.DataFrame({
            "symbol": sym,
            "date": [d.date() for d in sub.index],
            "close": sub["Close"].astype(float).to_list(),
            "volume": sub["Volume"].astype(float).to_list(),
        }, schema=EMPTY)
    no_data = {s for s, e in errors.items() if any(m in e.lower() for m in NO_DATA_MESSAGES)} - set(got)
    # Missing without any message, or with any other error (rate limits, timeouts): retry.
    failed = set(symbols) - set(got) - no_data
    return got, no_data, failed


def fetch(symbols: list[str], start: str, end: str) -> None:
    """Bring the store up to [start, end) for every symbol."""
    manifest = load_manifest()
    todo: dict[tuple[str, str], list[str]] = {}
    for s in symbols:
        for piece in missing_pieces(manifest.get(s), start, end):
            todo.setdefault(piece, []).append(s)
    n = sum(len(v) for v in todo.values())
    print(f"  {len(symbols):,} symbols; {n:,} pieces to fetch in {len(todo)} distinct windows")
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    backoff, redo_all = 60, []
    for (p_start, p_end), syms in sorted(todo.items()):
        queue = list(syms)
        while queue:
            chunk, queue = queue[:CHUNK], queue[CHUNK:]
            got, none, failed = download_chunk(chunk, p_start, p_end)
            redo = []
            for s, new in got.items():
                entry = manifest.get(s)
                old = pl.read_parquet(_path(s)) if entry and not entry.get("no_data") and _path(s).exists() \
                    else pl.DataFrame(schema=EMPTY)
                merged = stitch(old, new)
                if merged is None:      # can't stitch safely: fetch this symbol's whole window again
                    redo.append(s)
                    manifest.pop(s, None)
                    continue
                merged.write_parquet(_path(s))
                lo = min(p_start, entry["from"]) if entry and not entry.get("no_data") else p_start
                hi = max(p_end, entry["to"]) if entry and not entry.get("no_data") else p_end
                manifest[s] = {"from": lo, "to": hi}
            for s in none:
                entry = manifest.get(s)
                if entry and not entry.get("no_data"):
                    entry["from"], entry["to"] = min(p_start, entry["from"]), max(p_end, entry["to"])
                else:
                    manifest[s] = {"from": p_start, "to": p_end, "no_data": True}
            save_manifest(manifest)
            if redo:
                redo_all += redo
                print(f"  {len(redo)} symbols couldn't be stitched; will re-fetch their full window")
            if failed:  # rate limited: put them back and wait longer each time
                _record_rate_limit()
                queue = sorted(failed) + queue
                print(f"  rate limited on {len(failed)} symbols; waiting {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 600)
            else:
                backoff = 60
                time.sleep(PAUSE_SECONDS)
            print(f"  window {p_start}..{p_end}: {len(queue):,} left")
    # Symbols that couldn't be stitched had their manifest entry removed, so this
    # pass fetches their whole window fresh (and terminates: nothing to stitch).
    if redo_all:
        for s in redo_all:
            _path(s).unlink(missing_ok=True)
        fetch(sorted(set(redo_all)), start, end)


LAST_ERRORS: dict[str, str] = {}
SLOW_PAUSE = 3.0                  # seconds between requests in slow mode
SLOW_BACKOFF = 30 * 60            # after a rate limit
SLOW_COOLDOWN = 2 * 60 * 60       # no request until this long after the last rate limit


def _is_rate_limit(msg: str) -> bool:
    m = msg.lower()
    return "ratelimit" in m or "too many requests" in m or "rate limited" in m


def _wait_for_cooldown() -> None:
    from datetime import datetime
    if not RATE_LIMIT_PATH.exists():
        return
    last = datetime.fromisoformat(RATE_LIMIT_PATH.read_text().strip())
    wait = (last.timestamp() + SLOW_COOLDOWN) - time.time()
    if wait > 0:
        print(f"  last rate limit at {last:%H:%M:%S}; waiting {wait / 60:.0f} min before any request")
        time.sleep(wait)


def _record_rate_limit() -> None:
    from datetime import datetime
    RATE_LIMIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RATE_LIMIT_PATH.write_text(datetime.now().isoformat(timespec="seconds"))


def fetch_slow(symbols: list[str], start: str, end: str) -> None:
    """One symbol per request, ~3 s apart, in the given order (largest seed-fund
    value first). Waits 2 h after the last rate limit before starting and backs
    off 30 min on each new one. Other failures are retried a few times, then the
    symbol is skipped for this run (not marked "no data")."""
    _wait_for_cooldown()
    manifest = load_manifest()
    work = [(s, piece) for s in symbols for piece in missing_pieces(manifest.get(s), start, end)]
    print(f"  slow mode: {len(work):,} pieces for {len({s for s, _ in work}):,} symbols")
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    skipped, done = [], 0
    for sym, (p_start, p_end) in work:
        tries = 0
        while True:
            got, none, failed = download_chunk([sym], p_start, p_end)
            time.sleep(SLOW_PAUSE)
            if sym in got:
                entry = manifest.get(sym)
                old = pl.read_parquet(_path(sym)) if entry and not entry.get("no_data") and _path(sym).exists() \
                    else pl.DataFrame(schema=EMPTY)
                merged = stitch(old, got[sym])
                if merged is None:   # can't stitch: replace with a full-window download
                    p_start, p_end = start, end
                    manifest.pop(sym, None)
                    _path(sym).unlink(missing_ok=True)
                    continue
                merged.write_parquet(_path(sym))
                lo = min(p_start, entry["from"]) if entry and not entry.get("no_data") else p_start
                hi = max(p_end, entry["to"]) if entry and not entry.get("no_data") else p_end
                manifest[sym] = {"from": lo, "to": hi}
                break
            if sym in none:
                entry = manifest.get(sym)
                if entry and not entry.get("no_data"):
                    # Yahoo has nothing in this window for a symbol we hold (e.g. it listed
                    # later): record the window as covered so it isn't asked again.
                    entry["from"], entry["to"] = min(p_start, entry["from"]), max(p_end, entry["to"])
                else:
                    manifest[sym] = {"from": p_start, "to": p_end, "no_data": True}
                break
            msg = LAST_ERRORS.get(sym, "")
            if _is_rate_limit(msg):
                _record_rate_limit()
                print(f"  rate limited at {sym} ({done:,}/{len(work):,} done); backing off {SLOW_BACKOFF // 60} min")
                time.sleep(SLOW_BACKOFF)
                continue
            tries += 1
            if tries >= 3:
                skipped.append(sym)
                print(f"  skipped {sym} for this run: {msg[:120] or 'no data and no message'}")
                break
            time.sleep(30)
        done += 1
        if done % 25 == 0:
            save_manifest(manifest)
        if done % 250 == 0:
            print(f"  {done:,}/{len(work):,} pieces done")
    save_manifest(manifest)
    print(f"  slow mode finished: {done:,} pieces, {len(skipped)} skipped this run")


def symbols_by_value(con) -> list[str]:
    """Priced symbols ordered by total seed-fund value over all quarters, largest first."""
    priced = set(symbols_to_price(con))
    ranked = [r[0] for r in con.execute("""
        SELECT m.symbol FROM seed_positions p JOIN security_master m USING (cusip)
        WHERE m.symbol IS NOT NULL GROUP BY m.symbol ORDER BY sum(p.value_usd) DESC
    """).fetchall()]
    return [s for s in ranked if s in priced] + sorted(priced - set(ranked))


def load_store(symbols: list[str], start: str, end: str) -> pl.DataFrame:
    manifest = load_manifest()
    frames = [pl.read_parquet(_path(s)) for s in symbols
              if s in manifest and not manifest[s].get("no_data") and _path(s).exists()]
    if not frames:
        return pl.DataFrame(schema=EMPTY)
    return (pl.concat(frames).filter((pl.col("date") >= date.fromisoformat(start))
                                     & (pl.col("date") < date.fromisoformat(end)))
            .unique(["symbol", "date"]).sort("symbol", "date"))


def migrate_chunk_cache() -> None:
    """One-off: move the old per-run chunk folders (data/raw/prices/<start>_<end>/)
    into the per-symbol store. Later folders win; the old no_data.json becomes
    manifest entries for the newest window."""
    dirs = sorted(d for d in RAW_DIR.glob("*_*") if d.is_dir() and d.name != "store")
    if not dirs or MANIFEST_PATH.exists():  # once the store exists, old folders must never overwrite it
        return
    manifest = load_manifest()
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    for d in dirs:   # oldest first, so newer windows overwrite
        start, end = d.name.split("_")
        df = pl.concat([pl.read_parquet(f) for f in d.glob("*.parquet")])
        for (sym,), part in df.group_by(["symbol"]):
            part.select(list(EMPTY)).sort("date").write_parquet(_path(sym))
            manifest[sym] = {"from": start, "to": end}
    old_no_data = RAW_DIR / "no_data.json"
    if old_no_data.exists():
        start, end = dirs[-1].name.split("_")
        for s in json.loads(old_no_data.read_text()):
            manifest.setdefault(s, {"from": start, "to": end, "no_data": True})
    save_manifest(manifest)
    print(f"  migrated {len(dirs)} chunk folders into {STORE_DIR} ({len(manifest):,} symbols)")


# --- returns -------------------------------------------------------------------------

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


def build(start: str = START, end: str | None = None, slow: bool = False) -> None:
    end = end or date.today().isoformat()
    # Hold the database only briefly: the download takes a long time and other
    # steps need the database meanwhile.
    with db.connect(read_only=True) as con:
        symbols = symbols_by_value(con) if slow else symbols_to_price(con)
    migrate_chunk_cache()
    if slow:
        fetch_slow(symbols, start, end)
    else:
        fetch(symbols, start, end)
    prices = load_store(symbols, start, end)
    # Yahoo's dividend adjustment can push old adjusted closes to zero or below
    # (seen: BROGF, GMEX); the whole adjusted series is then unusable.
    broken = set(prices.filter(pl.col("close") <= 0)["symbol"].unique())
    prices = prices.filter(~pl.col("symbol").is_in(list(broken)))
    have = set(prices["symbol"].unique())
    missing = pl.DataFrame(
        [{"symbol": s, "reason": "non-positive adjusted close" if s in broken else "no data"}
         for s in sorted(set(symbols) - have)],
        schema={"symbol": pl.Utf8, "reason": pl.Utf8})
    returns = quarterly_returns(prices)
    with db.connect() as con:
        con.execute("CREATE OR REPLACE TABLE prices_daily AS SELECT * FROM prices")
        con.execute("CREATE OR REPLACE TABLE prices_missing AS SELECT * FROM missing")
        con.execute("CREATE OR REPLACE TABLE returns_quarterly AS SELECT * FROM returns")
    print(f"prices_daily: {len(prices):,} rows, {prices['symbol'].n_unique():,} symbols; "
          f"{len(missing):,} symbols missing ({len(broken)} dropped for non-positive adjusted closes)")


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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--slow", action="store_true",
                    help="one symbol per request, ~3 s apart, largest seed-fund value first; 2 h cooldown, 30 min backoff")
    args = ap.parse_args()
    build(args.start, args.end, args.slow)
    coverage_report(db.connect(read_only=True))
