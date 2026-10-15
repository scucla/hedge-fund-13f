"""Map the universe's CUSIPs to tickers: the security master.

Usage:
    uv run python -m src.securities.cusip_map

Sources, all cached under data/raw/:
- SEC Official List of Section 13(f) Securities (one TXT per quarter): issuer
  name and description. It has no tickers.
- OpenFIGI /v3/mapping: ticker, FIGI, exchange and security type. CUSIPs it
  can't map are retried with the FIGI some filers put in their 13F rows.
- SEC company_tickers.json: ticker -> issuer CIK (for XBRL fundamentals).

Writes `security_master` (one row per CUSIP) and prints value coverage.
"""

import json
import os
import re
import time
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv

from src.ingest import sec
from src.securities import db

LIST_PAGE = ("https://www.sec.gov/rules-regulations/staff-guidance/"
             "division-investment-management-frequently-asked-questions/official-list-section-13f-securities")
LIST_DIR = Path("data/raw/sec13flist")
FIGI_CACHE = Path("data/raw/openfigi/mapping.jsonl")
TICKERS_PATH = Path("data/raw/sec/company_tickers.json")

FIGI_URL = "https://api.openfigi.com/v3/mapping"
JOBS_PER_REQUEST = 100          # with an API key (10 without)
SECONDS_PER_REQUEST = 6 / 25    # 25 requests per 6 seconds with a key

# OpenFIGI securityType2 / securityType -> our coarse class.
CLASS_BY_TYPE = {
    "Common Stock": "common", "REIT": "common", "Tracking Stk": "common", "Ltd Part": "common",
    "MLP": "common", "Royalty Trst": "common", "Stapled Security": "common",
    "Depositary Receipt": "adr", "ADR": "adr", "GDR": "adr", "NY Reg Shrs": "adr",
    "ETP": "etf", "ETF": "etf", "Mutual Fund": "fund", "Closed-End Fund": "fund", "Open-End Fund": "fund",
    "Preference": "preferred", "Preferred Stock": "preferred",
    "Warrant": "warrant_right", "Right": "warrant_right", "Unit": "warrant_right",
}


# --- 13(f) list ---------------------------------------------------------------

def download_13f_lists(quarters: list[str]) -> list[Path]:
    html = sec.get(LIST_PAGE).text
    links = set(re.findall(r'href="([^"]*13flist(\d{4}q\d)[^"]*\.txt)"', html))
    paths = []
    for href, q in sorted(links):
        if q.upper() not in quarters:
            continue
        path = LIST_DIR / f"13flist{q}.txt"
        if not path.exists():
            LIST_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(sec.get("https://www.sec.gov" + href).content)
        paths.append(path)
    missing = set(quarters) - {p.stem[-6:].upper() for p in paths}
    if missing:
        raise RuntimeError(f"No 13(f) list found for {sorted(missing)}")
    return paths


def parse_13f_list(path: Path) -> pl.DataFrame:
    """Fixed width: CUSIP 1-9, options-listed flag 10, issuer 11-40,
    description 41-67, then *A*/*D* for additions/deletions."""
    rows = []
    for line in path.read_text(encoding="latin-1").splitlines():
        if not re.match(r"^[0-9A-Z]{9}[ *]", line):
            continue
        status = re.search(r"\*([AD])\*", line[60:])
        rows.append((line[:9], line[9] == "*", line[10:40].strip(), line[40:67].strip(),
                     status.group(1) if status else None, path.stem[-6:].upper()))
    return pl.DataFrame(rows, schema=["cusip", "options_listed", "issuer_13f", "description_13f", "status_13f",
                                      "list_quarter"], orient="row")


# --- OpenFIGI -----------------------------------------------------------------

def _cached_figi() -> dict[str, dict]:
    if not FIGI_CACHE.exists():
        return {}
    out = {}
    for line in FIGI_CACHE.read_text().splitlines():
        r = json.loads(line)
        out[(r["idType"], r["idValue"])] = r["response"]
    return out


def openfigi_map(id_type: str, values: list[str]) -> dict[str, dict]:
    """Map identifiers through OpenFIGI, caching every response (hits and misses)."""
    load_dotenv()
    key = os.environ.get("OPENFIGI_API_KEY", "").strip()
    per_request, wait = (JOBS_PER_REQUEST, SECONDS_PER_REQUEST) if key else (10, 60 / 25)
    headers = {"Content-Type": "application/json", **({"X-OPENFIGI-APIKEY": key} if key else {})}

    cache = _cached_figi()
    todo = [v for v in dict.fromkeys(values) if (id_type, v) not in cache]
    FIGI_CACHE.parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(todo), per_request):
        batch = todo[i:i + per_request]
        for attempt in range(5):
            resp = requests.post(FIGI_URL, json=[{"idType": id_type, "idValue": v} for v in batch],
                                 headers=headers, timeout=60)
            if resp.status_code != 429:
                break
            time.sleep(float(resp.headers.get("ratelimit-reset", 10)) + 1)
        resp.raise_for_status()
        with FIGI_CACHE.open("a") as f:
            for v, r in zip(batch, resp.json()):
                cache[(id_type, v)] = r
                f.write(json.dumps({"idType": id_type, "idValue": v, "response": r}) + "\n")
        print(f"  OpenFIGI {id_type}: {min(i + per_request, len(todo)):,}/{len(todo):,}")
        time.sleep(wait)
    return {v: cache[(id_type, v)] for v in values}


def best_listing(response: dict) -> dict | None:
    """Prefer the US composite listing; then any US exchange; then anything."""
    data = response.get("data") or []
    if not data:
        return None
    def rank(d):
        return (d.get("exchCode") != "US", not (d.get("exchCode") or "").startswith("U"),
                d.get("marketSector") != "Equity")
    return sorted(data, key=rank)[0]


def classify(listing: dict | None, description_13f: str | None) -> str:
    if listing is None:
        return "unmapped"
    if listing.get("marketSector") in ("Corp", "Govt", "Muni", "Mtge"):
        return "debt"
    for t in (listing.get("securityType"), listing.get("securityType2")):  # ETFs: "ETP" / "Mutual Fund"
        if t in CLASS_BY_TYPE:
            return CLASS_BY_TYPE[t]
    return "other"


def yahoo_symbol(ticker: str | None) -> str | None:
    # OpenFIGI writes share classes as BRK/B; Yahoo and the SEC use BRK-B.
    return ticker.replace("/", "-").replace(" ", "-") if ticker else None


# --- SEC ticker -> CIK --------------------------------------------------------

def ticker_ciks() -> pl.DataFrame:
    if not TICKERS_PATH.exists():
        TICKERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        TICKERS_PATH.write_bytes(sec.get("https://www.sec.gov/files/company_tickers.json").content)
    data = json.loads(TICKERS_PATH.read_text())
    return pl.DataFrame([{"symbol": d["ticker"].upper(), "issuer_cik": str(d["cik_str"]).zfill(10)}
                         for d in data.values()]).unique("symbol", keep="first")


# --- build --------------------------------------------------------------------

def build() -> pl.DataFrame:
    con = db.connect()
    universe = con.execute("""
        SELECT cusip, any_value(name_of_issuer) AS name_of_issuer,
               bool_and(put_call = '' AND shares = 0 AND principal > 0) AS principal_only
        FROM seed_positions GROUP BY cusip
    """).pl()

    lists = pl.concat([parse_13f_list(p) for p in download_13f_lists(db.QUARTERS)])
    latest_list = lists.sort("list_quarter").unique("cusip", keep="last")

    by_cusip = openfigi_map("ID_CUSIP", universe["cusip"].to_list())
    # CUSIPs starting with a letter are CINS codes (non-US domicile, e.g. G for
    # Cayman/Bermuda/Ireland, H for Switzerland). OpenFIGI only maps them as ID_CINS.
    cins = [c for c, r in by_cusip.items() if best_listing(r) is None and c[0].isalpha()]
    by_cins = openfigi_map("ID_CINS", cins) if cins else {}
    for c, r in by_cins.items():
        if best_listing(r) is not None:
            by_cusip[c] = r | {"_source": "openfigi_cins"}
    # Fallback: FIGIs that filers reported themselves for CUSIPs still unmapped.
    unmapped = [c for c, r in by_cusip.items() if best_listing(r) is None]
    filer_figis = con.execute("""
        SELECT cusip, mode(figi) AS figi FROM filings_db.infotable_raw
        WHERE figi IS NOT NULL AND cusip IN (SELECT unnest($c)) GROUP BY cusip
    """, {"c": unmapped}).pl()
    by_figi = openfigi_map("ID_BB_GLOBAL", filer_figis["figi"].to_list()) if len(filer_figis) else {}
    figi_for = dict(filer_figis.iter_rows())

    rows = []
    for cusip in universe["cusip"]:
        listing, source = best_listing(by_cusip[cusip]), by_cusip[cusip].get("_source", "openfigi_cusip")
        if listing is None and cusip in figi_for:
            listing, source = best_listing(by_figi.get(figi_for[cusip], {})), "openfigi_filer_figi"
        rows.append({
            "cusip": cusip,
            "ticker": listing.get("ticker") if listing else None,
            "exch_code": listing.get("exchCode") if listing else None,
            "figi": listing.get("figi") if listing else None,
            "composite_figi": listing.get("compositeFIGI") if listing else None,
            "figi_name": listing.get("name") if listing else None,
            "security_type": listing.get("securityType") if listing else None,
            "security_type2": listing.get("securityType2") if listing else None,
            "market_sector": listing.get("marketSector") if listing else None,
            "map_source": source if listing else None,
        })
    master = (
        universe.join(pl.DataFrame(rows), on="cusip", how="left")
        .join(latest_list.drop("list_quarter"), on="cusip", how="left")
        .with_columns(
            pl.struct("ticker", "market_sector", "security_type", "security_type2", "description_13f")
            .map_elements(lambda r: classify(
                None if r["ticker"] is None else
                {"marketSector": r["market_sector"], "securityType": r["security_type"],
                 "securityType2": r["security_type2"]},
                r["description_13f"]), return_dtype=pl.Utf8)
            .alias("security_class"),
            pl.col("ticker").map_elements(yahoo_symbol, return_dtype=pl.Utf8).alias("symbol"),
            pl.col("cusip").is_in(lists["cusip"].unique().to_list()).alias("on_13f_list"),
        )
        .with_columns(
            pl.when(pl.col("principal_only")).then(pl.lit("debt")).otherwise(pl.col("security_class"))
            .alias("security_class")
        )
        .join(ticker_ciks(), on="symbol", how="left")
        .with_columns(pl.lit(None, dtype=pl.Date).alias("known_at"))  # current mapping, not point-in-time
    )
    con.execute("CREATE OR REPLACE TABLE security_master AS SELECT * FROM master")
    return master


def coverage_report(con) -> None:
    by_q = con.execute("""
        SELECT p.quarter,
               sum(p.value_usd) / 1e12 AS value_tn,
               sum(p.value_usd) FILTER (WHERE m.ticker IS NOT NULL) / sum(p.value_usd) AS mapped,
               sum(p.value_usd) FILTER (WHERE m.ticker IS NOT NULL AND m.security_class <> 'debt')
                 / sum(p.value_usd) FILTER (WHERE m.security_class <> 'debt' OR m.security_class IS NULL)
                 AS mapped_ex_debt,
               sum(p.value_usd) FILTER (WHERE m.issuer_cik IS NOT NULL) / sum(p.value_usd) AS with_issuer_cik,
               count(DISTINCT p.cusip) AS cusips,
               count(DISTINCT p.cusip) FILTER (WHERE m.ticker IS NOT NULL) AS cusips_mapped
        FROM seed_positions p JOIN security_master m USING (cusip)
        GROUP BY 1 ORDER BY 1
    """).pl()
    print("\nCoverage by share of seed-fund value:")
    print(by_q)
    print("\nValue by security class (all quarters):")
    print(con.execute("""
        SELECT m.security_class, round(sum(p.value_usd) / 1e9) AS value_bn,
               round(sum(p.value_usd) / (SELECT sum(value_usd) FROM seed_positions), 4) AS share,
               count(DISTINCT cusip) AS cusips
        FROM seed_positions p JOIN security_master m USING (cusip) GROUP BY 1 ORDER BY 2 DESC
    """).pl())
    print("\nLargest unmapped positions (summed over funds, latest quarter each CUSIP appears):")
    with pl.Config(tbl_rows=25, fmt_str_lengths=40):
        print(con.execute("""
            SELECT p.cusip, any_value(m.name_of_issuer) AS issuer, any_value(m.description_13f) AS description,
                   max(p.quarter) AS last_quarter,
                   round(sum(p.value_usd) FILTER (WHERE p.quarter = '2026Q2') / 1e6, 1) AS value_2026q2_musd,
                   round(sum(p.value_usd) / 1e6, 1) AS value_all_q_musd,
                   count(DISTINCT p.cik) AS funds
            FROM seed_positions p JOIN security_master m USING (cusip)
            WHERE m.ticker IS NULL
            GROUP BY 1 ORDER BY value_all_q_musd DESC LIMIT 25
        """).pl())


if __name__ == "__main__":
    build()
    coverage_report(db.connect(read_only=True))
