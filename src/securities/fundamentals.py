"""Point-in-time fundamentals from the SEC XBRL companyfacts API, plus SIC codes.

Usage:
    uv run python -m src.securities.fundamentals

For every issuer CIK of a common stock or ADR in the security master:
- companyfacts JSON (all XBRL facts the company ever filed), cached gzipped
  under data/raw/edgar/companyfacts/. We keep the few concepts we need, and
  every reported version of each value: known_at = the date it was filed. A
  restated value is a new row with a later known_at, never an overwrite.
- the submissions profile (first page only) for the SIC code, cached under
  data/raw/edgar/issuer_profiles/.

Writes `fundamentals_facts` and `issuer_profile` to the securities database.
"""

import gzip
import json
from pathlib import Path

import polars as pl
import requests

from src.ingest import sec
from src.securities import db

FACTS_DIR = Path("data/raw/edgar/companyfacts")
# Hand-reviewed: an issuer whose current CIK replaced an older registrant (e.g.
# a new holding company) also gets the older CIK's facts.
LINKS_PATH = Path("data/reference/issuer_cik_links.csv")
PROFILE_DIR = Path("data/raw/edgar/issuer_profiles")

# item -> fallbacks in priority order (taxonomy, concept). Lower priority
# number wins when several are present for the same period.
CONCEPTS = {
    "assets": [("us-gaap", "Assets"), ("ifrs-full", "Assets")],
    "equity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        ("ifrs-full", "EquityAttributableToOwnersOfParent"),
        ("ifrs-full", "Equity"),
    ],
    "gross_profit": [("us-gaap", "GrossProfit"), ("ifrs-full", "GrossProfit")],
    "revenue": [
        ("us-gaap", "Revenues"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "SalesRevenueNet"),
        ("ifrs-full", "Revenue"),
    ],
    "cost_of_revenue": [
        ("us-gaap", "CostOfRevenue"),
        ("us-gaap", "CostOfGoodsAndServicesSold"),
        ("us-gaap", "CostOfGoodsSold"),
        ("ifrs-full", "CostOfSales"),
    ],
    # Cover-page count first. Multi-class companies tag it per class, which
    # companyfacts leaves out, hence the balance-sheet and weighted-average fallbacks.
    "shares_out": [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
    ],
}
UNIT = {"shares_out": "shares"}  # everything else must be USD; other currencies are skipped

SCHEMA = {
    "issuer_cik": pl.Utf8, "source_cik": pl.Utf8, "item": pl.Utf8, "priority": pl.Int32, "taxonomy": pl.Utf8, "concept": pl.Utf8,
    "unit": pl.Utf8, "start": pl.Date, "end": pl.Date, "val": pl.Float64, "accn": pl.Utf8, "fy": pl.Int32,
    "fp": pl.Utf8, "form": pl.Utf8, "filed": pl.Date, "known_at": pl.Date,
}


def _cached_json(url: str, path: Path) -> dict | None:
    """GET JSON through the gzip cache. A 404 (no XBRL facts) is cached as null."""
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    try:
        data = sec.get(url).json()
    except requests.HTTPError as e:
        if e.response is None or e.response.status_code != 404:
            raise
        data = None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(data).encode()))
    return data


def company_facts(cik: str) -> dict | None:
    return _cached_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                        FACTS_DIR / f"CIK{cik}.json.gz")


def issuer_profile(cik: str) -> dict | None:
    data = _cached_json(f"https://data.sec.gov/submissions/CIK{cik}.json", PROFILE_DIR / f"CIK{cik}.json.gz")
    if data is None:
        return None
    # Keep the cache small: drop the filing history, we only need the header.
    if "filings" in data:
        data.pop("filings")
        (PROFILE_DIR / f"CIK{cik}.json.gz").write_bytes(gzip.compress(json.dumps(data).encode()))
    return data


def extract_facts(cik: str, facts: dict | None, source_cik: str | None = None) -> list[dict]:
    rows = []
    if not facts:
        return rows
    for item, options in CONCEPTS.items():
        want_unit = UNIT.get(item, "USD")
        for priority, (taxonomy, concept) in enumerate(options):
            units = facts.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {})
            for f in units.get(want_unit, []):
                rows.append({
                    "issuer_cik": cik, "source_cik": source_cik or cik, "item": item, "priority": priority, "taxonomy": taxonomy,
                    "concept": concept, "unit": want_unit, "start": f.get("start"), "end": f["end"],
                    "val": float(f["val"]), "accn": f.get("accn"), "fy": f.get("fy"), "fp": f.get("fp"),
                    "form": f.get("form"), "filed": f["filed"], "known_at": f["filed"],
                })
    return rows


def facts_frame(rows: list[dict]) -> pl.DataFrame:
    dates = ("start", "end", "filed", "known_at")
    raw = pl.DataFrame(rows, schema={k: (pl.Utf8 if k in dates else v) for k, v in SCHEMA.items()})
    return raw.with_columns([pl.col(c).str.to_date() for c in dates]).unique()


def build() -> None:
    with db.connect(read_only=True) as con:
        ciks = [r[0] for r in con.execute("""
            SELECT DISTINCT issuer_cik FROM security_master
            WHERE issuer_cik IS NOT NULL AND security_class IN ('common', 'adr') ORDER BY 1
        """).fetchall()]
    links = {}
    if LINKS_PATH.exists():
        for issuer, linked in pl.read_csv(LINKS_PATH, schema_overrides={"issuer_cik": pl.Utf8, "linked_cik": pl.Utf8}) \
                .select("issuer_cik", "linked_cik").iter_rows():
            links.setdefault(issuer.zfill(10), []).append(linked.zfill(10))
    fact_rows, profiles = [], []
    for i, cik in enumerate(ciks, 1):
        fact_rows.extend(extract_facts(cik, company_facts(cik)))
        for linked in links.get(cik, []):
            fact_rows.extend(extract_facts(cik, company_facts(linked), source_cik=linked))
        p = issuer_profile(cik)
        profiles.append({
            "issuer_cik": cik,
            "issuer_name": (p or {}).get("name"),
            "sic": int(p["sic"]) if p and str(p.get("sic") or "").isdigit() else None,
            "sic_description": (p or {}).get("sicDescription"),
            "state_of_incorporation": (p or {}).get("stateOfIncorporation"),
            "fiscal_year_end": (p or {}).get("fiscalYearEnd"),
        })
        if i % 250 == 0 or i == len(ciks):
            print(f"  issuers {i:,}/{len(ciks):,}")

    facts = facts_frame(fact_rows)
    profile = pl.DataFrame(profiles, schema_overrides={"sic": pl.Int32})
    with db.connect() as con:
        con.execute("CREATE OR REPLACE TABLE fundamentals_facts AS SELECT * FROM facts")
        con.execute("CREATE OR REPLACE TABLE issuer_profile AS SELECT * FROM profile")
    print(f"fundamentals_facts: {len(facts):,} rows for {facts['issuer_cik'].n_unique():,} issuers; "
          f"issuer_profile: {len(profile):,} issuers, {profile['sic'].null_count()} without SIC")


if __name__ == "__main__":
    build()
