"""Compare a fund-quarter's holdings in our database with the filing on EDGAR.

Usage:
    uv run python -m src.ingest.check_units --cik 1336528 --quarter 2016Q4 [--db data/interim/13f_history.duckdb]

Fetches the filing's information-table XML (cached under data/raw/edgar/),
sums it per (CUSIP, put/call), and prints our top 10 next to EDGAR's. For
filings made before 2023-01-03 the XML <value> is in thousands of dollars, so
the expected value in dollars is value x 1000; from then on it is dollars.
"""

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date
from pathlib import Path

import duckdb

from src.ingest import sec
from src.ingest.load import DB_PATH, DOLLARS_FROM
from src.ingest.positions import fund_holdings

RAW = Path("data/raw/edgar/filings")


def edgar_positions(cik: str, accession: str) -> dict[tuple[str, str], float]:
    acc = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/"
    out_dir = RAW / acc
    out_dir.mkdir(parents=True, exist_ok=True)
    items = sec.get(base + "index.json").json()["directory"]["item"]
    xmls = [i["name"] for i in items if i["name"].lower().endswith(".xml") and "primary_doc" not in i["name"].lower()]
    path = out_dir / xmls[0]
    if not path.exists():
        path.write_bytes(sec.get(base + xmls[0]).content)
    strip = lambda t: t.split("}")[-1]  # noqa: E731
    pos = defaultdict(float)
    for it in ET.parse(path).getroot():
        d = {strip(c.tag): c for c in it}
        put_call = d["putCall"].text.strip().title() if "putCall" in d and d["putCall"].text else ""
        pos[(d["cusip"].text.strip().upper(), put_call)] += float(d["value"].text)
    return pos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cik", required=True)
    ap.add_argument("--quarter", required=True)
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()
    con = duckdb.connect(args.db, read_only=True)
    ours, used, _ = fund_holdings(con, args.cik, args.quarter)
    filed = used["filing_date"].max()
    unit = 1000 if filed < date.fromisoformat(DOLLARS_FROM) else 1
    edgar = defaultdict(float)
    for acc in used["accession_number"]:
        for k, v in edgar_positions(args.cik, acc).items():
            edgar[k] += v * unit
    print(f"{args.quarter} CIK {args.cik}: filings {used['accession_number'].to_list()} filed {filed}; "
          f"EDGAR XML values x {unit}")
    print(f"{'issuer':30s} {'cusip':9s} {'ours $M':>12s} {'EDGAR $M':>12s}  match")
    ok = True
    for r in ours.head(10).iter_rows(named=True):
        e = edgar.get((r["cusip"], r["put_call"].title() if r["put_call"] else ""), 0.0)
        match = abs(r["value_usd"] - e) <= max(1.0, 1e-6 * e)
        ok &= match
        print(f"{r['name_of_issuer'][:30]:30s} {r['cusip']:9s} {r['value_usd'] / 1e6:>12,.1f} {e / 1e6:>12,.1f}  {'yes' if match else 'NO'}")
    tot_ours, tot_edgar = ours["value_usd"].sum(), sum(edgar.values())
    print(f"total: ours ${tot_ours / 1e9:,.3f}B, EDGAR ${tot_edgar / 1e9:,.3f}B; positions {len(ours)} vs {len(edgar)}")
    raise SystemExit(0 if ok and abs(tot_ours - tot_edgar) <= max(1.0, 1e-6 * tot_edgar) else 1)


if __name__ == "__main__":
    main()
