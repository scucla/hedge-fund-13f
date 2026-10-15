"""Print a fund's top holdings for one quarter.

Usage:
    uv run python -m src.ingest.show_fund --cik 1336528 --quarter 2026Q2
    uv run python -m src.ingest.show_fund --cik 1336528 --quarter 2026Q2 --as-of 2026-08-01

--as-of shows only what had been filed by that date (point-in-time).
"""

import argparse
from datetime import date

import duckdb

from src.ingest.load import DB_PATH
from src.ingest.positions import fund_holdings, normalize_cik


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cik", required=True)
    parser.add_argument("--quarter", required=True, help="holdings quarter, e.g. 2026Q2")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    positions, used, warnings = fund_holdings(con, args.cik, args.quarter, args.as_of)
    for w in warnings:
        print(f"warning: {w}")
    if positions.is_empty():
        raise SystemExit(f"No 13F holdings for CIK {normalize_cik(args.cik)} in {args.quarter.upper()}"
                         + (f" filed by {args.as_of}" if args.as_of else "") + ".")

    print(f"{used['manager_name'][-1]}  (CIK {normalize_cik(args.cik)})  {args.quarter.upper()}")
    for f in used.iter_rows(named=True):
        kind = f["submission_type"] + (f" {f['amendment_type']}" if f["submission_type"] != "13F-HR" else "")
        print(f"  filing {f['accession_number']}  {kind}  filed {f['filing_date']}")
    total = positions["value_usd"].sum()
    print(f"  {len(positions)} positions, ${total / 1e9:,.2f}B reported value\n")

    print(f"{'#':>3}  {'Issuer':32s} {'CUSIP':9s} {'Opt':4s} {'Shares':>14s} {'Value ($M)':>12s} {'Weight':>7s}")
    for rank, p in enumerate(positions.head(args.top).iter_rows(named=True), 1):
        print(f"{rank:>3}  {p['name_of_issuer'][:32]:32s} {p['cusip']:9s} {p['put_call'][:4]:4s} "
              f"{p['shares']:>14,.0f} {p['value_usd'] / 1e6:>12,.1f} {p['weight']:>7.1%}")
    top = positions.head(args.top)["weight"].sum()
    print(f"\nTop {min(args.top, len(positions))} = {top:.1%} of reported value")


if __name__ == "__main__":
    main()
