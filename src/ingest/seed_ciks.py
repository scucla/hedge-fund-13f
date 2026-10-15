"""Map seed funds to every CIK they have filed 13F holdings reports under.

Usage:
    uv run python -m src.ingest.seed_ciks candidates "Pershing Square"   # search filer names
    uv run python -m src.ingest.seed_ciks build                           # refresh seed_funds.csv

data/reference/seed_cik_map.csv is the reviewed mapping (fund_name, cik,
role, note). It is maintained by hand: the name search only proposes
candidates, because similar names often belong to unrelated firms. `build`
joins it to the seed list, adds EDGAR facts per CIK and writes
data/reference/seed_funds.csv with one row per fund + CIK.

A fund is flagged `multi_cik_same_quarter` when two or more of its CIKs filed
an original 13F-HR for the same report period. That can mean two separate
books (e.g. a Hong Kong and a US entity) or the same book reported twice
(e.g. a firm and its founder); later milestones must not double-count.
"""

import argparse
import json
import re
from pathlib import Path

import polars as pl

from src.ingest import sec

REF_DIR = Path("data/reference")
SEED_PATH = REF_DIR / "seed_funds.csv"
MAP_PATH = REF_DIR / "seed_cik_map.csv"
INDEX_PATH = Path("data/interim/edgar_13f_index.parquet")
SUBMISSIONS_DIR = Path("data/raw/edgar/submissions")


def submissions(cik: str) -> dict:
    """EDGAR's filer profile, with every page of filing history merged, cached."""
    cik = cik.zfill(10)
    path = SUBMISSIONS_DIR / f"CIK{cik}.json"
    if path.exists():
        return json.loads(path.read_text())
    profile = sec.get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    history = profile["filings"]["recent"]
    for page in profile["filings"].get("files", []):
        older = sec.get(f"https://data.sec.gov/submissions/{page['name']}").json()
        for key in history:
            history[key] = history[key] + older.get(key, [])
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile))
    return profile


def thirteen_f_filings(cik: str) -> pl.DataFrame:
    f = submissions(cik)["filings"]["recent"]
    df = pl.DataFrame({k: f[k] for k in ("form", "filingDate", "reportDate", "accessionNumber")})
    return (
        df.filter(pl.col("form").str.starts_with("13F"))
        .with_columns(
            pl.col("filingDate").str.to_date(),
            pl.col("reportDate").replace("", None).str.to_date(),
        )
        .with_columns(
            (pl.col("reportDate").dt.year().cast(pl.Utf8) + "Q" + pl.col("reportDate").dt.quarter().cast(pl.Utf8))
            .alias("quarter")
        )
    )


def business_address(cik: str) -> str:
    a = submissions(cik).get("addresses", {}).get("business", {}) or {}
    return ", ".join(x for x in (a.get("street1"), a.get("city"), a.get("stateOrCountry")) if x)


def candidates(pattern: str) -> pl.DataFrame:
    """Every 13F filer whose name (at any time) contains `pattern` as whole words."""
    idx = pl.read_parquet(INDEX_PATH)
    regex = r"(?i)\b" + r"\W+".join(map(re.escape, pattern.split())) + r"\b"
    return (
        idx.filter(pl.col("company_name").str.contains(regex))
        .group_by("cik", "company_name")
        .agg(
            pl.col("form").str.starts_with("13F-HR").sum().alias("n_hr"),
            pl.len().alias("n_13f"),
            pl.col("date_filed").min().alias("first_filed"),
            pl.col("date_filed").max().alias("last_filed"),
        )
        .sort("cik", "first_filed")
    )


def build() -> pl.DataFrame:
    seed = pl.read_csv(SEED_PATH, columns=["fund_name", "style_guess"]).unique(maintain_order=True)
    cik_map = pl.read_csv(MAP_PATH, schema_overrides={"cik": pl.Utf8}).with_columns(pl.col("cik").str.zfill(10))

    missing = set(seed["fund_name"]) - set(cik_map["fund_name"])
    unknown = set(cik_map["fund_name"]) - set(seed["fund_name"])
    if missing or unknown:
        raise SystemExit(f"seed_cik_map.csv out of sync. Missing: {sorted(missing)} Unknown: {sorted(unknown)}")

    facts = []
    for cik in cik_map["cik"].unique().sort():
        prof = submissions(cik)
        f = thirteen_f_filings(cik)
        hr = f.filter(pl.col("form") == "13F-HR")
        former = [n["name"] for n in prof.get("formerNames", [])]
        facts.append({
            "cik": cik,
            "edgar_name": prof["name"],
            "former_names": "; ".join(former),
            "business_address": business_address(cik),
            "n_13f_hr": len(hr),
            "first_quarter": hr["quarter"].min() if len(hr) else None,
            "last_quarter": hr["quarter"].max() if len(hr) else None,
            "quarters": sorted(set(hr["quarter"].drop_nulls())),
        })
    facts = pl.DataFrame(facts)

    rows = seed.join(cik_map, on="fund_name", how="left").join(facts, on="cik", how="left")

    # Same-quarter overlap between a fund's CIKs (original 13F-HRs only).
    # CIKs whose link to the fund is unconfirmed don't count.
    overlap = (
        rows.filter(pl.col("role") != "uncertain")
        .select("fund_name", "cik", "quarters").explode("quarters").drop_nulls("quarters")
        .group_by("fund_name", "quarters").agg(pl.col("cik").n_unique().alias("n_ciks"))
        .filter(pl.col("n_ciks") > 1)
        .group_by("fund_name")
        .agg(
            pl.len().alias("n_overlap_quarters"),
            pl.col("quarters").min().alias("overlap_first"),
            pl.col("quarters").max().alias("overlap_last"),
        )
    )
    rows = (
        rows.join(overlap, on="fund_name", how="left")
        .with_columns(
            pl.col("n_overlap_quarters").fill_null(0),
            (pl.col("n_overlap_quarters").fill_null(0) > 0).alias("multi_cik_same_quarter"),
        )
        .drop("quarters")
    )
    order = {name: i for i, name in enumerate(seed["fund_name"])}
    rows = rows.with_columns(pl.col("fund_name").replace_strict(order).alias("_o")).sort("_o", "first_quarter").drop("_o")
    rows.write_csv(SEED_PATH)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("pattern")
    sub.add_parser("build")
    args = p.parse_args()
    if args.cmd == "candidates":
        with pl.Config(tbl_rows=200, fmt_str_lengths=60):
            print(candidates(args.pattern))
    else:
        rows = build()
        print(f"{rows['fund_name'].n_unique()} funds, {len(rows)} fund+CIK rows -> {SEED_PATH}")


if __name__ == "__main__":
    main()
