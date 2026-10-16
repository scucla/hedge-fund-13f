"""Corporate-event filings for the issuers our funds hold, from the EDGAR full-index.

Usage:
    uv run python -m src.securities.sec_events

The quarterly form index lists every filing once per company it names, so a
tender offer (SC TO-T) or a merger prospectus (425) appears under the target's
CIK as well as the bidder's. That makes one cheap pass over 2014-2026 enough to
see, per held issuer, when a deal was announced, when the stock was delisted
and who the counterparty was, without one request per company.

Writes `sec_events`: one row per (issuer CIK, form, accession) for the forms in
FORMS, with `other_ciks` = the other companies on the same accession (the
bidder or acquirer for deal forms), and `known_at` = the filing date.

Also writes `issuer_cik_fill`: CIKs for common stock and ADRs that the
security master left without one. The master's name fallback only knows
companies that filed XBRL in 2024-2025, so issuers that left the market
earlier have no CIK, and those are exactly the ones this module is about. Two
fallbacks, in order: another CUSIP with the same 6-character issuer prefix
that has a CIK (one CIK only), then a one-to-one name match against every
company that filed a 10-K, 10-Q, 20-F or 40-F in the index since 2014.
"""

import gzip
import re
from datetime import date
from pathlib import Path

import polars as pl

from src.ingest.edgar_index import RAW_DIR, download_quarter
from src.securities import db
from src.securities.cusip_map import normalize_name

FIRST_YEAR = 2014   # deals announced in 2014 can close in 2015, our first quarter

# form -> what it tells us
FORMS = {
    "25-NSE": "delisting",       # exchange removes the listing (filed by the exchange)
    "25": "delisting",           # issuer withdraws the listing
    "15-12B": "deregistration",
    "15-12G": "deregistration",
    "15-15D": "deregistration",
    "PREM14A": "merger_proxy",   # preliminary merger proxy (shareholder vote)
    "DEFM14A": "merger_proxy",
    "PREM14C": "merger_proxy",   # information statement (majority already consents)
    "DEFM14C": "merger_proxy",
    "SC TO-T": "tender_offer",   # third-party tender offer
    "SC 14D9": "tender_offer",   # target's response to a tender offer
    "SC 13E3": "going_private",
    "425": "stock_deal",         # prospectus communications in a stock deal
    "S-4": "stock_deal",
    "F-4": "stock_deal",
}
REPORT_FORMS = {"10-K": "report", "10-Q": "report", "20-F": "report", "40-F": "report"}
_FORM_RE = re.compile(r"^(.+?)\s{2,}")


def parse_events(path: Path, forms=FORMS) -> pl.DataFrame:
    """Rows of a form.idx whose form type (exact, so 'SC TO-T/A' is not 'SC TO-T') is in `forms`.
    Form types can contain one space ('SC TO-T'); the form column ends at the
    first run of two or more spaces. CIK, date and file name are read from the right."""
    rows = []
    with gzip.open(path, "rt", encoding="latin-1") as f:
        for line in f:
            m = _FORM_RE.match(line)
            if not m or m.group(1) not in forms:
                continue
            tokens = line.split()
            if len(tokens) < 4 or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", tokens[-2]):
                continue
            cik, filed, fname = tokens[-3:]
            name = " ".join(tokens[len(m.group(1).split()):-3])
            rows.append((m.group(1), name, cik.zfill(10), filed, fname))
    return pl.DataFrame(rows, schema=["form", "company_name", "cik", "date_filed", "file_name"], orient="row",
                        ).with_columns(pl.col("date_filed").str.to_date())


def index_paths() -> list[Path]:
    paths = []
    for year in range(FIRST_YEAR, date.today().year + 1):
        for q in range(1, 5):
            path = download_quarter(year, q, RAW_DIR)
            if path:
                paths.append(path)
    return paths


def fill_ciks(con, paths: list[Path]) -> pl.DataFrame:
    """issuer_cik for common/ADR CUSIPs without one (see module docstring)."""
    master = con.execute("""
        SELECT cusip, issuer_cik, security_class, figi_name, issuer_13f, name_of_issuer
        FROM security_master WHERE security_class IN ('common', 'adr')
    """).pl()
    missing = master.filter(pl.col("issuer_cik").is_null())
    by_prefix = (master.filter(pl.col("issuer_cik").is_not_null())
                 .group_by(pl.col("cusip").str.slice(0, 6).alias("prefix"))
                 .agg(pl.col("issuer_cik").unique().alias("ciks"))
                 .filter(pl.col("ciks").list.len() == 1)
                 .select("prefix", pl.col("ciks").list.first().alias("cik")))
    prefix_hits = dict(missing.with_columns(pl.col("cusip").str.slice(0, 6).alias("prefix"))
                       .join(by_prefix, on="prefix").select("cusip", "cik").iter_rows())

    filers = pl.concat([parse_events(p, REPORT_FORMS) for p in paths]).select("cik", "company_name").unique()
    filers = filers.with_columns(pl.col("company_name").map_elements(normalize_name, return_dtype=pl.Utf8).alias("name"))
    unique = (filers.drop_nulls("name").group_by("name").agg(pl.col("cik").unique())
              .filter(pl.col("cik").list.len() == 1))
    lookup = dict(unique.select("name", pl.col("cik").list.first()).iter_rows())
    name_hits = {}
    for cusip, *names in missing.filter(~pl.col("cusip").is_in(list(prefix_hits))) \
            .select("cusip", "figi_name", "issuer_13f", "name_of_issuer").iter_rows():
        found = {lookup[n] for n in map(normalize_name, names) if n and n in lookup}
        if len(found) == 1:
            name_hits[cusip] = found.pop()
    fill = pl.DataFrame(
        [(c, k, "cusip_prefix") for c, k in prefix_hits.items()] + [(c, k, "edgar_index_name") for c, k in name_hits.items()],
        schema=["cusip", "issuer_cik", "issuer_cik_source"], orient="row")
    con.execute("CREATE OR REPLACE TABLE issuer_cik_fill AS SELECT * FROM fill")
    return fill


def build(con) -> pl.DataFrame:
    paths = index_paths()
    fill_ciks(con, paths)
    allrows = pl.concat([parse_events(p) for p in paths]).unique()
    allrows = allrows.with_columns(
        pl.col("file_name").str.extract(r"(\d{10}-\d{2}-\d{6})").alias("accession"),
        pl.col("form").replace_strict(FORMS).alias("kind"),
    )
    held = {r[0] for r in con.execute("""
        SELECT issuer_cik FROM security_master WHERE issuer_cik IS NOT NULL
        UNION SELECT issuer_cik FROM issuer_cik_fill""").fetchall()}
    # every company on the accession, so a deal row knows its counterparty
    parties = allrows.group_by("accession").agg(pl.col("cik").unique().alias("parties"),
                                                pl.col("company_name").unique().alias("party_names"))
    events = (
        allrows.filter(pl.col("cik").is_in(held))
        .join(parties, on="accession")
        .with_columns(pl.col("parties").list.set_difference(pl.concat_list(pl.col("cik"))).alias("other_ciks"))
        .select("cik", "company_name", "form", "kind", "accession", "file_name", "other_ciks",
                pl.col("date_filed").alias("known_at"))
        .sort("cik", "known_at")
    )
    con.execute("CREATE OR REPLACE TABLE sec_events AS SELECT * FROM events")
    return events


if __name__ == "__main__":
    with db.connect() as con:
        ev = build(con)
        print(ev.group_by("form").len().sort("len", descending=True))
        print(f"{ev['cik'].n_unique():,} held issuers with at least one event")
        print(con.execute("SELECT issuer_cik_source, count(*) FROM issuer_cik_fill GROUP BY 1").fetchall())
