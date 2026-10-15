"""One book per seed fund per quarter: combine each fund's filer IDs (CIKs).

Usage:
    uv run python -m src.features.books

Rules, applied per fund-quarter (every decision is written to
`fund_book_decisions` with its reason):
1. CIKs in data/reference/fund_book_exclusions.csv are dropped (e.g. a
   broker-dealer's market-making inventory).
2. Two CIKs whose portfolios overlap by >= DUPLICATE_OVERLAP (sum over
   positions of the smaller of the two weights) report the same book twice:
   keep one, preferring primary > successor > affiliate > predecessor, then
   the larger book. Typical cases: the hand-over quarter when a fund changes
   filer, or a founder filing personally for the firm's book.
3. Everything else is a separate book of the same firm and is added up
   (e.g. Point72's regional entities, AQR vs AQR Arbitrage).

Writes `fund_book_decisions` and `fund_holdings` (fund_id x quarter x CUSIP x
put/call) to the securities database.
"""

import re
from pathlib import Path

import polars as pl

from src.securities import db

EXCLUSIONS_PATH = Path("data/reference/fund_book_exclusions.csv")
DUPLICATE_OVERLAP = 0.9
ROLE_RANK = {"primary": 0, "successor": 1, "affiliate": 2, "predecessor": 3}


def fund_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def decide(totals: pl.DataFrame, overlaps: pl.DataFrame, excluded: set[tuple[str, str]]) -> pl.DataFrame:
    """totals: fund_name, quarter, cik, role, value_usd. overlaps: fund_name,
    quarter, cik_a, cik_b, overlap. Returns one decision row per input row."""
    rows = []
    for (fund, quarter), grp in totals.group_by(["fund_name", "quarter"], maintain_order=True):
        ov = overlaps.filter((pl.col("fund_name") == fund) & (pl.col("quarter") == quarter))
        max_ov = {}
        for a, b, o in ov.select("cik_a", "cik_b", "overlap").iter_rows():
            max_ov[a] = max(max_ov.get(a, 0.0), o)
            max_ov[b] = max(max_ov.get(b, 0.0), o)
        best_first = sorted(grp.iter_rows(named=True), key=lambda r: (ROLE_RANK.get(r["role"], 9), -r["value_usd"]))
        kept: list[str] = []
        for r in best_first:
            cik = r["cik"]
            if (fund, cik) in excluded:
                included, reason = False, "excluded: see fund_book_exclusions.csv"
            else:
                dup_of = next(
                    (k for k in kept
                     for a, b, o in ov.select("cik_a", "cik_b", "overlap").iter_rows()
                     if {a, b} == {k, cik} and o >= DUPLICATE_OVERLAP),
                    None,
                )
                if dup_of:
                    included, reason = False, f"duplicate of {dup_of} (overlap >= {DUPLICATE_OVERLAP})"
                else:
                    kept.append(cik)
                    included = True
                    reason = "only filer" if len(grp) == 1 else "separate book: added"
            rows.append({"fund_id": fund_id(fund), "fund_name": fund, "quarter": quarter, "cik": cik,
                         "role": r["role"], "value_usd": r["value_usd"], "included": included,
                         "reason": reason, "max_overlap": max_ov.get(cik)})
    return pl.DataFrame(rows, schema_overrides={"max_overlap": pl.Float64})


def build(con) -> None:
    totals = con.execute("""
        SELECT fund_name, quarter, cik, role, sum(value_usd) AS value_usd
        FROM seed_positions GROUP BY ALL ORDER BY fund_name, quarter, cik
    """).pl()
    overlaps = con.execute("""
        WITH multi AS (SELECT fund_name, quarter FROM seed_positions GROUP BY ALL HAVING count(DISTINCT cik) > 1),
        p AS (SELECT s.fund_name, s.quarter, s.cik, s.cusip, s.put_call, s.weight
              FROM seed_positions s JOIN multi USING (fund_name, quarter))
        SELECT a.fund_name, a.quarter, a.cik AS cik_a, b.cik AS cik_b, sum(least(a.weight, b.weight)) AS overlap
        FROM p a JOIN p b ON a.fund_name = b.fund_name AND a.quarter = b.quarter AND a.cik < b.cik
                         AND a.cusip = b.cusip AND a.put_call = b.put_call
        GROUP BY ALL
    """).pl()
    excl = pl.read_csv(EXCLUSIONS_PATH, schema_overrides={"cik": pl.Utf8})
    excluded = {(f, c.zfill(10)) for f, c in excl.select("fund_name", "cik").iter_rows()}
    decisions = decide(totals, overlaps, excluded)
    con.execute("CREATE OR REPLACE TABLE fund_book_decisions AS SELECT * FROM decisions")
    con.execute("""
        CREATE OR REPLACE TABLE fund_holdings AS
        SELECT d.fund_id, any_value(d.fund_name) AS fund_name, s.quarter, s.cusip, s.put_call,
               any_value(s.name_of_issuer) AS name_of_issuer,
               sum(s.shares) AS shares, sum(s.principal) AS principal, sum(s.value_usd) AS value_usd,
               max(s.known_at) AS known_at, count(DISTINCT s.cik) AS n_ciks
        FROM seed_positions s
        JOIN fund_book_decisions d ON d.fund_name = s.fund_name AND d.quarter = s.quarter AND d.cik = s.cik
        WHERE d.included
          -- placeholder rows (no value, shares or principal) say "nothing to report"
          AND NOT (s.value_usd = 0 AND s.shares = 0 AND s.principal = 0)
        GROUP BY d.fund_id, s.quarter, s.cusip, s.put_call
    """)
    # A fund that filed a 13F-HR with nothing but a placeholder held no 13F
    # securities that quarter (e.g. it sold everything). Keep that fact.
    con.execute("""
        CREATE OR REPLACE TABLE empty_books AS
        SELECT d.fund_id, any_value(d.fund_name) AS fund_name, d.quarter, max(s.known_at) AS known_at
        FROM fund_book_decisions d
        JOIN seed_positions s ON s.fund_name = d.fund_name AND s.quarter = d.quarter AND s.cik = d.cik
        WHERE d.included
        GROUP BY d.fund_id, d.quarter
        HAVING (d.fund_id, d.quarter) NOT IN (SELECT fund_id, quarter FROM fund_holdings)
    """)
    # every fund-quarter's known_at is the latest filing among its included CIKs
    con.execute("""
        UPDATE fund_holdings h SET known_at = k.known_at
        FROM (SELECT fund_id, quarter, max(known_at) AS known_at FROM fund_holdings GROUP BY ALL) k
        WHERE h.fund_id = k.fund_id AND h.quarter = k.quarter
    """)


if __name__ == "__main__":
    with db.connect() as con:
        build(con)
        print(con.execute("""
            SELECT reason, count(*) AS n FROM fund_book_decisions GROUP BY 1 ORDER BY 2 DESC
        """).pl())
        print(con.execute("""
            SELECT fund_name, quarter, cik, role, round(value_usd / 1e9, 2) AS value_bn, round(max_overlap, 3) AS max_overlap, reason
            FROM fund_book_decisions WHERE reason <> 'only filer' ORDER BY fund_name, quarter, role
        """).pl())
