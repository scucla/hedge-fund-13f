"""Check that adding the holdout funds leaves every seed-fund table unchanged.

Usage (about 10 minutes; needs ~1 GB free for the copy):
    cp data/processed/securities.duckdb /tmp/seed_only.duckdb
    uv run python -c "import duckdb; c = duckdb.connect('/tmp/seed_only.duckdb'); \
        [c.execute(f\"DELETE FROM {t} WHERE cohort = 'holdout'\") for t in ('seed_positions', 'seed_positions_pit')]"
    for s in securities.cusip_map securities.stock_quarter features.books features.positions \
             features.fund_quarter features.fingerprint; do
        SECURITIES_DB=/tmp/seed_only.duckdb uv run python -m src.$s || break; done
    uv run python -m src.features.holdout_invariance data/processed/securities.duckdb /tmp/seed_only.duckdb

Compares the seed rows of each table in the full build with the seed-only
build: every non-float column must match exactly and floats within 1e-9
relative. On 2026-10-16 the same comparison (run as a one-off script before
this module was written) found all eight tables identical.
"""

import sys

import duckdb

TABLES = [
    ("security_master", ["cusip"], "WHERE a.cusip IN (SELECT cusip FROM so.security_master)"),
    ("stock_quarter", ["cusip", "quarter"], "WHERE (a.cusip, a.quarter) IN (SELECT (cusip, quarter) FROM so.stock_quarter)"),
    ("stock_zscores", ["symbol", "quarter"], "WHERE a.in_seed"),
    ("benchmark_weights", ["quarter", "issuer_key"], ""),
    ("stock_splits", ["cusip", "quarter"], ""),
    ("positions", ["fund_id", "quarter", "cusip", "put_call"], "WHERE a.cohort = 'seed'"),
    ("fund_quarter", ["fund_id", "quarter"], "WHERE a.cohort = 'seed'"),
    ("fund_fingerprint", ["fund_id", "quarter"], "WHERE a.cohort = 'seed'"),
]


def compare(con, table: str, keys: list[str], where: str) -> list[tuple]:
    cols = [(r[0], r[1]) for r in con.execute(f"DESCRIBE so.{table}").fetchall() if r[0] not in keys]
    on = " AND ".join(f"a.{k} IS NOT DISTINCT FROM b.{k}" for k in keys)
    n_a = con.execute(f"SELECT count(*) FROM {table} a {where}").fetchone()[0]
    n_b = con.execute(f"SELECT count(*) FROM so.{table} b {where.replace('a.', 'b.')}").fetchone()[0]
    n_ab = con.execute(f"SELECT count(*) FROM {table} a JOIN so.{table} b ON {on} {where}").fetchone()[0]
    diffs = [("row count", n_a, n_b)] if not n_a == n_b == n_ab else []
    for col, typ in cols:
        if typ in ("DOUBLE", "FLOAT"):
            bad = (f"abs(a.{col} - b.{col}) > 1e-9 * greatest(1, abs(b.{col})) "
                   f"OR (a.{col} IS NULL) <> (b.{col} IS NULL)")
        else:
            bad = f"a.{col} IS DISTINCT FROM b.{col}"
        n = con.execute(f"SELECT count(*) FILTER (WHERE {bad}) FROM {table} a JOIN so.{table} b ON {on} {where}").fetchone()[0]
        if n:
            diffs.append((col, n))
    return diffs


if __name__ == "__main__":
    full, seed_only = sys.argv[1], sys.argv[2]
    con = duckdb.connect(full, read_only=True)
    con.execute(f"ATTACH '{seed_only}' AS so (READ_ONLY)")
    ok = True
    for table, keys, where in TABLES:
        diffs = compare(con, table, keys, where)
        ok &= not diffs
        print(f"{table:18s}", diffs or "identical")
    sys.exit(0 if ok else 1)
