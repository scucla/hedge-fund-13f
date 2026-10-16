"""Load the downloaded 13F data sets into DuckDB at data/interim/13f.duckdb.

Usage:
    uv run python -m src.ingest.load

Rebuilds the database from every zip in data/raw/form13f/. Tables:
    filings        one row per 13F-HR / 13F-HR/A filing (notices dropped)
    infotable_raw  every information-table row, value normalized to dollars
    holdings       infotable_raw combined per (accession, CUSIP, put/call)
See docs/decisions.md for why each choice was made.
"""

import shutil
import tempfile
import zipfile
from pathlib import Path

import duckdb

RAW_DIR = Path("data/raw/form13f")
DB_PATH = Path("data/interim/13f.duckdb")
TABLES = ["SUBMISSION", "COVERPAGE", "INFOTABLE"]

# Market value switched from thousands to whole dollars for filings made on or
# after this date (SEC data set readme). It follows the filing date, not the
# report period: a 2001 holdings report filed in 2026 is already in dollars.
DOLLARS_FROM = "2023-01-03"

SCHEMA = """
CREATE TABLE filings (
    accession_number  VARCHAR PRIMARY KEY,
    cik               VARCHAR NOT NULL,
    manager_name      VARCHAR,
    submission_type   VARCHAR NOT NULL,   -- 13F-HR or 13F-HR/A
    is_amendment      BOOLEAN NOT NULL,
    amendment_no      INTEGER,
    amendment_type    VARCHAR,            -- RESTATEMENT, NEW HOLDINGS, or NULL
    report_type       VARCHAR,            -- HOLDINGS REPORT or COMBINATION REPORT
    period_of_report  DATE NOT NULL,
    quarter           VARCHAR NOT NULL,   -- e.g. 2026Q2, from period_of_report
    filing_date       DATE NOT NULL,
    known_at          DATE NOT NULL,      -- = filing_date: when the public could see it
    source_file       VARCHAR NOT NULL
);

CREATE TABLE infotable_raw (
    accession_number       VARCHAR NOT NULL,
    infotable_sk           BIGINT NOT NULL,
    name_of_issuer         VARCHAR,
    title_of_class         VARCHAR,
    cusip                  VARCHAR,
    figi                   VARCHAR,
    value_reported         DOUBLE,          -- as filed (thousands or dollars)
    value_usd              DOUBLE,          -- normalized to dollars
    shares_or_principal    DOUBLE,
    shares_or_principal_type VARCHAR,       -- SH or PRN
    put_call               VARCHAR,         -- Put, Call, or NULL for the security itself
    investment_discretion  VARCHAR,
    other_manager          VARCHAR,
    voting_sole            DOUBLE,
    voting_shared          DOUBLE,
    voting_none            DOUBLE,
    PRIMARY KEY (accession_number, infotable_sk)
);
"""

# Built one source file at a time: grouping all rows at once spills gigabytes
# of temp data to disk. Every filing lives in exactly one source file, so
# grouping per file gives the same result.
HOLDINGS_SQL = """
INSERT INTO holdings
SELECT
    i.accession_number,
    f.cik,
    f.quarter,
    f.period_of_report,
    f.known_at,
    f.submission_type,
    f.amendment_type,
    i.cusip,
    coalesce(i.put_call, '') AS put_call,   -- '' = the security itself, so it can be part of a key
    any_value(i.name_of_issuer)           AS name_of_issuer,
    any_value(i.title_of_class)           AS title_of_class,
    -- A few filings report one CUSIP partly in shares and partly in bond
    -- principal; never add those together.
    coalesce(sum(i.shares_or_principal) FILTER (WHERE i.shares_or_principal_type = 'SH'), 0)  AS shares,
    coalesce(sum(i.shares_or_principal) FILTER (WHERE i.shares_or_principal_type = 'PRN'), 0) AS principal,
    sum(i.value_usd)                      AS value_usd,
    count(*)                              AS n_raw_rows
FROM infotable_raw i
JOIN filings f USING (accession_number)
WHERE f.source_file = $source_file
GROUP BY ALL
"""


def _read(path: Path) -> str:
    # quote='' because the SEC files are plain tab-separated and issuer names
    # can contain stray double quotes.
    return f"read_csv('{path}', delim='\t', header=true, all_varchar=true, quote='')"


def load_zip(con: duckdb.DuckDBPyConnection, zip_path: Path, work_dir: Path,
             ciks: list[str] | None = None) -> None:
    """Load one data set zip. With `ciks`, keep only those filers (the
    seed-fund history backfill), so a file adds thousands of rows, not millions."""
    cik_filter = ""
    if ciks is not None:
        quoted = ", ".join(f"'{c}'" for c in ciks)
        cik_filter = f"AND lpad(s.CIK, 10, '0') IN ({quoted})"
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        paths = {}
        with zipfile.ZipFile(zip_path) as zf:
            # Most files keep the tables at the top level; some put them in a
            # folder (01jun2025-31aug2025 has 01JUN2025-31AUG2025_form13f/...).
            for t in TABLES:
                member = next(n for n in zf.namelist() if n.rsplit("/", 1)[-1] == f"{t}.tsv")
                paths[t] = Path(zf.extract(member, tmp))
        sub, cov, info = (_read(paths[t]) for t in TABLES)

        con.execute(f"""
            INSERT INTO filings
            SELECT
                s.ACCESSION_NUMBER,
                s.CIK,
                c.FILINGMANAGER_NAME,
                s.SUBMISSIONTYPE,
                s.SUBMISSIONTYPE = '13F-HR/A',
                try_cast(c.AMENDMENTNO AS INTEGER),
                nullif(c.AMENDMENTTYPE, ''),
                replace(c.REPORTTYPE, '13F ', ''),
                p.period,
                strftime(p.period, '%Y') || 'Q' || quarter(p.period),
                p.filed,
                p.filed,
                '{zip_path.name}'
            FROM {sub} s
            JOIN {cov} c USING (ACCESSION_NUMBER),
            LATERAL (SELECT strptime(s.PERIODOFREPORT, '%d-%b-%Y')::DATE AS period,
                            strptime(s.FILING_DATE, '%d-%b-%Y')::DATE AS filed) p
            WHERE s.SUBMISSIONTYPE IN ('13F-HR', '13F-HR/A')
              -- notice-only amendments carry no holdings
              AND c.REPORTTYPE <> '13F NOTICE'
              {cik_filter}
        """)

        con.execute(f"""
            INSERT INTO infotable_raw
            SELECT
                i.ACCESSION_NUMBER,
                i.INFOTABLE_SK::BIGINT,
                i.NAMEOFISSUER,
                i.TITLEOFCLASS,
                upper(trim(i.CUSIP)),
                nullif(i.FIGI, ''),
                i.VALUE::DOUBLE,
                i.VALUE::DOUBLE * CASE WHEN f.filing_date < DATE '{DOLLARS_FROM}' THEN 1000 ELSE 1 END,
                i.SSHPRNAMT::DOUBLE,
                i.SSHPRNAMTTYPE,
                nullif(i.PUTCALL, ''),
                i.INVESTMENTDISCRETION,
                nullif(i.OTHERMANAGER, ''),
                i.VOTING_AUTH_SOLE::DOUBLE,
                i.VOTING_AUTH_SHARED::DOUBLE,
                i.VOTING_AUTH_NONE::DOUBLE
            FROM {info} i
            JOIN filings f ON f.accession_number = i.ACCESSION_NUMBER
                          AND f.source_file = '{zip_path.name}'
        """)


def build(raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH) -> None:
    zips = sorted(raw_dir.glob("*_form13f.zip"))
    if not zips:
        raise SystemExit(f"No data sets in {raw_dir}; run src.ingest.download first.")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_db = db_path.with_suffix(".building.duckdb")
    tmp_db.unlink(missing_ok=True)

    con = duckdb.connect(str(tmp_db))
    con.execute(SCHEMA)
    for z in zips:
        print(f"loading {z.name}")
        load_zip(con, z, db_path.parent)
        if z is zips[0]:
            # Create the empty holdings table with the right column types.
            con.execute(f"CREATE TABLE holdings AS {HOLDINGS_SQL.split('INSERT INTO holdings', 1)[1]} LIMIT 0",
                        {"source_file": z.name})
        con.execute(HOLDINGS_SQL, {"source_file": z.name})
    for t in ["filings", "infotable_raw", "holdings"]:
        print(f"{t:14s} {con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]:>12,} rows")
    con.close()
    # Swap in only once the build succeeded, so a failed run never leaves a half-built DB.
    shutil.move(tmp_db, db_path)


if __name__ == "__main__":
    build()
