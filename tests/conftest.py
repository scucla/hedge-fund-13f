"""Fixtures: a tiny fake SEC data set built into a throwaway DuckDB, and the
real database when it exists."""

import zipfile
from pathlib import Path

import duckdb
import pytest

from src.ingest.load import DB_PATH, build

SUBMISSION = ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"]
COVERPAGE = ["ACCESSION_NUMBER", "ISAMENDMENT", "AMENDMENTNO", "AMENDMENTTYPE", "FILINGMANAGER_NAME", "REPORTTYPE"]
INFOTABLE = [
    "ACCESSION_NUMBER", "INFOTABLE_SK", "NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "FIGI", "VALUE",
    "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL", "INVESTMENTDISCRETION", "OTHERMANAGER",
    "VOTING_AUTH_SOLE", "VOTING_AUTH_SHARED", "VOTING_AUTH_NONE",
]

# (accession, filed, form, cik, period, amendment_type, report_type)
FILINGS = [
    # Fund 1, 2026Q2: original, a NEW HOLDINGS amendment, and a notice to drop.
    ("A1", "10-AUG-2026", "13F-HR", "0000000001", "30-JUN-2026", "", "13F HOLDINGS REPORT"),
    ("A2", "20-SEP-2026", "13F-HR/A", "0000000001", "30-JUN-2026", "NEW HOLDINGS", "13F HOLDINGS REPORT"),
    ("A3", "11-AUG-2026", "13F-NT", "0000000001", "30-JUN-2026", "", "13F NOTICE"),
    # Fund 2: one filing before the 2023-01-03 unit switch, one after.
    ("B1", "10-NOV-2022", "13F-HR", "0000000002", "30-SEP-2022", "", "13F HOLDINGS REPORT"),
    ("B2", "10-FEB-2023", "13F-HR", "0000000002", "31-DEC-2022", "", "13F HOLDINGS REPORT"),
    # Fund 3, 2026Q2: original, a RESTATEMENT, then an amendment with no type.
    ("C1", "01-AUG-2026", "13F-HR", "0000000003", "30-JUN-2026", "", "13F HOLDINGS REPORT"),
    ("C2", "20-AUG-2026", "13F-HR/A", "0000000003", "30-JUN-2026", "RESTATEMENT", "13F HOLDINGS REPORT"),
    ("C3", "25-AUG-2026", "13F-HR/A", "0000000003", "30-JUN-2026", "", "13F HOLDINGS REPORT"),
]

# (accession, cusip, name, value, amount, type, put/call, discretion)
ROWS = [
    ("A1", "AAA000001", "ALPHA INC", 1000, 100, "SH", "", "SOLE"),
    ("A1", "AAA000001", "ALPHA INC", 500, 50, "SH", "", "SOLE"),  # second lot, same key
    ("A1", "BBB000002", "BETA CORP", 3000, 300, "SH", "", "DFND"),
    ("A1", "BBB000002", "BETA CORP", 500, 100, "SH", "Call", "DFND"),
    ("A1", "CCC000003", "GAMMA CO", 100, 10, "SH", "", "SOLE"),
    ("A1", "CCC000003", "GAMMA CO", 900, 1000, "PRN", "", "SOLE"),  # bond lot of the same CUSIP
    ("A2", "DDD000004", "DELTA LTD", 2000, 20, "SH", "", "SOLE"),
    ("A3", "ZZZ000009", "SHOULD NOT LOAD", 1, 1, "SH", "", "SOLE"),
    ("B1", "EEE000005", "EPSILON", 5, 100, "SH", "", "SOLE"),  # thousands -> $5,000
    ("B2", "EEE000005", "EPSILON", 7000, 100, "SH", "", "SOLE"),  # dollars
    ("C1", "FFF000006", "PHI", 100, 10, "SH", "", "SOLE"),
    ("C2", "FFF000006", "PHI", 200, 10, "SH", "", "SOLE"),
    ("C2", "GGG000007", "PSI", 50, 5, "SH", "", "SOLE"),
    ("C3", "HHH000008", "OMEGA", 999, 9, "SH", "", "SOLE"),
]


def _tsv(header: list[str], rows: list[list]) -> str:
    return "\n".join("\t".join(map(str, r)) for r in [header, *rows]) + "\n"


def write_fake_dataset(path: Path) -> None:
    sub = [[a, filed, form, cik, period] for a, filed, form, cik, period, _, _ in FILINGS]
    cov = [
        [a, "Y" if form.endswith("/A") else "N", "1" if form.endswith("/A") else "", atype, f"FUND {cik[-1]}", rtype]
        for a, _, form, cik, _, atype, rtype in FILINGS
    ]
    info = [
        [a, sk, name, "COM", cusip, "", value, amount, kind, pc, disc, "", amount, 0, 0]
        for sk, (a, cusip, name, value, amount, kind, pc, disc) in enumerate(ROWS, 1)
    ]
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("SUBMISSION.tsv", _tsv(SUBMISSION, sub))
        zf.writestr("COVERPAGE.tsv", _tsv(COVERPAGE, cov))
        zf.writestr("INFOTABLE.tsv", _tsv(INFOTABLE, info))


@pytest.fixture(scope="session")
def fake_db(tmp_path_factory):
    root = tmp_path_factory.mktemp("fake13f")
    raw = root / "raw"
    raw.mkdir()
    write_fake_dataset(raw / "01jun2026-31aug2026_form13f.zip")
    db_path = root / "13f.duckdb"
    build(raw, db_path)
    con = duckdb.connect(str(db_path), read_only=True)
    yield con
    con.close()


@pytest.fixture(scope="session")
def real_db():
    if not DB_PATH.exists():
        pytest.skip(f"{DB_PATH} not built; run src.ingest.load")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    yield con
    con.close()


@pytest.fixture(scope="session")
def sec_db():
    from src.securities.db import DB_PATH as SEC_DB_PATH

    if not SEC_DB_PATH.exists():
        pytest.skip(f"{SEC_DB_PATH} not built; run the src.securities steps")
    con = duckdb.connect(str(SEC_DB_PATH), read_only=True)
    yield con
    con.close()


def table_exists(con, name: str) -> bool:
    return con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()[0] > 0
