"""Loader and point-in-time logic, checked on a tiny fake data set."""

from datetime import date

import pytest

from src.ingest.positions import fund_holdings


def scalar(con, sql, *params):
    return con.execute(sql, list(params)).fetchone()[0]


def test_notices_are_dropped(fake_db):
    assert scalar(fake_db, "SELECT count(*) FROM filings WHERE accession_number = 'A3'") == 0
    assert scalar(fake_db, "SELECT count(*) FROM infotable_raw WHERE accession_number = 'A3'") == 0


def test_values_switch_from_thousands_on_filing_date(fake_db):
    b1 = scalar(fake_db, "SELECT value_usd FROM infotable_raw WHERE accession_number = 'B1'")
    b2 = scalar(fake_db, "SELECT value_usd FROM infotable_raw WHERE accession_number = 'B2'")
    assert (b1, b2) == (5000, 7000)


def test_known_at_and_quarter(fake_db):
    row = fake_db.execute(
        "SELECT known_at, filing_date, quarter FROM filings WHERE accession_number = 'B2'"
    ).fetchone()
    assert row == (date(2023, 2, 10), date(2023, 2, 10), "2022Q4")


def test_lots_combine_but_shares_and_principal_stay_separate(fake_db):
    alpha = fake_db.execute(
        "SELECT shares, value_usd, n_raw_rows FROM holdings WHERE accession_number='A1' AND cusip='AAA000001'"
    ).fetchone()
    assert alpha == (150, 1500, 2)
    gamma = fake_db.execute(
        "SELECT shares, principal, value_usd FROM holdings WHERE accession_number='A1' AND cusip='CCC000003'"
    ).fetchone()
    assert gamma == (10, 1000, 1000)


def test_options_are_separate_positions(fake_db):
    rows = fake_db.execute(
        "SELECT put_call, value_usd FROM holdings WHERE accession_number='A1' AND cusip='BBB000002' ORDER BY 1"
    ).fetchall()
    assert rows == [("", 3000), ("Call", 500)]


def test_new_holdings_amendment_is_added_only_once_public(fake_db):
    before, _, _ = fund_holdings(fake_db, 1, "2026Q2", as_of=date(2026, 9, 1))
    after, used, _ = fund_holdings(fake_db, 1, "2026Q2", as_of=date(2026, 9, 30))
    assert "DDD000004" not in before["cusip"].to_list()
    assert "DDD000004" in after["cusip"].to_list()
    assert used["accession_number"].to_list() == ["A1", "A2"]
    assert after["weight"].sum() == pytest.approx(1.0)


def test_restatement_replaces_original_and_untyped_amendment_is_skipped(fake_db):
    early, _, _ = fund_holdings(fake_db, "3", "2026Q2", as_of=date(2026, 8, 10))
    assert early["value_usd"].to_list() == [100]
    late, used, warnings = fund_holdings(fake_db, "0000000003", "2026Q2", as_of=date(2026, 12, 31))
    assert sorted(late["cusip"].to_list()) == ["FFF000006", "GGG000007"]
    assert used["accession_number"].to_list() == ["C2"]
    assert len(warnings) == 1 and "C3" in warnings[0]


def test_nothing_before_first_filing(fake_db):
    positions, _, _ = fund_holdings(fake_db, 1, "2026Q2", as_of=date(2026, 8, 9))
    assert positions.is_empty()


def test_tables_inside_a_folder_in_the_zip(tmp_path):
    """Some SEC files put the tables in a subfolder (01jun2025-31aug2025)."""
    import zipfile

    import duckdb

    from src.ingest.load import build
    from tests.conftest import write_fake_dataset

    flat = tmp_path / "flat.zip"
    write_fake_dataset(flat)
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(flat) as src, zipfile.ZipFile(raw / "01jun2025-31aug2025_form13f.zip", "w") as dst:
        for name in src.namelist():
            dst.writestr(f"01JUN2025-31AUG2025_form13f/{name}", src.read(name))
    db_path = tmp_path / "13f.duckdb"
    build(raw, db_path)
    con = duckdb.connect(str(db_path), read_only=True)
    assert con.execute("SELECT count(*) FROM filings").fetchone()[0] == 7   # 8 minus the notice
