from datetime import date

import pytest

from src.ingest.download import (
    dataset_for_quarter,
    filing_deadline,
    parse_dataset,
    quarters_between,
)

BASE = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/"
DATASETS = [
    parse_dataset(BASE + name)
    for name in [
        "2023q3_form13f.zip",
        "2023q4_form13f.zip",
        "01jan2024-29feb2024_form13f.zip",
        "01dec2025-28feb2026_form13f.zip",
        "01mar2026-31may2026_form13f.zip",
        "01jun2026-31aug2026_form13f.zip",
    ]
]


def test_parse_window_and_calendar_names():
    window = parse_dataset(BASE + "01jun2026-31aug2026_form13f.zip")
    assert (window.first_filing_date, window.last_filing_date) == (date(2026, 6, 1), date(2026, 8, 31))
    calendar = parse_dataset(BASE + "2023q4_form13f.zip")
    assert (calendar.first_filing_date, calendar.last_filing_date) == (date(2023, 10, 1), date(2023, 12, 31))
    assert parse_dataset(BASE + "form_13f_readme.pdf") is None


def test_quarters_between_wraps_years():
    assert quarters_between("2025Q3", "2026Q2") == [(2025, 3), (2025, 4), (2026, 1), (2026, 2)]


@pytest.mark.parametrize(
    "year, q, expected",
    [
        (2026, 2, "01jun2026-31aug2026_form13f.zip"),  # due Aug 14
        (2026, 1, "01mar2026-31may2026_form13f.zip"),  # due May 15
        (2025, 4, "01dec2025-28feb2026_form13f.zip"),  # due Feb 14
        (2023, 4, "01jan2024-29feb2024_form13f.zip"),  # straddles the naming change
        (2023, 2, "2023q3_form13f.zip"),  # old scheme: filed in the next calendar quarter
    ],
)
def test_holdings_quarter_maps_to_file_with_deadline(year, q, expected):
    assert dataset_for_quarter(DATASETS, year, q).filename == expected


def test_deadline_is_45_days_after_quarter_end():
    assert filing_deadline(2026, 2) == date(2026, 8, 14)
