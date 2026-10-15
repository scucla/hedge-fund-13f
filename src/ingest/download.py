"""Download SEC Form 13F data sets into data/raw/.

Usage:
    uv run python -m src.ingest.download --start 2025Q3 --end 2026Q2

--start/--end are *holdings* quarters (the period a 13F reports on). Since
2024 the SEC publishes files by filing-date window (e.g. 01jun2026-31aug2026),
so each holdings quarter is mapped to the file whose window contains that
quarter's 45-day filing deadline. See docs/decisions.md.
"""

import argparse
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from src.ingest import sec

LISTING_URL = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
RAW_DIR = Path("data/raw/form13f")

_WINDOW_RE = re.compile(r"(\d{2}[a-z]{3}\d{4})-(\d{2}[a-z]{3}\d{4})_form13f\.zip$")
_CALENDAR_RE = re.compile(r"(\d{4})q([1-4])_form13f\.zip$")
_LINK_RE = re.compile(r'href="([^"]*_form13f\.zip)"')


@dataclass(frozen=True)
class DataSet:
    url: str
    filename: str
    first_filing_date: date
    last_filing_date: date


def parse_quarter(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{4})Q([1-4])", text.upper())
    if not m:
        raise ValueError(f"Quarter must look like 2026Q2, got {text!r}")
    return int(m.group(1)), int(m.group(2))


def quarter_end(year: int, q: int) -> date:
    month = 3 * q
    next_month_start = date(year + (month == 12), month % 12 + 1, 1)
    return next_month_start - timedelta(days=1)


def quarters_between(start: str, end: str) -> list[tuple[int, int]]:
    y, q = parse_quarter(start)
    ey, eq = parse_quarter(end)
    out = []
    while (y, q) <= (ey, eq):
        out.append((y, q))
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def filing_deadline(year: int, q: int) -> date:
    return quarter_end(year, q) + timedelta(days=45)


def parse_dataset(url: str) -> DataSet | None:
    """Work out which filing dates a data set file covers from its name."""
    filename = url.rsplit("/", 1)[-1]
    if m := _WINDOW_RE.search(filename):
        first, last = (datetime.strptime(s, "%d%b%Y").date() for s in m.groups())
        return DataSet(url, filename, first, last)
    if m := _CALENDAR_RE.search(filename):
        # Pre-2024 files hold the filings made during that calendar quarter.
        y, q = int(m.group(1)), int(m.group(2))
        first = date(y, 3 * q - 2, 1)
        return DataSet(url, filename, first, quarter_end(y, q))
    return None


def list_datasets() -> list[DataSet]:
    """Read the SEC listing page rather than guessing URLs: the newest file
    sometimes lives under a different path from the older ones."""
    html = sec.get(LISTING_URL).text
    found = {}
    for href in _LINK_RE.findall(html):
        url = href if href.startswith("http") else "https://www.sec.gov" + href
        ds = parse_dataset(url)
        if ds:
            found[ds.filename] = ds
    if not found:
        raise RuntimeError(f"No 13F data set links found on {LISTING_URL}; has the page changed?")
    return sorted(found.values(), key=lambda d: d.first_filing_date)


def dataset_for_quarter(datasets: list[DataSet], year: int, q: int) -> DataSet:
    deadline = filing_deadline(year, q)
    for ds in datasets:
        if ds.first_filing_date <= deadline <= ds.last_filing_date:
            return ds
    raise LookupError(f"No published data set covers the {year}Q{q} deadline ({deadline}) yet.")


def download(ds: DataSet, raw_dir: Path = RAW_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / ds.filename
    if target.exists():
        print(f"cached   {target}")
        return target
    print(f"download {ds.url}")
    resp = sec.get(ds.url, stream=True)
    partial = target.with_suffix(".part")
    with partial.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    partial.rename(target)  # only a complete file ever gets the final name
    print(f"saved    {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="first holdings quarter, e.g. 2025Q3")
    parser.add_argument("--end", required=True, help="last holdings quarter, e.g. 2026Q2")
    args = parser.parse_args()

    datasets = list_datasets()
    for year, q in quarters_between(args.start, args.end):
        ds = dataset_for_quarter(datasets, year, q)
        print(f"{year}Q{q} -> {ds.filename}")
        download(ds)


if __name__ == "__main__":
    main()
