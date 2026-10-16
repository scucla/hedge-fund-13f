"""Deal terms and 8-K items from SEC filings, cached under data/raw/edgar/.

- `deal_text(cik, accession, form)`: the text of the first ~800 KB of a deal
  filing's main document (merger proxy, 14D9, 425...), cached as gzip text in
  data/raw/edgar/deals/. Proxies state the consideration on their first pages
  (letter to shareholders, Q&A, summary).
- `parse_terms(text)`: cash per share and exchange ratio from the sentence
  that says what each target share becomes ("entitled to receive $54.20 in
  cash ... for each share", "$50.00 in cash, one share of Bristol-Myers Squibb
  common stock and one contingent value right"). CVRs, elections and proration
  are not modelled; CVRs are recorded as `has_cvr`.
- `items_8k(cik)`: (filing date, items) of the issuer's 8-Ks, from the
  submissions API, cached in data/raw/edgar/filings_8k/.
"""

import gzip
import html
import json
import re
from pathlib import Path

import requests

from src.ingest import sec

DEALS_DIR = Path("data/raw/edgar/deals")
FILINGS_8K_DIR = Path("data/raw/edgar/filings_8k")
MAX_DOC_BYTES = 800_000
MAX_TEXT_CHARS = 250_000

_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "one-half": 0.5, "one half": 0.5}
_NUM = r"(\d+(?:\.\d+)?|one-half|one half|one|two|three|four|five)"
# the sentence that defines the consideration
_CONSIDERATION = re.compile(
    r"(?:entitled to receive|right to receive|converted into|exchanged for|will receive)\s+"
    r"(?:the\s+(?:per\s+share\s+)?merger consideration\s+(?:of|consisting of)\s+)?"
    r"(.{0,400}?)\s+(?:for each|per share of|per\s+[A-Z][\w.&\- ]{0,40}?share|in exchange for each)",
    re.I | re.S)
_CASH = re.compile(r"\$\s?(\d{1,4}(?:,\d{3})*(?:\.\d+)?)\s*(?:per share\s*)?(?:,\s*)?(?:net to the (?:seller|holder)\s*)?in cash", re.I)
_STOCK = re.compile(_NUM + r"\s+(?:\(\d+(?:\.\d+)?\)\s+)?(?:of\s+an?\s+|validly issued[\w ,]*?\s+)?(?:fully paid[\w ,-]*?\s+)?"
                    r"(?:ordinary\s+)?shares?\s+of\s+(.{2,80}?)\s+(?:common|ordinary|class [ab]|series [ab]|american depositary)",
                    re.I | re.S)
_OFFER = re.compile(r"(?:offer price|at a price)\s+of\s+\$\s?(\d{1,4}(?:\.\d+)?)\s+per\s+(?:share|Share)", re.I)


def _write_atomic(path: Path, data: bytes) -> None:
    """Write then rename, so an interrupted download never leaves a truncated cache file."""
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(path)


def _to_text(raw: bytes) -> str:
    t = raw.decode("utf-8", errors="ignore")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t.replace("​", " ")).strip()[:MAX_TEXT_CHARS]


def main_document_url(cik: str, accession: str, form: str) -> str | None:
    """The filing's primary document from its -index.htm page (row whose Type is the form)."""
    folder = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
    page = sec.get(f"{folder}/{accession}-index.htm").text
    first = None
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        href = re.search(r'href="([^"]+)"', row)
        if len(cells) >= 4 and href:
            first = first or href.group(1)
            if cells[3] == form:
                return "https://www.sec.gov" + href.group(1)
    return "https://www.sec.gov" + first if first else None


def deal_text(cik: str, accession: str, form: str, fetch: bool = True) -> str | None:
    path = DEALS_DIR / f"{accession}.txt.gz"
    if path.exists():
        return gzip.decompress(path.read_bytes()).decode() or None
    if not fetch:
        return None
    try:
        url = main_document_url(cik, accession, form)
        text = _to_text(sec.get_head(url, MAX_DOC_BYTES)) if url else ""
    except requests.HTTPError as e:
        if e.response is None or e.response.status_code != 404:
            raise
        text = ""   # cached as empty: a missing document stays missing
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, gzip.compress(text.encode()))
    return text or None


def _num(s: str) -> float:
    s = s.lower().replace(",", "")
    return float(_WORDS.get(s, s)) if s in _WORDS else float(s)


def parse_terms(text: str) -> dict | None:
    """{'cash': float|None, 'ratio': float|None, 'acquirer_name': str|None, 'has_cvr': bool, 'snippet': str}
    from the first consideration sentence that names cash or stock; else a
    tender offer price; else None."""
    for m in _CONSIDERATION.finditer(text):
        phrase = m.group(1)
        cash = _CASH.search(phrase)
        stock = _STOCK.search(phrase)
        if not cash and not stock:
            continue
        return {
            "cash": _num(cash.group(1)) if cash else None,
            "ratio": _num(stock.group(1)) if stock else None,
            "acquirer_name": stock.group(2).strip(" ,") if stock else None,
            "has_cvr": "contingent value right" in phrase.lower(),
            "snippet": m.group(0)[:500],
        }
    offer = _OFFER.search(text)
    if offer:
        return {"cash": _num(offer.group(1)), "ratio": None, "acquirer_name": None, "has_cvr": False,
                "snippet": text[max(offer.start() - 100, 0):offer.end() + 100]}
    return None


def items_8k(cik: str, fetch: bool = True) -> list[tuple[str, str]] | None:
    """[(filing date, items)] for the issuer's 8-Ks (most recent 1,000 filings; enough
    for the companies here, which stopped filing when they were delisted)."""
    path = FILINGS_8K_DIR / f"CIK{cik}.json.gz"
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    if not fetch:
        return None
    try:
        recent = sec.get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()["filings"]["recent"]
        rows = [(d, i) for f, d, i in zip(recent["form"], recent["filingDate"], recent["items"]) if f.startswith("8-K")]
    except requests.HTTPError as e:
        if e.response is None or e.response.status_code != 404:
            raise
        rows = []
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, gzip.compress(json.dumps(rows).encode()))
    return rows
