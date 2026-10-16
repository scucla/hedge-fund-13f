"""Why stocks drop out of our price data, and what holders earned when they did.

Usage:
    uv run python -m src.securities.delistings            # classify + returns
    uv run python -m src.securities.delistings --fetch    # also download deal documents (SEC, cached)

Yahoo has no prices at all for stocks that stopped trading before today, so
every quarter of such a stock is unpriced, not just the last one. The ordinary
quarters get 13F-implied returns (src.securities.implied_prices). This module
handles the **last** quarter: a stock held by some seed fund at quarter q and by
none at q+1 ("exit") either was delisted during q+1 or was simply sold by all of
them.

For each exit of an unpriced common stock or ADR:

1. **Delisting date**: the first Form 25 / 25-NSE / 15-12B / 15-12G of the issuer
   from DELIST_EARLY_DAYS before q's end through DELIST_LATE_DAYS after q+1's
   end. No such filing -> not delisted in q+1 (`sold_out`, or `cusip_change` if
   the same issuer is held under another CUSIP next quarter).
2. **Why** (`exit_class`), from the issuer's filings in the DEAL_LOOKBACK_DAYS
   before the delisting:
   - `cash_acquisition`: a merger proxy, tender-offer response (SC 14D9) or
     going-private schedule (SC 13E3), and no stock-deal filing.
   - `stock_acquisition`: a 425 or S-4/F-4 naming another company. Terms decide
     whether it is pure stock or `mixed` (cash + stock).
   - `performance`: no deal filing, and an 8-K with item 1.03 (bankruptcy) or
     3.01 (listing-standards notice) in the year before, or the last 13F price
     below $1.
   - `other`: none of the above (voluntary delisting, foreign issuer moving
     its listing, deregistration of a small company).
3. **Delisting return** over q+1, from q's 13F-implied price P:
   - cash: deal price / P - 1
   - stock: exchange ratio x acquirer close on the delisting date / P - 1
   - mixed: (cash + ratio x acquirer close) / P - 1
   - performance: Shumway (1997): -30%. Shumway & Warther (1999) find about
     -55% for Nasdaq; reported as a sensitivity, never used in the label.
   - deal terms not found: last 13F price, i.e. 0% (`terms_source = 'last_13f_price'`).
   - other: no return (stays missing, with the reason recorded).

Writes `delisting_events` (one row per exit of an unpriced stock).
"""

import argparse
import re

import polars as pl

from src.securities import db
from src.securities.deal_docs import deal_text, items_8k, parse_terms

DELIST_EARLY_DAYS = 30      # a fund can report a position at q's end that was delisted just before
DELIST_LATE_DAYS = 15       # 15-12B is filed ~10 days after the delisting takes effect
DEAL_LOOKBACK_DAYS = 730    # announcement to close is rarely more than two years
DISTRESS_LOOKBACK_DAYS = 365
PENNY = 1.0
SHUMWAY_RET = -0.30
SHUMWAY_WARTHER_NASDAQ_RET = -0.55
# a parsed cash price this far from the last 13F price is a parse error, not a deal
# (announced targets trade near the offer; 0.5x-2x leaves room for hostile bumps)
TERMS_SANE = (0.5, 2.0)
DOC_PRIORITY = ["DEFM14A", "DEFM14C", "SC 14D9", "PREM14A", "PREM14C", "SC 13E3", "425"]
MAX_DOCS_PER_DEAL = 3
SPAC_SIC = 6770
_SPAC_NAME = re.compile(r"\bACQUISITION\b|\bACQ\b|\bSPAC\b", re.I)

QEND = "last_day(make_date(left({q}, 4)::INT, right({q}, 1)::INT * 3, 1))"
NEXTQ = """CASE WHEN right({q}, 1) = '4' THEN (left({q}, 4)::INT + 1)::VARCHAR || 'Q1'
               ELSE left({q}, 5) || (right({q}, 1)::INT + 1)::VARCHAR END"""


def exits(con) -> None:
    """Exits of unpriced common/ADR stocks, with the issuer's delisting and deal filings."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ex AS
        WITH held AS (
            SELECT DISTINCT p.cusip, p.quarter
            FROM fund_holdings p JOIN security_master m USING (cusip)
            WHERE p.cohort = 'seed' AND p.put_call = '' AND m.security_class IN ('common', 'adr')
        ),
        cik AS (
            SELECT m.cusip, m.security_class, m.symbol, m.name_of_issuer, m.figi_name,
                   coalesce(m.issuer_cik, f.issuer_cik) AS issuer_cik
            FROM security_master m LEFT JOIN issuer_cik_fill f USING (cusip)
        ),
        unpriced AS (SELECT cusip, quarter FROM stock_quarter WHERE price IS NULL)
        SELECT h.cusip, h.quarter, {NEXTQ.format(q='h.quarter')} AS next_quarter,
               {QEND.format(q='h.quarter')} AS q_end,
               {QEND.format(q=NEXTQ.format(q='h.quarter'))} AS next_q_end,
               c.issuer_cik, c.security_class, c.symbol, coalesce(c.figi_name, c.name_of_issuer) AS name
        FROM held h
        JOIN unpriced u USING (cusip, quarter)
        JOIN cik c USING (cusip)
        WHERE h.quarter < '{db.LAST_QUARTER}'
          AND NOT EXISTS (SELECT 1 FROM held n WHERE n.cusip = h.cusip AND n.quarter = {NEXTQ.format(q='h.quarter')})
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ex_delist AS
        SELECT e.cusip, e.quarter, min(s.known_at) AS delist_date,
               arg_min(s.form, s.known_at) AS delist_form
        FROM ex e JOIN sec_events s ON s.cik = e.issuer_cik
        WHERE s.kind IN ('delisting', 'deregistration')
          AND s.known_at > e.q_end - INTERVAL {DELIST_EARLY_DAYS} DAY
          AND s.known_at <= e.next_q_end + INTERVAL {DELIST_LATE_DAYS} DAY
        GROUP BY ALL
    """)
    # deal filings in the two years up to the delisting
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ex_deal AS
        SELECT e.cusip, e.quarter,
               bool_or(s.kind = 'merger_proxy') AS has_proxy,
               bool_or(s.form = 'SC 14D9') AS has_14d9,
               bool_or(s.form = 'SC 13E3') AS has_13e3,
               bool_or(s.kind = 'stock_deal' AND len(s.other_ciks) > 0) AS has_stock_deal,
               min(s.known_at) FILTER (WHERE s.kind IN ('merger_proxy', 'tender_offer', 'going_private', 'stock_deal'))
                   AS first_deal_filing,
               list(DISTINCT s.accession ORDER BY s.accession) FILTER (WHERE s.kind IN ('merger_proxy', 'tender_offer', 'going_private', 'stock_deal'))
                   AS deal_accessions
        FROM ex e JOIN ex_delist d USING (cusip, quarter)
        JOIN sec_events s ON s.cik = e.issuer_cik
        WHERE s.known_at BETWEEN d.delist_date - INTERVAL {DEAL_LOOKBACK_DAYS} DAY AND d.delist_date
        GROUP BY ALL
    """)


def candidate_docs(con) -> pl.DataFrame:
    """Deal filings to read per exit, best first: the definitive proxy states the final terms."""
    order = "CASE s.form " + " ".join(f"WHEN '{f}' THEN {i}" for i, f in enumerate(DOC_PRIORITY)) + " END"
    return con.execute(f"""
        SELECT e.cusip, e.quarter, e.issuer_cik, s.accession, s.form, s.known_at,
               row_number() OVER (PARTITION BY e.cusip, e.quarter ORDER BY {order}, s.known_at DESC) AS rank
        FROM ex e JOIN ex_delist d USING (cusip, quarter) JOIN sec_events s ON s.cik = e.issuer_cik
        WHERE s.form IN {tuple(DOC_PRIORITY)}
          AND s.known_at BETWEEN d.delist_date - INTERVAL {DEAL_LOOKBACK_DAYS} DAY AND d.delist_date
        QUALIFY rank <= {MAX_DOCS_PER_DEAL}
        ORDER BY e.cusip, e.quarter, rank
    """).pl()


def deal_terms(con, fetch: bool) -> pl.DataFrame:
    """Parsed terms per exit: the first candidate document whose terms parse."""
    docs = candidate_docs(con)
    out, seen = [], set()
    for i, (cusip, quarter, cik, acc, form) in enumerate(
            docs.select("cusip", "quarter", "issuer_cik", "accession", "form").iter_rows()):
        if (cusip, quarter) in seen:
            continue
        if fetch and i % 200 == 0:
            print(f"  deal documents: {i:,}/{len(docs):,}", flush=True)
        text = deal_text(cik, acc, form, fetch=fetch)
        terms = parse_terms(text) if text else None
        if terms:
            seen.add((cusip, quarter))
            out.append({"cusip": cusip, "quarter": quarter, "terms_accession": acc, "terms_form": form, **terms})
    schema = {"cusip": pl.Utf8, "quarter": pl.Utf8, "terms_accession": pl.Utf8, "terms_form": pl.Utf8,
              "cash": pl.Float64, "ratio": pl.Float64, "acquirer_name": pl.Utf8, "has_cvr": pl.Boolean,
              "snippet": pl.Utf8}
    return pl.DataFrame(out, schema=schema)


def distress_8k(con, fetch: bool) -> pl.DataFrame:
    """Latest 8-K item 1.03 (bankruptcy) / 3.01 (listing notice) date per delisted exit with no deal."""
    rows = con.execute(f"""
        SELECT e.cusip, e.quarter, e.issuer_cik, d.delist_date
        FROM ex e JOIN ex_delist d USING (cusip, quarter) LEFT JOIN ex_deal x USING (cusip, quarter)
        WHERE NOT coalesce(x.has_proxy OR x.has_14d9 OR x.has_13e3 OR x.has_stock_deal, false)
    """).pl()
    out = []
    for i, (cusip, quarter, cik, delist) in enumerate(rows.iter_rows()):
        if fetch and i % 200 == 0:
            print(f"  8-K items: {i:,}/{len(rows):,}", flush=True)
        items = items_8k(cik, fetch=fetch)
        if items is None:
            continue
        for date, its in items:
            d = pl.Series([date]).str.to_date()[0]
            if 0 <= (delist - d).days <= DISTRESS_LOOKBACK_DAYS:
                if "1.03" in its:
                    out.append((cusip, quarter, "1.03", d))
                if "3.01" in its:
                    out.append((cusip, quarter, "3.01", d))
    return pl.DataFrame(out, schema=["cusip", "quarter", "item", "date"], orient="row")


def classify(con, fetch: bool = False) -> None:
    terms = deal_terms(con, fetch)
    distress = distress_8k(con, fetch)
    con.execute("CREATE OR REPLACE TEMP TABLE ex_terms AS SELECT * FROM terms")
    con.execute("""CREATE OR REPLACE TEMP TABLE ex_distress AS
                   SELECT cusip, quarter, bool_or(item = '1.03') AS item_103, bool_or(item = '3.01') AS item_301
                   FROM distress GROUP BY ALL""")
    # acquirer for stock legs: a counterparty on the issuer's 425 / S-4 filings that we can price
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ex_acquirer AS
        WITH cp0 AS (
            SELECT e.cusip, e.quarter, unnest(s.other_ciks) AS acq_cik
            FROM ex e JOIN ex_delist d USING (cusip, quarter) JOIN sec_events s ON s.cik = e.issuer_cik
            WHERE s.kind = 'stock_deal'
              AND s.known_at BETWEEN d.delist_date - INTERVAL {DEAL_LOOKBACK_DAYS} DAY AND d.delist_date
        ),
        cp AS (SELECT cusip, quarter, acq_cik, count(*) AS n_filings FROM cp0 GROUP BY ALL),
        priced AS (
            SELECT DISTINCT m.issuer_cik, m.symbol, m.cusip
            FROM security_master m WHERE m.security_class = 'common'
              AND m.symbol IN (SELECT symbol FROM returns_quarterly)
        )
        SELECT cp.cusip, cp.quarter,
               arg_max(p.symbol, cp.n_filings) AS acq_symbol, arg_max(p.cusip, cp.n_filings) AS acq_cusip,
               arg_max(cp.acq_cik, cp.n_filings) AS acq_cik
        FROM cp JOIN priced p ON p.issuer_cik = cp.acq_cik
        GROUP BY ALL
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE delisting_events AS
        WITH base AS (
            SELECT e.*, d.delist_date, d.delist_form,
                   x.first_deal_filing, x.has_proxy, x.has_14d9, x.has_13e3, x.has_stock_deal,
                   coalesce(x.has_proxy OR x.has_14d9 OR x.has_13e3 OR x.has_stock_deal, false) AS has_deal,
                   t.terms_accession, t.terms_form, t.cash, t.ratio, t.acquirer_name, t.has_cvr, t.snippet,
                   a.acq_symbol, a.acq_cik,
                   ds.item_103, ds.item_301,
                   ip.px AS px_q, ip.px_disagree,
                   ip.known_at AS px_known_at,
                   (pr.sic = {SPAC_SIC} OR regexp_matches(e.name, '{_SPAC_NAME.pattern}', 'i')) AS is_spac,
                   EXISTS (SELECT 1 FROM fund_holdings h JOIN security_master m USING (cusip)
                           LEFT JOIN issuer_cik_fill f USING (cusip)
                           WHERE coalesce(m.issuer_cik, f.issuer_cik) = e.issuer_cik AND h.cusip <> e.cusip
                             AND h.quarter = e.next_quarter AND h.cohort = 'seed' AND h.put_call = '')
                       AS issuer_held_next_q
            FROM ex e
            LEFT JOIN ex_delist d USING (cusip, quarter)
            LEFT JOIN ex_deal x USING (cusip, quarter)
            LEFT JOIN ex_terms t USING (cusip, quarter)
            LEFT JOIN ex_acquirer a USING (cusip, quarter)
            LEFT JOIN ex_distress ds USING (cusip, quarter)
            LEFT JOIN implied_prices ip USING (cusip, quarter)
            LEFT JOIN issuer_profile pr ON pr.issuer_cik = e.issuer_cik
        ),
        -- acquirer's nominal price on the delisting date: its median 13F price at q's end
        -- (nominal, like the exchange ratio) times its adjusted-close return since then
        acq AS (
            SELECT b.cusip, b.quarter,
                   aip.px * (SELECT arg_max(close, date) FROM prices_daily WHERE symbol = b.acq_symbol AND date <= b.delist_date)
                          / (SELECT arg_max(close, date) FROM prices_daily WHERE symbol = b.acq_symbol AND date <= b.q_end)
                       AS acq_px_delist
            FROM base b
            JOIN security_master am ON am.symbol = b.acq_symbol AND am.security_class = 'common'
            JOIN implied_prices aip ON aip.cusip = am.cusip AND aip.quarter = b.quarter
            WHERE b.ratio IS NOT NULL
            QUALIFY row_number() OVER (PARTITION BY b.cusip, b.quarter ORDER BY aip.n_funds DESC) = 1
        ),
        cls AS (
            SELECT b.*, acq.acq_px_delist,
                CASE
                    WHEN b.delist_date IS NULL AND b.issuer_held_next_q THEN 'cusip_change'
                    WHEN b.delist_date IS NULL AND b.issuer_cik IS NULL THEN 'no_cik'
                    WHEN b.delist_date IS NULL THEN 'sold_out'
                    WHEN b.is_spac THEN 'other'
                    WHEN b.has_deal AND b.ratio IS NOT NULL AND b.cash IS NOT NULL THEN 'stock_acquisition'
                    WHEN b.has_deal AND b.ratio IS NOT NULL THEN 'stock_acquisition'
                    WHEN b.has_deal AND b.cash IS NOT NULL THEN 'cash_acquisition'
                    WHEN b.has_deal AND b.has_stock_deal THEN 'stock_acquisition'
                    WHEN b.has_deal THEN 'cash_acquisition'
                    WHEN b.item_103 OR b.item_301 OR b.px_q < {PENNY} THEN 'performance'
                    ELSE 'other'
                END AS exit_class,
                CASE
                    WHEN b.delist_date IS NULL THEN NULL
                    WHEN b.is_spac THEN 'spac'
                    WHEN b.has_deal AND b.ratio IS NOT NULL AND b.cash IS NOT NULL THEN 'mixed'
                    WHEN b.has_deal AND b.ratio IS NOT NULL THEN 'stock'
                    WHEN b.has_deal AND b.cash IS NOT NULL THEN 'cash'
                    WHEN b.has_deal THEN 'terms_not_found'
                    WHEN b.item_103 THEN 'bankruptcy'
                    WHEN b.item_301 THEN 'listing_notice'
                    WHEN b.px_q < {PENNY} THEN 'penny'
                END AS exit_subclass
            FROM base b LEFT JOIN acq USING (cusip, quarter)
        ),
        raw AS (
            SELECT *,
                CASE exit_subclass
                    WHEN 'cash' THEN cash / px_q - 1
                    WHEN 'stock' THEN ratio * acq_px_delist / px_q - 1
                    WHEN 'mixed' THEN (cash + ratio * acq_px_delist) / px_q - 1
                END AS terms_ret
            FROM cls
        )
        SELECT *,
            -- terms that put the payout outside 0.5x-2x of the last 13F price are parse errors
            terms_ret IS NOT NULL AND (terms_ret + 1 < {TERMS_SANE[0]} OR terms_ret + 1 > {TERMS_SANE[1]}) AS terms_suspect,
            CASE
                WHEN exit_class IN ('cash_acquisition', 'stock_acquisition') AND px_q IS NOT NULL THEN
                    CASE WHEN terms_ret IS NOT NULL AND terms_ret + 1 BETWEEN {TERMS_SANE[0]} AND {TERMS_SANE[1]}
                         THEN terms_ret ELSE 0.0 END
                WHEN exit_class = 'performance' THEN {SHUMWAY_RET}
            END AS delist_ret,
            CASE
                WHEN exit_class IN ('cash_acquisition', 'stock_acquisition') AND px_q IS NOT NULL THEN
                    CASE WHEN terms_ret IS NOT NULL AND terms_ret + 1 BETWEEN {TERMS_SANE[0]} AND {TERMS_SANE[1]}
                         THEN 'delist_' || exit_subclass ELSE 'delist_last_13f_price' END
                WHEN exit_class = 'performance' THEN 'delist_shumway'
            END AS ret_source,
            CASE WHEN exit_class = 'performance' THEN {SHUMWAY_WARTHER_NASDAQ_RET} END AS delist_ret_sensitivity,
            -- the return is public once the stock is delisted and the q 13F price is out
            greatest(delist_date, px_known_at) AS ret_known_at
        FROM raw
    """)


def summary(con) -> pl.DataFrame:
    return con.execute("""
        SELECT (d.cusip IS NOT NULL) AS delisted_in_next_q,
               e.issuer_cik IS NOT NULL AS has_cik,
               coalesce(x.has_proxy OR x.has_14d9 OR x.has_13e3, false) AS cash_evidence,
               coalesce(x.has_stock_deal, false) AS stock_evidence,
               count(*) AS exits, count(DISTINCT e.cusip) AS cusips
        FROM ex e LEFT JOIN ex_delist d USING (cusip, quarter) LEFT JOIN ex_deal x USING (cusip, quarter)
        GROUP BY ALL ORDER BY ALL
    """).pl()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    with db.connect() as con:
        exits(con)
        classify(con, fetch=args.fetch)
        print(con.execute("""
            SELECT exit_class, exit_subclass, ret_source, count(*) AS exits,
                   round(median(delist_ret), 3) AS median_ret, round(avg(delist_ret), 3) AS mean_ret,
                   sum(terms_suspect::INT) AS terms_suspect
            FROM delisting_events GROUP BY ALL ORDER BY exits DESC
        """).pl())
