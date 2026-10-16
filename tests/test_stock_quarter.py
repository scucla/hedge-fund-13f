"""stock_quarter: synthetic cases with known answers, plus checks on the real table."""

from datetime import date, timedelta

import duckdb
import polars as pl
import pytest

from src.securities.stock_quarter import build
from tests.conftest import table_exists

Q = "2026Q1"
Q_END = date(2026, 3, 31)


def business_days(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def make_tables(ff_until: date | None = None):
    days = business_days(date(2024, 12, 1), date(2026, 6, 30))
    # Market: alternating +1% / -0.5% days; RF zero to keep beta exact.
    mkt = [0.01 if i % 2 else -0.005 for i in range(len(days))]
    ff = pl.DataFrame({"date": days, "Mkt_RF": mkt, "RF": [0.0] * len(days)})
    if ff_until:  # French's file ends before the quarter does
        ff = ff.filter(pl.col("date") <= ff_until)

    def path(rets, start=10.0):
        out, p = [], start
        for r in rets:
            p *= 1 + r
            out.append(p)
        return out

    prices = pl.concat([
        # AAA moves one-for-one with the market -> beta 1. Its quarter-end price is set below.
        pl.DataFrame({"symbol": "AAA", "date": days, "close": path(mkt)}),
        pl.DataFrame({"symbol": "SPY", "date": days, "close": path(mkt, 100.0)}),  # market proxy, not in the universe
        pl.DataFrame({"symbol": "NEG", "date": days, "close": [20.0] * len(days)}),
        pl.DataFrame({"symbol": "NOBE", "date": days, "close": [5.0] * len(days)}),
        pl.DataFrame({"symbol": "ETF", "date": days, "close": path([m / 2 for m in mkt])}),
        pl.DataFrame({"symbol": "ADR", "date": days, "close": [30.0] * len(days)}),
    ]).with_columns(pl.lit(0.0).alias("volume"))
    aaa_q_end = prices.filter((pl.col("symbol") == "AAA") & (pl.col("date") <= Q_END))["close"][-1]

    master = pl.DataFrame({
        "cusip": ["A00000001", "N00000002", "B00000003", "E00000004", "D00000005"],
        "symbol": ["AAA", "NEG", "NOBE", "ETF", "ADR"],
        "security_class": ["common", "common", "common", "etf", "adr"],
        "issuer_cik": ["1", "2", "3", None, "5"],
    })
    positions = master.select("cusip").with_columns(pl.lit(Q).alias("quarter"))

    def fact(cik, item, end, val, filed, start=None, priority=0):
        return {"issuer_cik": cik, "item": item, "priority": priority, "start": start, "end": end,
                "val": val, "known_at": filed}

    facts = pl.DataFrame([
        # AAA: book equity 500 at FY2025, filed Feb 2026 -> used.
        fact("1", "equity", date(2025, 12, 31), 500.0, date(2026, 2, 1)),
        # ...restated to 600 in June 2026 and a Q1 balance sheet filed in May: both after Q_END, ignored.
        fact("1", "equity", date(2025, 12, 31), 600.0, date(2026, 6, 1)),
        fact("1", "equity", date(2026, 3, 31), 999.0, date(2026, 5, 1)),
        fact("1", "shares_out", date(2026, 1, 20), 100.0, date(2026, 2, 1)),
        fact("1", "assets", date(2025, 12, 31), 200.0, date(2026, 2, 1)),
        fact("1", "assets", date(2024, 12, 31), 160.0, date(2025, 2, 1)),
        # No GrossProfit tag: falls back to revenue - cost of revenue = 50.
        fact("1", "revenue", date(2025, 12, 31), 80.0, date(2026, 2, 1), start=date(2025, 1, 1)),
        fact("1", "cost_of_revenue", date(2025, 12, 31), 30.0, date(2026, 2, 1), start=date(2025, 1, 1)),
        # NEG: negative book equity.
        fact("2", "equity", date(2025, 12, 31), -40.0, date(2026, 2, 1)),
        fact("2", "shares_out", date(2026, 1, 20), 10.0, date(2026, 2, 1)),
        # ...and a period ending after Q_END that was filed before it: ignored too.
        fact("1", "equity", date(2026, 6, 30), 777.0, date(2026, 3, 15)),
        # NOBE: shares but no book equity at all. A later share count of 0 is a tagging error: ignored.
        fact("3", "shares_out", date(2026, 1, 20), 10.0, date(2026, 2, 1)),
        fact("3", "shares_out", date(2026, 2, 15), 0.0, date(2026, 2, 20)),
        # ADR: has a share count, but it counts ordinary shares -> no market cap.
        fact("5", "shares_out", date(2026, 1, 20), 1000.0, date(2026, 2, 1)),
        fact("5", "equity", date(2025, 12, 31), 900.0, date(2026, 2, 1)),
    ])
    profile = pl.DataFrame({"issuer_cik": ["1", "2", "3", "5"], "sic": [3571, 4911, None, 9999]},
                           schema={"issuer_cik": pl.Utf8, "sic": pl.Int32})
    ff12 = pl.DataFrame({"ff12": [6, 8], "ff12_code": ["BusEq", "Utils"], "ff12_name": ["x", "y"],
                         "sic_from": [3570, 4900], "sic_to": [3579, 4949]})
    returns = pl.DataFrame({"symbol": ["AAA", "AAA"], "quarter": ["2026Q1", "2026Q2"],
                            "ret_q": [0.05, 0.10], "known_at": [Q_END, date(2026, 6, 30)]})

    tables = {"seed_positions": positions, "seed_positions_pit": positions, "security_master": master, "prices_daily": prices,
              "ff_factors_daily": ff, "fundamentals_facts": facts, "issuer_profile": profile,
              "ff12_sic_ranges": ff12, "returns_quarterly": returns}
    return tables, aaa_q_end


def build_rows(tables):
    con = duckdb.connect()
    for name, df in tables.items():
        con.register(f"{name}_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_df")
    build(con)
    return {r["symbol"]: r for r in con.execute("SELECT * FROM stock_quarter").pl().iter_rows(named=True)}


@pytest.fixture(scope="module")
def synth():
    tables, aaa_q_end = make_tables()
    return build_rows(tables), aaa_q_end


@pytest.fixture(scope="module")
def synth_proxy():
    tables, _ = make_tables(ff_until=date(2026, 2, 27))
    return build_rows(tables)


def test_every_security_kept(synth):
    rows, _ = synth
    assert set(rows) == {"AAA", "NEG", "NOBE", "ETF", "ADR"}


def test_negative_book_equity_gives_null_bm(synth):
    rows, _ = synth
    assert rows["NEG"]["be"] == -40.0
    assert rows["NEG"]["bm"] is None
    assert rows["NEG"]["me"] == pytest.approx(200.0)  # size is still computed


def test_missing_book_equity_gives_null_bm(synth):
    rows, _ = synth
    assert rows["NOBE"]["be"] is None and rows["NOBE"]["bm"] is None
    assert rows["NOBE"]["me"] == pytest.approx(50.0)


def test_book_equity_is_point_in_time(synth):
    rows, aaa_price = synth
    assert rows["AAA"]["be"] == 500.0  # not the June restatement (600) or May filing (999)
    assert rows["AAA"]["me"] == pytest.approx(100 * aaa_price)
    assert rows["AAA"]["bm"] == pytest.approx(500 / (100 * aaa_price))
    assert rows["AAA"]["fundamentals_known_at"] <= Q_END


def test_gross_profitability_and_asset_growth(synth):
    rows, _ = synth
    assert rows["AAA"]["gp_at"] == pytest.approx(50 / 200)
    assert rows["AAA"]["asset_growth"] == pytest.approx(200 / 160 - 1)


def test_beta_and_vol(synth):
    rows, _ = synth
    assert rows["AAA"]["beta"] == pytest.approx(1.0)
    assert rows["ETF"]["beta"] == pytest.approx(0.5)
    assert rows["NEG"]["beta"] == pytest.approx(0.0)
    assert rows["NEG"]["vol"] == pytest.approx(0.0)


def test_beta_uses_ff_when_available(synth):
    rows, _ = synth
    assert rows["AAA"]["beta_market"] == "ff"
    assert rows["AAA"]["n_proxy_days"] == 0


def test_beta_falls_back_to_spy_after_ff_ends(synth_proxy):
    aaa = synth_proxy["AAA"]
    assert aaa["beta_market"] == "ff+spy"
    assert aaa["n_proxy_days"] == len(business_days(date(2026, 3, 2), Q_END))
    assert aaa["beta"] == pytest.approx(1.0)  # AAA moves one-for-one with the market and with SPY
    assert aaa["n_days"] >= 250             # no days dropped for lack of factor data


def test_momentum_skips_last_month(synth):
    rows, _ = synth
    assert rows["NEG"]["mom_12_1"] == pytest.approx(0.0)
    assert rows["AAA"]["mom_12_1"] is not None


def test_no_market_cap_for_adr_or_etf(synth):
    rows, _ = synth
    assert rows["ADR"]["me"] is None and rows["ADR"]["bm"] is None
    assert rows["ADR"]["be"] == 900.0
    assert rows["ETF"]["me"] is None and rows["ETF"]["ff12"] is None


def test_sector_mapping(synth):
    rows, _ = synth
    assert (rows["AAA"]["ff12"], rows["AAA"]["ff12_code"]) == (6, "BusEq")
    assert (rows["NEG"]["ff12"], rows["NEG"]["ff12_code"]) == (8, "Utils")
    assert rows["NOBE"]["ff12"] is None              # no SIC
    assert (rows["ADR"]["ff12"], rows["ADR"]["ff12_code"]) == (12, "Other")  # SIC outside all ranges


def test_label_is_next_quarter(synth):
    rows, _ = synth
    assert rows["AAA"]["ret_q"] == 0.05
    assert rows["AAA"]["ret_next_q"] == 0.10
    assert rows["AAA"]["label_known_at"] == date(2026, 6, 30)
    assert rows["AAA"]["known_at"] == Q_END


# --- real table ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def sq(sec_db):
    if not table_exists(sec_db, "stock_quarter"):
        pytest.skip("stock_quarter not built")
    return sec_db


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


@pytest.mark.data
def test_real_unique_key(sq):
    assert scalar(sq, "SELECT count(*) - count(DISTINCT (cusip, quarter)) FROM stock_quarter") == 0


@pytest.mark.data
def test_real_nothing_from_after_quarter_end(sq):
    assert scalar(sq, """
        SELECT count(*) FROM stock_quarter
        WHERE price_date > known_at OR fundamentals_known_at > known_at
           OR be_end > known_at OR fy_end > known_at OR label_known_at <= known_at""") == 0


@pytest.mark.data
def test_real_bm_null_unless_positive_book_equity(sq):
    assert scalar(sq, "SELECT count(*) FROM stock_quarter WHERE bm IS NOT NULL AND NOT (be > 0)") == 0
    # and those stocks are still in the table
    assert scalar(sq, "SELECT count(*) FROM stock_quarter WHERE be <= 0") > 0


@pytest.mark.data
def test_real_ranges(sq):
    assert scalar(sq, """
        SELECT count(*) FROM stock_quarter
        WHERE me <= 0 OR bm <= 0 OR vol < 0 OR price <= 0
           OR mom_12_1 <= -1 OR ret_next_q <= -1 OR ff12 NOT BETWEEN 1 AND 12""") == 0


@pytest.mark.data
@pytest.mark.parametrize("symbol, lo, hi", [("AAPL", 2e12, 6e12), ("MSFT", 2e12, 6e12), ("KO", 1.5e11, 4e11)])
def test_real_market_caps_plausible(sq, symbol, lo, hi):
    me = scalar(sq, f"SELECT max(me) FROM stock_quarter WHERE symbol = '{symbol}' AND quarter = '2026Q2'")
    assert me is not None and lo <= me <= hi


@pytest.mark.data
def test_real_linked_cik_supplies_history(sq):
    # XOM's current registrant (a 2026 holding company) has no filings before
    # 2026-08-03; its history comes from the linked old CIK.
    row = sq.execute("""SELECT max(me), max(be), max(asset_growth) FROM stock_quarter
                        WHERE symbol = 'XOM' AND quarter = '2026Q2'""").fetchone()
    assert all(v is not None for v in row)
    assert 3e11 <= row[0] <= 1e12


@pytest.mark.data
def test_real_spy_beta_near_one(sq):
    beta = scalar(sq, "SELECT max(beta) FROM stock_quarter WHERE symbol = 'SPY' AND quarter = '2026Q2'")
    assert 0.9 <= beta <= 1.1
