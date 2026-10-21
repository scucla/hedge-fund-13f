"""Event study (src/analysis/event_study.py): timing, abnormal returns, weighting, statistics."""

from datetime import date, datetime, timedelta

import duckdb
import numpy as np
import polars as pl
import pytest

from src.analysis import event_study as es

D = date.fromisoformat


def weekdays(start: str, n: int) -> list[date]:
    d, out = D(start), []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# --- day 0 ---------------------------------------------------------------------------------------

def test_day0_rules():
    days = np.array(weekdays("2026-08-10", 10), dtype="datetime64[D]")   # Mon 10 .. Fri 21
    i = lambda d: int(np.searchsorted(days, np.datetime64(d)))
    assert es.day0_index(datetime(2026, 8, 12, 15, 59), D("2026-08-12"), days) == (i("2026-08-12"), "intraday")
    assert es.day0_index(datetime(2026, 8, 12, 16, 0), D("2026-08-12"), days) == (i("2026-08-13"), "after_close")
    assert es.day0_index(datetime(2026, 8, 14, 21, 30), D("2026-08-17"), days) == (i("2026-08-17"), "after_close")  # Fri night
    assert es.day0_index(datetime(2026, 8, 15, 11, 0), D("2026-08-17"), days) == (i("2026-08-17"), "after_close")   # Saturday
    # no header: the filing date, day 0 = the next trading day
    assert es.day0_index(None, D("2026-08-12"), days) == (i("2026-08-13"), "fallback")


# --- statistics ----------------------------------------------------------------------------------

def test_clustered_mean_matches_hand_calculation():
    x = np.array([1.0, 3.0, 2.0, 6.0])
    g = np.array(["a", "a", "b", "b"])
    m, t = es.clustered_mean(x, g)
    assert m == 3.0
    sums = np.array([(1 - 3) + (3 - 3), (2 - 3) + (6 - 3)])          # cluster sums of deviations
    se = np.sqrt((sums ** 2).sum() * 2 / 1) / 4
    assert t == pytest.approx(3.0 / se)


def test_fund_weighting_gives_each_fund_one_vote():
    df = pl.DataFrame({"fund_id": ["big"] * 3 + ["small"], "car": [0.1, 0.1, 0.1, -0.1],
                       "day0": [D("2026-08-14"), D("2026-08-14"), D("2026-08-17"), D("2026-08-18")]})
    s = es.summarise(df)
    assert s["mean"] == pytest.approx(0.05)
    assert s["fund_wtd_mean"] == pytest.approx(0.0)
    assert s["fund_median"] == pytest.approx(0.0) and s["funds"] == 2


# --- abnormal returns on a synthetic price history -----------------------------------------------

@pytest.fixture
def market():
    days = weekdays("2026-01-05", 40)
    rows = []
    for k, d in enumerate(days):
        rows.append(("SPY", d, 100.0))                       # flat market: abnormal = raw return
        rows.append(("UP", d, 100.0 * 1.01 ** k))            # +1% a day, all period
        if k <= 12:
            rows.append(("GONE", d, 50.0))                   # last close day 12, then a deal at 100
            rows.append(("QUIET", d, 20.0))                  # last close day 12, no delisting record
    con = duckdb.connect()
    con.execute("CREATE TABLE prices_daily (symbol VARCHAR, date DATE, close DOUBLE)")
    con.executemany("INSERT INTO prices_daily VALUES (?, ?, ?)", rows)
    cal = pl.DataFrame({"i": range(len(days)), "date": days})
    yield con, cal, days
    con.close()


def test_windows_delisting_adjustment_and_exclusions(market):
    con, cal, days = market
    events = pl.DataFrame({"fund_id": ["f"] * 4, "quarter": ["2025Q4"] * 4, "cusip": ["u", "g", "q", "n"],
                           "symbol": ["UP", "GONE", "QUIET", None], "day0_i": [10] * 4})
    span = es.daily_abnormal(con, ["UP", "GONE", "QUIET"], cal)
    delist = pl.DataFrame({"cusip": ["g"], "delist_date": [days[12] + timedelta(days=3)], "exit_class": ["cash_acquisition"],
                           "kind": ["deal"], "deal_value": [100.0]})
    cars = es.window_cars(con, events, span, delist, cal)
    get = lambda c, w: cars.filter((pl.col("cusip") == c) & (pl.col("window") == w)).row(0, named=True)
    assert get("u", "[0,+1]")["car"] == pytest.approx(0.02)            # two days of +1%
    assert get("u", "[+2,+10]")["car"] == pytest.approx(0.09)
    assert get("u", "[+11,+60]")["status"] == "window not over yet"   # day 10+60 is past the data
    g = get("g", "[+2,+10]")                                            # series ends on day 12
    assert g["status"] == "ok, delisting-adjusted" and g["car"] == pytest.approx(1.0)   # 0 drift + 50 -> 100
    assert get("q", "[+2,+10]")["status"] == "price history ends, no delisting record"
    assert get("q", "[0,+1]")["car"] == pytest.approx(0.0)              # still priced over [0,+1]
    assert get("n", "[0,+1]")["status"] == "no daily prices"


def test_car_path_is_cumulative(market):
    con, cal, _ = market
    events = pl.DataFrame({"fund_id": ["f"], "quarter": ["2025Q4"], "cusip": ["u"], "symbol": ["UP"], "day0_i": [10]})
    span = es.daily_abnormal(con, ["UP"], cal)
    es.PATH_DAYS, saved = (-5, 20), es.PATH_DAYS
    try:
        path = es.car_path(con, events, span)
    finally:
        es.PATH_DAYS = saved
    assert path.filter(pl.col("t") == -5)["event_weighted"][0] == pytest.approx(0.01)
    assert path.filter(pl.col("t") == 20)["event_weighted"][0] == pytest.approx(0.26)


def test_winsorize_clips_within_each_window_and_keeps_raw():
    cars = pl.DataFrame({"window": ["a"] * 1000 + ["b"] * 1000,
                         "car": list(np.linspace(-1, 1, 999)) + [900.0] + list(np.linspace(0, 0.1, 1000))})
    w = es.winsorize(cars)
    a = w.filter(pl.col("window") == "a")
    assert a["car_raw"].max() == 900.0 and a["car"].max() < 2          # the glitch is clipped
    assert w.filter(pl.col("window") == "b")["car"].max() <= 0.1        # window b keeps its own bounds
