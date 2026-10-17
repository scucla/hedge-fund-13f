"""Milestone 6 (Task C): point-in-time panel, embargo, leakage audit, metrics."""

from datetime import date

import duckdb
import numpy as np
import polars as pl
import pytest

from src.features import task_c
from src.models import leakage_audit as la
from src.models.task_c2 import rank_metrics
from src.models.task_c_common import CLASSES, c1_metrics, embargo, frames
from tests.conftest import table_exists

D = date.fromisoformat


# --- synthetic build: a late peer filing and a not-yet-public clustering are ignored ---------

def _tables() -> dict[str, pl.DataFrame]:
    """Target fund F files 2022Q1 on 05-10. Peer A filed 2022Q1 earlier (05-01); peer B files
    2022Q1 later (05-15) and exits S there, so on F's date B is still seen through its 2021Q4
    book, which holds S. The 2022Q1 clustering is published 05-20 (after F) and puts F with A;
    the 2021Q4 one (02-20) puts F with B."""
    pos = []
    def p(f, q, cusip, known, is_new=False, sc=0.0):
        pos.append({"fund_id": f, "cohort": "seed", "quarter": q, "known_at": D(known), "put_call": "", "cusip": cusip,
                    "symbol": cusip, "is_new": is_new, "share_change": sc, "weight_long": 0.5, "shares": 100.0,
                    "value_usd": 1000.0, "split_ratio": 1.0, "age_q": 2, "shares_suspect": False, "rank": 1,
                    "top10": True, "age_censored": False})
    for f, k0, k1 in (("F", "2022-02-10", "2022-05-10"), ("A", "2022-02-01", "2022-05-01"), ("B", "2022-02-10", "2022-05-15")):
        p(f, "2021Q4", "S", k0)
        p(f, "2021Q4", "X", k0)
        if f != "B":
            p(f, "2022Q1", "S", k1)
        p(f, "2022Q1", "X", k1)
    positions = pl.DataFrame(pos)
    moves = pl.DataFrame([
        {"fund_id": "F", "cohort": "seed", "cusip": "S", "quarter": "2022Q1", "known_at": D("2022-08-10"),
         "action": "hold", "forced_exit": False, "share_change_next": 0.0},
        {"fund_id": "B", "cohort": "seed", "cusip": "S", "quarter": "2021Q4", "known_at": D("2022-05-15"),
         "action": "exit", "forced_exit": False, "share_change_next": None},
    ])
    clusters = pl.DataFrame([
        {"method": "fund2vec", "algo": "hdbscan", "fund_id": f, "quarter": q, "cluster": c, "known_at": D(k)}
        for f, q, c, k in (("F", "2021Q4", 1, "2022-02-20"), ("B", "2021Q4", 1, "2022-02-20"), ("A", "2021Q4", 2, "2022-02-20"),
                           ("F", "2022Q1", 3, "2022-05-20"), ("A", "2022Q1", 3, "2022-05-20"), ("B", "2022Q1", 4, "2022-05-20"))])
    fq = positions.select("fund_id", "quarter", "known_at").unique().with_columns(
        pl.lit(2).alias("n_positions"), pl.lit(1.0).alias("top10_weight"), pl.lit(0.5).alias("hhi"),
        pl.lit(0.1).alias("turnover_clipped"), pl.lit(0.1).alias("gross_turnover_clipped"),
        pl.lit(0.0).alias("options_share"), pl.lit(2000.0).alias("long_value"))
    sq = pl.DataFrame([{"cusip": c, "quarter": q, "known_at": D(e), "fundamentals_known_at": D(e), "log_me": 1.0,
                        "bm": 0.5, "gp_at": 0.2, "asset_growth": 0.1, "mom_12_1": 0.1, "beta": 1.0, "vol": 0.2,
                        "ret_q": 0.0, "ff12": 6, "ret_source": "yahoo"}
                       for c in ("S", "X") for q, e in (("2021Q4", "2021-12-31"), ("2022Q1", "2022-03-31"))])
    prices = pl.DataFrame({"symbol": ["S", "S"], "date": [D("2022-03-31"), D("2022-05-09")], "close": [10.0, 11.0]})
    holdings = positions.select("fund_id", "cohort", "cusip", "quarter", "put_call")
    empty = pl.DataFrame(schema={"fund_id": pl.Utf8, "cohort": pl.Utf8, "quarter": pl.Utf8, "known_at": pl.Date})
    return {"positions": positions, "position_moves": moves, "fund_clusters": clusters, "fund_quarter": fq,
            "stock_quarter": sq, "prices_daily": prices, "fund_holdings": holdings, "empty_books": empty}


@pytest.fixture(scope="module")
def synthetic_panel():
    con = duckdb.connect()
    for name, df in _tables().items():
        con.register(f"{name}_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_df")
    task_c.build(con)
    yield con
    con.close()


def test_late_peer_filing_is_not_used(synthetic_panel):
    r = synthetic_panel.execute("""SELECT n_peer_hold, n_peer_exit, share_peers_filed_q, peer_known_at
                                   FROM c1_panel WHERE fund_id = 'F' AND cusip = 'S'""").fetchone()
    # A (2022Q1 book) and B (2021Q4 book) hold S; B's 2022Q1 exit is published after F's date
    assert r[0] == 2 and r[1] == 0
    assert r[2] == 0.5 and r[3] == D("2022-05-01")


def test_cluster_published_after_prediction_is_not_used(synthetic_panel):
    r = synthetic_panel.execute("""SELECT cl_quarter, cluster_known_at, cl_size, cl_n_hold
                                   FROM c1_panel WHERE fund_id = 'F' AND cusip = 'S'""").fetchone()
    assert r[0] == "2021Q4" and r[1] == D("2022-02-20")
    assert r[2] == 1 and r[3] == 1   # B is the only cluster peer, and it holds S in its visible book


def test_return_since_quarter_end_uses_closes_up_to_prediction(synthetic_panel):
    r = synthetic_panel.execute("SELECT ret_since_qend FROM c1_panel WHERE fund_id = 'F' AND cusip = 'S'").fetchone()
    assert r[0] == pytest.approx(0.1)


# --- embargo and frames --------------------------------------------------------------------

def _rows(quarters, pred, label, cohort="seed"):
    return pl.DataFrame({"fund_id": ["f"] * len(quarters), "cohort": [cohort] * len(quarters), "quarter": quarters,
                         "pred_date": [D(x) for x in pred], "label_known_at": [D(x) for x in label]})


def test_embargo_drops_labels_public_after_first_scored_prediction():
    fit = _rows(["2021Q3", "2021Q4"], ["2021-11-14", "2022-02-14"], ["2022-02-14", "2022-05-16"])
    scored = _rows(["2022Q1"], ["2022-05-04"], ["2022-08-14"])
    kept, cutoff = embargo(fit, scored)
    assert cutoff == D("2022-05-04") and kept["quarter"].to_list() == ["2021Q3"]


def test_frames_embargo_both_boundaries_and_seed_only():
    df = pl.concat([
        _rows(["2021Q3", "2021Q4", "2022Q1", "2022Q4", "2023Q1"],
              ["2021-11-14", "2022-02-14", "2022-05-04", "2023-02-14", "2023-05-09"],
              ["2022-02-14", "2022-05-16", "2022-08-14", "2023-05-15", "2023-08-14"]),
        _rows(["2021Q3"], ["2021-11-14"], ["2022-02-14"], cohort="holdout")])
    fr = frames(df)
    assert fr["train"]["quarter"].to_list() == ["2021Q3"]
    assert sorted(fr["trainval"]["quarter"].to_list()) == ["2021Q3", "2021Q4", "2022Q1"]   # 2022Q4 label after 2023-05-09
    assert set(fr["train"]["cohort"]) == {"seed"}
    assert la.check_embargo(fr, {"val": fr["val"], "test": fr["test"]})[0]["violations"] == 0


# --- audit checks catch injected leaks -----------------------------------------------------

def _panel_row(**over):
    row = {c: 0.0 for g in task_c.FEATURE_GROUPS.values() for c in g}
    row |= {k: D("2022-05-01") for k in la.GROUP_KNOWN_AT.values()} | {"max_peer_known_at": D("2022-05-01")}
    row |= {"pred_date": D("2022-05-10"), "label_known_at": D("2022-08-10")}
    return row | over


def test_known_at_check_flags_late_and_undated_features():
    ok = la.check_known_at(pl.DataFrame([_panel_row()]))
    assert sum(r["violations"] for r in ok) == 0
    late = {r["check"]: r["violations"] for r in la.check_known_at(pl.DataFrame([_panel_row(stock_known_at=D("2022-06-01"))]))}
    assert late["known_at:stock"] == 1
    undated = {r["check"]: r["violations"] for r in la.check_known_at(pl.DataFrame([_panel_row(cluster_known_at=None)]))}
    assert undated["known_at:cluster"] == 1
    future_label = {r["check"]: r["violations"] for r in la.check_known_at(pl.DataFrame([_panel_row(label_known_at=D("2022-05-10"))]))}
    assert future_label["label_after_pred"] == 1


def test_peer_recompute_flags_a_count_that_used_a_late_filing():
    t = _tables()
    q_idx = pl.col("quarter").str.slice(0, 4).cast(pl.Int32) * 4 + pl.col("quarter").str.slice(5, 1).cast(pl.Int32)
    positions = t["positions"].select("fund_id", "quarter", "cusip").with_columns(q_idx.alias("q_idx"))
    filings = (t["positions"].group_by("fund_id", "cohort", "quarter").agg(pl.col("known_at").max())
               .with_columns(q_idx.alias("q_idx")))
    def panel(n):
        return pl.DataFrame({"fund_id": ["F"], "quarter": ["2022Q1"], "q_idx": [2022 * 4 + 1],
                             "pred_date": [D("2022-05-10")], "cusip": ["S"], "n_peer_hold": [n]})
    assert la.peer_recompute(positions, filings, panel(2)) == (1, 0)
    assert la.peer_recompute(positions, filings, panel(1)) == (1, 1)   # what a quarter-q count would say


def test_forbidden_feature_check():
    assert la.check_features({"m": ["weight_long", "ret_next_q", "crowding"]})[0]["violations"] == 2
    assert la.check_features({"m": ["weight_long"]})[0]["violations"] == 0


# --- metrics -------------------------------------------------------------------------------

def test_rank_metrics():
    m = rank_metrics(["a", "b", "c"], {"b", "z"})
    assert m["recall"] == 0.5 and m["hit"] == 1.0
    assert m["ndcg"] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))


def test_c1_metrics_perfect_and_uniform():
    y = np.array([0, 1, 2, 3])
    m = c1_metrics(y, np.eye(4))
    assert m["macro_f1"] == 1.0 and m["exit_auc"] == 1.0
    u = c1_metrics(y, np.full((4, 4), 0.25))
    assert u["log_loss"] == pytest.approx(np.log(4))


# --- real data -----------------------------------------------------------------------------

@pytest.mark.data
def test_c1_panel_keys_and_labels(sec_db):
    if not table_exists(sec_db, "c1_panel"):
        pytest.skip("c1_panel not built; run src.features.task_c")
    dup, null_cusip, bad_action, null_clean, hold = sec_db.execute(f"""
        SELECT count(*) - count(DISTINCT (fund_id, cusip, quarter)), count(*) FILTER (WHERE cusip IS NULL),
               count(*) FILTER (WHERE action NOT IN {tuple(CLASSES)}), count(*) FILTER (WHERE ret_clean IS NULL),
               avg((action = 'hold')::INT)
        FROM c1_panel""").fetchone()
    assert dup == 0 and null_cusip == 0 and bad_action == 0 and null_clean == 0
    assert 0.1 < hold < 0.4   # hold is the smallest class for these active books (docs/task_c_baselines.md)


@pytest.mark.data
def test_new_buys_keys_and_not_held_before(sec_db):
    if not table_exists(sec_db, "new_buys"):
        pytest.skip("new_buys not built; run src.features.task_c")
    dup, null_cusip, held_before = sec_db.execute("""
        SELECT count(*) - count(DISTINCT (fund_id, cusip, quarter)), count(*) FILTER (WHERE cusip IS NULL),
               (SELECT count(*) FROM new_buys n JOIN fund_holdings h
                  ON h.fund_id = n.fund_id AND h.cusip = n.cusip AND h.quarter = n.quarter AND h.put_call = '')
        FROM new_buys""").fetchone()
    assert dup == 0 and null_cusip == 0 and held_before == 0


@pytest.mark.data
def test_leakage_audit_passes(sec_db):
    if not table_exists(sec_db, "c1_panel"):
        pytest.skip("c1_panel not built; run src.features.task_c")
    res = la.audit(sec_db)
    failed = res.filter(pl.col("violations") > 0)
    assert failed.is_empty(), failed.to_dicts()
