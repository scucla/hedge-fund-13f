"""Milestone 5: fund embeddings, clusters and crowding."""

import json
from datetime import date

import numpy as np
import polars as pl
import pytest
from scipy.stats import special_ortho_group

from src.features import embeddings as emb
from src.models import clustering as cl
from src.models import crowding as cr
from tests.conftest import table_exists


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


# --- embeddings, by hand -------------------------------------------------------------

def test_asof_is_quarter_end_plus_window():
    assert emb.q_end("2026Q1") == date(2026, 3, 31)
    assert emb.q_end("2025Q4") == date(2025, 12, 31)
    assert emb.asof("2026Q1") == date(2026, 5, 30)


def test_sqrt_rows_have_unit_norm_and_coverage_counts_off_vocab_weight():
    books = pl.DataFrame({"fund_id": ["a", "a", "b", "b"], "issuer_key": ["X", "Y", "X", "Z"],
                          "w": [0.36, 0.64, 0.75, 0.25]})
    m, cov = emb.sqrt_matrix(books, ["a", "b"], {"X": 0, "Y": 1})
    d = m.toarray()
    assert d[0] == pytest.approx([0.6, 0.8])
    assert np.linalg.norm(d[0]) == pytest.approx(1.0)
    assert cov == pytest.approx([1.0, 0.75])     # Z is off-vocabulary


def test_procrustes_recovers_a_rotation_from_the_anchor_stocks():
    rng = np.random.default_rng(1)
    prev = {f"s{i}": rng.normal(size=4) for i in range(30)}
    rot = special_ortho_group.rvs(4, random_state=2)
    cur = {k: v @ rot.T for k, v in prev.items()}       # same stocks, basis rotated
    cur["new"] = rng.normal(size=4)                     # a stock only in the current quarter
    r = emb.procrustes_align(cur, prev)
    for k in prev:
        assert cur[k] @ r == pytest.approx(prev[k], abs=1e-8)


def test_nmf_themes_are_matched_by_permutation():
    rng = np.random.default_rng(3)
    prev = {f"s{i}": rng.random(3) for i in range(20)}
    perm = [2, 0, 1]
    cur = {k: v[perm] for k, v in prev.items()}         # theme j of cur = theme perm[j] of prev
    got = emb.match_themes(cur, prev, 3)
    for k in prev:
        assert cur[k][got] == pytest.approx(prev[k])


def test_fund2vec_corpus_is_point_in_time():
    """Only seed books from the last W2V_WINDOW_Q quarters filed by asof(q)."""
    def book(f, q, known, cohort="seed", keys=("A", "B")):
        return [{"fund_id": f, "cohort": cohort, "quarter": q, "known_at": known, "issuer_key": k,
                 "w": 1 / len(keys)} for k in keys]
    rows = (book("f1", "2026Q1", date(2026, 5, 10)) + book("f2", "2026Q1", date(2026, 5, 12))
            + book("f3", "2026Q1", date(2026, 7, 1), keys=("LATE", "A"))        # filed after asof
            + book("f4", "2026Q2", date(2026, 8, 10), keys=("FUTURE", "A"))    # next quarter
            + book("f5", "2026Q1", date(2026, 5, 10), "holdout", ("HOLD", "A"))
            + book("f6", "2026Q1", date(2026, 5, 10), keys=("LATE", "FUTURE", "HOLD", "B")))
    sentences, vocab = emb.w2v_corpus(pl.DataFrame(rows), "2026Q1", np.random.default_rng(0))
    assert vocab == {"A", "B"}
    assert {t for s in sentences for t in s} <= {"A", "B"}
    assert len(sentences) == 3 * emb.W2V_SENTENCES     # f1, f2, f6


# --- clustering, by hand -------------------------------------------------------------

def test_noise_becomes_singletons():
    assert cl.singletons(np.array([0, -1, 0, -1])).tolist() == [0, 1, 0, 2]


def test_stability_counts_noise_as_singletons():
    prev = {"a": 0, "b": 0, "c": -1, "d": -1}
    assert cl.stability(prev, dict(prev)) == pytest.approx(1.0)
    # c and d were noise, now clustered together with nothing else: not identical
    assert cl.stability(prev, {"a": 0, "b": 0, "c": 1, "d": 1}) < 1.0
    assert np.isnan(cl.stability(None, prev))


def test_purity_skips_noise():
    clusters = np.array([0, 0, 0, 1, 1, -1])
    labels = np.array(["x", "x", "y", "y", "y", "z"])
    assert cl.purity(clusters, labels) == pytest.approx(4 / 5)


def test_selection_rule_ranks_and_breaks_ties_on_noise():
    s = pl.DataFrame({"method": ["a", "b", "c"], "algo": ["hdbscan"] * 3,
                      "stability_ari": [0.9, 0.8, 0.1], "label_ari": [0.1, 0.2, 0.0],
                      "econ_gap": [0.5, 0.5, 0.1], "noise_share": [0.5, 0.2, 0.0]})
    # a: 1 + 2 + 1 = 4, b: 2 + 1 + 1 = 4 -> tie, b has less noise
    assert cl.choose(s)["method"].to_list() == ["b", "a", "c"]


# --- crowding, by hand ---------------------------------------------------------------

def test_crowding_needs_conviction_holders_inside_one_cluster():
    clusters = {"f1": 0, "f2": 0, "f3": 0, "f4": 0, "f5": 1, "f6": -1}
    rows = [("f1", "A", 0.05), ("f2", "A", 0.05), ("f3", "A", 0.05),      # 3 of 4 in cluster 0
            ("f1", "B", 0.05), ("f5", "B", 0.05), ("f6", "B", 0.05),      # 3 holders, spread out
            ("f1", "C", 0.001), ("f2", "C", 0.001), ("f3", "C", 0.001),   # below the weight floor
            ("f4", "C", 0.001)]
    book = pl.DataFrame(rows, schema=["fund_id", "issuer_key", "w"], orient="row")
    out = {r["issuer_key"]: r for r in cr.crowding_measures(book, clusters).iter_rows(named=True)}
    assert out["A"]["crowded"] and out["A"]["max_cluster_n"] == 3 and out["A"]["cluster_pairs"] == 3
    assert out["A"]["bucket"] == cr.CROWDED and out["A"]["crowded_in"] == [0]
    assert not out["B"]["crowded"] and out["B"]["bucket"] == cr.WIDE and out["B"]["n_holders"] == 3
    assert out["C"]["n_holders"] == 0 and not out["C"]["crowded"] and out["C"]["bucket"] == cr.FEW


def test_episode_return_compounds_and_stops_at_delisting():
    assert cr.compound([0.1, -0.5], ["yahoo", "yahoo"]) == pytest.approx(1.1 * 0.5 - 1)
    assert cr.compound([0.2, None], ["delist_cash", None]) == pytest.approx(0.2)
    assert cr.compound([0.2, None], ["yahoo", None]) is None


# --- the real tables -----------------------------------------------------------------

@pytest.fixture(scope="module")
def m5(sec_db):
    if not table_exists(sec_db, "fund_embeddings"):
        pytest.skip("run src.features.embeddings")
    return sec_db


def test_embedding_keys_unique_and_vectors_unit_norm(m5):
    assert scalar(m5, """SELECT count(*) - count(DISTINCT (method, fund_id, quarter)) FROM fund_embeddings""") == 0
    assert scalar(m5, "SELECT count(*) FROM fund_embeddings WHERE vec IS NULL OR known_at IS NULL") == 0
    assert scalar(m5, "SELECT count(*) FROM fund_embeddings WHERE abs(list_dot_product(vec, vec) - 1) > 1e-6") == 0


def test_embedding_known_at_is_the_funds_filing_or_the_fit(m5):
    """known_at is the fund's own filing or, if later, the fitted model's inputs,
    which never go past asof(quarter)."""
    rows = m5.execute("""
        SELECT e.quarter, e.known_at, f.known_at AS own
        FROM fund_embeddings e
        JOIN (SELECT fund_id, quarter, max(known_at) AS known_at FROM positions GROUP BY ALL) f USING (fund_id, quarter)
    """).fetchall()
    assert rows
    for q, known, own in rows:
        assert known >= own
        assert known <= max(own, emb.asof(q))


def test_background_is_projected_and_never_clustered(m5):
    assert scalar(m5, "SELECT min(quarter) FROM fund_embeddings WHERE cohort = 'background'") >= emb.BG_FIRST_QUARTER
    assert scalar(m5, """SELECT max(n) FROM (SELECT count(DISTINCT fund_id) AS n FROM fund_embeddings
                         WHERE cohort = 'background' GROUP BY method, quarter)""") <= emb.BACKGROUND_N
    assert scalar(m5, """SELECT count(*) FROM fund_clusters c JOIN fund_embeddings e USING (fund_id, quarter, method)
                         WHERE e.cohort <> 'seed'""") == 0
    assert scalar(m5, "SELECT count(*) FROM fund_clusters WHERE fund_id LIKE 'bg_%'") == 0


def test_seed_books_sum_to_one(m5):
    books = emb.load_books(m5)
    s = books.group_by("fund_id", "quarter").agg(pl.col("w").sum())["w"]
    assert s.min() == pytest.approx(1.0) and s.max() == pytest.approx(1.0)
    assert books["issuer_key"].null_count() == 0


def test_cluster_tables(m5):
    assert scalar(m5, "SELECT count(*) - count(DISTINCT (method, algo, fund_id, quarter)) FROM fund_clusters") == 0
    assert scalar(m5, "SELECT count(*) - count(DISTINCT (method, algo, quarter)) FROM cluster_metrics") == 0
    assert scalar(m5, "SELECT count(*) FROM fund_clusters WHERE known_at IS NULL") == 0
    assert scalar(m5, "SELECT count(*) FROM fund_clusters WHERE algo = 'kmeans' AND cluster < 0") == 0
    assert scalar(m5, "SELECT max(noise_share) FROM cluster_metrics WHERE algo = 'kmeans'") == 0


def test_choice_was_made_on_pre_2023_quarters():
    choice = json.loads(cl.CHOICE_PATH.read_text())
    assert choice["method"] in emb.METHODS
    assert choice["quarters"][1] <= "2022Q4" and choice["econ_last"] <= "2021Q4"
    assert max(cl.SELECT_QUARTERS) <= "2022Q4"


def test_crowding_tables(m5):
    if not table_exists(m5, "crowding_stocks"):
        pytest.skip("run src.models.crowding")
    assert scalar(m5, "SELECT count(*) - count(DISTINCT (episode, issuer_key)) FROM crowding_stocks") == 0
    assert scalar(m5, "SELECT count(*) FROM crowding_stocks WHERE cusip IS NULL OR ret IS NULL") == 0
    assert scalar(m5, "SELECT count(DISTINCT episode) FROM crowding_stocks") == len(cr.EPISODES)
    assert scalar(m5, "SELECT count(*) FROM cluster_overlap WHERE within < 0 OR within > 1 OR across > within") == 0


def test_background_filers_pass_the_etf_cap(m5):
    from src.securities.db import FILINGS_DB_PATH

    m5.execute(f"ATTACH IF NOT EXISTS '{FILINGS_DB_PATH}' AS filings_db (READ_ONLY)")
    bg = emb.background_books(m5)
    assert bg.group_by("quarter").agg(pl.col("fund_id").n_unique())["fund_id"].max() <= emb.BACKGROUND_N
    # a filer can pass the screen in one quarter and fail it in another: match on both
    shares = m5.execute("""SELECT max(w.etf_share) FROM bg_w w
                           JOIN (SELECT DISTINCT fund_id, quarter FROM fund_embeddings WHERE cohort = 'background') e
                             ON e.fund_id = 'bg_' || w.cik::VARCHAR AND e.quarter = w.quarter""").fetchone()[0]
    assert shares <= emb.BG_MAX_ETF_SHARE


def test_crowding_floor_raises_the_bar():
    """A higher conviction floor can only remove holders, never add crowded names."""
    clusters = {"f1": 0, "f2": 0, "f3": 0}
    book = pl.DataFrame({"fund_id": ["f1", "f2", "f3"], "issuer_key": ["A"] * 3, "w": [0.006, 0.015, 0.03]})
    n = [cr.crowding_measures(book, clusters, f)["n_holders"][0] for f in [0.005, 0.01, 0.02]]
    assert n == [3, 2, 1]
    assert cr.crowding_measures(book, clusters, 0.005)["crowded"][0]
    assert not cr.crowding_measures(book, clusters, 0.01)["crowded"][0]


def test_crowding_sensitivity_table(m5):
    if not table_exists(m5, "crowding_sensitivity"):
        pytest.skip("run src.models.crowding")
    df = m5.execute("SELECT episode, floor, n_crowded FROM crowding_sensitivity ORDER BY episode, floor").pl()
    assert df.height == len(cr.EPISODES) * len(cr.SENSITIVITY_FLOORS)
    for _, g in df.group_by("episode"):
        assert g.sort("floor")["n_crowded"].to_list() == sorted(g["n_crowded"], reverse=True)
