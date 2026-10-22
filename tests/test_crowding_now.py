"""Crowded now (src/analysis/crowding_now.py): ranking, crowding flag, ETF exclusion, cluster names."""

import polars as pl
import pytest

from src.analysis import crowding_now as cn
from src.models import style_classifier


@pytest.fixture
def labels(monkeypatch):
    lab = pl.DataFrame({"fund_id": ["a1", "a2", "a3", "a4", "g1", "g2", "e1"],
                        "label_v2": ["growth_tech"] * 4 + ["growth_tech"] * 2 + ["event_driven"]})
    monkeypatch.setattr(style_classifier, "labels", lambda: lab)


def _book(rows):
    return pl.DataFrame(rows, schema=["fund_id", "issuer_key", "w"], orient="row")


def test_ranking_crowding_and_etf_exclusion(labels):
    clusters = {"a1": 0, "a2": 0, "a3": 0, "a4": 0, "g1": 1, "g2": 1, "e1": -1}
    book = _book([("a1", "X", 0.05), ("a2", "X", 0.05), ("a3", "X", 0.02), ("e1", "X", 0.30),   # 4 holders, 3 in cluster 0
                  ("a1", "Y", 0.10), ("g1", "Y", 0.10), ("g2", "Y", 0.10), ("e1", "Y", 0.10), ("a2", "Y", 0.005),
                  ("a1", "SPY", 0.5), ("a2", "SPY", 0.5), ("a3", "SPY", 0.5), ("a4", "SPY", 0.5), ("g1", "SPY", 0.5)])
    top = cn.snapshot(book, clusters, {}, {"X": "XX", "Y": "YY"}, exclude={"SPY"})
    assert top["issuer_key"].to_list() == ["X", "Y"]          # SPY has the most holders but is left out
    x, y = top.row(0, named=True), top.row(1, named=True)
    assert x["n_holders"] == 4 and x["max_cluster_n"] == 3 and x["crowded"]    # 3 of cluster 0's 4 funds
    assert y["n_holders"] == 4 and not y["crowded"]            # a2's 0.5% is below the 1% floor; 2 per cluster
    assert x["top_holders"].startswith("e1 (no cluster, 30.0%)")


def test_cluster_names_majority_style_and_duplicates(labels):
    names = cn.cluster_names({"a1": 0, "a2": 0, "g1": 1, "g2": 1, "e1": 2, "zz": 3})
    assert names[0] == "growth_tech" and names[1] == "growth_tech #2" and names[2] == "event_driven"
    assert names[3] == "cluster 3" and names[-1] == "no cluster"


def test_short_holder_list():
    assert cn._short("; ".join(f"f{i} (c, 1.0%)" for i in range(10)), 8).endswith("; and 2 more")
