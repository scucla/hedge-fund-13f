"""Checks on the committed seed list (data/reference/seed_funds.csv)."""

import polars as pl
import pytest

from src.ingest.seed_ciks import MAP_PATH, SEED_PATH

STYLES = {
    "activist", "growth_tech", "value_concentrated", "discretionary_other",
    "event_driven", "quant", "multi_manager", "healthcare",
}
ROLES = {"primary", "predecessor", "successor", "affiliate", "uncertain"}


@pytest.fixture(scope="module")
def seed():
    return pl.read_csv(SEED_PATH, schema_overrides={"cik": pl.Utf8})


def test_unique_fund_cik(seed):
    assert seed.select("fund_name", "cik").is_duplicated().sum() == 0


def test_no_missing_ciks(seed):
    assert seed["cik"].null_count() == 0
    assert seed["cik"].str.contains(r"^\d{10}$").all()


def test_each_fund_has_exactly_one_primary(seed):
    counts = seed.group_by("fund_name").agg((pl.col("role") == "primary").sum().alias("n"))
    assert counts.filter(pl.col("n") != 1).is_empty()


def test_known_styles_and_roles(seed):
    assert set(seed["style_guess"]) <= STYLES
    assert set(seed["role"]) <= ROLES


def test_one_style_per_fund(seed):
    assert seed.group_by("fund_name").agg(pl.col("style_guess").n_unique()).filter(pl.col("style_guess") > 1).is_empty()


def test_every_cik_filed_holdings(seed):
    assert (seed["n_13f_hr"] > 0).all()


def test_flag_matches_overlap_count(seed):
    assert (seed["multi_cik_same_quarter"] == (seed["n_overlap_quarters"] > 0)).all()


def test_map_and_seed_agree():
    cik_map = pl.read_csv(MAP_PATH, schema_overrides={"cik": pl.Utf8})
    seed = pl.read_csv(SEED_PATH, schema_overrides={"cik": pl.Utf8})
    assert len(cik_map) == len(seed)
    assert seed["fund_name"].n_unique() == 77
