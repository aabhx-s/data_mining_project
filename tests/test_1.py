"""Pytest checks for the processed cricket ETL outputs."""

from pathlib import Path

import pandas as pd
import pytest

from src.transformation.cricket_etl import DATASETS, project_root_from_file


PROJECT_ROOT = project_root_from_file()


@pytest.fixture(params=list(DATASETS.items()), ids=list(DATASETS))
def competition_tables(request):
    output_name, (raw_name, competition) = request.param
    processed = PROJECT_ROOT / "data" / "processed" / output_name
    return {
        "competition": competition,
        "raw_directory": PROJECT_ROOT / "data" / "raw" / raw_name,
        "matches": pd.read_csv(processed / "matches.csv", dtype={"match_id": "string"}),
        "batting": pd.read_csv(
            processed / "batting_match.csv", dtype={"match_id": "string", "player_id": "string"}
        ),
        "bowling": pd.read_csv(
            processed / "bowling_match.csv", dtype={"match_id": "string", "player_id": "string"}
        ),
    }


def test_raw_and_processed_match_ids_match(competition_tables):
    raw_ids = {path.stem for path in competition_tables["raw_directory"].glob("*.json")}
    processed_ids = set(competition_tables["matches"]["match_id"].dropna())
    assert raw_ids == processed_ids


def test_match_table_has_unique_ids(competition_tables):
    matches = competition_tables["matches"]
    assert not matches["match_id"].duplicated().any()
    assert set(matches["competition"].dropna()) == {competition_tables["competition"]}


@pytest.mark.parametrize("table_name", ["batting", "bowling"])
def test_player_match_keys_and_match_references(competition_tables, table_name):
    table = competition_tables[table_name]
    valid_match_ids = set(competition_tables["matches"]["match_id"])
    assert table["player_id"].notna().all()
    assert not table.duplicated(["match_id", "player_id"]).any()
    assert set(table["match_id"]).issubset(valid_match_ids)


def test_statistics_are_nonnegative(competition_tables):
    batting = competition_tables["batting"]
    bowling = competition_tables["bowling"]
    assert (batting[["runs", "balls", "fours", "sixes"]] >= 0).all().all()
    assert (bowling[["balls", "runs_conceded", "wickets"]] >= 0).all().all()


def test_undefined_rates_are_explained_by_zero_denominators(competition_tables):
    batting = competition_tables["batting"]
    bowling = competition_tables["bowling"]
    assert batting.loc[batting["strike_rate"].isna(), "balls"].eq(0).all()
    assert bowling.loc[bowling["economy"].isna(), "balls"].eq(0).all()
    assert bowling.loc[bowling["bowling_strike_rate"].isna(), "wickets"].eq(0).all()
