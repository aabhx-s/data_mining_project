"""Create IPL player-season and latest recent-form feature tables.

Run after cricket_etl.py and player_master_etl.py:

    python -m src.transformation.feature_builder --as-of-season 2026
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.transformation.cricket_etl import project_root_from_file


PLAYER_SEASON_COLUMNS = [
    "player_id", "player", "season", "season_end_year", "batting_matches", "runs",
    "batting_balls", "fours", "sixes", "strike_rate", "boundary_runs",
    "boundary_percentage", "bowling_matches", "bowling_balls", "runs_conceded",
    "wickets", "overs", "economy", "bowling_strike_rate",
]


def season_end_year(season: object) -> int:
    """Convert labels such as 2020/21 or 2026 into one comparable ending year."""

    values = re.findall(r"\d{2,4}", str(season))
    if not values:
        raise ValueError(f"Cannot find a year in season value {season!r}")
    if len(values) == 1:
        return int(values[0])
    first_year = int(values[0])
    final_part = values[-1]
    return int(final_part) if len(final_part) == 4 else (first_year // 100) * 100 + int(final_part)


def stable_player_name(values: pd.Series) -> str | None:
    """Return a repeatable display name for a player ID."""

    values = values.dropna().astype(str)
    if values.empty:
        return None
    counts = values.value_counts()
    return sorted(counts[counts.eq(counts.max())].index)[0]


def safe_ratio(numerator: pd.Series, denominator: pd.Series, multiplier: float = 1.0) -> pd.Series:
    """Calculate a ratio while preserving undefined values as missing, never infinity."""

    return numerator.div(denominator.replace(0, np.nan)).mul(multiplier)


def load_ipl_match_tables(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the canonical IPL batting and bowling outputs."""

    directory = project_root / "data" / "processed" / "ipl"
    batting = pd.read_csv(directory / "batting_match.csv", dtype={"player_id": "string"})
    bowling = pd.read_csv(directory / "bowling_match.csv", dtype={"player_id": "string"})
    for name, table in {"batting": batting, "bowling": bowling}.items():
        required = {"player_id", "player", "season", "match_id"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{name} table is missing {sorted(missing)}. Run cricket_etl.py first.")
        if table["player_id"].isna().any():
            raise ValueError(f"{name} table has missing player IDs.")
    return batting, bowling


def build_player_season_features(project_root: str | Path | None = None) -> pd.DataFrame:
    """Aggregate every IPL season; no fixed five-season filter is applied here."""

    root = Path(project_root) if project_root else project_root_from_file()
    batting, bowling = load_ipl_match_tables(root)

    batting_season = (
        batting.groupby(["player_id", "season"], as_index=False)
        .agg(
            player=("player", stable_player_name),
            batting_matches=("match_id", "nunique"),
            runs=("runs", "sum"),
            batting_balls=("balls", "sum"),
            fours=("fours", "sum"),
            sixes=("sixes", "sum"),
        )
    )

    bowling_season = (
        bowling.groupby(["player_id", "season"], as_index=False)
        .agg(
            bowling_player=("player", stable_player_name),
            bowling_matches=("match_id", "nunique"),
            bowling_balls=("balls", "sum"),
            runs_conceded=("runs_conceded", "sum"),
            wickets=("wickets", "sum"),
        )
    )

    player_season = batting_season.merge(
        bowling_season, on=["player_id", "season"], how="outer", validate="one_to_one"
    )
    player_season["player"] = player_season["player"].fillna(player_season.pop("bowling_player"))

    count_columns = [
        "batting_matches", "runs", "batting_balls", "fours", "sixes", "bowling_matches",
        "bowling_balls", "runs_conceded", "wickets",
    ]
    player_season[count_columns] = player_season[count_columns].fillna(0)
    player_season["season_end_year"] = player_season["season"].map(season_end_year)
    player_season["strike_rate"] = safe_ratio(
        player_season["runs"], player_season["batting_balls"], 100
    ).round(2)
    player_season["boundary_runs"] = player_season["fours"] * 4 + player_season["sixes"] * 6
    player_season["boundary_percentage"] = safe_ratio(
        player_season["boundary_runs"], player_season["runs"], 100
    ).round(2)
    player_season["overs"] = (player_season["bowling_balls"] / 6).round(1)
    player_season["economy"] = safe_ratio(
        player_season["runs_conceded"], player_season["bowling_balls"], 6
    ).round(2)
    player_season["bowling_strike_rate"] = safe_ratio(
        player_season["bowling_balls"], player_season["wickets"]
    ).round(2)

    player_season = (
        player_season[PLAYER_SEASON_COLUMNS]
        .sort_values(["season_end_year", "player_id"])
        .reset_index(drop=True)
    )
    output = root / "data" / "processed" / "ipl" / "player_season_features.csv"
    player_season.to_csv(output, index=False)
    return player_season


def build_recent_form_features(
    project_root: str | Path | None = None,
    as_of_season: int | None = None,
    recent_seasons: int = 3,
) -> pd.DataFrame:
    """Build an as-of snapshot from IPL player-season features without future leakage."""

    root = Path(project_root) if project_root else project_root_from_file()
    path = root / "data" / "processed" / "ipl" / "player_season_features.csv"
    features = pd.read_csv(path, dtype={"player_id": "string"})
    if features.empty:
        raise ValueError("player_season_features.csv is empty")

    as_of_year = as_of_season or int(features["season_end_year"].max())
    history = features.loc[features["season_end_year"].le(as_of_year)].copy()
    if history.empty:
        raise ValueError(f"No IPL player-season data is available through {as_of_year}")

    latest = (
        history.sort_values(["player_id", "season_end_year"])
        .groupby("player_id", as_index=False)
        .tail(1)
        .set_index("player_id")
    )
    recent_start = as_of_year - recent_seasons + 1
    recent = history.loc[history["season_end_year"].between(recent_start, as_of_year)].copy()
    recent["season_weight"] = recent["season_end_year"] - recent_start + 1
    recent["season_weight"] = recent["season_weight"] / recent_seasons
    for stat in ("runs", "batting_balls", "wickets", "bowling_balls", "runs_conceded"):
        recent[f"weighted_{stat}"] = recent[stat] * recent["season_weight"]

    aggregate = (
        recent.groupby("player_id", as_index=False)
        .agg(
            player=("player", stable_player_name),
            recent_seasons_available=("season_end_year", "nunique"),
            recent_batting_matches=("batting_matches", "sum"),
            recent_runs=("runs", "sum"),
            recent_batting_balls=("batting_balls", "sum"),
            recent_fours=("fours", "sum"),
            recent_sixes=("sixes", "sum"),
            recent_bowling_matches=("bowling_matches", "sum"),
            recent_bowling_balls=("bowling_balls", "sum"),
            recent_runs_conceded=("runs_conceded", "sum"),
            recent_wickets=("wickets", "sum"),
            weighted_runs=("weighted_runs", "sum"),
            weighted_batting_balls=("weighted_batting_balls", "sum"),
            weighted_wickets=("weighted_wickets", "sum"),
            weighted_bowling_balls=("weighted_bowling_balls", "sum"),
            weighted_runs_conceded=("weighted_runs_conceded", "sum"),
        )
    )
    aggregate["recent_strike_rate"] = safe_ratio(
        aggregate["recent_runs"], aggregate["recent_batting_balls"], 100
    ).round(2)
    aggregate["recent_economy"] = safe_ratio(
        aggregate["recent_runs_conceded"], aggregate["recent_bowling_balls"], 6
    ).round(2)
    aggregate["recent_bowling_strike_rate"] = safe_ratio(
        aggregate["recent_bowling_balls"], aggregate["recent_wickets"]
    ).round(2)
    aggregate["weighted_strike_rate"] = safe_ratio(
        aggregate["weighted_runs"], aggregate["weighted_batting_balls"], 100
    ).round(2)
    aggregate["weighted_economy"] = safe_ratio(
        aggregate["weighted_runs_conceded"], aggregate["weighted_bowling_balls"], 6
    ).round(2)

    career = (
        history.groupby("player_id", as_index=False)
        .agg(
            first_season=("season_end_year", "min"),
            career_runs=("runs", "sum"),
            career_wickets=("wickets", "sum"),
            runs_standard_deviation=("runs", "std"),
            strike_rate_standard_deviation=("strike_rate", "std"),
            wickets_standard_deviation=("wickets", "std"),
            economy_standard_deviation=("economy", "std"),
        )
    )

    latest_columns = latest[
        ["season", "season_end_year", "runs", "strike_rate", "wickets", "economy"]
    ].rename(
        columns={
            "season": "latest_ipl_season",
            "season_end_year": "latest_ipl_season_end_year",
            "runs": "latest_ipl_runs",
            "strike_rate": "latest_ipl_strike_rate",
            "wickets": "latest_ipl_wickets",
            "economy": "latest_ipl_economy",
        }
    )
    snapshot = (
        aggregate.merge(career, on="player_id", how="left", validate="one_to_one")
        .merge(latest_columns, on="player_id", how="left", validate="one_to_one")
        .assign(as_of_season_end_year=as_of_year)
        .sort_values("player_id")
        .reset_index(drop=True)
    )
    output = root / "data" / "processed" / "ipl" / "recent_form_features.csv"
    snapshot.to_csv(output, index=False)
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--as-of-season", type=int, default=None)
    parser.add_argument("--recent-seasons", type=int, default=3)
    arguments = parser.parse_args()

    player_season = build_player_season_features(arguments.project_root)
    snapshot = build_recent_form_features(
        arguments.project_root,
        as_of_season=arguments.as_of_season,
        recent_seasons=arguments.recent_seasons,
    )
    print(f"Saved {len(player_season):,} IPL player-season rows.")
    print(f"Saved {len(snapshot):,} recent-form player snapshots.")


if __name__ == "__main__":
    main()
