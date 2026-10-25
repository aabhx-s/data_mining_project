"""Extract Cricsheet JSON archives into model-ready match, batting and bowling tables.

Run all configured competitions from the repository root with:

    python -m src.transformation.cricket_etl --all

The output tables deliberately retain Cricsheet's registry player ID. Names are
useful for display, but player_id is the stable key used for joins and modelling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


BOWLER_WICKET_TYPES = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
}

DATASETS = {
    "ipl": ("ipl_male_json", "IPL"),
    "odi": ("odis_male_json", "ODI"),
    "t20": ("t20s_male_json", "T20"),
    "bbl": ("bbl_male_json", "BBL"),
    "smat": ("sma_male_json", "SMAT"),
    "t20_wc": ("icc_mens_t20_world_cup_male_json", "T20_WC"),
}

# Cricsheet match 1130677 uses "KV Sharma" for two different opponents.
# The registry maps that abbreviation only to Karn Sharma (Vidarbha). The
# Railways batter is Karan Sharma, so give that otherwise-unidentifiable
# player a deterministic local ID instead of merging two people together.
# Keep source-specific corrections here, with match ID + team + source name
# as the key, so normal registry IDs remain unchanged everywhere else.
MATCH_PLAYER_OVERRIDES: dict[tuple[str, str, str], tuple[str, str]] = {
    ("1130677", "Railways", "KV Sharma"): (
        "source_1130677_railways_karan_sharma",
        "Karan Sharma",
    ),
    ("1130677", "Vidarbha", "KV Sharma"): ("119678fd", "Karn Sharma"),
}

MATCH_COLUMNS = [
    "match_id", "competition", "season", "date", "match_type", "team1", "team2",
    "venue", "city", "toss_winner", "toss_decision", "winner", "result",
    "winner_margin_runs", "winner_margin_wickets", "match_method", "player_of_match",
]

BATTING_COLUMNS = [
    "match_id", "competition", "season", "date", "team", "opposition", "player_id",
    "player", "runs", "balls", "fours", "sixes", "strike_rate",
]

BOWLING_COLUMNS = [
    "match_id", "competition", "season", "date", "bowling_team", "opposition",
    "player_id", "player", "balls", "runs_conceded", "wickets", "overs", "economy",
    "bowling_strike_rate",
]


def project_root_from_file() -> Path:
    """Return the repository root regardless of the current working directory."""

    return Path(__file__).resolve().parents[2]


def is_legal_ball(delivery: dict[str, Any]) -> bool:
    """Return whether a delivery counts as a legal ball for batting and bowling."""

    extras = delivery.get("extras", {})
    return "wides" not in extras and "noballs" not in extras


def calculate_bowler_runs(delivery: dict[str, Any]) -> int:
    """Return runs charged to the bowler, excluding byes, leg-byes and penalties."""

    extras = delivery.get("extras", {})
    total = delivery["runs"]["total"]
    return (
        total
        - extras.get("byes", 0)
        - extras.get("legbyes", 0)
        - extras.get("penalty", 0)
    )


def get_bowler_wickets(delivery: dict[str, Any]) -> int:
    """Count only dismissal types credited to the bowler."""

    return sum(
        wicket.get("kind") in BOWLER_WICKET_TYPES
        for wicket in delivery.get("wickets", [])
    )


def safe_rate(numerator: pd.Series, denominator: pd.Series, multiplier: float = 1.0) -> pd.Series:
    """Calculate a rate without attempting to round missing values.

    A batter can face only wides, and a bowler can deliver only illegal balls.
    Their rate is undefined, so the output intentionally contains a blank value.
    """

    result = pd.Series(index=numerator.index, dtype="float64")
    valid_denominator = denominator.gt(0)
    result.loc[valid_denominator] = (
        numerator.loc[valid_denominator]
        .div(denominator.loc[valid_denominator])
        .mul(multiplier)
        .round(2)
    )
    return result


def registry_people(match_info: dict[str, Any]) -> dict[str, str]:
    """Return the Cricsheet display-name -> canonical-ID registry for a match."""

    people = match_info.get("registry", {}).get("people", {})
    return {str(name): str(player_id) for name, player_id in people.items()}


def resolve_player(
    match_id: str,
    team: str | None,
    player_name: str,
    player_registry: dict[str, str],
) -> tuple[str | None, str]:
    """Return the most accurate ID and display name for a match participant."""

    override = MATCH_PLAYER_OVERRIDES.get((match_id, team or "", player_name))
    if override:
        return override
    return player_registry.get(player_name), player_name


def other_team(teams: list[str], batting_team: str) -> str | None:
    """Return the other team in a normal two-team match."""

    return next((team for team in teams if team != batting_team), None)


def extract_outcome(match_info: dict[str, Any]) -> dict[str, Any]:
    """Normalise optional Cricsheet outcome fields into stable match-table columns."""

    outcome = match_info.get("outcome", {})
    margin = outcome.get("by", {})
    players_of_match = match_info.get("player_of_match", [])

    if outcome.get("winner"):
        result = "win"
    elif outcome.get("result"):
        result = str(outcome["result"])
    elif outcome.get("eliminator"):
        result = "eliminator"
    else:
        result = None

    return {
        "winner": outcome.get("winner") or outcome.get("eliminator"),
        "result": result,
        "winner_margin_runs": margin.get("runs"),
        "winner_margin_wickets": margin.get("wickets"),
        "match_method": outcome.get("method"),
        "player_of_match": " | ".join(players_of_match) if players_of_match else None,
    }


def process_match(file_path: str | Path, competition: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process one Cricsheet JSON match into match, batting and bowling tables."""

    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as source_file:
        match = json.load(source_file)

    info = match["info"]
    teams = list(info.get("teams", []))
    if len(teams) != 2:
        raise ValueError(f"Expected two teams, found {teams!r}")

    player_registry = registry_people(info)
    deliveries: list[dict[str, Any]] = []

    for innings in match.get("innings", []):
        batting_team = innings["team"]
        bowling_team = other_team(teams, batting_team)

        for over in innings.get("overs", []):
            for delivery in over.get("deliveries", []):
                batter = delivery["batter"]
                bowler = delivery["bowler"]
                batter_id, batter_name = resolve_player(
                    file_path.stem, batting_team, batter, player_registry
                )
                bowler_id, bowler_name = resolve_player(
                    file_path.stem, bowling_team, bowler, player_registry
                )
                deliveries.append(
                    {
                        "match_id": file_path.stem,
                        "competition": competition,
                        "season": str(info.get("season", "")),
                        "date": str(info.get("dates", [None])[0]),
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "batter_id": batter_id,
                        "batter": batter_name,
                        "bowler_id": bowler_id,
                        "bowler": bowler_name,
                        "batter_runs": delivery["runs"]["batter"],
                        "is_legal_ball": is_legal_ball(delivery),
                        "bowler_runs": calculate_bowler_runs(delivery),
                        "bowler_wickets": get_bowler_wickets(delivery),
                        "four": int(delivery["runs"]["batter"] == 4),
                        "six": int(delivery["runs"]["batter"] == 6),
                    }
                )

    deliveries_df = pd.DataFrame(deliveries)
    if deliveries_df.empty:
        raise ValueError("Match contains no deliveries")

    batting = (
        deliveries_df.groupby(
            [
                "match_id", "competition", "season", "date", "batting_team", "bowling_team",
                "batter_id", "batter",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            runs=("batter_runs", "sum"),
            balls=("is_legal_ball", "sum"),
            fours=("four", "sum"),
            sixes=("six", "sum"),
        )
        .rename(
            columns={
                "batting_team": "team",
                "bowling_team": "opposition",
                "batter_id": "player_id",
                "batter": "player",
            }
        )
    )
    batting["strike_rate"] = safe_rate(batting["runs"], batting["balls"], 100)
    batting = batting[BATTING_COLUMNS]

    bowling = (
        deliveries_df.groupby(
            [
                "match_id", "competition", "season", "date", "bowling_team", "batting_team",
                "bowler_id", "bowler",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            balls=("is_legal_ball", "sum"),
            runs_conceded=("bowler_runs", "sum"),
            wickets=("bowler_wickets", "sum"),
        )
        .rename(
            columns={
                "batting_team": "opposition",
                "bowler_id": "player_id",
                "bowler": "player",
            }
        )
    )
    bowling["overs"] = (
        (bowling["balls"] // 6).astype(str) + "." + (bowling["balls"] % 6).astype(str)
    )
    bowling["economy"] = safe_rate(bowling["runs_conceded"], bowling["balls"], 6)
    bowling["bowling_strike_rate"] = safe_rate(bowling["balls"], bowling["wickets"])
    bowling = bowling[BOWLING_COLUMNS]

    toss = info.get("toss", {})
    match_row = {
        "match_id": file_path.stem,
        "competition": competition,
        "season": str(info.get("season", "")),
        "date": str(info.get("dates", [None])[0]),
        "match_type": info.get("match_type"),
        "team1": teams[0],
        "team2": teams[1],
        "venue": info.get("venue"),
        "city": info.get("city"),
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        **extract_outcome(info),
    }
    matches = pd.DataFrame([match_row], columns=MATCH_COLUMNS)

    return matches, batting, bowling


def validate_output(matches: pd.DataFrame, batting: pd.DataFrame, bowling: pd.DataFrame) -> None:
    """Fail early if an ETL result violates its essential keys."""

    if matches["match_id"].duplicated().any():
        raise ValueError("Duplicate match_id values found in matches output")

    for name, table in {"batting": batting, "bowling": bowling}.items():
        if table["player_id"].isna().any():
            missing_names = sorted(table.loc[table["player_id"].isna(), "player"].dropna().unique())
            sample = ", ".join(missing_names[:10])
            raise ValueError(f"{name} output has missing player IDs. Sample: {sample}")
        number_columns = [
            column for column in table.columns
            if column in {"runs", "balls", "fours", "sixes", "runs_conceded", "wickets"}
        ]
        if (table[number_columns] < 0).any().any():
            raise ValueError(f"{name} output has negative statistics")


def process_competition(
    input_directory: str | Path,
    output_directory: str | Path,
    competition: str,
) -> dict[str, int]:
    """Process a complete Cricsheet archive and write its three processed tables."""

    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    json_files = sorted(input_directory.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON match files found in {input_directory}")

    all_matches: list[pd.DataFrame] = []
    all_batting: list[pd.DataFrame] = []
    all_bowling: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    for index, file_path in enumerate(json_files, start=1):
        try:
            matches, batting, bowling = process_match(file_path, competition)
            all_matches.append(matches)
            all_batting.append(batting)
            all_bowling.append(bowling)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append({"file": file_path.name, "error": str(error)})

        if index % 250 == 0 or index == len(json_files):
            print(f"{competition}: processed {index:,}/{len(json_files):,} files")

    error_path = output_directory / "etl_errors.csv"
    if errors:
        output_directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(errors).to_csv(error_path, index=False)
        raise RuntimeError(f"{competition}: {len(errors)} files failed. See {error_path}")

    # A previous unsuccessful run may have left a log behind. It no longer
    # describes the current successful output, so remove just that generated log.
    if error_path.exists():
        error_path.unlink()

    matches_df = pd.concat(all_matches, ignore_index=True).sort_values("match_id")
    batting_df = pd.concat(all_batting, ignore_index=True).sort_values(["match_id", "player_id"])
    bowling_df = pd.concat(all_bowling, ignore_index=True).sort_values(["match_id", "player_id"])
    validate_output(matches_df, batting_df, bowling_df)

    output_directory.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(output_directory / "matches.csv", index=False)
    batting_df.to_csv(output_directory / "batting_match.csv", index=False)
    bowling_df.to_csv(output_directory / "bowling_match.csv", index=False)

    return {
        "matches": len(matches_df),
        "batting_rows": len(batting_df),
        "bowling_rows": len(bowling_df),
    }


def run_default_pipeline(project_root: str | Path | None = None) -> pd.DataFrame:
    """Run ETL for every configured source directory in this repository."""

    root = Path(project_root) if project_root else project_root_from_file()
    results: list[dict[str, Any]] = []

    for output_name, (raw_directory_name, competition) in DATASETS.items():
        print(f"\nRunning {competition} ETL")
        counts = process_competition(
            root / "data" / "raw" / raw_directory_name,
            root / "data" / "processed" / output_name,
            competition,
        )
        results.append({"dataset": output_name, "competition": competition, **counts})

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Run all configured competitions.")
    parser.add_argument("--competition", choices=DATASETS.keys(), help="Run one configured competition.")
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    arguments = parser.parse_args()

    if arguments.all:
        print(run_default_pipeline(arguments.project_root).to_string(index=False))
        return

    if arguments.competition:
        raw_name, competition = DATASETS[arguments.competition]
        result = process_competition(
            arguments.project_root / "data" / "raw" / raw_name,
            arguments.project_root / "data" / "processed" / arguments.competition,
            competition,
        )
        print(result)
        return

    parser.error("Choose --all or --competition.")


if __name__ == "__main__":
    main()
