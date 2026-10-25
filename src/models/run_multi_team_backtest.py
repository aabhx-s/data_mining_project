"""Run one leakage-safe historical backtest for every team profile in a CSV.

This is the efficient companion to ``run_historical_backtest.py``. It trains
the fair-value and performance models once for a target season, then applies
the resulting candidate forecasts to each team's separately documented needs.

Example:

    python -m src.models.run_multi_team_backtest --target-season 2026

The CSV at ``data/reference/team_needs.csv`` must contain one unique row per
team. Results are saved in one shared directory beneath ``data/processed``;
each team still receives its own ranking CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.models.common import ensure_directory, write_json
from src.models.score_auction_candidates import score_auction_candidates
from src.models.score_team_fit import REQUIRED_PROFILE_COLUMNS, score_team_fit
from src.models.train_fair_value_model import train_fair_value_model
from src.models.train_performance_model import train_performance_model
from src.transformation.cricket_etl import project_root_from_file


def run_multi_team_backtest(
    target_season: int = 2026,
    project_root: str | Path | None = None,
    team_needs_path: str | Path | None = None,
    top_n: int = 5,
) -> tuple[Path, pd.DataFrame]:
    """Score every unique team profile using the same historical model snapshot."""

    if target_season < 2015:
        raise ValueError("target_season must be 2015 or later for a meaningful historical split.")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    root = Path(project_root) if project_root else project_root_from_file()
    profile_path = Path(team_needs_path) if team_needs_path else root / "data" / "reference" / "team_needs.csv"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Missing {profile_path}. Copy and edit data/reference/team_needs_template.csv first."
        )

    profiles = pd.read_csv(profile_path)
    missing = REQUIRED_PROFILE_COLUMNS.difference(profiles.columns)
    if missing:
        raise ValueError(f"{profile_path} is missing required columns: {sorted(missing)}")
    teams = profiles["team"].astype("string").str.strip()
    if teams.isna().any() or teams.eq("").any():
        raise ValueError("Every team profile needs a non-empty team name.")
    duplicates = teams.loc[teams.duplicated(keep=False)].tolist()
    if duplicates:
        raise ValueError(f"Each team needs exactly one profile; duplicates found: {sorted(set(duplicates))}")

    cutoff_season = target_season - 1
    output_directory = ensure_directory(
        root / "data" / "processed" / "backtests" / f"ipl_{target_season}_all_teams"
    )

    fair_value_result = train_fair_value_model(
        project_root=root,
        test_years=2,
        through_auction_year=cutoff_season,
        output_directory=output_directory,
    )
    candidate_result = score_auction_candidates(
        project_root=root,
        pool_year=target_season,
        auction_year=target_season,
        recent_years=3,
        model_directory=output_directory,
        candidate_scope="latest_sold_players",
    )
    candidate_ids = pd.read_csv(
        output_directory / "auction_candidate_fair_values.csv", dtype={"player_id": "string"}
    )["player_id"].dropna().tolist()
    performance_result = train_performance_model(
        project_root=root,
        as_of_season=cutoff_season,
        lookback_seasons=4,
        test_seasons=2,
        output_directory=output_directory,
        candidate_player_ids=candidate_ids,
    )

    summaries: list[dict[str, object]] = []
    for team in teams.tolist():
        ranking = score_team_fit(
            team=team,
            project_root=root,
            team_needs_path=profile_path,
            model_directory=output_directory,
        )
        for _, row in ranking.head(top_n).iterrows():
            summaries.append(
                {
                    "team": team,
                    "rank": int(row["rank"]),
                    "player_id": row["player_id"],
                    "player_name_canonical": row["player_name_canonical"],
                    "team_fit_score": row["team_fit_score"],
                    "recommendation": row["recommendation"],
                    "within_budget": row["within_budget"],
                    "role_match": row["role_match"],
                    "predicted_fair_value_inr": row["predicted_fair_value_inr"],
                    "expected_runs": row["expected_runs"],
                    "expected_wickets": row["expected_wickets"],
                }
            )

    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(output_directory / "all_teams_top_recommendations.csv", index=False)
    write_json(
        {
            "target_season": target_season,
            "performance_data_through_season": cutoff_season,
            "fair_value_training_through_auction_year": cutoff_season,
            "candidate_pool_year": target_season,
            "candidate_pool_limitation": "Sold players only; unsold auction candidates are unavailable.",
            "teams": teams.tolist(),
            "top_n_per_team": top_n,
            "fair_value_training": fair_value_result,
            "performance_training": performance_result,
            "candidate_scoring": candidate_result,
            "summary_file": str(output_directory / "all_teams_top_recommendations.csv"),
        },
        output_directory / "multi_team_backtest_summary.json",
    )
    return output_directory, summary_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-season", type=int, default=2026)
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--team-needs", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=5)
    arguments = parser.parse_args()
    directory, summary = run_multi_team_backtest(
        target_season=arguments.target_season,
        project_root=arguments.project_root,
        team_needs_path=arguments.team_needs,
        top_n=arguments.top_n,
    )
    print(f"Saved {len(summary):,} recommendations for {summary['team'].nunique():,} teams under {directory}.")


if __name__ == "__main__":
    main()
