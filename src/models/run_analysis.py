"""Single command entry point for forward forecasts and historical backtests.

Examples:

    # Current forward analysis (defaults to the latest auction year + one).
    python -m src.models.run_analysis --mode future --skip-team-fit

    # Forecast one team's recommendations from a chosen profile file.
    python -m src.models.run_analysis --mode future --team "Royal Challengers Bengaluru" --team-needs data/reference/team_needs.csv

    # Recreate one team's pre-season recommendation.
    python -m src.models.run_analysis --mode historical --target-season 2026 --team "Royal Challengers Bengaluru"

    # Compare every profile in a historical season.
    python -m src.models.run_analysis --mode historical --target-season 2026 --all-teams

``future`` trains on all data available through the chosen pool year.
``historical`` trains only through the season before ``target-season``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.run_future_analysis import run_future_analysis
from src.models.run_historical_backtest import run_historical_backtest
from src.models.run_multi_team_backtest import run_multi_team_backtest
from src.transformation.cricket_etl import project_root_from_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["future", "historical"], required=True)
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--team", default=None, help="One team profile to score.")
    parser.add_argument("--all-teams", action="store_true", help="Score every profile in a historical run.")
    parser.add_argument("--team-needs", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--pool-year", type=int, default=None, help="Reference auction year for future mode.")
    parser.add_argument("--forecast-year", type=int, default=None, help="Prediction year for future mode.")
    parser.add_argument("--target-season", type=int, default=None, help="Season to recreate in historical mode.")
    parser.add_argument("--player-search", default=None, help="Optional player text to display in a single-team historical run.")
    parser.add_argument("--skip-team-fit", action="store_true", help="Future mode: create player forecasts and values only.")
    parser.add_argument("--activity-years", type=int, default=2, help="Future candidate qualification window; default: the last two completed years.")
    parser.add_argument("--min-qualifying-matches", type=int, default=1, help="Future mode: minimum distinct recent IPL/international participations.")
    parser.add_argument("--availability-file", type=Path, default=None, help="Optional season-specific status reference; defaults to data/reference/player_availability.csv if present.")
    parser.add_argument("--require-confirmed-availability", action="store_true", help="Future mode: reject unknown availability as well as blocked statuses.")
    arguments = parser.parse_args()

    if arguments.mode == "future":
        if arguments.all_teams:
            parser.error("--all-teams is only available in historical mode; future mode scores all profiles by default.")
        directory, summary = run_future_analysis(
            project_root=arguments.project_root,
            pool_year=arguments.pool_year,
            forecast_year=arguments.forecast_year,
            team_needs_path=arguments.team_needs,
            top_n=arguments.top_n,
            skip_team_fit=arguments.skip_team_fit,
            teams=[arguments.team] if arguments.team else None,
            activity_years=arguments.activity_years,
            min_qualifying_matches=arguments.min_qualifying_matches,
            availability_file=arguments.availability_file,
            require_confirmed_availability=arguments.require_confirmed_availability,
        )
        if summary.empty:
            print(f"Saved forward player forecasts and fair values under {directory}.")
        else:
            print(f"Saved {len(summary):,} forward team recommendations under {directory}.")
        return

    if arguments.target_season is None:
        parser.error("--target-season is required in historical mode.")
    if arguments.activity_years != 2 or arguments.min_qualifying_matches != 1 or arguments.availability_file or arguments.require_confirmed_availability:
        parser.error("The activity/availability screening options are for future mode only.")
    if arguments.skip_team_fit:
        parser.error("--skip-team-fit is only available in future mode.")
    if arguments.team and arguments.all_teams:
        parser.error("Choose either --team or --all-teams in historical mode, not both.")
    if not arguments.team and not arguments.all_teams:
        parser.error("Historical mode requires --team or --all-teams.")

    if arguments.all_teams:
        directory, summary = run_multi_team_backtest(
            target_season=arguments.target_season,
            project_root=arguments.project_root,
            team_needs_path=arguments.team_needs,
            top_n=arguments.top_n,
        )
        print(f"Saved {len(summary):,} historical recommendations under {directory}.")
        return

    directory, ranking = run_historical_backtest(
        team=arguments.team,
        target_season=arguments.target_season,
        player_search=arguments.player_search,
        project_root=arguments.project_root,
        team_needs_path=arguments.team_needs,
    )
    print(f"Saved {len(ranking):,} historical candidate rankings under {directory}.")


if __name__ == "__main__":
    main()
