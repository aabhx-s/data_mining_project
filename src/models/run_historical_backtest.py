"""Run a leakage-safe historical team-fit backtest.

Example: recreate a pre-IPL-2026 recommendation using only data available
through the 2025 IPL season and auction history through auction-year 2025:

    python -m src.models.run_historical_backtest --team "Royal Challengers Bengaluru" --target-season 2026 --player-search Iyer

The 2026 sold-player dataset supplies a retrospective candidate pool only. It
does not contain unsold players, so the test answers whether a known auction
candidate would have ranked well, not whether the model could predict a sale.
All backtest artifacts are written below ``data/processed/backtests`` and do
not overwrite the current model outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.models.common import ensure_directory, write_json
from src.models.score_auction_candidates import score_auction_candidates
from src.models.score_team_fit import score_team_fit
from src.models.train_fair_value_model import train_fair_value_model
from src.models.train_performance_model import train_performance_model
from src.transformation.cricket_etl import project_root_from_file


def run_historical_backtest(
    team: str,
    target_season: int = 2026,
    player_search: str | None = None,
    project_root: str | Path | None = None,
    team_needs_path: str | Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Create a team-fit ranking using only data known before ``target_season``."""

    if target_season < 2015:
        raise ValueError("target_season must be 2015 or later for a meaningful historical split.")

    root = Path(project_root) if project_root else project_root_from_file()
    cutoff_season = target_season - 1
    output_directory = ensure_directory(
        root / "data" / "processed" / "backtests" / f"ipl_{target_season}_{team.casefold().replace(' ', '_')}"
    )

    # The final fair-value model uses rows through the last known auction year.
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
        # The 2025 auction median is used as the pre-auction 2026 market baseline.
        market_median_inr=None,
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
    ranking = score_team_fit(
        team=team,
        project_root=root,
        team_needs_path=team_needs_path,
        model_directory=output_directory,
    )

    if player_search:
        matches = ranking.loc[
            ranking["player_name_canonical"].astype("string").str.contains(player_search, case=False, na=False)
        ]
        if matches.empty:
            print(f"No backtest candidate matched {player_search!r}.")
        else:
            columns = [
                "rank", "player_name_canonical", "team_fit_score", "recommendation",
                "predicted_fair_value_inr", "expected_runs", "expected_wickets", "reason",
            ]
            print(matches[columns].to_string(index=False))

    summary = {
        "team": team,
        "target_season": target_season,
        "performance_data_through_season": cutoff_season,
        "fair_value_training_through_auction_year": cutoff_season,
        "candidate_pool_year": target_season,
        "candidate_pool_limitation": "Sold players only; unsold auction candidates are unavailable.",
        "fair_value_training": fair_value_result,
        "performance_training": performance_result,
        "candidate_scoring": candidate_result,
        "ranking_file": str(output_directory / f"team_fit_{''.join(character if character.isalnum() else '_' for character in team).strip('_')}.csv"),
    }
    write_json(summary, output_directory / "backtest_summary.json")
    return output_directory, ranking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", required=True, help="Team name exactly as entered in team_needs.csv")
    parser.add_argument("--target-season", type=int, default=2026)
    parser.add_argument("--player-search", default=None, help="Optional text to display a candidate, for example Iyer")
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--team-needs", type=Path, default=None)
    arguments = parser.parse_args()
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
