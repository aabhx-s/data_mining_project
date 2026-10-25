"""Create the project's current forward-looking IPL auction analysis.

The command trains on every available historical record, forecasts next-season
IPL performance, estimates a fair auction value for the selected candidate
pool, and optionally ranks candidates for every team profile provided.

With data through IPL 2026 and the 2026 auction, the default output is a 2027
forecast:

    python -m src.models.run_future_analysis --top-n 10

Auction price training records sold players only. Candidate rankings require
recent IPL or supported international activity and exclude season-specific
unavailability notes. Unknown availability is explicit; recent history is not
proof of next-season medical fitness, retention, registration or auction sale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.models.common import ensure_directory, write_json
from src.models.candidate_eligibility import load_country_policy
from src.models.score_auction_candidates import score_auction_candidates
from src.models.score_team_fit import REQUIRED_PROFILE_COLUMNS, score_team_fit
from src.models.train_fair_value_model import train_fair_value_model
from src.models.train_performance_model import train_performance_model
from src.transformation.cricket_etl import project_root_from_file


def available_auction_years(root: Path) -> list[int]:
    """Read valid auction years from the cleaned sold-player source."""

    auction_path = root / "data" / "processed" / "auction" / "auction_clean.csv"
    if not auction_path.exists():
        raise FileNotFoundError(f"Missing {auction_path}. Run auction_etl.py first.")
    years = pd.to_numeric(pd.read_csv(auction_path, usecols=["auction_year"])["auction_year"], errors="coerce")
    output = sorted(years.dropna().astype(int).unique().tolist())
    if not output:
        raise ValueError("auction_clean.csv does not contain a valid auction_year.")
    return output


def load_profile_teams(profile_path: Path) -> list[str]:
    """Validate a profile file and return its unique team names."""

    profiles = pd.read_csv(profile_path)
    missing = REQUIRED_PROFILE_COLUMNS.difference(profiles.columns)
    if missing:
        raise ValueError(f"{profile_path} is missing required columns: {sorted(missing)}")
    teams = profiles["team"].astype("string").str.strip()
    if teams.isna().any() or teams.eq("").any():
        raise ValueError("Every team profile needs a non-empty team name.")
    if teams.duplicated().any():
        raise ValueError("Each team must appear only once in the team-needs profile file.")
    return teams.tolist()


def run_future_analysis(
    project_root: str | Path | None = None,
    pool_year: int | None = None,
    forecast_year: int | None = None,
    team_needs_path: str | Path | None = None,
    top_n: int = 10,
    skip_team_fit: bool = False,
    teams: list[str] | None = None,
    activity_years: int = 2,
    min_qualifying_matches: int = 1,
    availability_file: str | Path | None = None,
    require_confirmed_availability: bool = False,
) -> tuple[Path, pd.DataFrame]:
    """Run the final forward-analysis workflow and return its comparison table."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if activity_years < 1 or min_qualifying_matches < 1:
        raise ValueError("activity_years and min_qualifying_matches must be at least 1")

    root = Path(project_root) if project_root else project_root_from_file()
    years = available_auction_years(root)
    selected_pool_year = pool_year or max(years)
    if selected_pool_year not in years:
        raise ValueError(f"pool_year {selected_pool_year} is unavailable; available years: {years}")
    selected_forecast_year = forecast_year or selected_pool_year + 1
    if selected_forecast_year <= selected_pool_year:
        raise ValueError("forecast_year must be later than pool_year for a forward analysis.")
    # Validate the selected season's rule before expensive model training. Do
    # not silently carry the user's 2027 country scenario into later seasons.
    load_country_policy(root / "data" / "reference" / "candidate_country_policy.json", selected_forecast_year)

    output_directory = ensure_directory(root / "data" / "processed" / "model")
    fair_value_result = train_fair_value_model(project_root=root, test_years=2, output_directory=output_directory)
    candidate_result = score_auction_candidates(
        project_root=root,
        pool_year=selected_pool_year,
        auction_year=selected_forecast_year,
        recent_years=3,
        model_directory=output_directory,
        candidate_scope="recent_ipl_or_international",
        activity_years=activity_years,
        min_qualifying_matches=min_qualifying_matches,
        availability_path=availability_file,
        require_confirmed_availability=require_confirmed_availability,
    )
    # Forecast only the screened candidate IDs. Historical model training is
    # deliberately not filtered using future candidate or availability labels.
    candidate_ids = pd.read_csv(
        output_directory / "auction_candidate_fair_values.csv", dtype={"player_id": "string"}
    )["player_id"].dropna().tolist()
    performance_result = train_performance_model(
        project_root=root,
        as_of_season=selected_forecast_year - 1,
        lookback_seasons=4,
        test_seasons=2,
        output_directory=output_directory,
        candidate_player_ids=candidate_ids,
    )

    summaries: list[dict[str, object]] = []
    profile_teams: list[str] = []
    profile_path: Path | None = None
    if not skip_team_fit:
        default_profile_path = root / "data" / "reference" / "team_needs.csv"
        profile_path = Path(team_needs_path) if team_needs_path else default_profile_path
        if not profile_path.exists():
            raise FileNotFoundError(
                f"Missing team-needs profile: {profile_path}. "
                "Copy a team_needs CSV into data/reference or use --skip-team-fit."
            )
    if profile_path is not None:
        profile_teams = load_profile_teams(profile_path)
        requested_teams = [team.strip() for team in teams] if teams else profile_teams
        unavailable = sorted(set(requested_teams).difference(profile_teams))
        if unavailable:
            raise ValueError(f"Requested teams are not in {profile_path}: {unavailable}")
        for team in requested_teams:
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
                        "investment_score": row["investment_score"],
                        "investment_label": row["investment_label"],
                        "within_budget": row["within_budget"],
                        "role_match": row["role_match"],
                        "predicted_fair_value_inr": row["predicted_fair_value_inr"],
                        "expected_runs": row["expected_runs"],
                        "expected_wickets": row["expected_wickets"],
                        "pre_budget_fit_score": row.get("pre_budget_fit_score", row["team_fit_score"]),
                        "availability_status": row.get("availability_status", "unknown"),
                        "recent_ipl_matches": row.get("recent_ipl_matches"),
                        "recent_international_matches": row.get("recent_international_matches"),
                        "eligibility_reason": row.get("eligibility_reason"),
                    }
                )

    summary_table = pd.DataFrame(summaries)
    if profile_teams:
        summary_table.to_csv(output_directory / "future_team_recommendations.csv", index=False)

    write_json(
        {
            "pool_year": selected_pool_year,
            "forecast_year": selected_forecast_year,
            "performance_trained_through_season": selected_forecast_year - 1,
            "candidate_pool_assumption": (
                f"Requires at least {min_qualifying_matches} recorded IPL or supported international "
                f"match participation(s) during {selected_forecast_year - activity_years}–{selected_forecast_year - 1}. "
                "BBL/SMAT-only history does not qualify a candidate. Season-specific blocked availability "
                "statuses are excluded. "
                + ("Only reference records marked available are admitted. " if require_confirmed_availability else "Unknown availability is assumed available, not verified. ")
                + "Activity screening does not confirm medical fitness, retention/release, registration, selection or auction sale."
                + " Country screening applies the season-specific project policy: "
                + ", ".join(candidate_result["country_policy"]["excluded_countries"])
                + ". These exclusions are recorded as project planning assumptions, not a claim that every country has a confirmed blanket IPL ban."
            ),
            "team_profiles_used": requested_teams if profile_path is not None and profile_path.exists() else [],
            "top_n_per_team": top_n if profile_teams else 0,
            "fair_value_training": fair_value_result,
            "performance_training": performance_result,
            "candidate_scoring": candidate_result,
            "team_recommendation_file": str(output_directory / "future_team_recommendations.csv") if profile_teams else None,
        },
        output_directory / "future_analysis_summary.json",
    )
    return output_directory, summary_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--pool-year", type=int, default=None)
    parser.add_argument("--forecast-year", type=int, default=None)
    parser.add_argument("--team-needs", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--activity-years", type=int, default=2)
    parser.add_argument("--min-qualifying-matches", type=int, default=1)
    parser.add_argument("--availability-file", type=Path, default=None)
    parser.add_argument("--require-confirmed-availability", action="store_true")
    parser.add_argument(
        "--skip-team-fit",
        action="store_true",
        help="Create player forecasts and fair values only, without reading a team-needs profile.",
    )
    arguments = parser.parse_args()
    directory, summary = run_future_analysis(
        project_root=arguments.project_root,
        pool_year=arguments.pool_year,
        forecast_year=arguments.forecast_year,
        team_needs_path=arguments.team_needs,
        top_n=arguments.top_n,
        skip_team_fit=arguments.skip_team_fit,
        activity_years=arguments.activity_years,
        min_qualifying_matches=arguments.min_qualifying_matches,
        availability_file=arguments.availability_file,
        require_confirmed_availability=arguments.require_confirmed_availability,
    )
    if summary.empty:
        print(f"Saved player forecasts and fair values under {directory}; no team profile file was supplied.")
    else:
        print(f"Saved {len(summary):,} team recommendations under {directory}.")


if __name__ == "__main__":
    main()
