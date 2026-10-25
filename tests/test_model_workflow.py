"""End-to-end regression test for the modelling and historical-backtest workflow."""

from __future__ import annotations

import importlib
import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.run_historical_backtest import run_historical_backtest
from src.models.run_multi_team_backtest import run_multi_team_backtest
from src.models.run_future_analysis import run_future_analysis
from src.models.score_auction_candidates import score_auction_candidates
from src.models.score_team_fit import score_team_fit
from src.transformation.build_model_table import build_model_table
from src.transformation.cricket_etl import DATASETS


PLAYERS = [
    ("ishan", "Ishan Kishan", "Wicketkeeper", "Wicketkeeper Batter"),
    ("batter_2", "Top Batter", "Batter", "Batter"),
    ("batter_3", "Budget Batter", "Batter", "Batter"),
    ("bowler_1", "Top Bowler", "Bowler", "Bowler"),
    ("bowler_2", "Budget Bowler", "Bowler", "Bowler"),
    ("allrounder_1", "Batting Allrounder", "All-rounder", "Batting Allrounder"),
    ("allrounder_2", "Bowling Allrounder", "All-rounder", "Bowling Allrounder"),
    ("batter_4", "Middle Batter", "Batter", "Batter"),
]


def fast_forest(**kwargs):
    """Keep the integration test quick; production still uses 500 trees."""

    kwargs["n_estimators"] = 8
    kwargs["n_jobs"] = 1
    return RandomForestRegressor(**kwargs)


def create_mini_project(root):
    """Create minimal, internally consistent processed data for all model stages."""

    processed = root / "data" / "processed"
    (root / "data" / "reference").mkdir(parents=True)
    (processed / "auction").mkdir(parents=True)

    for directory in DATASETS:
        base = processed / directory
        base.mkdir(parents=True)
        batting_rows, bowling_rows = [], []
        for year in range(2017, 2027):
            for index, (player_id, player, role, _) in enumerate(PLAYERS):
                runs = 340 - index * 24 + (year - 2017) * (10 if index < 3 else 4)
                wickets = 0 if "Batter" in role or "Wicketkeeper" in role else 4 + index
                key = f"{directory}_{year}_{player_id}"
                batting_rows.append(
                    {
                        "match_id": key,
                        "date": f"{year}-06-01",
                        "player_id": player_id,
                        "player": player,
                        "runs": runs,
                        "balls": max(1, runs // 2),
                        "fours": runs // 20,
                        "sixes": runs // 30,
                    }
                )
                bowling_rows.append(
                    {
                        "match_id": key,
                        "date": f"{year}-06-01",
                        "player_id": player_id,
                        "player": player,
                        "balls": wickets * 12,
                        "runs_conceded": wickets * 18,
                        "wickets": wickets,
                    }
                )
        pd.DataFrame(batting_rows).to_csv(base / "batting_match.csv", index=False)
        pd.DataFrame(bowling_rows).to_csv(base / "bowling_match.csv", index=False)

    season_rows = []
    for year in range(2017, 2027):
        for index, (player_id, player, role, _) in enumerate(PLAYERS):
            runs = 450 - index * 30 + (year - 2017) * (14 if index < 3 else 6)
            wickets = 0 if "Batter" in role or "Wicketkeeper" in role else 5 + index
            season_rows.append(
                {
                    "player_id": player_id,
                    "player": player,
                    "season_end_year": year,
                    "batting_matches": 12,
                    "runs": runs,
                    "batting_balls": max(1, runs // 2),
                    "fours": runs // 18,
                    "sixes": runs // 28,
                    "bowling_matches": 12 if wickets else 0,
                    "bowling_balls": wickets * 15,
                    "runs_conceded": wickets * 19,
                    "wickets": wickets,
                }
            )
    (processed / "ipl").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(season_rows).to_csv(processed / "ipl" / "player_season_features.csv", index=False)

    pd.DataFrame(
        [
            {
                "player_id": player_id,
                "player": player,
                "country": "India",
                "batting_style": "Left hand Bat",
                "bowling_style": "Right arm Medium",
                "playing_role": playing_role,
            }
            for player_id, player, _, playing_role in PLAYERS
        ]
    ).to_csv(processed / "player_master.csv", index=False)

    auction_rows = []
    for year in range(2020, 2027):
        current_year_rows = []
        for index, (player_id, player, role, _) in enumerate(PLAYERS):
            price = 24_000_000 + index * 5_000_000 + (year - 2020) * 1_000_000
            current_year_rows.append(
                {
                    "auction_year": year,
                    "player_id": player_id,
                    "player_name_auction": player,
                    "player_name_canonical": player,
                    "player_match_status": "exact",
                    "team": "Example Team",
                    "role": role,
                    "nationality": "Indian",
                    "price_original": price,
                    "price_currency": "INR",
                    "exchange_rate_to_inr": 1.0,
                    "price_inr": price,
                    "source_file": f"{year}.csv",
                }
            )
        median = pd.DataFrame(current_year_rows)["price_inr"].median()
        for row in current_year_rows:
            row["price_inr_year_median_ratio"] = row["price_inr"] / median
        auction_rows.extend(current_year_rows)
    pd.DataFrame(auction_rows).to_csv(processed / "auction" / "auction_clean.csv", index=False)

    pd.DataFrame(
        [
            {
                "team": "Example Team",
                "desired_role": "Batter|Wicketkeeper",
                "performance_weight": 0.45,
                "value_weight": 0.25,
                "batting_weight": 0.25,
                "bowling_weight": 0.05,
                "max_fair_value_inr": 100_000_000,
            }
        ]
    ).to_csv(root / "data" / "reference" / "team_needs.csv", index=False)


def test_complete_model_and_backtest_workflow(tmp_path, monkeypatch):
    """Exercise model-table build, both models, scoring and historical output."""

    fair_value = importlib.import_module("src.models.train_fair_value_model")
    performance = importlib.import_module("src.models.train_performance_model")
    monkeypatch.setattr(fair_value, "RandomForestRegressor", fast_forest)
    monkeypatch.setattr(performance, "RandomForestRegressor", fast_forest)
    create_mini_project(tmp_path)

    model_table = build_model_table(tmp_path)
    assert model_table["eligible_for_price_model"].all()
    fair_result = fair_value.train_fair_value_model(tmp_path, test_years=2)
    performance_result = performance.train_performance_model(
        tmp_path, as_of_season=2026, lookback_seasons=3, test_seasons=2
    )
    candidate_result = score_auction_candidates(tmp_path, pool_year=2026, auction_year=2027)
    ranking = score_team_fit(" Example Team ", tmp_path)

    assert fair_result["test_rows"] > 0
    assert performance_result["forecast_rows"] == len(PLAYERS)
    assert candidate_result["scored_players"] == len(PLAYERS)
    assert len(ranking) == len(PLAYERS)
    assert ranking["team_fit_score"].between(0, 100).all()
    assert ranking["investment_score"].between(0, 100).all()
    assert set(ranking["investment_label"]).issubset(
        {"High value", "Fair value", "Expensive / limited value"}
    )
    assert set(ranking["recommendation"]).issubset({"Strong fit", "Consider", "Low fit"})
    assert ranking.loc[ranking["player_name_canonical"].eq("Ishan Kishan"), "role_match"].item()
    scored_candidates = pd.read_csv(tmp_path / "data" / "processed" / "model" / "auction_candidate_fair_values.csv")
    assert set(scored_candidates["source_pool_year"]) == {2026}
    assert set(scored_candidates["forecast_auction_year"]) == {2027}
    assert set(scored_candidates["candidate_pool_scope"]) == {"recent_ipl_or_international"}
    assert scored_candidates["recent_qualifying_matches"].ge(1).all()
    assert (tmp_path / "data" / "processed" / "model" / "auction_candidate_eligibility.csv").exists()
    assert "price_inr" not in scored_candidates.columns
    role_shortlists = pd.read_csv(tmp_path / "data" / "processed" / "model" / "team_role_shortlists_Example_Team.csv")
    assert role_shortlists["role_rank"].max() <= 5
    assert {"investment_score", "investment_label"}.issubset(role_shortlists.columns)

    backtest_dir, backtest_ranking = run_historical_backtest(
        team="Example Team", target_season=2026, player_search="Kishan", project_root=tmp_path
    )
    assert (backtest_dir / "backtest_summary.json").exists()
    assert len(backtest_ranking) == len(PLAYERS)
    assert set(backtest_ranking["recommendation"]).issubset(
        {"Strong fit", "Consider", "Low fit"}
    )
    summary = json.loads((backtest_dir / "backtest_summary.json").read_text(encoding="utf-8"))
    assert summary["performance_data_through_season"] == 2025
    assert summary["fair_value_training_through_auction_year"] == 2025
    assert summary["candidate_pool_year"] == 2026

    multi_directory, multi_summary = run_multi_team_backtest(
        target_season=2026, project_root=tmp_path, top_n=3
    )
    assert len(multi_summary) == 3
    assert multi_summary["team"].nunique() == 1
    assert (multi_directory / "all_teams_top_recommendations.csv").exists()
    assert (multi_directory / "multi_team_backtest_summary.json").exists()

    future_directory, future_summary = run_future_analysis(
        project_root=tmp_path, pool_year=2026, forecast_year=2027, top_n=3
    )
    assert future_directory == tmp_path / "data" / "processed" / "model"
    assert len(future_summary) == 3
    future_metadata = json.loads((future_directory / "future_analysis_summary.json").read_text(encoding="utf-8"))
    assert future_metadata["pool_year"] == 2026
    assert future_metadata["forecast_year"] == 2027
    assert (future_directory / "upcoming_performance_predictions.csv").exists()
    assert (future_directory / "auction_candidate_fair_values.csv").exists()
    assert (future_directory / "future_team_recommendations.csv").exists()
    forecast = pd.read_csv(future_directory / "upcoming_performance_predictions.csv")
    assert {
        "has_recent_t20_history", "has_recent_odi_history",
        "t20_recent_matches", "odi_recent_matches",
        "forecast_method", "expected_runs_conservative", "expected_runs_upside",
        "expected_wickets_conservative", "expected_wickets_upside",
        "historical_median_ipl_runs", "historical_median_runs_per_batting_match",
    }.issubset(forecast.columns)
    assert (forecast["expected_runs_conservative"] <= forecast["expected_runs"]).all()
    assert (forecast["expected_runs"] <= forecast["expected_runs_upside"]).all()
    calibration = pd.read_csv(future_directory / "performance_calibration_comparison.csv")
    assert set(calibration["forecast_method"]) == {"direct_total", "opportunity_rate_calibrated"}
    performance_artifact = __import__("joblib").load(future_directory / "performance_model.joblib")
    assert {"t20_recent_runs", "odi_recent_runs"}.issubset(performance_artifact["feature_columns"])
    assert "opportunity_rate_pipeline" in performance_artifact


def test_cross_format_features_do_not_use_matches_after_the_cutoff():
    """A 2027 match must not affect an as-of-2026 forecast feature."""

    performance = importlib.import_module("src.models.train_performance_model")
    batting_columns = ["match_id", "date", "player_id", "player", "runs", "balls"]
    bowling_columns = ["match_id", "date", "player_id", "player", "balls", "runs_conceded", "wickets"]
    empty_batting = pd.DataFrame(columns=batting_columns)
    empty_bowling = pd.DataFrame(columns=bowling_columns)
    tables = {
        directory: (empty_batting.copy(), empty_bowling.copy())
        for directory in ("t20", "bbl", "smat", "t20_wc", "odi")
    }
    tables["t20"] = (
        pd.DataFrame(
            [
                ["known_match", "2026-06-01", "form_player", "Form Player", 80, 40],
                ["future_match", "2027-02-01", "form_player", "Form Player", 999, 300],
            ],
            columns=batting_columns,
        ).assign(date=lambda table: pd.to_datetime(table["date"])),
        empty_bowling.copy(),
    )

    features = performance.build_cross_format_features(tables, as_of_season=2026, lookback_seasons=3)
    player = features.loc[features["player_id"].eq("form_player")].iloc[0]
    assert player["t20_recent_runs"] == 80
    assert player["t20_recent_matches"] == 1
    assert bool(player["has_recent_odi_history"]) is False
