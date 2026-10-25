"""Checks for forecast coverage, role handling, stale files and offline reports."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.visualization.create_prediction_visualizations import (
    build_payload,
    create_prediction_visualizations,
    disciplines,
    load_outputs,
    scenario_values,
)
from src.visualization.prediction_report import write_report


def create_fixture(root: Path) -> Path:
    model = root / "data" / "processed" / "model"
    model.mkdir(parents=True)
    fair = pd.DataFrame(
        [
            ["bat", "Example Batter", "Batter", "Top order Batter", "India", 70_000_000, 2027],
            ["bowl", "Example Bowler", "Bowler", "Bowler", "India", 40_000_000, 2027],
            ["keep", "Example Keeper", "Wicketkeeper", "Wicketkeeper Batter", "India", 50_000_000, 2027],
            ["all", "Example All-rounder", "All-rounder", "Batting Allrounder", "India", 60_000_000, 2027],
            ["other", "Role Unavailable", None, None, "India", 10_000_000, 2027],
        ],
        columns=["player_id", "player_name_canonical", "role", "metadata_playing_role", "nationality", "predicted_fair_value_inr", "forecast_auction_year"],
    )
    fair.to_csv(model / "auction_candidate_fair_values.csv", index=False)
    forecast = pd.DataFrame(
        [
            ["bat", 560.0, 0.0, 13.5, 0.0],
            ["bowl", 10.0, 18.0, 3.0, 13.0],
            ["keep", 420.0, 0.0, 12.0, 0.0],
            ["all", 260.0, 12.0, 12.0, 11.0],
            ["other", 20.0, 1.0, 2.0, 2.0],
        ],
        columns=["player_id", "expected_runs", "expected_wickets", "expected_batting_matches", "expected_bowling_matches"],
    )
    forecast["as_of_season"] = 2026
    forecast["predicted_ipl_season"] = 2027
    forecast["has_recent_ipl_history"] = True
    forecast["has_recent_t20_history"] = True
    forecast["has_recent_odi_history"] = False
    forecast["forecast_method"] = "opportunity_rate_calibrated"
    for measure, error in (("runs", 60), ("wickets", 2)):
        forecast[f"expected_{measure}_conservative"] = (forecast[f"expected_{measure}"] - error).clip(lower=0)
        forecast[f"expected_{measure}_upside"] = forecast[f"expected_{measure}"] + error
    forecast.to_csv(model / "upcoming_performance_predictions.csv", index=False)
    summaries = []
    for team in ("Team One", "Team Two"):
        table = fair.merge(forecast, on="player_id", validate="one_to_one")
        table["team"] = team
        table["rank"] = range(1, len(table) + 1)
        table["team_fit_score"] = [90, 82, 75, 65, 40]
        table["investment_score"] = [72, 80, 65, 76, 35]
        table["investment_label"] = "Fair value"
        table["role_match"] = [True, False, True, True, False]
        table["within_budget"] = [True, True, True, True, False]
        table["recommendation"] = ["Strong fit"] * 4 + ["Low fit"]
        table["reason"] = "Saved role and per-player price check."
        suffix = team.replace(" ", "_")
        table.to_csv(model / f"team_fit_{suffix}.csv", index=False)
        roles = table.iloc[:4].copy()
        roles["shortlist_role"] = ["Batter", "Bowler", "Wicketkeeper-batter", "All-rounder"]
        roles["role_rank"] = 1
        roles.to_csv(model / f"team_role_shortlists_{suffix}.csv", index=False)
        summaries.append(table.head(2))
    pd.concat(summaries).to_csv(model / "future_team_recommendations.csv", index=False)
    (model / "future_analysis_summary.json").write_text(
        json.dumps({"forecast_year": 2027, "team_profiles_used": ["Team One", "Team Two"], "candidate_pool_assumption": "Every active player is assumed available."}), encoding="utf-8"
    )
    (model / "performance_model.joblib").write_bytes(b"not executable: the report must never unpickle this file")
    return model


class PredictionReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = create_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_every_player_and_team_is_in_payload(self):
        data = load_outputs(self.root)
        payload = build_payload(data, self.model / "visualizations", [], [], top_n=1)
        self.assertEqual(len(payload["players"]), 5)
        self.assertEqual(set(payload["teams"]), {"Team One", "Team Two"})
        self.assertEqual(len(payload["teams"]["Team One"]), 5)
        roles = data.players.set_index("player_id")["role_group"].to_dict()
        self.assertEqual(roles["bat"], "Batter")
        self.assertEqual(roles["keep"], "Wicketkeeper-batter")
        self.assertEqual(roles["other"], "Other")
        self.assertEqual(disciplines(roles["bat"]), ["runs"])
        self.assertEqual(disciplines(roles["bowl"]), ["wickets"])

    def test_legacy_forecasts_do_not_get_fabricated_ranges(self):
        file = self.model / "upcoming_performance_predictions.csv"
        forecast = pd.read_csv(file)
        forecast = forecast.drop(columns=[c for c in forecast if c.endswith(("_conservative", "_upside"))])
        forecast.to_csv(file, index=False)
        data = load_outputs(self.root)
        self.assertEqual(scenario_values(data.players.iloc[0], "runs"), (560.0, None, None))
        self.assertTrue(any("scenario ranges are absent" in note for note in data.warnings))

    def test_stale_team_forecasts_are_rejected(self):
        file = self.model / "team_fit_Team_One.csv"
        table = pd.read_csv(file)
        table.loc[0, "expected_runs"] = 373
        table.to_csv(file, index=False)
        with self.assertRaisesRegex(ValueError, "is stale"):
            load_outputs(self.root)

    def test_stale_combined_recommendations_are_rejected(self):
        file = self.model / "future_team_recommendations.csv"
        table = pd.read_csv(file)
        table.loc[0, "team_fit_score"] = 10
        table.to_csv(file, index=False)
        with self.assertRaisesRegex(ValueError, "outdated team_fit_score"):
            load_outputs(self.root)

    def test_invalid_scenario_order_is_rejected(self):
        file = self.model / "upcoming_performance_predictions.csv"
        table = pd.read_csv(file)
        table.loc[0, "expected_runs_conservative"] = 999
        table.to_csv(file, index=False)
        with self.assertRaisesRegex(ValueError, "scenario ordering"):
            load_outputs(self.root)

    def test_actual_history_is_restricted_to_forecast_cutoff(self):
        directory = self.root / "data" / "processed" / "ipl"
        directory.mkdir()
        pd.DataFrame([
            {"player_id": "bat", "season_end_year": 2026, "runs": 600, "wickets": 0},
            {"player_id": "bat", "season_end_year": 2027, "runs": 999, "wickets": 0},
        ]).to_csv(directory / "player_season_features.csv", index=False)
        data = load_outputs(self.root)
        self.assertEqual(data.history["season_end_year"].tolist(), [2026])
        self.assertEqual(data.history["runs"].tolist(), [600])

    def test_embedded_json_cannot_close_script_tag(self):
        path = self.root / "escape.html"
        payload = {"name": "</script><img src=x onerror=alert(1)>"}
        write_report(payload, path)
        text = path.read_text(encoding="utf-8")
        self.assertIn("\\u003c/script\\u003e", text)
        self.assertNotIn(payload["name"], text)

    def test_undefined_metrics_are_unavailable_not_nonstandard_json(self):
        path = self.root / "metrics.html"
        write_report({"r2": float("nan"), "rmse": float("inf")}, path)
        text = path.read_text(encoding="utf-8")
        self.assertIn('"r2":null', text)
        self.assertIn('"rmse":null', text)

    def test_complete_report_and_optional_player_png_do_not_modify_sources(self):
        original = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.model.iterdir() if p.is_file()}
        report, manifest = create_prediction_visualizations(self.root, top_n=2, all_players=True)
        self.assertTrue(report.exists())
        self.assertEqual(manifest["forecast_players"], 5)
        self.assertEqual(len(manifest["teams"]), 2)
        self.assertTrue((report.parent / "players" / "example_batter_bat.png").exists())
        self.assertTrue((report.parent / "players" / "example_bowler_bowl.png").exists())
        self.assertEqual(len(list((report.parent / "players").glob("*.png"))), 5)
        for chart in manifest["charts"]:
            self.assertTrue((report.parent / chart["file"]).exists())
        source_names = {row["file"] for row in manifest["source_files"]}
        self.assertIn("performance_model.joblib", source_names)
        for path, digest in original.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
