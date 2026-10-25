"""Next-season qualification must use recent, pre-cutoff cricket evidence."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.models.candidate_eligibility import assess_candidate_eligibility, load_availability


def empty_tables():
    empty = pd.DataFrame(columns=["player_id", "match_id", "date"])
    return {name: (empty.copy(), empty.copy()) for name in ("ipl", "t20", "odi", "t20_wc", "bbl", "smat")}


def events(*rows):
    return pd.DataFrame(rows, columns=["player_id", "match_id", "date"])


class CandidateEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.availability = Path(self.temp.name) / "availability.csv"
        self.candidates = pd.DataFrame({"player_id": ["recent_ipl", "international", "old_ipl", "domestic", "future_only", "inactive"]})
        self.tables = empty_tables()
        self.tables["ipl"] = (
            events(["recent_ipl", "ipl_2026", "2026-05-01"], ["international", "ipl_old", "2019-04-01"], ["old_ipl", "ipl_2024", "2024-05-01"], ["future_only", "future", "2027-04-01"]),
            events(["recent_ipl", "ipl_2026", "2026-05-01"]),
        )
        self.tables["odi"] = (events(["international", "odi_2025", "2025-01-01"]), events())
        self.tables["bbl"] = (events(["domestic", "bbl_2026", "2026-12-01"]), events())

    def tearDown(self):
        self.temp.cleanup()

    def assess(self, **kwargs):
        return assess_candidate_eligibility(self.candidates, self.tables, 2027, **kwargs).set_index("player_id")

    def write_notes(self, rows):
        pd.DataFrame(rows, columns=["player_id", "forecast_year", "availability_status", "known_as_of", "reason", "source_url"]).to_csv(self.availability, index=False)

    def test_recent_ipl_or_international_is_required_not_both(self):
        audit = self.assess()
        self.assertTrue(audit.loc["recent_ipl", "eligible_for_candidate_pool"])
        self.assertTrue(audit.loc["international", "eligible_for_candidate_pool"])
        self.assertEqual(audit.loc["international", "recent_ipl_matches"], 0)
        self.assertEqual(audit.loc["international", "recent_international_matches"], 1)
        self.assertFalse(audit.loc["old_ipl", "eligible_for_candidate_pool"])
        self.assertEqual(audit.loc["old_ipl", "last_ipl_match_date"], "2024-05-01")
        self.assertFalse(audit.loc["inactive", "eligible_for_candidate_pool"])
        self.assertEqual(audit.loc["recent_ipl", "activity_window_start"], "2025-01-01")
        self.assertEqual(audit.loc["recent_ipl", "activity_window_end"], "2026-12-31")

    def test_domestic_only_is_excluded_with_explanation(self):
        audit = self.assess()
        self.assertFalse(audit.loc["domestic", "eligible_for_candidate_pool"])
        self.assertEqual(audit.loc["domestic", "exclusion_reason"], "domestic_history_only")
        self.assertEqual(audit.loc["domestic", "recent_domestic_matches"], 1)

    def test_future_matches_cannot_qualify_a_player(self):
        audit = self.assess()
        self.assertFalse(audit.loc["future_only", "eligible_for_candidate_pool"])
        self.assertEqual(audit.loc["future_only", "recent_qualifying_matches"], 0)

    def test_batting_and_bowling_do_not_double_count_a_match(self):
        audit = self.assess()
        self.assertEqual(audit.loc["recent_ipl", "recent_ipl_matches"], 1)
        self.assertEqual(audit.loc["recent_ipl", "recent_qualifying_matches"], 1)

    def test_world_cup_duplicates_do_not_satisfy_minimum(self):
        self.tables["odi"] = (events(), events())
        duplicate = events(["international", "same_global_id", "2026-06-01"])
        self.tables["t20"] = (duplicate, duplicate.copy())
        self.tables["t20_wc"] = (duplicate.copy(), duplicate.copy())
        audit = self.assess(min_qualifying_matches=2)
        self.assertEqual(audit.loc["international", "recent_international_matches"], 1)
        self.assertFalse(audit.loc["international", "eligible_for_candidate_pool"])
        self.assertEqual(audit.loc["international", "exclusion_reason"], "insufficient_recent_qualifying_matches")

    def test_recorded_ipl_retirement_and_ruled_out_status_exclude_recent_players(self):
        self.write_notes([
            ["recent_ipl", 2027, "retired_from_ipl", "2026-09-01", "Example status", ""],
            ["international", 2027, "ruled_out", "2026-10-01", "Example status for this season", ""],
        ])
        audit = self.assess(availability_path=self.availability)
        self.assertFalse(audit.loc["recent_ipl", "eligible_for_candidate_pool"])
        self.assertFalse(audit.loc["international", "eligible_for_candidate_pool"])
        self.assertIn("retired_from_ipl", audit.loc["recent_ipl", "exclusion_reason"])
        self.assertIn("ruled_out", audit.loc["international", "exclusion_reason"])

    def test_previous_season_unavailability_is_not_permanent(self):
        self.write_notes([["recent_ipl", 2025, "ruled_out", "2024-12-01", "Old injury", ""]])
        audit = self.assess(availability_path=self.availability)
        self.assertTrue(audit.loc["recent_ipl", "eligible_for_candidate_pool"])
        self.assertEqual(audit.loc["recent_ipl", "availability_status"], "unknown")

    def test_latest_known_update_can_clear_a_season_specific_block(self):
        self.write_notes([
            ["recent_ipl", 2027, "ruled_out", "2026-08-01", "Earlier note", ""],
            ["recent_ipl", 2027, "available", "2026-11-01", "Later note", ""],
        ])
        audit = self.assess(availability_path=self.availability, require_confirmed_availability=True)
        self.assertTrue(audit.loc["recent_ipl", "eligible_for_candidate_pool"])
        self.assertEqual(audit.loc["recent_ipl", "availability_status"], "available")

    def test_future_availability_note_is_not_used_before_cutoff(self):
        self.write_notes([["recent_ipl", 2027, "available", "2027-01-15", "Future note", ""]])
        audit = self.assess(availability_path=self.availability, require_confirmed_availability=True)
        self.assertFalse(audit.loc["recent_ipl", "eligible_for_candidate_pool"])
        self.assertEqual(audit.loc["recent_ipl", "availability_status"], "unknown")

    def test_available_status_cannot_bypass_recent_history_requirement(self):
        self.write_notes([["inactive", 2027, "available", "2026-10-01", "Example note", ""]])
        audit = self.assess(availability_path=self.availability)
        self.assertFalse(audit.loc["inactive", "eligible_for_candidate_pool"])

    def test_unknown_is_explicit_and_strict_mode_rejects_it(self):
        normal = self.assess()
        strict = self.assess(require_confirmed_availability=True)
        self.assertEqual(normal.loc["recent_ipl", "availability_status"], "unknown")
        self.assertIn("assumed", normal.loc["recent_ipl", "availability_evidence"])
        self.assertIn("Not medically assessed", normal.loc["recent_ipl", "fitness_status"])
        self.assertFalse(strict["eligible_for_candidate_pool"].any())

    def test_international_retirement_is_not_silently_treated_as_ipl_retirement(self):
        # No format-retirement inference exists: recent IPL activity qualifies.
        self.assertTrue(self.assess().loc["recent_ipl", "eligible_for_candidate_pool"])
        self.write_notes([["recent_ipl", 2027, "international_retired", "2026-10-01", "Wrong kind of status", ""]])
        with self.assertRaisesRegex(ValueError, "International retirement alone"):
            self.assess(availability_path=self.availability)

    def test_duplicate_status_on_same_date_is_rejected(self):
        self.write_notes([
            ["recent_ipl", 2027, "available", "2026-10-01", "First", ""],
            ["recent_ipl", 2027, "ruled_out", "2026-10-01", "Conflicting", ""],
        ])
        with self.assertRaisesRegex(ValueError, "Duplicate availability"):
            self.assess(availability_path=self.availability)

    def test_header_only_status_file_is_supported(self):
        self.write_notes([])
        audit = self.assess(availability_path=self.availability)
        self.assertTrue(audit.loc["recent_ipl", "eligible_for_candidate_pool"])

    def test_invalid_window_or_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            self.assess(activity_years=0)
        with self.assertRaises(ValueError):
            self.assess(min_qualifying_matches=0)

    def test_full_pipeline_and_report_exclude_inactive_domestic_only_and_blocked_players(self):
        from test_model_workflow import create_mini_project, fast_forest
        from src.models import train_fair_value_model as fair
        from src.models import train_performance_model as performance
        from src.models.run_future_analysis import run_future_analysis
        from src.models.score_auction_candidates import score_auction_candidates
        from src.transformation.build_model_table import build_model_table
        from src.visualization.create_prediction_visualizations import create_prediction_visualizations

        root = Path(self.temp.name) / "project"
        create_mini_project(root)
        processed = root / "data" / "processed"
        master_path = processed / "player_master.csv"
        master = pd.read_csv(master_path)
        extra_master = pd.DataFrame([
            {"player_id": "domestic_only", "player": "Domestic Only", "country": "India", "batting_style": "Right hand Bat", "bowling_style": "Right arm Medium", "playing_role": "Batter"},
            {"player_id": "inactive_old", "player": "Old IPL Participant", "country": "India", "batting_style": "Right hand Bat", "bowling_style": "Right arm Medium", "playing_role": "Batter"},
        ])
        pd.concat([master, extra_master], ignore_index=True).to_csv(master_path, index=False)
        for directory, year, player_id, player in (("bbl", 2026, "domestic_only", "Domestic Only"), ("ipl", 2019, "inactive_old", "Old IPL Participant")):
            path = processed / directory / "batting_match.csv"
            source = pd.read_csv(path)
            row = source.iloc[0].copy()
            row["player_id"], row["player"] = player_id, player
            row["match_id"], row["date"] = f"{directory}_{player_id}_{year}", f"{year}-06-01"
            pd.concat([source, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
        pd.DataFrame([{
            "player_id": "batter_2", "forecast_year": 2027,
            "availability_status": "retired_from_ipl", "known_as_of": "2026-10-01",
            "reason": "Synthetic test retirement; not a real player status",
        }]).to_csv(root / "data" / "reference" / "player_availability.csv", index=False)
        build_model_table(root)
        with patch.object(fair, "RandomForestRegressor", fast_forest), patch.object(performance, "RandomForestRegressor", fast_forest):
            model, summary = run_future_analysis(root, pool_year=2026, forecast_year=2027, top_n=3)
        excluded = {"domestic_only", "inactive_old", "batter_2"}
        for filename in ("auction_candidate_fair_values.csv", "upcoming_performance_predictions.csv", "team_fit_Example_Team.csv", "team_role_shortlists_Example_Team.csv", "future_team_recommendations.csv"):
            frame = pd.read_csv(model / filename)
            self.assertTrue(excluded.isdisjoint(set(frame["player_id"])), filename)
        audit = pd.read_csv(model / "auction_candidate_eligibility.csv").set_index("player_id")
        self.assertEqual(audit.loc["domestic_only", "exclusion_reason"], "domestic_history_only")
        self.assertIn("availability_retired_from_ipl", audit.loc["batter_2", "exclusion_reason"])
        self.assertEqual(audit.loc["inactive_old", "exclusion_reason"], "no_recent_ipl_or_international_history")
        report, manifest = create_prediction_visualizations(root, top_n=3)
        self.assertEqual(manifest["forecast_players"], 7)
        self.assertTrue((report.parent / "candidate_eligibility.png").exists())
        represented = [row for row in manifest["source_files"] if row["file"] == "auction_candidate_eligibility.csv"]
        self.assertIn("candidate-pool screen", represented[0]["view"])
        # If a subsequent strict run has no confirmed candidates, don't leave
        # the old non-empty fair-value pool masquerading as its new result.
        with self.assertRaisesRegex(ValueError, "No players pass"):
            score_auction_candidates(root, pool_year=2026, auction_year=2027, require_confirmed_availability=True)
        self.assertTrue(pd.read_csv(model / "auction_candidate_fair_values.csv").empty)


if __name__ == "__main__":
    unittest.main()
