"""Pre-budget fit must remain visible without bypassing purchase constraints."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.models.score_team_fit import score_team_fit
from test_prediction_visualizations import create_fixture


class BudgetScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = create_fixture(self.root)
        reference = self.root / "data" / "reference"
        reference.mkdir()
        self.profile_path = reference / "team_needs.csv"
        self.profile = {
            "team": "Team One", "desired_role": "Batter",
            "performance_weight": 0.45, "value_weight": 0.2,
            "batting_weight": 0.3, "bowling_weight": 0.05,
            "max_fair_value_inr": 50_000_000,
        }
        pd.DataFrame([self.profile]).to_csv(self.profile_path, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_over_budget_player_keeps_potential_but_final_score_is_zero(self):
        ranking = score_team_fit("Team One", self.root)
        batter = ranking.set_index("player_id").loc["bat"]
        self.assertFalse(bool(batter["within_budget"]))
        self.assertGreater(batter["pre_budget_fit_score"], 60)
        self.assertEqual(batter["team_fit_score"], 0)
        self.assertEqual(batter["recommendation"], "Low fit")
        self.assertEqual(batter["max_fair_value_inr"], 50_000_000)
        self.assertIn("fit before budget gate", batter["reason"])
        # A candidate right at the cap remains allowed.
        keeper = ranking.set_index("player_id").loc["keep"]
        self.assertTrue(bool(keeper["within_budget"]))
        self.assertEqual(keeper["team_fit_score"], keeper["pre_budget_fit_score"])
        affordable = ranking["within_budget"].astype(bool).tolist()
        self.assertEqual(affordable, sorted(affordable, reverse=True))

    def test_original_weighted_score_and_rank_policy_are_unchanged(self):
        ranking = score_team_fit("Team One", self.root)
        expected = 100 * ranking["role_alignment"] * (
            0.45 * ranking["performance_score"] + 0.2 * ranking["value_score"]
            + 0.3 * ranking["batting_score"] + 0.05 * ranking["bowling_score"]
        )
        pd.testing.assert_series_equal(
            ranking["pre_budget_fit_score"], expected.round(1), check_names=False
        )
        final = expected.round(1).where(ranking["within_budget"], 0.0)
        pd.testing.assert_series_equal(ranking["team_fit_score"], final, check_names=False)
        shortlist = pd.read_csv(self.model / "team_role_shortlists_Team_One.csv")
        self.assertIn("pre_budget_fit_score", shortlist)
        self.assertIn("max_fair_value_inr", shortlist)

    def test_blank_price_limit_does_not_block_anyone(self):
        self.profile["max_fair_value_inr"] = None
        pd.DataFrame([self.profile]).to_csv(self.profile_path, index=False)
        ranking = score_team_fit("Team One", self.root)
        self.assertTrue(ranking["within_budget"].all())
        self.assertTrue(ranking["max_fair_value_inr"].isna().all())
        pd.testing.assert_series_equal(
            ranking["team_fit_score"], ranking["pre_budget_fit_score"], check_names=False
        )


if __name__ == "__main__":
    unittest.main()
