"""Regression checks for the role labels shown on the auction candidate board."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.visualization.create_model_visualizations import (
    candidate_role_group,
    plot_role_aware_candidate_board,
    select_role_aware_candidates,
)


class CandidateRoleBoardTests(unittest.TestCase):
    def setUp(self):
        self.values = pd.DataFrame(
            [
                ["kohli", "V Kohli", "Batter", "Top order Batter", 150_000_000],
                ["gill", "Shubman Gill", "Batter", "Opening Batter", 160_000_000],
                ["klaasen", "H Klaasen", "Wicketkeeper", "Wicketkeeper Batter", 90_000_000],
                ["keeper", "Another Keeper", "Wicketkeeper", None, 40_000_000],
                ["bowler", "Example Bowler", "Bowler", "Bowler", 70_000_000],
                ["allrounder", "Example All-rounder", "All-rounder", "Batting Allrounder", 80_000_000],
            ],
            columns=["player_id", "player_name_canonical", "role", "metadata_playing_role", "predicted_fair_value_inr"],
        )
        self.predictions = pd.DataFrame(
            [
                ["kohli", 559.0, 0.0],
                ["gill", 563.0, 0.0],
                ["klaasen", 478.0, 0.0],
                ["keeper", 250.0, 0.0],
                ["bowler", 10.0, 18.0],
                ["allrounder", 240.0, 12.0],
            ],
            columns=["player_id", "expected_runs", "expected_wickets"],
        )

    def test_batting_does_not_imply_wicketkeeping(self):
        for role in ("Batter", "Opening Batter", "Top order Batter", "Middle order Batter"):
            with self.subTest(role=role):
                self.assertEqual(candidate_role_group(role), "Batter")
        self.assertEqual(candidate_role_group("Wicketkeeper Batter"), "Wicketkeeper-batter")
        self.assertEqual(candidate_role_group("Bowler"), "Bowler")
        self.assertEqual(candidate_role_group("Bowling Allrounder"), "All-rounder")

    def test_missing_metadata_uses_auction_role(self):
        self.assertEqual(candidate_role_group(None, "Wicketkeeper"), "Wicketkeeper-batter")
        self.assertEqual(candidate_role_group(pd.NA, "Batter"), "Batter")
        self.assertEqual(candidate_role_group("Unknown", "Bowler"), "Bowler")
        self.assertEqual(candidate_role_group(None, None), "Other")

    def test_each_role_has_its_own_shortlist(self):
        board = select_role_aware_candidates(self.values, self.predictions, players_per_role=1)
        self.assertEqual(
            board["role_group"].tolist(),
            ["Batter", "Wicketkeeper-batter", "Bowler", "All-rounder"],
        )
        self.assertEqual(board["player_id"].tolist(), ["gill", "klaasen", "bowler", "allrounder"])
        self.assertEqual(board["player_id"].nunique(), len(board))

    def test_existing_prediction_values_are_not_changed(self):
        original_values = self.values.copy(deep=True)
        original_predictions = self.predictions.copy(deep=True)
        board = select_role_aware_candidates(self.values, self.predictions)
        indexed = board.set_index("player_id")
        self.assertEqual(indexed.loc["kohli", "role_group"], "Batter")
        self.assertEqual(indexed.loc["gill", "role_group"], "Batter")
        self.assertEqual(indexed.loc["kohli", "expected_runs"], 559.0)
        pd.testing.assert_frame_equal(self.values, original_values)
        pd.testing.assert_frame_equal(self.predictions, original_predictions)

    def test_chart_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            chart = plot_role_aware_candidate_board(self.values, self.predictions, Path(directory))
            self.assertEqual(chart.name, "role_aware_candidate_board.png")
            self.assertGreater(chart.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
