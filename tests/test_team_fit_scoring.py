"""Unit tests for transparent, role-aware team-fit scoring rules."""

import pandas as pd

from src.models.score_team_fit import (
    desired_role_capabilities,
    recommendation_from_score,
    role_aware_performance_score,
    role_capabilities,
)


def test_wicketkeeper_batter_is_eligible_for_batting_need():
    candidate = role_capabilities("Wicketkeeper", "Wicketkeeper Batter")
    desired = desired_role_capabilities("Batter")
    assert candidate.intersection(desired)


def test_profile_can_name_multiple_acceptable_roles():
    desired = desired_role_capabilities("Batter|Wicketkeeper")
    assert {"batter", "wicketkeeper"}.issubset(desired)


def test_batter_requirement_does_not_penalise_specialist_for_bowling():
    batting = pd.Series([0.8, 0.2])
    bowling = pd.Series([0.0, 1.0])
    result = role_aware_performance_score(batting, bowling, {"batter"})
    assert result.tolist() == [0.8, 0.2]


def test_fixed_recommendation_thresholds():
    assert recommendation_from_score(60.1) == "Strong fit"
    assert recommendation_from_score(60.0) == "Consider"
    assert recommendation_from_score(40.1) == "Consider"
    assert recommendation_from_score(40.0) == "Low fit"
