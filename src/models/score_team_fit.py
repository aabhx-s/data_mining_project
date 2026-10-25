"""Rank auction candidates against a transparent, user-defined team-needs profile.

This is deliberately an explainable decision aid, not a claim to know a team's
private strategy or squad availability. Edit ``data/reference/team_needs.csv``
before running it.

    python -m src.models.score_team_fit --team "Example Team"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.models.common import ensure_directory
from src.transformation.cricket_etl import project_root_from_file


REQUIRED_PROFILE_COLUMNS = {
    "team",
    "desired_role",
    "performance_weight",
    "value_weight",
    "batting_weight",
    "bowling_weight",
    "max_fair_value_inr",
}


ROLE_LABELS = {
    "all_rounder": "All-rounder",
    "batter": "Batter",
    "bowler": "Bowler",
    "wicketkeeper": "Wicketkeeper",
}


def normalise_role_text(value: object) -> str:
    """Convert a possibly missing source role into safe comparison text."""

    return "" if pd.isna(value) else str(value).lower().replace("-", " ")


def role_capabilities(*values: object) -> set[str]:
    """Return the cricketing capabilities described by one or more role labels.

    Auction files frequently call Ishan Kishan a ``Wicketkeeper`` while player
    metadata calls him a ``Wicketkeeper Batter``. Both descriptions should make
    him eligible for a batting requirement.  Similarly, an all-rounder should
    not be excluded from a batting or bowling requirement merely because the
    source uses a single broad label.
    """

    text = " ".join(normalise_role_text(value) for value in values)
    capabilities: set[str] = set()
    is_all_rounder = "all" in text and "round" in text

    if "wicket" in text or "keeper" in text:
        capabilities.update({"wicketkeeper", "batter"})
    if "bat" in text:
        capabilities.add("batter")
    if "bowl" in text:
        capabilities.add("bowler")
    if is_all_rounder:
        capabilities.add("all_rounder")
        # A generic all-rounder can contribute in both disciplines.  More
        # specific labels (Batting/Bowling All-rounder) are handled above too.
        capabilities.update({"batter", "bowler"})

    return capabilities


def desired_role_capabilities(value: object) -> set[str]:
    """Parse a profile role; multiple acceptable roles may use ``|`` or ``/``.

    Examples: ``Batter`` and ``Batter|Wicketkeeper``.  The latter is useful
    when a team needs a top-order batter and is open to a keeper-batter.
    """

    parts = re.split(r"[|/;]", normalise_role_text(value))
    capabilities: set[str] = set()
    for part in parts:
        text = normalise_role_text(part)
        # This is intentionally not the same as candidate capabilities. A
        # batter can satisfy a batting need, but cannot satisfy a request for
        # an all-rounder merely because an all-rounder can bat.
        if "all" in text and "round" in text:
            capabilities.add("all_rounder")
        elif "wicket" in text or "keeper" in text:
            capabilities.add("wicketkeeper")
        elif "bowl" in text:
            capabilities.add("bowler")
        elif "bat" in text:
            capabilities.add("batter")
    return capabilities


def percentile_score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Score candidates from 0 to 1 by their standing within this candidate pool.

    Percentile scoring is more stable than min-max scaling: one unusually high
    forecast cannot force almost every other candidate close to zero.
    """

    numeric = pd.to_numeric(values, errors="coerce")
    valid_count = int(numeric.notna().sum())
    if valid_count < 2:
        return pd.Series(0.5, index=values.index, dtype=float)

    ascending_rank = numeric.rank(method="average", ascending=True)
    if higher_is_better:
        scores = ascending_rank / valid_count
    else:
        scores = (valid_count - ascending_rank + 1) / valid_count
    return scores.fillna(0.5).astype(float)


def role_aware_performance_score(
    batting_score: pd.Series,
    bowling_score: pd.Series,
    desired_capabilities: set[str],
) -> pd.Series:
    """Use the performance dimension that actually matters for the team need."""

    needs_batting = bool(desired_capabilities.intersection({"batter", "wicketkeeper"}))
    needs_bowling = "bowler" in desired_capabilities
    needs_all_rounder = "all_rounder" in desired_capabilities

    if needs_batting and not needs_bowling and not needs_all_rounder:
        return batting_score
    if needs_bowling and not needs_batting and not needs_all_rounder:
        return bowling_score
    return (batting_score + bowling_score) / 2


def recommendation_from_score(score: float) -> str:
    """Map the documented fixed thresholds to a human-readable label."""

    if score > 60:
        return "Strong fit"
    if score > 40:
        return "Consider"
    return "Low fit"


def investment_label(score: float) -> str:
    """Turn the performance-for-price score into an interpretable label."""

    if score > 70:
        return "High value"
    if score > 45:
        return "Fair value"
    return "Expensive / limited value"


def primary_role_group(role: object, metadata_role: object) -> str:
    """Assign each candidate to one non-overlapping auction-shortlist group."""

    text = f"{normalise_role_text(role)} {normalise_role_text(metadata_role)}"
    if "all" in text and "round" in text:
        return "All-rounder"
    if "wicket" in text or "keeper" in text:
        return "Wicketkeeper-batter"
    if "bowl" in text:
        return "Bowler"
    if "bat" in text or "order" in text:
        return "Batter"
    return "Other"


def build_role_shortlists(ranking: pd.DataFrame, players_per_role: int = 5) -> pd.DataFrame:
    """Return the highest-ranked affordable targets in each primary role group."""

    if players_per_role < 1:
        raise ValueError("players_per_role must be at least 1")
    shortlist = ranking.copy()
    shortlist["shortlist_role"] = shortlist["primary_role_group"]
    role_order = ["Batter", "Wicketkeeper-batter", "Bowler", "All-rounder"]
    shortlist = shortlist.loc[shortlist["shortlist_role"].isin(role_order)].copy()
    shortlist["role_order"] = pd.Categorical(shortlist["shortlist_role"], categories=role_order, ordered=True)
    shortlist = shortlist.sort_values(
        ["role_order", "within_budget", "team_fit_score", "investment_score"],
        ascending=[True, False, False, False],
    )
    shortlist["role_rank"] = shortlist.groupby("shortlist_role").cumcount() + 1
    shortlist = shortlist.loc[shortlist["role_rank"].le(players_per_role)].copy()
    columns = [
        "team", "shortlist_role", "role_rank", "player_id", "player_name_canonical",
        "team_fit_score", "recommendation", "investment_score", "investment_label",
        "expected_runs", "expected_runs_conservative", "expected_runs_upside",
        "expected_wickets", "expected_wickets_conservative", "expected_wickets_upside",
        "predicted_fair_value_inr", "within_budget",
        "role_match", "candidate_role_capabilities", "reason",
    ]
    columns.extend(column for column in ("pre_budget_fit_score", "max_fair_value_inr") if column in shortlist)
    columns.extend(column for column in (
        "recent_ipl_matches", "recent_international_matches", "last_ipl_match_date",
        "last_international_match_date", "availability_status", "availability_evidence",
        "fitness_status", "eligibility_reason",
    ) if column in shortlist)
    shortlist = shortlist.sort_values(["role_order", "role_rank"])
    return shortlist[columns].reset_index(drop=True)


def role_based_investment_performance(row: pd.Series) -> float:
    """Select the output dimension that defines value for the player's own role."""

    if row["primary_role_group"] == "Bowler":
        return float(row["bowling_score"])
    if row["primary_role_group"] == "All-rounder":
        return float((row["batting_score"] + row["bowling_score"]) / 2)
    return float(row["batting_score"])


def load_team_profile(path: Path, team: str) -> pd.Series:
    """Load exactly one needs profile for the requested team."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Copy and edit data/reference/team_needs_template.csv first."
        )
    profiles = pd.read_csv(path)
    missing = REQUIRED_PROFILE_COLUMNS.difference(profiles.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    matches = profiles.loc[
        profiles["team"].astype(str).str.strip().str.casefold().eq(team.strip().casefold())
    ]
    if len(matches) != 1:
        available = ", ".join(profiles["team"].astype(str).tolist()) or "none"
        raise ValueError(f"Expected one profile for {team!r}. Available teams: {available}")
    return matches.iloc[0]


def score_team_fit(
    team: str,
    project_root: str | Path | None = None,
    team_needs_path: str | Path | None = None,
    model_directory: str | Path | None = None,
) -> pd.DataFrame:
    """Score known auction-pool candidates for one team's stated needs and budget."""

    root = Path(project_root) if project_root else project_root_from_file()
    model_directory = Path(model_directory) if model_directory else root / "data" / "processed" / "model"
    value_path = model_directory / "auction_candidate_fair_values.csv"
    performance_path = model_directory / "upcoming_performance_predictions.csv"
    profile_path = Path(team_needs_path) if team_needs_path else root / "data" / "reference" / "team_needs.csv"
    if not value_path.exists() or not performance_path.exists():
        raise FileNotFoundError(
            "Missing candidate fair values or performance predictions. "
            "Run score_auction_candidates.py and train_performance_model.py first."
        )

    profile = load_team_profile(profile_path, team)
    performance = pd.read_csv(performance_path, dtype={"player_id": "string"})
    performance_columns = [
        "player_id", "expected_runs", "expected_wickets", "expected_batting_matches", "expected_bowling_matches",
        "expected_runs_conservative", "expected_runs_upside",
        "expected_wickets_conservative", "expected_wickets_upside",
    ]
    available_performance_columns = [column for column in performance_columns if column in performance.columns]
    candidates = pd.read_csv(value_path, dtype={"player_id": "string"}).merge(
        performance[available_performance_columns],
        on="player_id",
        how="inner",
        validate="one_to_one",
    )
    if candidates.empty:
        raise ValueError("No candidates appear in both the value and performance prediction outputs.")

    desired_capabilities = desired_role_capabilities(profile["desired_role"])
    if not desired_capabilities:
        raise ValueError(
            "desired_role must describe at least one of Batter, Bowler, "
            "Wicketkeeper or All-rounder. Multiple acceptable roles may use '|', "
            "for example Batter|Wicketkeeper."
        )

    candidates["candidate_role_capabilities"] = candidates.apply(
        lambda row: " | ".join(
            ROLE_LABELS[capability]
            for capability in sorted(role_capabilities(row["role"], row["metadata_playing_role"]))
        )
        or "Unknown",
        axis=1,
    )
    candidate_capabilities = candidates.apply(
        lambda row: role_capabilities(row["role"], row["metadata_playing_role"]), axis=1
    )
    candidates["role_match"] = candidate_capabilities.map(
        lambda capabilities: bool(capabilities.intersection(desired_capabilities))
    )
    # Missing role data should reduce confidence, but must not discard a player
    # purely because a source omitted a role. A known incompatible role receives
    # a clearer penalty.
    candidates["role_alignment"] = candidate_capabilities.map(
        lambda capabilities: 1.0
        if capabilities.intersection(desired_capabilities)
        else 0.75
        if not capabilities
        else 0.55
    )

    candidates["batting_score"] = percentile_score(candidates["expected_runs"])
    candidates["bowling_score"] = percentile_score(candidates["expected_wickets"])
    candidates["primary_role_group"] = candidates.apply(
        lambda row: primary_role_group(row["role"], row["metadata_playing_role"]), axis=1
    )
    candidates["performance_score"] = role_aware_performance_score(
        candidates["batting_score"], candidates["bowling_score"], desired_capabilities
    )
    # Investment score is role-specific but team-independent: a bowler is judged
    # on predicted wickets, a batter on predicted runs, and an all-rounder on
    # both. It identifies market value; team fit and budget remain separate.
    candidates["market_affordability_score"] = percentile_score(
        candidates["predicted_fair_value_inr"], higher_is_better=False
    )
    candidates["investment_performance_score"] = candidates.apply(
        role_based_investment_performance, axis=1
    )
    candidates["investment_score"] = (
        100 * (0.70 * candidates["investment_performance_score"] + 0.30 * candidates["market_affordability_score"])
    ).round(1)
    candidates["investment_label"] = candidates["investment_score"].map(investment_label)

    weights = pd.Series(
        {
            "performance_score": float(profile["performance_weight"]),
            "value_score": float(profile["value_weight"]),
            "batting_score": float(profile["batting_weight"]),
            "bowling_score": float(profile["bowling_weight"]),
        }
    )
    if weights.isna().any() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Team-profile weights must be numeric, non-negative and sum to more than zero.")
    weights = weights / weights.sum()

    max_value = pd.to_numeric(pd.Series([profile["max_fair_value_inr"]]), errors="coerce").iloc[0]
    if pd.notna(max_value) and max_value <= 0:
        raise ValueError("max_fair_value_inr must be a positive INR amount or blank.")
    candidates["max_fair_value_inr"] = max_value
    candidates["within_budget"] = candidates["predicted_fair_value_inr"].le(max_value) if pd.notna(max_value) else True
    # A player at the stated budget remains viable (score 0.50), while a
    # cheaper player gets more value credit. This avoids treating a fair but
    # expensive player as poor value solely because another candidate is cheap.
    if pd.notna(max_value) and max_value > 0:
        budget_share = (candidates["predicted_fair_value_inr"] / max_value).clip(lower=0, upper=1)
        candidates["value_score"] = 1 - (0.5 * budget_share)
    else:
        candidates["value_score"] = percentile_score(
            candidates["predicted_fair_value_inr"], higher_is_better=False
        )

    weighted_score = sum(candidates[column] * weight for column, weight in weights.items())
    # Preserve the weighted fit before the hard budget block. An expensive
    # player can be a good potential fit while remaining ineligible under the
    # team's current price limit. Keep final purchase rankings unchanged.
    candidates["pre_budget_fit_score"] = (100 * weighted_score * candidates["role_alignment"]).round(1)
    candidates["team_fit_score"] = candidates["pre_budget_fit_score"]
    candidates.loc[~candidates["within_budget"], "team_fit_score"] = 0.0
    candidates["recommendation"] = candidates["team_fit_score"].map(recommendation_from_score)
    for measure in (
        "expected_runs_conservative", "expected_runs_upside",
        "expected_wickets_conservative", "expected_wickets_upside",
    ):
        if measure not in candidates:
            candidates[measure] = candidates["expected_runs"] if "runs" in measure else candidates["expected_wickets"]
    candidates["reason"] = candidates.apply(
        lambda row: (
            f"Expected {row.expected_runs:.0f} runs ({row.expected_runs_conservative:.0f}-{row.expected_runs_upside:.0f}) "
            f"and {row.expected_wickets:.1f} wickets ({row.expected_wickets_conservative:.1f}-{row.expected_wickets_upside:.1f}); "
            f"fair value INR {row.predicted_fair_value_inr:,.0f}; "
            f"fit before budget gate {row.pre_budget_fit_score:.1f}; "
            f"{row.investment_label.lower()} investment score {row.investment_score:.1f}; "
            f"{'role match' if row.role_match else 'role mismatch'}; "
            f"{'within budget' if row.within_budget else 'over budget'}"
        ),
        axis=1,
    )
    output = candidates.sort_values(
        ["within_budget", "team_fit_score", "predicted_fair_value_inr"], ascending=[False, False, True]
    ).reset_index(drop=True)
    output.insert(0, "team", team)
    output.insert(1, "rank", range(1, len(output) + 1))

    output_directory = ensure_directory(model_directory)
    safe_team = "".join(character if character.isalnum() else "_" for character in team).strip("_")
    output.to_csv(output_directory / f"team_fit_{safe_team}.csv", index=False)
    role_shortlists = build_role_shortlists(output)
    role_shortlists.to_csv(output_directory / f"team_role_shortlists_{safe_team}.csv", index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", required=True, help="Team name exactly as entered in team_needs.csv")
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--team-needs", type=Path, default=None)
    arguments = parser.parse_args()
    output = score_team_fit(arguments.team, arguments.project_root, arguments.team_needs)
    print(f"Saved {len(output):,} ranked candidates for {arguments.team}.")
    print(output[["rank", "player_name_canonical", "team_fit_score", "recommendation"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
