"""Estimate fair value for active players using the trained price model.

The fair-value model learns from historical sold-player prices.  For a future
analysis, the default pool requires recent IPL or supported international
participation. Domestic-only activity is not sufficient. Season-specific
retirements/withdrawals/ruled-out statuses can be excluded using a reference
file; recent match history alone does not verify next-season medical fitness.

Run after ``train_fair_value_model.py``:

    python -m src.models.score_auction_candidates --pool-year 2026 --auction-year 2027
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.common import ensure_directory, prepare_feature_frame
from src.models.candidate_eligibility import AUDIT_COLUMNS, assess_candidate_eligibility, load_country_policy
from src.transformation.build_model_table import competition_features, load_performance_tables
from src.transformation.cricket_etl import DATASETS, project_root_from_file


CANDIDATE_SCOPES = {"recent_ipl_or_international", "all_recent_active", "latest_sold_players"}


def role_from_metadata(value: object) -> str | None:
    """Convert rich player metadata to the broad auction roles used in training."""

    text = "" if pd.isna(value) else str(value).casefold()
    if "all" in text and "round" in text:
        return "All-rounder"
    if "wicket" in text or "keeper" in text:
        return "Wicketkeeper"
    if "bowl" in text:
        return "Bowler"
    if "bat" in text or "order" in text:
        return "Batter"
    return None


def build_candidate_pool(
    root: Path,
    auction: pd.DataFrame,
    pool_year: int,
    candidate_scope: str,
) -> pd.DataFrame:
    """Return either every known player or the legacy sold-player reference pool."""

    if candidate_scope not in CANDIDATE_SCOPES:
        choices = ", ".join(sorted(CANDIDATE_SCOPES))
        raise ValueError(f"candidate_scope must be one of: {choices}")

    if candidate_scope == "latest_sold_players":
        candidates = auction.loc[
            auction["auction_year"].eq(pool_year) & auction["player_id"].notna()
        ].copy()
        return candidates.sort_values("player_name_auction").drop_duplicates("player_id")

    master_path = root / "data" / "processed" / "player_master.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing {master_path}. Run player_master_etl.py first.")
    master = pd.read_csv(master_path, dtype={"player_id": "string"}).dropna(subset=["player_id", "player"])
    candidates = master[["player_id", "player", "country", "playing_role"]].copy()
    candidates = candidates.rename(columns={"player": "player_name_auction", "country": "nationality"})
    candidates["role"] = candidates["playing_role"].map(role_from_metadata)
    candidates["nationality"] = candidates["nationality"].fillna("Unknown")
    candidates["player_match_status"] = "metadata_candidate"
    return candidates.drop(columns="playing_role").drop_duplicates("player_id")


def build_candidate_features(
    root: Path,
    candidates: pd.DataFrame,
    auction_year: int,
    recent_years: int,
    performances: dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    """Create the same pre-auction performance features used during training."""

    performances = performances if performances is not None else load_performance_tables(root)
    cutoff_year = auction_year - 1
    start_date = pd.Timestamp(year=cutoff_year - recent_years + 1, month=1, day=1)
    end_date = pd.Timestamp(year=cutoff_year, month=12, day=31)
    features = candidates.copy()
    features["auction_year"] = auction_year
    features["performance_window_start"] = start_date.date().isoformat()
    features["performance_window_end"] = end_date.date().isoformat()

    for directory, (batting, bowling) in performances.items():
        competition = competition_features(batting, bowling, directory, start_date, end_date)
        features = features.merge(competition, on="player_id", how="left", validate="one_to_one")
        history_column = f"has_{directory}_history"
        if history_column not in features:
            features[history_column] = False
        else:
            features[history_column] = features[history_column].astype("boolean").fillna(False).astype(bool)

    history_columns = [
        f"has_{directory}_history" for directory in DATASETS if f"has_{directory}_history" in features
    ]
    features["has_any_recent_performance"] = features[history_columns].any(axis=1)

    master_path = root / "data" / "processed" / "player_master.csv"
    master = pd.read_csv(master_path, dtype={"player_id": "string"})
    context = master[
        ["player_id", "player", "country", "batting_style", "bowling_style", "playing_role"]
    ].rename(
        columns={
            "player": "player_name_canonical",
            "country": "metadata_country",
            "batting_style": "metadata_batting_style",
            "bowling_style": "metadata_bowling_style",
            "playing_role": "metadata_playing_role",
        }
    )
    features = features.drop(columns=["player_name_canonical"], errors="ignore").merge(
        context, on="player_id", how="left", validate="one_to_one"
    )
    return features


def score_auction_candidates(
    project_root: str | Path | None = None,
    pool_year: int | None = None,
    auction_year: int | None = None,
    recent_years: int = 3,
    market_median_inr: float | None = None,
    model_directory: str | Path | None = None,
    candidate_scope: str = "recent_ipl_or_international",
    activity_years: int = 2,
    min_qualifying_matches: int = 1,
    availability_path: str | Path | None = None,
    require_confirmed_availability: bool = False,
) -> dict[str, object]:
    """Score active or legacy sold-player candidates for a later auction year."""

    if recent_years < 1:
        raise ValueError("recent_years must be at least 1")
    if activity_years < 1 or min_qualifying_matches < 1:
        raise ValueError("activity_years and min_qualifying_matches must be at least 1")
    root = Path(project_root) if project_root else project_root_from_file()
    model_directory = Path(model_directory) if model_directory else root / "data" / "processed" / "model"
    artifact_path = model_directory / "fair_value_model.joblib"
    auction_path = root / "data" / "processed" / "auction" / "auction_clean.csv"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing {artifact_path}. Train the fair-value model first.")
    if not auction_path.exists():
        raise FileNotFoundError(f"Missing {auction_path}. Run auction_etl.py first.")

    artifact = joblib.load(artifact_path)
    auction = pd.read_csv(auction_path, dtype={"player_id": "string"})
    available_years = sorted(auction["auction_year"].dropna().astype(int).unique())
    selected_pool_year = pool_year or max(available_years)
    selected_auction_year = auction_year or selected_pool_year + 1
    candidates = build_candidate_pool(root, auction, selected_pool_year, candidate_scope)
    if candidates.empty:
        raise ValueError(f"No candidates are available for scope {candidate_scope!r}.")

    reference = Path(availability_path) if availability_path else root / "data" / "reference" / "player_availability.csv"
    if availability_path and not reference.is_absolute():
        reference = root / reference
    if availability_path and not reference.exists():
        raise FileNotFoundError(f"The requested availability reference does not exist: {reference}")
    performances = load_performance_tables(root)
    country_policy = load_country_policy(
        root / "data" / "reference" / "candidate_country_policy.json",
        selected_auction_year,
        required=candidate_scope == "recent_ipl_or_international",
    )
    activity = assess_candidate_eligibility(
        candidates, performances, selected_auction_year,
        activity_years=activity_years, min_qualifying_matches=min_qualifying_matches,
        availability_path=reference,
        require_confirmed_availability=require_confirmed_availability,
        require_recent_ipl_or_international=candidate_scope == "recent_ipl_or_international",
        country_policy=country_policy,
    )
    features = build_candidate_features(root, candidates, selected_auction_year, recent_years, performances=performances)
    features = features.drop(columns=[field for field in AUDIT_COLUMNS if field in features])
    features = features.merge(activity, on="player_id", how="left", validate="one_to_one")
    missing_form = features["eligible_for_candidate_pool"] & ~features["has_any_recent_performance"]
    features.loc[missing_form, "exclusion_reason"] = "no_model_feature_history_in_window"
    features.loc[missing_form, "eligibility_reason"] = "no_model_feature_history_in_window"
    features["eligible_for_candidate_pool"] &= features["has_any_recent_performance"]
    eligible = features["eligible_for_candidate_pool"]
    scored = features.loc[eligible].copy()
    output_directory = ensure_directory(model_directory)
    audit_columns = ["player_id", "player_name_canonical", "nationality", "role", *AUDIT_COLUMNS]
    features[audit_columns].to_csv(output_directory / "auction_candidate_eligibility.csv", index=False)
    features.loc[~eligible].to_csv(output_directory / "auction_candidate_excluded_players.csv", index=False)
    output_columns = [
        "player_id", "player_name_canonical", "role", "nationality",
        "metadata_playing_role", "has_any_recent_performance", *AUDIT_COLUMNS,
    ]
    if scored.empty:
        # Clear the primary candidate snapshot rather than leaving an old pool
        # that could be mistaken for the successful result of this failed run.
        pd.DataFrame(columns=[*output_columns, "source_pool_year", "candidate_pool_scope", "forecast_auction_year", "market_median_inr_assumption", "predicted_price_median_ratio", "predicted_fair_value_inr"]).to_csv(output_directory / "auction_candidate_fair_values.csv", index=False)
        raise ValueError(
            "No players pass the candidate-history/availability screen. "
            "See auction_candidate_eligibility.csv for exclusion reasons. "
            "Check source-data recency and any require-confirmed-availability setting; no recommendations were generated."
        )
    numeric_features = artifact["numeric_features"]
    categorical_features = artifact["categorical_features"]
    model_input = prepare_feature_frame(scored, numeric_features, categorical_features)
    ratio = np.exp(artifact["pipeline"].predict(model_input))
    baseline_median = market_median_inr or artifact["reference_market_median_inr"]

    output = scored[output_columns].copy()
    output["source_pool_year"] = selected_pool_year
    output["candidate_pool_scope"] = candidate_scope
    output["forecast_auction_year"] = selected_auction_year
    output["market_median_inr_assumption"] = baseline_median
    output["predicted_price_median_ratio"] = ratio
    output["predicted_fair_value_inr"] = (ratio * baseline_median).round(0)
    output = output.sort_values("predicted_fair_value_inr", ascending=False).reset_index(drop=True)

    output.to_csv(output_directory / "auction_candidate_fair_values.csv", index=False)
    return {
        "pool_players": len(candidates),
        "scored_players": len(output),
        "excluded_players": int((~eligible).sum()),
        "auction_year": selected_auction_year,
        "candidate_scope": candidate_scope,
        "activity_years": activity_years,
        "min_qualifying_matches": min_qualifying_matches,
        "activity_window_start": activity["activity_window_start"].iloc[0],
        "activity_window_end": activity["activity_window_end"].iloc[0],
        "require_confirmed_availability": require_confirmed_availability,
        "availability_reference_used": str(reference) if reference.exists() else None,
        "eligible_availability_unknown": int(output["availability_status"].eq("unknown").sum()),
        "excluded_reason_counts": features.loc[~eligible, "exclusion_reason"].value_counts().to_dict(),
        "country_policy": country_policy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--pool-year", type=int, default=None)
    parser.add_argument("--auction-year", type=int, default=None)
    parser.add_argument("--recent-years", type=int, default=3)
    parser.add_argument("--market-median-inr", type=float, default=None)
    parser.add_argument(
        "--candidate-scope",
        choices=sorted(CANDIDATE_SCOPES),
        default="recent_ipl_or_international",
        help="Default: recent IPL or supported international history. Other scopes retain legacy behaviour.",
    )
    parser.add_argument("--activity-years", type=int, default=2, help="Qualification window in completed calendar years, separate from the feature lookback.")
    parser.add_argument("--min-qualifying-matches", type=int, default=1)
    parser.add_argument("--availability-file", type=Path, default=None)
    parser.add_argument("--require-confirmed-availability", action="store_true", help="Only admit recent-history players explicitly marked available for the forecast year.")
    arguments = parser.parse_args()
    result = score_auction_candidates(
        arguments.project_root,
        arguments.pool_year,
        arguments.auction_year,
        arguments.recent_years,
        arguments.market_median_inr,
        candidate_scope=arguments.candidate_scope,
        activity_years=arguments.activity_years,
        min_qualifying_matches=arguments.min_qualifying_matches,
        availability_path=arguments.availability_file,
        require_confirmed_availability=arguments.require_confirmed_availability,
    )
    print(f"Scored {result['scored_players']:,}/{result['pool_players']:,} candidates for {result['auction_year']}.")
    print(f"Excluded by history/availability checks: {result['excluded_players']:,}")


if __name__ == "__main__":
    main()
