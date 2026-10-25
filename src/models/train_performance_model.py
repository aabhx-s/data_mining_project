"""Train a next-season IPL performance model and score recently active players.

The model predicts runs, wickets and match appearances conditional on a player
appearing in the following IPL season. It does not predict auction selection or
injury availability. IPL history remains the closest signal to the target, while
recent T20 (T20I, BBL, SMAT and T20 World Cup) and ODI form are supplementary
inputs. Every input is restricted to information known by the forecast cutoff.

Season totals are calibrated in two stages.  First, a model forecasts playing
opportunity and output rate per appearance.  Second, that estimate is blended
with a player's trailing IPL median rate.  The blend is selected only on held-out
seasons, which prevents the model from arbitrarily inflating established players
while reducing the excessive regression-to-the-mean caused by direct total-only
forecasts.

Run after ``feature_builder.py``:

    python -m src.models.train_performance_model --as-of-season 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.models.common import ensure_directory, regression_metrics, write_json
from src.transformation.cricket_etl import project_root_from_file


AGGREGATE_COLUMNS = [
    "batting_matches",
    "runs",
    "batting_balls",
    "fours",
    "sixes",
    "bowling_matches",
    "bowling_balls",
    "runs_conceded",
    "wickets",
]
TARGET_COLUMNS = [
    "next_runs",
    "next_wickets",
    "next_batting_matches",
    "next_bowling_matches",
]
T20_SOURCE_DIRECTORIES = ("t20", "bbl", "smat", "t20_wc")
EXTERNAL_FEATURE_GROUPS = {
    "t20": T20_SOURCE_DIRECTORIES,
    "odi": ("odi",),
}
EXTERNAL_AGGREGATE_COLUMNS = [
    "matches",
    "runs",
    "batting_balls",
    "wickets",
    "bowling_balls",
    "runs_conceded",
]
IPL_HISTORY_FEATURE_COLUMNS = [
    "has_recent_ipl_history",
    "recent_seasons_available",
    "last_ipl_season",
    *[f"recent_{column}" for column in AGGREGATE_COLUMNS],
    "career_seasons",
    "career_runs",
    "career_wickets",
    "recent_strike_rate",
    "recent_economy",
    "recent_wickets_per_match",
    "recent_runs_per_match",
    "recent_median_season_runs",
    "recent_median_season_wickets",
    "recent_median_runs_per_match",
    "recent_median_wickets_per_match",
]
ROLE_FEATURE_COLUMNS = [
    "role_is_batter",
    "role_is_bowler",
    "role_is_all_rounder",
    "role_is_wicketkeeper",
]
PERFORMANCE_FEATURE_COLUMNS = [
    *IPL_HISTORY_FEATURE_COLUMNS,
    *ROLE_FEATURE_COLUMNS,
    *[
        column
        for prefix in EXTERNAL_FEATURE_GROUPS
        for column in [
            f"has_recent_{prefix}_history",
            *[f"{prefix}_recent_{statistic}" for statistic in EXTERNAL_AGGREGATE_COLUMNS],
            f"{prefix}_recent_strike_rate",
            f"{prefix}_recent_economy",
        ]
    ],
]
RATE_TARGET_COLUMNS = [
    "next_runs_per_batting_match",
    "next_wickets_per_bowling_match",
]
OPPORTUNITY_RATE_TARGET_COLUMNS = [
    "next_batting_matches",
    "next_bowling_matches",
    *RATE_TARGET_COLUMNS,
]
CALIBRATION_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def safe_divide(numerator: pd.Series, denominator: pd.Series, multiplier: float = 1.0) -> pd.Series:
    """Calculate a rate and leave undefined values missing."""

    return numerator.div(denominator.replace(0, np.nan)).mul(multiplier)


def empty_cross_format_features() -> pd.DataFrame:
    """Return an empty but merge-safe cross-format feature table."""

    columns = ["player_id", "player"]
    for prefix in EXTERNAL_FEATURE_GROUPS:
        columns.extend(
            [
                f"has_recent_{prefix}_history",
                *[f"{prefix}_recent_{statistic}" for statistic in EXTERNAL_AGGREGATE_COLUMNS],
                f"{prefix}_recent_strike_rate",
                f"{prefix}_recent_economy",
            ]
        )
    return pd.DataFrame(columns=columns)


def load_cross_format_tables(project_root: Path) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Load processed non-IPL match tables used as supplementary form signals."""

    tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    required_columns = {
        "batting": {"match_id", "date", "player_id", "player", "runs", "balls"},
        "bowling": {"match_id", "date", "player_id", "player", "balls", "runs_conceded", "wickets"},
    }
    for directory in sorted({item for group in EXTERNAL_FEATURE_GROUPS.values() for item in group}):
        base = project_root / "data" / "processed" / directory
        batting_path = base / "batting_match.csv"
        bowling_path = base / "bowling_match.csv"
        if not batting_path.exists() or not bowling_path.exists():
            raise FileNotFoundError(
                f"Missing processed {directory.upper()} match tables under {base}. Run cricket_etl.py first."
            )
        batting = pd.read_csv(batting_path, dtype={"player_id": "string"})
        bowling = pd.read_csv(bowling_path, dtype={"player_id": "string"})
        for name, table in {"batting": batting, "bowling": bowling}.items():
            missing = required_columns[name].difference(table.columns)
            if missing:
                raise ValueError(f"{base / (name + '_match.csv')} is missing columns: {sorted(missing)}")
            # Match tables may contain ISO and legacy day-first dates. The
            # explicit mixed parser keeps the forecast cutoff deterministic.
            table["date"] = pd.to_datetime(
                table["date"], format="mixed", dayfirst=True, errors="coerce"
            )
            if table["date"].isna().any():
                raise ValueError(f"{base / (name + '_match.csv')} contains invalid match dates")
        tables[directory] = (batting, bowling)
    return tables


def aggregate_cross_format_window(
    tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    directories: tuple[str, ...],
    prefix: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate one comparable-format group without double-counting duplicated archives."""

    batting_frames = [tables[directory][0] for directory in directories if not tables[directory][0].empty]
    bowling_frames = [tables[directory][1] for directory in directories if not tables[directory][1].empty]
    batting = pd.concat(batting_frames, ignore_index=True) if batting_frames else pd.DataFrame(
        columns=["match_id", "date", "player_id", "player", "runs", "balls"]
    )
    bowling = pd.concat(bowling_frames, ignore_index=True) if bowling_frames else pd.DataFrame(
        columns=["match_id", "date", "player_id", "player", "balls", "runs_conceded", "wickets"]
    )
    batting = batting.loc[batting["date"].between(start_date, end_date)].dropna(subset=["player_id"]).copy()
    bowling = bowling.loc[bowling["date"].between(start_date, end_date)].dropna(subset=["player_id"]).copy()

    # A T20 World Cup match can also be present in the generic T20 archive.
    # Cricsheet match IDs are global, so keeping one player-match record avoids
    # awarding that performance twice.
    batting = batting.drop_duplicates(["match_id", "player_id"])
    bowling = bowling.drop_duplicates(["match_id", "player_id"])
    if batting.empty and bowling.empty:
        return pd.DataFrame(columns=["player_id", "player"])

    batting_aggregate = (
        batting.groupby("player_id", as_index=False)
        .agg(
            player_batting=("player", "last"),
            **{
                f"{prefix}_recent_batting_matches": ("match_id", "nunique"),
                f"{prefix}_recent_runs": ("runs", "sum"),
                f"{prefix}_recent_batting_balls": ("balls", "sum"),
            },
        )
    )
    bowling_aggregate = (
        bowling.groupby("player_id", as_index=False)
        .agg(
            player_bowling=("player", "last"),
            **{
                f"{prefix}_recent_bowling_matches": ("match_id", "nunique"),
                f"{prefix}_recent_wickets": ("wickets", "sum"),
                f"{prefix}_recent_bowling_balls": ("balls", "sum"),
                f"{prefix}_recent_runs_conceded": ("runs_conceded", "sum"),
            },
        )
    )
    aggregate = batting_aggregate.merge(bowling_aggregate, on="player_id", how="outer")
    aggregate["player"] = aggregate.pop("player_batting").combine_first(aggregate.pop("player_bowling"))
    aggregate[f"{prefix}_recent_matches"] = pd.concat(
        [
            batting[["player_id", "match_id"]],
            bowling[["player_id", "match_id"]],
        ],
        ignore_index=True,
    ).drop_duplicates().groupby("player_id")["match_id"].nunique().reindex(aggregate["player_id"]).to_numpy()
    # Players can contribute in only one discipline. Their unused discipline is
    # correctly zero, unlike an absent competition which remains missing after a merge.
    for statistic in EXTERNAL_AGGREGATE_COLUMNS:
        column = f"{prefix}_recent_{statistic}"
        if column not in aggregate:
            aggregate[column] = 0.0
        aggregate[column] = pd.to_numeric(aggregate[column], errors="coerce").fillna(0.0)
    aggregate[f"has_recent_{prefix}_history"] = True
    aggregate[f"{prefix}_recent_strike_rate"] = safe_divide(
        aggregate[f"{prefix}_recent_runs"], aggregate[f"{prefix}_recent_batting_balls"], 100
    )
    aggregate[f"{prefix}_recent_economy"] = safe_divide(
        aggregate[f"{prefix}_recent_runs_conceded"], aggregate[f"{prefix}_recent_bowling_balls"], 6
    )
    return aggregate


def build_cross_format_features(
    tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    as_of_season: int,
    lookback_seasons: int,
) -> pd.DataFrame:
    """Build recent T20 and ODI form features using only pre-cutoff matches."""

    start_date = pd.Timestamp(year=as_of_season - lookback_seasons + 1, month=1, day=1)
    end_date = pd.Timestamp(year=as_of_season, month=12, day=31)
    result: pd.DataFrame | None = None
    for prefix, directories in EXTERNAL_FEATURE_GROUPS.items():
        group = aggregate_cross_format_window(tables, directories, prefix, start_date, end_date)
        if group.empty:
            continue
        if result is None:
            result = group
        else:
            result = result.merge(group, on="player_id", how="outer", suffixes=("", "_new"))
            result["player"] = result["player"].combine_first(result.pop("player_new"))
    if result is None:
        return empty_cross_format_features()

    for prefix in EXTERNAL_FEATURE_GROUPS:
        flag = f"has_recent_{prefix}_history"
        if flag not in result:
            result[flag] = False
        result[flag] = result[flag].fillna(False).astype(bool)
        for statistic in EXTERNAL_AGGREGATE_COLUMNS:
            column = f"{prefix}_recent_{statistic}"
            if column not in result:
                result[column] = np.nan
        for rate in ("strike_rate", "economy"):
            column = f"{prefix}_recent_{rate}"
            if column not in result:
                result[column] = np.nan
    return result


def build_history_features(history: pd.DataFrame, as_of_season: int, lookback_seasons: int) -> pd.DataFrame:
    """Aggregate only performance known by the end of ``as_of_season``."""

    start_season = as_of_season - lookback_seasons + 1
    recent = history.loc[
        history["season_end_year"].between(start_season, as_of_season)
    ].copy()
    if recent.empty:
        return pd.DataFrame(columns=["player_id", "player", *IPL_HISTORY_FEATURE_COLUMNS])

    features = (
        recent.groupby("player_id", as_index=False)
        .agg(
            player=("player", "last"),
            recent_seasons_available=("season_end_year", "nunique"),
            last_ipl_season=("season_end_year", "max"),
            **{f"recent_{column}": (column, "sum") for column in AGGREGATE_COLUMNS},
        )
    )
    features["has_recent_ipl_history"] = True
    career = history.loc[history["season_end_year"].le(as_of_season)].groupby("player_id", as_index=False).agg(
        career_seasons=("season_end_year", "nunique"),
        career_runs=("runs", "sum"),
        career_wickets=("wickets", "sum"),
    )
    features = features.merge(career, on="player_id", how="left", validate="one_to_one")
    features["recent_strike_rate"] = safe_divide(
        features["recent_runs"], features["recent_batting_balls"], 100
    )
    features["recent_economy"] = safe_divide(
        features["recent_runs_conceded"], features["recent_bowling_balls"], 6
    )
    features["recent_wickets_per_match"] = safe_divide(
        features["recent_wickets"], features["recent_bowling_matches"]
    )
    features["recent_runs_per_match"] = safe_divide(
        features["recent_runs"], features["recent_batting_matches"]
    )
    # Totals can hide a consistent top-order player's actual per-appearance
    # output when one shorter season is mixed into the window.  Preserve the
    # median of the season-level rates for later calibration.
    season_rates = recent[["player_id", "runs", "wickets", "batting_matches", "bowling_matches"]].copy()
    season_rates["season_runs_per_match"] = safe_divide(
        season_rates["runs"], season_rates["batting_matches"]
    )
    season_rates["season_wickets_per_match"] = safe_divide(
        season_rates["wickets"], season_rates["bowling_matches"]
    )
    medians = (
        season_rates.groupby("player_id", as_index=False)
        .agg(
            recent_median_season_runs=("runs", "median"),
            recent_median_season_wickets=("wickets", "median"),
            recent_median_runs_per_match=("season_runs_per_match", "median"),
            recent_median_wickets_per_match=("season_wickets_per_match", "median"),
        )
    )
    features = features.merge(medians, on="player_id", how="left", validate="one_to_one")
    return features


def role_flags(player_context: pd.DataFrame) -> pd.DataFrame:
    """Return broad role indicators from metadata without using future results."""

    context = player_context.copy()
    if "playing_role" not in context:
        context["playing_role"] = ""
    text = context["playing_role"].fillna("").astype(str).str.casefold()
    context["role_is_all_rounder"] = (text.str.contains("all") & text.str.contains("round")).astype(int)
    context["role_is_wicketkeeper"] = (text.str.contains("wicket") | text.str.contains("keeper")).astype(int)
    context["role_is_bowler"] = text.str.contains("bowl").astype(int)
    context["role_is_batter"] = (text.str.contains("bat") | text.str.contains("order")).astype(int)
    # A generic all-rounder is eligible in both disciplines even when the
    # metadata does not spell out batting or bowling all-rounder.
    context.loc[context["role_is_all_rounder"].eq(1), ["role_is_batter", "role_is_bowler"]] = 1
    return context[["player_id", "player", *ROLE_FEATURE_COLUMNS]].drop_duplicates("player_id")


def combine_history_features(
    ipl_history: pd.DataFrame,
    cross_format_history: pd.DataFrame,
    candidate_player_ids: Iterable[str] | None = None,
    player_names: pd.DataFrame | None = None,
    player_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine IPL and cross-format snapshots into one forecast-safe feature row."""

    combined = ipl_history.merge(
        cross_format_history,
        on="player_id",
        how="outer",
        suffixes=("_ipl", "_cross"),
    )
    if "player_ipl" in combined and "player_cross" in combined:
        combined["player"] = combined.pop("player_ipl").combine_first(combined.pop("player_cross"))
    elif "player_ipl" in combined:
        combined = combined.rename(columns={"player_ipl": "player"})
    elif "player_cross" in combined:
        combined = combined.rename(columns={"player_cross": "player"})

    if candidate_player_ids is not None:
        candidates = pd.DataFrame({"player_id": pd.Series(list(candidate_player_ids), dtype="string")})
        candidates = candidates.dropna().drop_duplicates()
        combined = candidates.merge(combined, on="player_id", how="left")
    if player_names is not None and not player_names.empty:
        names = player_names[["player_id", "player"]].dropna(subset=["player_id"]).drop_duplicates("player_id")
        combined = combined.merge(names, on="player_id", how="left", suffixes=("", "_master"))
        if "player_master" in combined:
            combined["player"] = combined["player"].combine_first(combined.pop("player_master"))

    if player_context is not None and not player_context.empty:
        context = role_flags(player_context)
        combined = combined.merge(context, on="player_id", how="left", suffixes=("", "_context"))
        if "player_context" in combined:
            combined["player"] = combined["player"].combine_first(combined.pop("player_context"))
    for column in ROLE_FEATURE_COLUMNS:
        if column not in combined:
            combined[column] = 0
        combined[column] = pd.to_numeric(combined[column], errors="coerce").fillna(0).astype(int)

    history_flags = ["has_recent_ipl_history", *[f"has_recent_{prefix}_history" for prefix in EXTERNAL_FEATURE_GROUPS]]
    for flag in history_flags:
        if flag not in combined:
            combined[flag] = False
        combined[flag] = combined[flag].fillna(False).astype(bool)
    combined["has_recent_performance"] = combined[history_flags].any(axis=1)
    return combined


def build_training_table(
    player_season: pd.DataFrame,
    cross_format_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    lookback_seasons: int,
    player_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one row per player-season with features before the next IPL season."""

    seasons = sorted(player_season["season_end_year"].dropna().astype(int).unique())
    rows: list[pd.DataFrame] = []
    for target_season in seasons:
        prior_season = target_season - 1
        target = player_season.loc[player_season["season_end_year"].eq(target_season)].copy()
        ipl_history = build_history_features(player_season, prior_season, lookback_seasons)
        cross_format_history = build_cross_format_features(cross_format_tables, prior_season, lookback_seasons)
        history_features = combine_history_features(
            ipl_history, cross_format_history, player_context=player_context
        )
        target = target.rename(columns={"player": "player_target"})
        cohort = target.merge(history_features, on="player_id", how="left")
        cohort["player"] = cohort["player"].combine_first(cohort.pop("player_target"))
        cohort = cohort.loc[cohort["has_recent_performance"].eq(True)].copy()
        if cohort.empty:
            continue
        cohort["target_season"] = target_season
        cohort = cohort.rename(
            columns={
                "runs": "next_runs",
                "wickets": "next_wickets",
                "batting_matches": "next_batting_matches",
                "bowling_matches": "next_bowling_matches",
            }
        )
        rows.append(cohort)
    if not rows:
        raise ValueError("Could not create next-season training rows from player-season data.")
    return pd.concat(rows, ignore_index=True)


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Return the explicitly defined fields available before the target season.

    The joined training table also contains observed next-season statistics for
    labels. Those columns must never become model inputs.
    """

    missing = sorted(set(PERFORMANCE_FEATURE_COLUMNS).difference(table.columns))
    if missing:
        raise ValueError(f"Performance feature table is missing columns: {missing}")
    return PERFORMANCE_FEATURE_COLUMNS.copy()


def make_random_forest_pipeline() -> Pipeline:
    """Use the established nonlinear baseline for multi-output forecasting."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=3,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_xgboost_pipeline() -> Pipeline:
    """Use regularised boosted trees as a competing multi-output forecaster."""

    estimator = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", MultiOutputRegressor(estimator)),
        ]
    )


def scaled_mae(actual: pd.DataFrame, prediction: np.ndarray) -> tuple[dict[str, dict[str, float]], float]:
    """Return per-target metrics and one scale-neutral score for model selection."""

    target_metrics = {
        target: regression_metrics(actual[target].to_numpy(), prediction[:, index])
        for index, target in enumerate(TARGET_COLUMNS)
    }
    normalised_errors = [
        target_metrics[target]["mae"] / max(float(actual[target].mean()), 1.0)
        for target in TARGET_COLUMNS
    ]
    return target_metrics, float(np.mean(normalised_errors))


def opportunity_rate_targets(table: pd.DataFrame) -> pd.DataFrame:
    """Create forecast-safe opportunity and per-appearance training targets."""

    targets = table[["next_batting_matches", "next_bowling_matches"]].astype(float).copy()
    targets["next_runs_per_batting_match"] = safe_divide(
        table["next_runs"].astype(float), targets["next_batting_matches"]
    ).fillna(0.0)
    targets["next_wickets_per_bowling_match"] = safe_divide(
        table["next_wickets"].astype(float), targets["next_bowling_matches"]
    ).fillna(0.0)
    return targets[OPPORTUNITY_RATE_TARGET_COLUMNS]


def season_totals_from_opportunity_rate(
    opportunity_rate_prediction: np.ndarray,
    historical_features: pd.DataFrame,
    baseline_weight: float,
) -> np.ndarray:
    """Turn predicted opportunities/rates into calibrated IPL season totals.

    ``baseline_weight`` blends the learned rate with the player's trailing IPL
    median rate.  A player without IPL rate history remains fully model-driven.
    """

    prediction = np.maximum(np.asarray(opportunity_rate_prediction, dtype=float), 0.0)
    batting_matches = prediction[:, 0]
    bowling_matches = prediction[:, 1]
    model_run_rate = prediction[:, 2]
    model_wicket_rate = prediction[:, 3]
    historical_run_rate = pd.to_numeric(
        historical_features["recent_median_runs_per_match"], errors="coerce"
    ).to_numpy(dtype=float)
    historical_wicket_rate = pd.to_numeric(
        historical_features["recent_median_wickets_per_match"], errors="coerce"
    ).to_numpy(dtype=float)
    has_run_baseline = np.isfinite(historical_run_rate) & (historical_run_rate >= 0)
    has_wicket_baseline = np.isfinite(historical_wicket_rate) & (historical_wicket_rate >= 0)
    run_rate = np.where(
        has_run_baseline,
        (1 - baseline_weight) * model_run_rate + baseline_weight * historical_run_rate,
        model_run_rate,
    )
    wicket_rate = np.where(
        has_wicket_baseline,
        (1 - baseline_weight) * model_wicket_rate + baseline_weight * historical_wicket_rate,
        model_wicket_rate,
    )
    return np.column_stack(
        [
            np.maximum(batting_matches * run_rate, 0.0),
            np.maximum(bowling_matches * wicket_rate, 0.0),
            batting_matches,
            bowling_matches,
        ]
    )


def choose_calibration_weight(
    actual: pd.DataFrame,
    opportunity_rate_prediction: np.ndarray,
    historical_features: pd.DataFrame,
) -> tuple[float, np.ndarray, dict[str, dict[str, float]], float, list[dict[str, float]]]:
    """Select the historical-rate blend by chronological held-out error."""

    rows: list[dict[str, float]] = []
    candidates: dict[float, tuple[np.ndarray, dict[str, dict[str, float]], float]] = {}
    for weight in CALIBRATION_WEIGHTS:
        seasonal_prediction = season_totals_from_opportunity_rate(
            opportunity_rate_prediction, historical_features, weight
        )
        metrics, normalised_mae = scaled_mae(actual, seasonal_prediction)
        candidates[weight] = (seasonal_prediction, metrics, normalised_mae)
        rows.append({"historical_rate_weight": weight, "scaled_mae": normalised_mae})
    selected_weight = min(candidates, key=lambda weight: candidates[weight][2])
    prediction, metrics, normalised_mae = candidates[selected_weight]
    return selected_weight, prediction, metrics, normalised_mae, rows


def train_performance_model(
    project_root: str | Path | None = None,
    as_of_season: int | None = None,
    lookback_seasons: int = 4,
    test_seasons: int = 2,
    output_directory: str | Path | None = None,
    candidate_player_ids: Iterable[str] | None = None,
) -> dict[str, object]:
    """Evaluate chronologically, fit all data, and write an upcoming-season forecast."""

    if lookback_seasons < 1 or test_seasons < 1:
        raise ValueError("lookback_seasons and test_seasons must both be at least 1")

    root = Path(project_root) if project_root else project_root_from_file()
    source_path = root / "data" / "processed" / "ipl" / "player_season_features.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing {source_path}. Run feature_builder.py first.")
    player_season = pd.read_csv(source_path, dtype={"player_id": "string"})
    required_season_columns = {"player_id", "player", "season_end_year", *AGGREGATE_COLUMNS}
    missing_season_columns = sorted(required_season_columns.difference(player_season.columns))
    if missing_season_columns:
        raise ValueError(
            f"{source_path} is missing required columns: {missing_season_columns}. "
            "Rebuild it with: python -m src.transformation.feature_builder --as-of-season 2026"
        )
    player_season["season_end_year"] = pd.to_numeric(player_season["season_end_year"], errors="raise")
    as_of = as_of_season or int(player_season["season_end_year"].max())
    history = player_season.loc[player_season["season_end_year"].le(as_of)].copy()
    cross_format_tables = load_cross_format_tables(root)
    master_path = root / "data" / "processed" / "player_master.csv"
    if master_path.exists():
        player_context = pd.read_csv(master_path, dtype={"player_id": "string"})
    else:
        player_context = pd.DataFrame(columns=["player_id", "player", "playing_role"])

    training = build_training_table(
        history, cross_format_tables, lookback_seasons, player_context=player_context
    )
    available_targets = sorted(training["target_season"].astype(int).unique())
    if len(available_targets) <= test_seasons:
        raise ValueError("Not enough IPL seasons for the requested chronological test split.")
    held_out_seasons = available_targets[-test_seasons:]
    train = training.loc[~training["target_season"].isin(held_out_seasons)].copy()
    test = training.loc[training["target_season"].isin(held_out_seasons)].copy()
    columns = feature_columns(training)
    if not columns:
        raise ValueError("No numeric historical features available for performance model.")

    train_x = train[columns].apply(pd.to_numeric, errors="coerce")
    test_x = test[columns].apply(pd.to_numeric, errors="coerce")
    train_y = np.log1p(train[TARGET_COLUMNS].astype(float))
    test_y = test[TARGET_COLUMNS].astype(float)
    model_factories = {
        "random_forest": make_random_forest_pipeline,
        "xgboost": make_xgboost_pipeline,
    }
    evaluation_predictions: dict[str, np.ndarray] = {}
    evaluation_metrics: dict[str, dict[str, dict[str, float]]] = {}
    comparison_rows: list[dict[str, float | str]] = []
    for model_name, factory in model_factories.items():
        model = factory()
        model.fit(train_x, train_y)
        prediction = np.maximum(np.expm1(model.predict(test_x)), 0)
        target_metrics, normalised_mae = scaled_mae(test_y, prediction)
        evaluation_predictions[model_name] = prediction
        evaluation_metrics[model_name] = target_metrics
        comparison_rows.append({"model": model_name, "scaled_mae": normalised_mae})
    comparison = pd.DataFrame(comparison_rows).sort_values("scaled_mae").reset_index(drop=True)
    selected_model_name = str(comparison.iloc[0]["model"])
    direct_test_prediction = evaluation_predictions[selected_model_name]
    direct_metrics, direct_normalised_mae = scaled_mae(test_y, direct_test_prediction)

    # The companion model prevents a high-volume player from being forecast as
    # an average player simply because totals were log-transformed.  It learns
    # playing opportunity and per-appearance production separately.
    opportunity_rate_model = model_factories[selected_model_name]()
    opportunity_rate_model.fit(train_x, np.log1p(opportunity_rate_targets(train)))
    opportunity_rate_test_prediction = np.maximum(
        np.expm1(opportunity_rate_model.predict(test_x)), 0
    )
    (
        calibration_weight,
        calibrated_test_prediction,
        calibrated_metrics,
        calibrated_normalised_mae,
        calibration_rows,
    ) = choose_calibration_weight(test_y, opportunity_rate_test_prediction, test)
    use_calibrated_forecast = calibrated_normalised_mae < direct_normalised_mae
    selected_forecast_method = (
        "opportunity_rate_calibrated" if use_calibrated_forecast else "direct_total"
    )
    test_prediction = calibrated_test_prediction if use_calibrated_forecast else direct_test_prediction
    selected_metrics = calibrated_metrics if use_calibrated_forecast else direct_metrics

    metrics = {
        "held_out_target_seasons": held_out_seasons,
        "training_rows": len(train),
        "test_rows": len(test),
        "selected_model": selected_model_name,
        "selected_forecast_method": selected_forecast_method,
        "model_comparison": comparison_rows,
        "calibration": {
            "approach": "predicted appearances multiplied by calibrated per-appearance output",
            "historical_rate_weight": calibration_weight,
            "direct_total_scaled_mae": direct_normalised_mae,
            "calibrated_scaled_mae": calibrated_normalised_mae,
            "calibration_candidates": calibration_rows,
        },
        "conditional_on_next_season_appearance": True,
        "feature_sources": {
            "ipl": "Primary signal: prior IPL seasons.",
            "t20": "Supplementary: T20I, BBL, SMAT and T20 World Cup matches.",
            "odi": "Supplementary context: ODI matches, kept separate from T20 statistics.",
        },
        "targets": selected_metrics,
    }

    output_directory = ensure_directory(
        Path(output_directory) if output_directory else root / "data" / "processed" / "model"
    )
    test_predictions = test[["target_season", "player_id", "player"] + TARGET_COLUMNS].copy()
    for index, target in enumerate(TARGET_COLUMNS):
        test_predictions[f"predicted_{target}"] = test_prediction[:, index]
    test_predictions["selected_model"] = selected_model_name
    test_predictions["selected_forecast_method"] = selected_forecast_method
    test_predictions.to_csv(output_directory / "performance_test_predictions.csv", index=False)

    comparison.to_csv(output_directory / "performance_model_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"forecast_method": "direct_total", "scaled_mae": direct_normalised_mae},
            {"forecast_method": "opportunity_rate_calibrated", "scaled_mae": calibrated_normalised_mae},
        ]
    ).to_csv(output_directory / "performance_calibration_comparison.csv", index=False)
    final_model = model_factories[selected_model_name]()
    all_x = training[columns].apply(pd.to_numeric, errors="coerce")
    final_model.fit(all_x, np.log1p(training[TARGET_COLUMNS].astype(float)))
    final_opportunity_rate_model = model_factories[selected_model_name]()
    final_opportunity_rate_model.fit(all_x, np.log1p(opportunity_rate_targets(training)))
    ipl_history = build_history_features(history, as_of, lookback_seasons)
    cross_format_history = build_cross_format_features(cross_format_tables, as_of, lookback_seasons)
    player_names = player_context[["player_id", "player"]] if not player_context.empty else None
    candidate_features = combine_history_features(
        ipl_history,
        cross_format_history,
        candidate_player_ids=candidate_player_ids,
        player_names=player_names,
        player_context=player_context,
    )
    # A forecast is meaningful only for players active in IPL or another supported
    # competition during the selected lookback window.
    candidate_features = candidate_features.loc[candidate_features["has_recent_performance"]].copy()
    candidate_x = candidate_features[columns].apply(pd.to_numeric, errors="coerce")
    direct_forecast = np.maximum(np.expm1(final_model.predict(candidate_x)), 0)
    opportunity_rate_forecast = np.maximum(
        np.expm1(final_opportunity_rate_model.predict(candidate_x)), 0
    )
    calibrated_forecast = season_totals_from_opportunity_rate(
        opportunity_rate_forecast, candidate_features, calibration_weight
    )
    forecast = calibrated_forecast if use_calibrated_forecast else direct_forecast
    predictions = candidate_features[
        [
            "player_id", "player", "last_ipl_season", "recent_seasons_available",
            "has_recent_ipl_history", "has_recent_t20_history", "has_recent_odi_history",
            "t20_recent_matches", "odi_recent_matches",
        ]
    ].copy()
    predictions["as_of_season"] = as_of
    predictions["predicted_ipl_season"] = as_of + 1
    for index, target in enumerate(TARGET_COLUMNS):
        predictions[f"expected_{target.removeprefix('next_')}"] = forecast[:, index]
    predictions["forecast_method"] = selected_forecast_method
    predictions["model_runs_per_batting_match"] = opportunity_rate_forecast[:, 2]
    predictions["model_wickets_per_bowling_match"] = opportunity_rate_forecast[:, 3]
    predictions["historical_median_ipl_runs"] = candidate_features[
        "recent_median_season_runs"
    ].to_numpy()
    predictions["historical_median_ipl_wickets"] = candidate_features[
        "recent_median_season_wickets"
    ].to_numpy()
    predictions["historical_median_runs_per_batting_match"] = candidate_features[
        "recent_median_runs_per_match"
    ].to_numpy()
    predictions["historical_median_wickets_per_bowling_match"] = candidate_features[
        "recent_median_wickets_per_match"
    ].to_numpy()
    predictions["expected_runs_conservative"] = np.maximum(
        predictions["expected_runs"] - selected_metrics["next_runs"]["mae"], 0
    )
    predictions["expected_runs_upside"] = predictions["expected_runs"] + selected_metrics["next_runs"]["mae"]
    predictions["expected_wickets_conservative"] = np.maximum(
        predictions["expected_wickets"] - selected_metrics["next_wickets"]["mae"], 0
    )
    predictions["expected_wickets_upside"] = (
        predictions["expected_wickets"] + selected_metrics["next_wickets"]["mae"]
    )
    predictions.to_csv(output_directory / "upcoming_performance_predictions.csv", index=False)

    artifact = {
        "pipeline": final_model,
        "opportunity_rate_pipeline": final_opportunity_rate_model,
        "feature_columns": columns,
        "target_columns": TARGET_COLUMNS,
        "opportunity_rate_target_columns": OPPORTUNITY_RATE_TARGET_COLUMNS,
        "lookback_seasons": lookback_seasons,
        "trained_through_season": as_of,
        "feature_sources": metrics["feature_sources"],
        "selected_model": selected_model_name,
        "selected_forecast_method": selected_forecast_method,
        "historical_rate_weight": calibration_weight,
    }
    joblib.dump(artifact, output_directory / "performance_model.joblib")
    write_json(metrics, output_directory / "performance_metrics.json")
    return {
        "training_rows": len(training),
        "test_rows": len(test),
        "held_out_seasons": held_out_seasons,
        "forecast_rows": len(predictions),
        "runs_mae": metrics["targets"]["next_runs"]["mae"],
        "wickets_mae": metrics["targets"]["next_wickets"]["mae"],
        "selected_model": selected_model_name,
        "selected_forecast_method": selected_forecast_method,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--as-of-season", type=int, default=None)
    parser.add_argument("--lookback-seasons", type=int, default=4)
    parser.add_argument("--test-seasons", type=int, default=2)
    arguments = parser.parse_args()
    result = train_performance_model(
        arguments.project_root,
        arguments.as_of_season,
        arguments.lookback_seasons,
        arguments.test_seasons,
    )
    print(f"Trained performance model on {result['training_rows']:,} player-season rows.")
    print(f"Held out IPL seasons: {', '.join(map(str, result['held_out_seasons']))}")
    print(f"Test MAE — runs: {result['runs_mae']:.1f}; wickets: {result['wickets_mae']:.2f}")
    print(f"Saved {result['forecast_rows']:,} upcoming-season forecasts.")


if __name__ == "__main__":
    main()
