"""Create a leakage-safe player-auction training table from cleaned ETL outputs.

Run after the cricket, player-master, feature and auction ETL steps:

    python -m src.transformation.build_model_table --recent-years 3

Each auction row only receives performances dated before 1 January of its auction
year. This is deliberately conservative because the raw auction files identify a
year but not the precise auction date.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.transformation.cricket_etl import DATASETS, project_root_from_file


def safe_ratio(numerator: pd.Series, denominator: pd.Series, multiplier: float = 1.0) -> pd.Series:
    """Calculate ratios without introducing infinity for players with no attempts."""

    return numerator.div(denominator.replace(0, np.nan)).mul(multiplier)


def load_performance_tables(project_root: Path) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Load every processed competition table required for recent-form features."""

    tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for directory in DATASETS:
        base = project_root / "data" / "processed" / directory
        batting = pd.read_csv(base / "batting_match.csv", dtype={"player_id": "string"})
        bowling = pd.read_csv(base / "bowling_match.csv", dtype={"player_id": "string"})
        for name, table in {"batting": batting, "bowling": bowling}.items():
            if "player_id" not in table.columns:
                raise ValueError(f"{base / (name + '_match.csv')} has no player_id. Run cricket_etl.py first.")
            # Processed files can contain ISO dates and legacy day-first dates
            # such as 31-03-24. Parsing each value explicitly prevents an
            # ambiguous date from entering the pre-auction feature window.
            table["date"] = pd.to_datetime(
                table["date"], format="mixed", dayfirst=True, errors="coerce"
            )
            if table["date"].isna().any():
                raise ValueError(f"{directory} {name} data has invalid match dates")
        tables[directory] = (batting, bowling)
    return tables


def competition_features(
    batting: pd.DataFrame,
    bowling: pd.DataFrame,
    prefix: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate one competition in a fixed pre-auction window by canonical player ID."""

    batting = batting.loc[batting["date"].between(start_date, end_date)].copy()
    bowling = bowling.loc[bowling["date"].between(start_date, end_date)].copy()
    batting = batting.dropna(subset=["player_id"])
    bowling = bowling.dropna(subset=["player_id"])

    batting_aggregate = (
        batting.groupby("player_id", as_index=False)
        .agg(
            **{
                f"{prefix}_batting_matches": ("match_id", "nunique"),
                f"{prefix}_runs": ("runs", "sum"),
                f"{prefix}_balls": ("balls", "sum"),
                f"{prefix}_fours": ("fours", "sum"),
                f"{prefix}_sixes": ("sixes", "sum"),
            }
        )
    )
    bowling_aggregate = (
        bowling.groupby("player_id", as_index=False)
        .agg(
            **{
                f"{prefix}_bowling_matches": ("match_id", "nunique"),
                f"{prefix}_bowling_balls": ("balls", "sum"),
                f"{prefix}_runs_conceded": ("runs_conceded", "sum"),
                f"{prefix}_wickets": ("wickets", "sum"),
            }
        )
    )
    features = batting_aggregate.merge(bowling_aggregate, on="player_id", how="outer")
    if features.empty:
        return pd.DataFrame(columns=["player_id"])

    for column in features.columns:
        if column != "player_id":
            features[column] = features[column].fillna(0)
    features[f"{prefix}_strike_rate"] = safe_ratio(
        features[f"{prefix}_runs"], features[f"{prefix}_balls"], 100
    ).round(2)
    features[f"{prefix}_economy"] = safe_ratio(
        features[f"{prefix}_runs_conceded"], features[f"{prefix}_bowling_balls"], 6
    ).round(2)
    features[f"{prefix}_bowling_strike_rate"] = safe_ratio(
        features[f"{prefix}_bowling_balls"], features[f"{prefix}_wickets"]
    ).round(2)
    features[f"has_{prefix}_history"] = True
    return features


def build_model_table(project_root: str | Path | None = None, recent_years: int = 3) -> pd.DataFrame:
    """Build one model-ready row per sold-player auction observation."""

    if recent_years < 1:
        raise ValueError("recent_years must be at least one")

    root = Path(project_root) if project_root else project_root_from_file()
    auction_path = root / "data" / "processed" / "auction" / "auction_clean.csv"
    if not auction_path.exists():
        raise FileNotFoundError(f"Missing {auction_path}. Run auction_etl.py first.")
    auction = pd.read_csv(auction_path, dtype={"player_id": "string"})
    performances = load_performance_tables(root)

    master_path = root / "data" / "processed" / "player_master.csv"
    master = pd.read_csv(master_path, dtype={"player_id": "string"})
    player_context = master[
        ["player_id", "country", "batting_style", "bowling_style", "playing_role"]
    ].rename(
        columns={
            "country": "metadata_country",
            "batting_style": "metadata_batting_style",
            "bowling_style": "metadata_bowling_style",
            "playing_role": "metadata_playing_role",
        }
    )

    cohorts: list[pd.DataFrame] = []
    for auction_year, cohort in auction.groupby("auction_year", sort=True):
        cohort = cohort.copy()
        cutoff_year = int(auction_year) - 1
        start_date = pd.Timestamp(year=cutoff_year - recent_years + 1, month=1, day=1)
        end_date = pd.Timestamp(year=cutoff_year, month=12, day=31)
        cohort["performance_window_start"] = start_date.date().isoformat()
        cohort["performance_window_end"] = end_date.date().isoformat()

        for directory, (batting, bowling) in performances.items():
            feature_table = competition_features(batting, bowling, directory, start_date, end_date)
            cohort = cohort.merge(feature_table, on="player_id", how="left", validate="many_to_one")
            history_flag = f"has_{directory}_history"
            if history_flag not in cohort:
                cohort[history_flag] = False
            cohort[history_flag] = cohort[history_flag].fillna(False)
        cohorts.append(cohort)

    model_table = pd.concat(cohorts, ignore_index=True).merge(
        player_context, on="player_id", how="left", validate="many_to_one"
    )
    history_columns = [column for column in model_table if column.startswith("has_") and column.endswith("_history")]
    model_table["has_any_recent_performance"] = model_table[history_columns].any(axis=1)
    model_table["eligible_for_price_model"] = (
        model_table["player_id"].notna()
        & model_table["price_inr"].notna()
        & model_table["has_any_recent_performance"]
    )

    output_directory = root / "data" / "processed" / "model"
    output_directory.mkdir(parents=True, exist_ok=True)
    model_table.to_csv(output_directory / "player_auction_training.csv", index=False)
    model_table.loc[~model_table["eligible_for_price_model"]].to_csv(
        output_directory / "player_auction_excluded_rows.csv", index=False
    )
    return model_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--recent-years", type=int, default=3)
    arguments = parser.parse_args()
    table = build_model_table(arguments.project_root, arguments.recent_years)
    print(f"Saved {len(table):,} auction observations.")
    print(f"Eligible for a price model: {table['eligible_for_price_model'].sum():,}")


if __name__ == "__main__":
    main()
