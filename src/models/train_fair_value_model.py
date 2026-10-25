"""Train a leakage-aware IPL auction fair-value model.

The target is a player's sale price divided by that auction year's median sale
price. This removes most year-to-year market growth. At scoring time the ratio
is converted to INR using a chosen auction-market median.

Run after ``build_model_table.py``:

    python -m src.models.train_fair_value_model --test-years 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.models.common import (
    ensure_directory,
    make_preprocessor,
    prepare_feature_frame,
    regression_metrics,
    to_boolean,
    write_json,
)
from src.transformation.cricket_etl import project_root_from_file


CATEGORICAL_FEATURES = [
    "role",
    "nationality",
    "metadata_country",
    "metadata_batting_style",
    "metadata_bowling_style",
    "metadata_playing_role",
]

NON_FEATURE_COLUMNS = {
    "auction_year",  # The target is already normalised by each year's market median.
    "player_id",
    "player_name_auction",
    "player_name_canonical",
    "player_match_status",
    "team",  # The actual buying team must not influence fair player value.
    "price_original",
    "price_currency",
    "exchange_rate_to_inr",
    "price_inr",
    "price_inr_year_median_ratio",  # This is the target, not an input.
    "source_file",
    "performance_window_start",
    "performance_window_end",
    "eligible_for_price_model",
}


def choose_features(table: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Choose only pre-auction performance/context fields for the value model."""

    categorical = [column for column in CATEGORICAL_FEATURES if column in table.columns]
    numeric = [
        column
        for column in table.columns
        if column not in NON_FEATURE_COLUMNS
        and column not in categorical
        and (pd.api.types.is_numeric_dtype(table[column]) or column.startswith("has_"))
    ]
    if not numeric:
        raise ValueError("No numeric pre-auction features were found in the model table.")
    return sorted(numeric), sorted(categorical)


def make_random_forest_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """Build the established nonlinear baseline."""

    return Pipeline(
        [
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
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


def make_xgboost_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """Build a regularised gradient-boosting competitor for tabular features."""

    return Pipeline(
        [
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                XGBRegressor(
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
                ),
            ),
        ]
    )


def feature_importance_table(model: Pipeline) -> pd.DataFrame:
    """Expose transformed-feature importance for report writing and debugging."""

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    return (
        pd.DataFrame({"feature": names, "importance": estimator.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_fair_value_model(
    project_root: str | Path | None = None,
    test_years: int = 2,
    through_auction_year: int | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate chronologically, then fit and save the final fair-value model."""

    if test_years < 1:
        raise ValueError("test_years must be at least 1")

    root = Path(project_root) if project_root else project_root_from_file()
    source_path = root / "data" / "processed" / "model" / "player_auction_training.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing {source_path}. Run build_model_table.py first.")

    table = pd.read_csv(source_path, dtype={"player_id": "string"})
    eligible = to_boolean(table["eligible_for_price_model"])
    table = table.loc[eligible & table["price_inr_year_median_ratio"].gt(0)].copy()
    if through_auction_year is not None:
        table = table.loc[table["auction_year"].le(through_auction_year)].copy()
    if table.empty:
        raise ValueError("No eligible price-model rows are available.")

    years = sorted(pd.to_numeric(table["auction_year"], errors="coerce").dropna().astype(int).unique())
    if len(years) <= test_years:
        raise ValueError("Not enough auction years for the requested chronological test split.")
    held_out_years = years[-test_years:]
    train = table.loc[~table["auction_year"].isin(held_out_years)].copy()
    test = table.loc[table["auction_year"].isin(held_out_years)].copy()

    numeric_features, categorical_features = choose_features(table)
    target_column = "price_inr_year_median_ratio"
    train_features = prepare_feature_frame(train, numeric_features, categorical_features)
    test_features = prepare_feature_frame(test, numeric_features, categorical_features)
    train_target = np.log(train[target_column].astype(float))
    test_target = np.log(test[target_column].astype(float))

    test_medians = test["price_inr"].div(test[target_column]).to_numpy(dtype=float)
    model_factories = {
        "random_forest": make_random_forest_pipeline,
        "xgboost": make_xgboost_pipeline,
    }
    evaluation_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    comparison_rows: list[dict[str, float | str]] = []
    for model_name, factory in model_factories.items():
        model = factory(numeric_features, categorical_features)
        model.fit(train_features, train_target)
        predicted_ratio = np.exp(model.predict(test_features))
        predicted_inr = predicted_ratio * test_medians
        evaluation_predictions[model_name] = (predicted_ratio, predicted_inr)
        ratio_metrics = regression_metrics(test[target_column].to_numpy(), predicted_ratio)
        price_metrics = regression_metrics(test["price_inr"].to_numpy(), predicted_inr)
        comparison_rows.append(
            {
                "model": model_name,
                "ratio_mae": ratio_metrics["mae"],
                "ratio_rmse": ratio_metrics["rmse"],
                "price_mae_inr": price_metrics["mae"],
                "price_rmse_inr": price_metrics["rmse"],
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values("ratio_mae").reset_index(drop=True)
    selected_model_name = str(comparison.iloc[0]["model"])
    predicted_ratio, predicted_inr = evaluation_predictions[selected_model_name]
    metrics = {
        "test_years": held_out_years,
        "training_rows": len(train),
        "test_rows": len(test),
        "selected_model": selected_model_name,
        "model_comparison": comparison_rows,
        "median_price_ratio": regression_metrics(test[target_column].to_numpy(), predicted_ratio),
        "price_inr": regression_metrics(test["price_inr"].to_numpy(), predicted_inr),
    }

    output_directory = ensure_directory(
        Path(output_directory) if output_directory else root / "data" / "processed" / "model"
    )
    predictions = test[
        ["auction_year", "player_id", "player_name_auction", "price_inr", target_column]
    ].copy()
    predictions["predicted_price_median_ratio"] = predicted_ratio
    predictions["predicted_fair_value_inr"] = predicted_inr.round(0)
    predictions["selected_model"] = selected_model_name
    predictions["absolute_error_inr"] = (predictions["price_inr"] - predictions["predicted_fair_value_inr"]).abs()
    predictions.to_csv(output_directory / "fair_value_test_predictions.csv", index=False)

    comparison.to_csv(output_directory / "fair_value_model_comparison.csv", index=False)
    # Refit the selected model on every eligible historical row after chronological validation.
    final_model = model_factories[selected_model_name](numeric_features, categorical_features)
    all_features = prepare_feature_frame(table, numeric_features, categorical_features)
    final_model.fit(all_features, np.log(table[target_column].astype(float)))
    latest_year = max(years)
    reference_median = float(table.loc[table["auction_year"].eq(latest_year), "price_inr"].median())
    artifact = {
        "pipeline": final_model,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": "log_price_inr_year_median_ratio",
        "reference_auction_year": latest_year,
        "reference_market_median_inr": reference_median,
        "selected_model": selected_model_name,
    }
    joblib.dump(artifact, output_directory / "fair_value_model.joblib")
    feature_importance_table(final_model).to_csv(
        output_directory / "fair_value_feature_importance.csv", index=False
    )
    write_json(metrics, output_directory / "fair_value_metrics.json")

    return {
        "training_rows": len(table),
        "test_rows": len(test),
        "held_out_years": held_out_years,
        "ratio_mae": metrics["median_price_ratio"]["mae"],
        "price_mae_inr": metrics["price_inr"]["mae"],
        "selected_model": selected_model_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--test-years", type=int, default=2)
    parser.add_argument(
        "--through-auction-year",
        type=int,
        default=None,
        help="Exclude later auction rows; use this for a historical backtest.",
    )
    arguments = parser.parse_args()
    result = train_fair_value_model(
        arguments.project_root,
        arguments.test_years,
        arguments.through_auction_year,
    )
    print(f"Trained fair-value model on {result['training_rows']:,} rows.")
    print(f"Held out auction years: {', '.join(map(str, result['held_out_years']))}")
    print(f"Test price MAE: INR {result['price_mae_inr']:,.0f}")


if __name__ == "__main__":
    main()
