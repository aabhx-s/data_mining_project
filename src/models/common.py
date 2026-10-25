"""Shared utilities for reproducible cricket-model training and scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def ensure_directory(path: Path) -> Path:
    """Create and return an output directory."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write human-readable metrics/configuration with NumPy values converted safely."""

    def convert(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialise {type(value).__name__}")

    path.write_text(json.dumps(payload, indent=2, default=convert), encoding="utf-8")


def to_boolean(values: pd.Series) -> pd.Series:
    """Read boolean CSV columns consistently whether pandas inferred bool or string."""

    return values.astype("string").str.strip().str.lower().isin({"true", "1", "yes"})


def prepare_feature_frame(
    table: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """Select model features and make missing values safe for sklearn transformers."""

    required = set(numeric_features).union(categorical_features)
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Model input is missing required columns: {missing}")

    features = table[numeric_features + categorical_features].copy()
    for column in numeric_features:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in categorical_features:
        features[column] = features[column].astype("string").fillna("Unknown").astype(str)
    return features


def make_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Return a reusable numeric/categorical preprocessing stage."""

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return compact, dependency-free regression metrics."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    denominator = np.sum(np.square(actual - np.mean(actual)))
    r2 = float(1 - np.sum(np.square(errors)) / denominator) if denominator else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def min_max_score(values: pd.Series) -> pd.Series:
    """Scale values to 0–1, returning a neutral score when all values are equal."""

    values = pd.to_numeric(values, errors="coerce")
    lower = values.min()
    upper = values.max()
    if pd.isna(lower) or pd.isna(upper):
        return pd.Series(0.5, index=values.index, dtype=float)
    if upper == lower:
        return pd.Series(0.5, index=values.index, dtype=float)
    return (values - lower) / (upper - lower)
