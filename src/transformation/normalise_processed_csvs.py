"""Store whole-number fields as integers in the existing processed CSV files.

This edits files under ``data/processed`` in place. It converts only values
that are intrinsically whole numbers, such as runs, wickets, balls, counts,
years and INR amounts. Rates, ratios and model forecasts remain decimal.

Run after the ETL/model pipeline whenever new files are written:

    python -m src.transformation.normalise_processed_csvs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.transformation.cricket_etl import project_root_from_file


INTEGER_COLUMNS = {
    "runs",
    "balls",
    "fours",
    "sixes",
    "wickets",
    "runs_conceded",
    "innings_batted",
    "dismissals",
    "auction_year",
    "as_of_season",
    "as_of_season_end_year",
    "predicted_ipl_season",
    "season_end_year",
    "target_season",
    "first_season",
    "last_season",
    "last_ipl_season",
    "latest_ipl_season_end_year",
    "latest_ipl_runs",
    "latest_ipl_wickets",
    "price_original",
    "price_inr",
    "predicted_fair_value_inr",
    "absolute_error_inr",
    "market_median_inr_assumption",
    "max_fair_value_inr",
    "rank",
}

INTEGER_SUFFIXES = (
    "_matches",
    "_runs",
    "_balls",
    "_fours",
    "_sixes",
    "_wickets",
    "_runs_conceded",
    "_innings_batted",
    "_dismissals",
    "_seasons_available",
    "_seasons",
)


def is_integer_column(column: str) -> bool:
    """Return whether a column represents a count, year or whole rupee amount."""

    if column in INTEGER_COLUMNS:
        return True
    # Model estimates are averages and must retain their useful fractional part.
    if column.startswith(("expected_", "predicted_", "weighted_")):
        return False
    return column.endswith(INTEGER_SUFFIXES)


def cast_whole_number_columns(table: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    """Use pandas' nullable integer dtype without rounding or losing missing values."""

    result = table.copy()
    for column in result.columns:
        if not is_integer_column(column) or not pd.api.types.is_numeric_dtype(result[column]):
            continue
        values = result[column].dropna()
        if not np.isclose(values.to_numpy(dtype=float) % 1, 0).all():
            raise ValueError(
                f"{source_path}: {column!r} contains a non-whole value and was not changed."
            )
        result[column] = result[column].astype("Int64")
    return result


def normalise_processed_csvs(project_root: str | Path | None = None) -> list[Path]:
    """Rewrite the existing processed CSVs with clean integer representations."""

    root = Path(project_root) if project_root else project_root_from_file()
    processed_root = root / "data" / "processed"
    if not processed_root.exists():
        raise FileNotFoundError(f"Missing {processed_root}. Run the ETL first.")

    changed: list[Path] = []
    for path in sorted(processed_root.rglob("*.csv")):
        # Ignore the optional presentation copies from an earlier workflow.
        if "readable" in path.relative_to(processed_root).parts:
            continue
        table = pd.read_csv(path, dtype={"player_id": "string", "match_id": "string"})
        normalised = cast_whole_number_columns(table, path)
        normalised.to_csv(path, index=False)
        changed.append(path)
    if not changed:
        raise FileNotFoundError(f"No CSV files found under {processed_root}.")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    arguments = parser.parse_args()
    changed = normalise_processed_csvs(arguments.project_root)
    print(f"Updated {len(changed):,} processed CSV files in place.")


if __name__ == "__main__":
    main()
