"""Run post-ETL integrity checks over every processed Cricsheet competition."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# ``python src/check.py`` puts src/ rather than the project root on sys.path.
# Add the root so both direct execution and ``python -m src.check`` work.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transformation.cricket_etl import DATASETS, project_root_from_file


def check_competition(project_root: Path, output_name: str, raw_name: str, competition: str) -> list[str]:
    """Return a list of integrity errors for one competition."""

    errors: list[str] = []
    raw_files = sorted((project_root / "data" / "raw" / raw_name).glob("*.json"))
    processed_directory = project_root / "data" / "processed" / output_name
    files = {
        "matches": processed_directory / "matches.csv",
        "batting": processed_directory / "batting_match.csv",
        "bowling": processed_directory / "bowling_match.csv",
    }
    if not raw_files:
        return [f"{competition}: no raw JSON files found"]
    if missing_files := [str(path) for path in files.values() if not path.exists()]:
        return [f"{competition}: missing processed files: {', '.join(missing_files)}"]

    matches = pd.read_csv(files["matches"], dtype={"match_id": "string"})
    batting = pd.read_csv(files["batting"], dtype={"match_id": "string", "player_id": "string"})
    bowling = pd.read_csv(files["bowling"], dtype={"match_id": "string", "player_id": "string"})
    raw_ids = {path.stem for path in raw_files}
    processed_ids = set(matches["match_id"].dropna())

    if raw_ids != processed_ids:
        errors.append(f"{competition}: raw and processed match IDs differ")
    if matches["match_id"].duplicated().any():
        errors.append(f"{competition}: duplicate match IDs")
    if set(matches["competition"].dropna()) != {competition}:
        errors.append(f"{competition}: wrong competition value in matches table")

    for table_name, table in {"batting": batting, "bowling": bowling}.items():
        if "player_id" not in table.columns:
            errors.append(
                f"{competition}: {table_name} table has no player_id; rerun cricket_etl.py"
            )
            continue
        if table["player_id"].isna().any():
            errors.append(f"{competition}: {table_name} rows have missing player IDs")
        if not set(table["match_id"].dropna()).issubset(processed_ids):
            errors.append(f"{competition}: {table_name} rows reference missing matches")
        if table.duplicated(["match_id", "player_id"]).any():
            errors.append(f"{competition}: duplicate {table_name} player-match keys")

    if (batting[["runs", "balls", "fours", "sixes"]] < 0).any().any():
        errors.append(f"{competition}: negative batting statistics")
    if (bowling[["balls", "runs_conceded", "wickets"]] < 0).any().any():
        errors.append(f"{competition}: negative bowling statistics")
    return errors


def main() -> None:
    root = project_root_from_file()
    all_errors: list[str] = []
    for output_name, (raw_name, competition) in DATASETS.items():
        errors = check_competition(root, output_name, raw_name, competition)
        all_errors.extend(errors)
        status = "PASS" if not errors else "FAIL"
        print(f"{status}: {competition}")
        for error in errors:
            print(f"  - {error}")

    if all_errors:
        raise SystemExit(f"Integrity checks failed: {len(all_errors)} issue(s)")
    print("All ETL integrity checks passed.")


if __name__ == "__main__":
    main()
