"""Small manual smoke test for one JSON file from every configured archive.

This is optional for a quick sanity check. Use ``pytest`` for the full test
suite in tests/test_1.py.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transformation.cricket_etl import DATASETS, process_match, project_root_from_file


def main() -> None:
    root = project_root_from_file()
    for output_name, (raw_name, competition) in DATASETS.items():
        files = sorted((root / "data" / "raw" / raw_name).glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON files found for {competition}")
        matches, batting, bowling = process_match(files[0], competition)
        print(
            f"{output_name}: match={len(matches)}, batting={len(batting)}, "
            f"bowling={len(bowling)}, player_ids_missing="
            f"{batting['player_id'].isna().sum() + bowling['player_id'].isna().sum()}"
        )


if __name__ == "__main__":
    main()
