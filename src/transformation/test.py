from pathlib import Path
from cricket_etl import process_match


# ============================================================
# testing whether the pipeline loads for one match from each data file.
# ============================================================

TEST_FILES = {
    "IPL": Path("data/raw/ipl_male_json"),
    "ODI": Path("data/raw/odis_male_json"),
    "T20": Path("data/raw/t20s_male_json"),
    "BBL": Path("data/raw/bbl_male_json"),
    "SMAT": Path("data/raw/sma_male_json"),
    "T20_WC": Path("data/raw/icc_mens_t20_world_cup_male_json"),
}


# ============================================================
# TEST
# ============================================================

for competition, directory in TEST_FILES.items():

    print("\n" + "=" * 60)
    print(f"TESTING {competition}")
    print("=" * 60)

    json_files = sorted(directory.glob("*.json"))

    if not json_files:
        print(f"NO JSON FILES FOUND: {directory}")
        continue

    test_file = json_files[0]

    print(f"File: {test_file.name}")

    try:

        match, batting, bowling = process_match(
            test_file,
            competition
        )

        print("SUCCESS")

        print(f"Match rows:   {len(match)}")
        print(f"Batting rows: {len(batting)}")
        print(f"Bowling rows: {len(bowling)}")

        print("\nMatch columns:")
        print(match.columns.tolist())

        print("\nBatting columns:")
        print(batting.columns.tolist())

        print("\nBowling columns:")
        print(bowling.columns.tolist())

    except Exception as e:

        print("FAILED")
        print(f"Error: {e}")