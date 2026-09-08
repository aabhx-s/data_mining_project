import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DIR = Path("data/processed")

COMPETITIONS = [
    "ipl",
    "odi",
    "t20",
    "bbl",
    "smat",
    "t20_wc",
]


# ============================================================
# TEST HELPERS
# ============================================================

def load_data(competition):
    """
    Load the three processed datasets for a competition.
    """

    directory = PROCESSED_DIR / competition

    matches = pd.read_csv(
        directory / "matches.csv"
    )

    batting = pd.read_csv(
        directory / "batting_match.csv"
    )

    bowling = pd.read_csv(
        directory / "bowling_match.csv"
    )

    return matches, batting, bowling


def test_file_existence():

    print("\n" + "=" * 60)
    print("TEST 1 — FILE EXISTENCE")
    print("=" * 60)

    passed = True

    for competition in COMPETITIONS:

        directory = PROCESSED_DIR / competition

        required_files = [
            "matches.csv",
            "batting_match.csv",
            "bowling_match.csv",
        ]

        for filename in required_files:

            file_path = directory / filename

            if file_path.exists():
                print(f"PASS: {file_path}")
            else:
                print(f"FAIL: {file_path}")
                passed = False

    return passed


def test_row_counts():

    print("\n" + "=" * 60)
    print("TEST 2 — ROW COUNTS")
    print("=" * 60)

    passed = True

    for competition in COMPETITIONS:

        matches, batting, bowling = load_data(
            competition
        )

        print(f"\n{competition.upper()}")

        print(f"Matches:  {len(matches):,}")
        print(f"Batting:  {len(batting):,}")
        print(f"Bowling:  {len(bowling):,}")

        if len(matches) == 0:
            print("FAIL: Matches dataset is empty")
            passed = False

        if len(batting) == 0:
            print("FAIL: Batting dataset is empty")
            passed = False

        if len(bowling) == 0:
            print("FAIL: Bowling dataset is empty")
            passed = False

    return passed


def test_duplicates():

    print("\n" + "=" * 60)
    print("TEST 3 — DUPLICATES")
    print("=" * 60)

    passed = True

    for competition in COMPETITIONS:

        matches, batting, bowling = load_data(
            competition
        )

        match_duplicates = matches.duplicated().sum()

        batting_duplicates = batting.duplicated().sum()

        bowling_duplicates = bowling.duplicated().sum()

        print(f"\n{competition.upper()}")

        print(
            f"Match duplicates:   {match_duplicates}"
        )

        print(
            f"Batting duplicates: {batting_duplicates}"
        )

        print(
            f"Bowling duplicates: {bowling_duplicates}"
        )

        if (
            match_duplicates > 0
            or batting_duplicates > 0
            or bowling_duplicates > 0
        ):
            passed = False

    return passed


def test_missing_values():

    print("\n" + "=" * 60)
    print("TEST 4 — MISSING VALUES")
    print("=" * 60)

    passed = True

    for competition in COMPETITIONS:

        matches, batting, bowling = load_data(
            competition
        )

        print(f"\n{competition.upper()}")

        datasets = {
            "matches": matches,
            "batting": batting,
            "bowling": bowling,
        }

        for name, df in datasets.items():

            missing = df.isnull().sum()

            missing = missing[
                missing > 0
            ]

            if missing.empty:

                print(
                    f"PASS: {name} has no missing values"
                )

            else:

                print(
                    f"WARNING: {name} has missing values:"
                )

                print(missing)

    return passed


def test_competition_column():

    print("\n" + "=" * 60)
    print("TEST 5 — COMPETITION COLUMN")
    print("=" * 60)

    passed = True

    expected_names = {
        "ipl": "IPL",
        "odi": "ODI",
        "t20": "T20",
        "bbl": "BBL",
        "smat": "SMAT",
        "t20_wc": "T20_WC",
    }

    for competition in COMPETITIONS:

        matches, batting, bowling = load_data(
            competition
        )

        expected = expected_names[competition]

        datasets = {
            "matches": matches,
            "batting": batting,
            "bowling": bowling,
        }

        for name, df in datasets.items():

            if "competition" not in df.columns:

                print(
                    f"FAIL: {competition} {name} "
                    "missing competition column"
                )

                passed = False

            else:

                values = set(
                    df["competition"].dropna().unique()
                )

                if values == {expected}:

                    print(
                        f"PASS: {competition} {name}"
                    )

                else:

                    print(
                        f"FAIL: {competition} {name}"
                        f" → {values}"
                    )

                    passed = False

    return passed


def test_batting_logic():

    print("\n" + "=" * 60)
    print("TEST 6 — BATTING LOGIC")
    print("=" * 60)

    passed = True

    for competition in COMPETITIONS:

        _, batting, _ = load_data(
            competition
        )

        # Runs cannot be negative
        if (batting["runs"] < 0).any():

            print(
                f"FAIL: Negative runs in {competition}"
            )

            passed = False

        # Balls cannot be negative
        if (batting["balls"] < 0).any():

            print(
                f"FAIL: Negative balls in {competition}"
            )

            passed = False

        # Strike rate should be non-negative
        if (
            batting["strike_rate"].dropna() < 0
        ).any():

            print(
                f"FAIL: Negative strike rate in {competition}"
            )

            passed = False

        else:

            print(
                f"PASS: {competition.upper()}"
            )

    return passed


def test_bowling_logic():

    print("\n" + "=" * 60)
    print("TEST 7 — BOWLING LOGIC")
    print("=" * 60)

    passed = True

    for competition in COMPETITIONS:

        _, _, bowling = load_data(
            competition
        )

        # Balls cannot be negative
        if (bowling["balls"] < 0).any():

            print(
                f"FAIL: Negative balls in {competition}"
            )

            passed = False

        # Runs conceded cannot be negative
        if (
            bowling["runs_conceded"] < 0
        ).any():

            print(
                f"FAIL: Negative runs conceded "
                f"in {competition}"
            )

            passed = False

        # Wickets cannot be negative
        if (bowling["wickets"] < 0).any():

            print(
                f"FAIL: Negative wickets in {competition}"
            )

            passed = False

        # Economy cannot be negative
        if (
            bowling["economy"].dropna() < 0
        ).any():

            print(
                f"FAIL: Negative economy in {competition}"
            )

            passed = False

        else:

            print(
                f"PASS: {competition.upper()}"
            )

    return passed


def test_match_references():

    print("\n" + "=" * 60)
    print("TEST 8 — MATCH REFERENCES")
    print("=" * 60)

    passed = True

    for competition in COMPETITIONS:

        matches, batting, bowling = load_data(
            competition
        )

        match_ids = set(
            matches["match_id"]
        )

        batting_ids = set(
            batting["match_id"]
        )

        bowling_ids = set(
            bowling["match_id"]
        )

        missing_batting = (
            batting_ids - match_ids
        )

        missing_bowling = (
            bowling_ids - match_ids
        )

        if missing_batting:

            print(
                f"FAIL: {competition} has batting "
                "records without a match"
            )

            passed = False

        if missing_bowling:

            print(
                f"FAIL: {competition} has bowling "
                "records without a match"
            )

            passed = False

        if (
            not missing_batting
            and not missing_bowling
        ):

            print(
                f"PASS: {competition.upper()}"
            )

    return passed


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    results = {}

    results["File existence"] = test_file_existence()
    results["Row counts"] = test_row_counts()
    results["Duplicates"] = test_duplicates()
    results["Missing values"] = test_missing_values()
    results["Competition column"] = test_competition_column()
    results["Batting logic"] = test_batting_logic()
    results["Bowling logic"] = test_bowling_logic()
    results["Match references"] = test_match_references()


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL ETL TEST REPORT")
    print("=" * 60)

    for test_name, result in results.items():

        status = "PASS" if result else "FAIL"

        print(
            f"{status:6} | {test_name}"
        )

    passed = sum(
        results.values()
    )

    total = len(results)

    print()
    print(
        f"Tests passed: {passed}/{total}"
    )

    if passed == total:

        print(
            "\nALL TESTS PASSED"
        )

    else:

        print(
            "\nSOME TESTS FAILED — REVIEW ABOVE"
        )