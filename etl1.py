import json
import pandas as pd
from pathlib import Path


# ============================================================
# WICKET TYPES CREDITED TO THE BOWLER
# ============================================================

BOWLER_WICKET_TYPES = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_legal_ball(delivery):
    """
    Returns True if the delivery counts as a legal ball.
    Wides and no-balls do not count as legal balls.
    """

    extras = delivery.get("extras", {})

    return (
        "wides" not in extras
        and "noballs" not in extras
    )


def calculate_bowler_runs(delivery):
    """
    Calculates runs charged to the bowler.

    Byes and leg-byes are not charged to the bowler.
    """

    total = delivery["runs"]["total"]
    extras = delivery.get("extras", {})

    total -= extras.get("byes", 0)
    total -= extras.get("legbyes", 0)
    total -= extras.get("penalty", 0)

    return total


def get_bowler_wickets(delivery):
    """
    Counts wickets credited to the bowler.
    Run-outs and other non-bowler dismissals are excluded.
    """

    wickets = delivery.get("wickets", [])

    return sum(
        1
        for wicket in wickets
        if wicket["kind"] in BOWLER_WICKET_TYPES
    )


# ============================================================
# PROCESS ONE IPL MATCH
# ============================================================

def process_ipl_match(file_path):
    """
    Reads one Cricsheet IPL JSON file and returns:

    1. Match information
    2. Player batting statistics
    3. Player bowling statistics
    """

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    with open(file_path, "r", encoding="utf-8") as f:
        match = json.load(f)

    info = match["info"]

    match_id = file_path.stem
    season = info["season"]
    date = info["dates"][0]

    teams = info["teams"]

    venue = info.get("venue")
    city = info.get("city")


    # --------------------------------------------------------
    # Extract delivery-level data
    # --------------------------------------------------------

    deliveries = []

    for innings in match["innings"]:

        batting_team = innings["team"]

        # Find bowling team
        bowling_team = next(
            team for team in teams
            if team != batting_team
        )

        for over in innings["overs"]:

            for delivery in over["deliveries"]:

                extras = delivery.get("extras", {})

                deliveries.append({

                    "match_id": match_id,
                    "season": season,
                    "date": date,

                    "venue": venue,
                    "city": city,

                    "batting_team": batting_team,
                    "bowling_team": bowling_team,

                    "batter": delivery["batter"],
                    "bowler": delivery["bowler"],
                    "non_striker": delivery["non_striker"],

                    "delivery": delivery["actual_delivery"],

                    "batter_runs": delivery["runs"]["batter"],
                    "extra_runs": delivery["runs"]["extras"],
                    "total_runs": delivery["runs"]["total"],

                    "is_legal_ball": is_legal_ball(delivery),

                    "bowler_runs": calculate_bowler_runs(delivery),

                    "bowler_wickets": get_bowler_wickets(delivery),

                    "four": int(
                        delivery["runs"]["batter"] == 4
                    ),

                    "six": int(
                        delivery["runs"]["batter"] == 6
                    )
                })


    deliveries_df = pd.DataFrame(deliveries)


    # ========================================================
    # BATTING STATISTICS
    # ========================================================

    batting_match = (
        deliveries_df
        .groupby(
            [
                "match_id",
                "season",
                "date",
                "batting_team",
                "batter"
            ],
            as_index=False
        )
        .agg(

            runs=("batter_runs", "sum"),

            balls=("is_legal_ball", "sum"),

            fours=("four", "sum"),

            sixes=("six", "sum")
        )
    )


    # Strike rate

    batting_match["strike_rate"] = (
        batting_match["runs"]
        .div(
            batting_match["balls"].replace(
                0,
                float("nan")
            )
        )
        .mul(100)
        .round(2)
    )


    # Rename batting columns

    batting_match = batting_match.rename(
        columns={
            "batting_team": "team",
            "batter": "player"
        }
    )


    # ========================================================
    # BOWLING STATISTICS
    # ========================================================

    bowling_match = (
        deliveries_df
        .groupby(
            [
                "match_id",
                "season",
                "date",
                "bowling_team",
                "batting_team",
                "bowler"
            ],
            as_index=False
        )
        .agg(

            balls=("is_legal_ball", "sum"),

            runs_conceded=("bowler_runs", "sum"),

            wickets=("bowler_wickets", "sum")
        )
    )


    # Overs

    bowling_match["overs"] = (
        (bowling_match["balls"] // 6).astype(str)
        + "."
        + (bowling_match["balls"] % 6).astype(str)
    )


    # Economy

    bowling_match["economy"] = (
        bowling_match["runs_conceded"]
        .div(
            bowling_match["balls"].replace(
                0,
                float("nan")
            )
        )
        .mul(6)
        .round(2)
    )


    # Bowling strike rate

    bowling_match["bowling_strike_rate"] = (
        bowling_match["balls"]
        .div(
            bowling_match["wickets"].replace(
                0,
                float("nan")
            )
        )
        .round(2)
    )


    # Rename bowler

    bowling_match = bowling_match.rename(
        columns={
            "bowler": "player",
            "batting_team": "opposition"
        }
    )


    # ========================================================
    # MATCH INFORMATION
    # ========================================================

    match_data = pd.DataFrame([{

        "match_id": match_id,

        "season": season,

        "date": date,

        "team1": teams[0],

        "team2": teams[1],

        "venue": venue,

        "city": city
    }])


    # ========================================================
    # RETURN
    # ========================================================

    return (
        match_data,
        batting_match,
        bowling_match
    )


# ============================================================
# PROCESS ALL IPL MATCHES
# ============================================================

def process_all_ipl_matches(
    input_directory,
    output_directory
):
    """
    Processes every IPL JSON file and saves
    consolidated CSV files.
    """

    input_directory = Path(input_directory)
    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    json_files = sorted(
        input_directory.glob("*.json")
    )

    print(
        f"Found {len(json_files)} IPL match files."
    )


    all_matches = []
    all_batting = []
    all_bowling = []

    errors = []


    # --------------------------------------------------------
    # Process every match
    # --------------------------------------------------------

    for i, file_path in enumerate(
        json_files,
        start=1
    ):

        try:

            match_data, batting, bowling = (
                process_ipl_match(file_path)
            )

            all_matches.append(match_data)

            all_batting.append(batting)

            all_bowling.append(bowling)


            if i % 100 == 0:

                print(
                    f"Processed {i}/{len(json_files)} matches..."
                )


        except Exception as e:

            errors.append({

                "file": file_path.name,

                "error": str(e)
            })

            print(
                f"ERROR: {file_path.name} → {e}"
            )


    # --------------------------------------------------------
    # Combine everything
    # --------------------------------------------------------

    matches_df = pd.concat(
        all_matches,
        ignore_index=True
    )

    batting_df = pd.concat(
        all_batting,
        ignore_index=True
    )

    bowling_df = pd.concat(
        all_bowling,
        ignore_index=True
    )


    # --------------------------------------------------------
    # Save processed data
    # --------------------------------------------------------

    matches_df.to_csv(
        output_directory / "matches.csv",
        index=False
    )

    batting_df.to_csv(
        output_directory / "batting_match.csv",
        index=False
    )

    bowling_df.to_csv(
        output_directory / "bowling_match.csv",
        index=False
    )


    # Save errors if any

    if errors:

        pd.DataFrame(errors).to_csv(
            output_directory / "etl_errors.csv",
            index=False
        )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nETL COMPLETE")
    print("-------------------------")

    print(
        f"Matches processed: {len(matches_df)}"
    )

    print(
        f"Batting records: {len(batting_df)}"
    )

    print(
        f"Bowling records: {len(bowling_df)}"
    )

    all_players = set(
        batting_df["player"]
    ).union(
        set(bowling_df["player"])
    )

    print(
        f"Unique players: {len(all_players)}"
    )

    print(
        f"Errors: {len(errors)}"
    )

    return (
        matches_df,
        batting_df,
        bowling_df
    )