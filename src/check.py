from pathlib import Path
import pandas as pd


DATASETS = {
    "IPL": {
        "raw": Path("data/raw/ipl_male_json"),
        "processed": Path("data/processed/ipl/matches.csv"),
    },

    "ODI": {
        "raw": Path("data/raw/odis_male_json"),
        "processed": Path("data/processed/odi/matches.csv"),
    },

    "T20": {
        "raw": Path("data/raw/t20s_male_json"),
        "processed": Path("data/processed/t20/matches.csv"),
    },

    "BBL": {
        "raw": Path("data/raw/bbl_male_json"),
        "processed": Path("data/processed/bbl/matches.csv"),
    },

    "SMAT": {
        "raw": Path("data/raw/sma_male_json"),
        "processed": Path("data/processed/smat/matches.csv"),
    },

    "T20_WC": {
        "raw": Path("data/raw/icc_mens_t20_world_cup_male_json"),
        "processed": Path("data/processed/t20_wc/matches.csv"),
    },
}


for competition, paths in DATASETS.items():

    raw_files = sorted(
        paths["raw"].glob("*.json")
    )

    processed = pd.read_csv(
        paths["processed"]
    )

    raw_ids = {
        file.stem
        for file in raw_files
    }

    processed_ids = set(
        processed["match_id"].astype(str)
    )

    missing = raw_ids - processed_ids
    extra = processed_ids - raw_ids

    print("\n" + "=" * 60)
    print(competition)
    print("=" * 60)

    print(f"Raw JSON files:       {len(raw_files)}")
    print(f"Processed matches:    {len(processed)}")
    print(f"Missing from output:  {len(missing)}")
    print(f"Extra in output:      {len(extra)}")

    if missing:
        print("\nMissing match IDs:")

        for match_id in sorted(missing):
            print(f"  {match_id}")

    if extra:
        print("\nExtra match IDs:")

        for match_id in sorted(extra):
            print(f"  {match_id}")