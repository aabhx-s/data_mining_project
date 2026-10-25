"""Build a canonical player master from processed Cricsheet IDs and metadata.

Run after cricket_etl.py:

    python -m src.transformation.player_master_etl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.transformation.cricket_etl import DATASETS, project_root_from_file


MASTER_COLUMNS = [
    "player_id",
    "player",
    "aliases",
    "full_name",
    "unique_name",
    "cricinfo_id",
    "country",
    "dob",
    "batting_style",
    "bowling_style",
    "birthplace",
    "playing_role",
    "metadata_found",
]


def first_non_empty(values: pd.Series) -> str | None:
    """Return the most frequent non-empty value, then use alphabetical tie-breaking."""

    cleaned = values.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned.ne("")]
    if cleaned.empty:
        return None
    counts = cleaned.value_counts()
    return sorted(counts[counts.eq(counts.max())].index)[0]


def load_observed_players(processed_root: Path) -> pd.DataFrame:
    """Collect every player ID and display-name alias from every processed competition."""

    sources: list[pd.DataFrame] = []
    for directory in DATASETS:
        for filename in ("batting_match.csv", "bowling_match.csv"):
            path = processed_root / directory / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Run cricket_etl.py before building player_master.csv."
                )
            table = pd.read_csv(path, usecols=["player_id", "player"], dtype="string")
            sources.append(table)

    players = pd.concat(sources, ignore_index=True).dropna(subset=["player_id", "player"])
    players["player_id"] = players["player_id"].str.strip()
    players["player"] = players["player"].str.strip()
    players = players[players["player_id"].ne("") & players["player"].ne("")]

    return (
        players.groupby("player_id", as_index=False)
        .agg(
            player=("player", first_non_empty),
            aliases=("player", lambda values: " | ".join(sorted(set(values.dropna())))),
        )
        .sort_values("player_id")
        .reset_index(drop=True)
    )


def load_metadata(raw_root: Path) -> pd.DataFrame:
    """Load the supplied R-package metadata, keeping one strongest record per player ID."""

    path = raw_root / "player_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata source: {path}")

    metadata = pd.read_csv(path, dtype="string")
    metadata = metadata.rename(columns={"cricsheet_id": "player_id"})
    metadata["player_id"] = metadata["player_id"].str.strip()
    metadata = metadata.dropna(subset=["player_id"])
    metadata = metadata[metadata["player_id"].ne("")]

    wanted = [
        "player_id",
        "cricinfo_id",
        "unique_name",
        "full_name",
        "name",
        "country",
        "dob",
        "batting_style",
        "bowling_style",
        "birthplace",
        "playing_role",
    ]
    metadata = metadata[[column for column in wanted if column in metadata.columns]].copy()
    metadata["metadata_quality"] = metadata.notna().sum(axis=1)
    return (
        metadata.sort_values(["player_id", "metadata_quality"], ascending=[True, False])
        .drop_duplicates("player_id", keep="first")
        .drop(columns="metadata_quality")
    )


def load_people_lookup(raw_root: Path) -> pd.DataFrame:
    """Load Cricsheet people names as a fallback when metadata has missing display names."""

    path = raw_root / "people.csv"
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "people_name", "people_unique_name"])

    people = pd.read_csv(path, usecols=["identifier", "name", "unique_name"], dtype="string")
    people = people.rename(
        columns={
            "identifier": "player_id",
            "name": "people_name",
            "unique_name": "people_unique_name",
        }
    )
    people["player_id"] = people["player_id"].str.strip()
    return (
        people.dropna(subset=["player_id"])
        .groupby("player_id", as_index=False)
        .agg(
            people_name=("people_name", first_non_empty),
            people_unique_name=("people_unique_name", first_non_empty),
        )
    )


def build_player_master(project_root: str | Path | None = None) -> pd.DataFrame:
    """Create an all-competition player master and an IPL-only compatibility copy."""

    root = Path(project_root) if project_root else project_root_from_file()
    processed_root = root / "data" / "processed"
    raw_root = root / "data" / "raw"

    observed = load_observed_players(processed_root)
    metadata = load_metadata(raw_root)
    people = load_people_lookup(raw_root)
    master = observed.merge(metadata, on="player_id", how="left").merge(people, on="player_id", how="left")

    master["full_name"] = master["full_name"].fillna(master.get("people_name"))
    master["unique_name"] = master["unique_name"].fillna(master.get("people_unique_name"))
    master["metadata_found"] = master["country"].notna()
    master = master.drop(columns=["name", "people_name", "people_unique_name"], errors="ignore")

    for column in MASTER_COLUMNS:
        if column not in master.columns:
            master[column] = pd.NA
    master = master[MASTER_COLUMNS].sort_values("player_id").reset_index(drop=True)

    if master["player_id"].duplicated().any():
        raise ValueError("player_master has duplicate player IDs")

    processed_root.mkdir(parents=True, exist_ok=True)
    master.to_csv(processed_root / "player_master.csv", index=False)

    ipl_batting = pd.read_csv(processed_root / "ipl" / "batting_match.csv", usecols=["player_id"], dtype="string")
    ipl_bowling = pd.read_csv(processed_root / "ipl" / "bowling_match.csv", usecols=["player_id"], dtype="string")
    ipl_player_ids = set(pd.concat([ipl_batting, ipl_bowling])["player_id"].dropna())
    master[master["player_id"].isin(ipl_player_ids)].to_csv(
        processed_root / "ipl" / "player_master.csv", index=False
    )

    return master


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    arguments = parser.parse_args()
    master = build_player_master(arguments.project_root)
    print(f"Saved {len(master):,} canonical player records.")
    print(f"Players without metadata: {(~master['metadata_found']).sum():,}")


if __name__ == "__main__":
    main()
