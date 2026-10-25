"""Standardise IPL sold-player auction CSV files and map them to canonical player IDs.

Place source files in data/raw/auction/, then run:

    python -m src.transformation.auction_etl

The source files contain sold players only. Therefore this output is suitable for
estimating price conditional on a sale, not for predicting whether a player is sold.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.transformation.cricket_etl import project_root_from_file


USD_TO_INR_BY_YEAR = {
    2013: 58.60,
    2014: 61.03,
    2015: 64.15,
    2016: 67.20,
    2017: 65.12,
    2018: 68.39,
    2019: 70.39,
    2020: 74.10,
    2021: 73.92,
    2022: 77.52,
    2023: 82.58,
    2024: 83.50,
    2025: 85.00,
    2026: 86.00,
}

OUTPUT_COLUMNS = [
    "auction_year", "player_id", "player_name_auction", "player_name_canonical",
    "player_match_status", "team", "role", "nationality", "nationality_source",
    "nationality_match_status", "price_original",
    "price_currency", "exchange_rate_to_inr", "price_inr", "price_inr_year_median_ratio",
    "source_file",
]


def normalise_column_name(value: str) -> str:
    """Make a CSV heading safe for case-insensitive matching."""

    return re.sub(r"[^a-z0-9]", "", value.lower())


def normalise_player_name(value: object) -> str:
    """Make a conservative matching key; the original name is always retained."""

    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", text.lower())


def player_name_tokens(value: object) -> set[str]:
    """Return comparable name words, preserving enough information to avoid loose matching."""

    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def first_existing_column(columns: pd.Index, choices: list[str]) -> str | None:
    """Return the first matching source heading from a list of accepted alternatives."""

    by_normalised_name = {normalise_column_name(column): column for column in columns}
    return next((by_normalised_name.get(normalise_column_name(choice)) for choice in choices if normalise_column_name(choice) in by_normalised_name), None)


def parse_money(value: object) -> float:
    """Parse values such as 2500000, ₹ 25 Lakh, 2.5 Cr or $400,000."""

    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().lower().replace(",", "")
    if text in {"", "-", "nan", "na", "n/a"}:
        return np.nan
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return np.nan
    amount = float(match.group())
    if "crore" in text or re.search(r"\bcr\b", text):
        return amount * 10_000_000
    if "lakh" in text or "lac" in text:
        return amount * 100_000
    if "million" in text or re.search(r"\bmn\b", text):
        return amount * 1_000_000
    return amount


def infer_currency(price_column: str) -> str:
    """Infer the currency from the supplied price heading."""

    lowered = price_column.lower()
    if "$" in price_column or "usd" in lowered or "dollar" in lowered:
        return "USD"
    return "INR"


def infer_auction_year(filename: str) -> int:
    """Read a four-digit auction year from a filename such as IPL_Auction_2025.csv."""

    match = re.search(r"(20\d{2})", filename)
    if not match:
        raise ValueError(f"Could not infer auction year from filename: {filename}")
    return int(match.group(1))


def load_name_overrides(path: Path) -> dict[str, tuple[str, str | None]]:
    """Load explicit name corrections keyed by normalised auction-player name."""

    if not path.exists():
        return {}
    overrides = pd.read_csv(path, dtype="string").dropna(subset=["source_name", "player_id"])
    required = {"source_name", "player_id"}
    missing = required.difference(overrides.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return {
        normalise_player_name(row.source_name): (str(row.player_id), getattr(row, "canonical_name", None))
        for row in overrides.itertuples(index=False)
    }


def build_player_lookup(master: pd.DataFrame) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build exact and full-name-token lookups, preserving genuine ambiguities."""

    exact_lookup: dict[str, set[str]] = {}
    token_lookup: dict[str, set[str]] = {}
    name_columns = ["player", "aliases", "full_name", "unique_name"]
    for row in master.itertuples(index=False):
        player_id = str(row.player_id)
        for column in name_columns:
            value = getattr(row, column, None)
            if pd.isna(value) or not str(value).strip():
                continue
            # ``aliases`` is a pipe-separated list produced by player_master_etl.py.
            values = str(value).split("|") if column == "aliases" else [str(value)]
            for name in values:
                key = normalise_player_name(name)
                if key:
                    exact_lookup.setdefault(key, set()).add(player_id)
                token_lookup.setdefault(player_id, set()).update(player_name_tokens(name))
    return exact_lookup, token_lookup


def tokens_describe_same_name(source_tokens: set[str], candidate_tokens: set[str]) -> bool:
    """Match full names to names with middle names/initials, but not unrelated spellings.

    For example, ``Abhishek Nayar`` matches ``Abhishek Mohan Nayar`` and
    ``Chris Gayle`` matches ``Christopher Henry Gayle``. A candidate must
    satisfy every source word and a player is used only if it is unique.
    """

    if len(source_tokens) < 2:
        return False
    for source in source_tokens:
        if source in candidate_tokens:
            continue
        # Allow a first-name shortening such as Chris/Christopher, but never
        # use one- or two-letter initials for this fallback.
        if len(source) >= 4 and any(
            len(candidate) >= 4
            and (source.startswith(candidate) or candidate.startswith(source))
            for candidate in candidate_tokens
        ):
            continue
        return False
    return True


def map_player(
    name: object,
    overrides: dict[str, tuple[str, str | None]],
    exact_lookup: dict[str, set[str]],
    token_lookup: dict[str, set[str]],
    master_names: dict[str, str],
) -> tuple[str | None, str | None, str]:
    """Map an auction name with override, exact, then unique full-name matching."""

    key = normalise_player_name(name)
    if key in overrides:
        player_id, canonical_name = overrides[key]
        return player_id, canonical_name or master_names.get(player_id), "override"
    candidates = exact_lookup.get(key, set())
    if len(candidates) == 1:
        player_id = next(iter(candidates))
        return player_id, master_names.get(player_id), "exact_normalised_name"
    if len(candidates) > 1:
        return None, None, "ambiguous"

    source_tokens = player_name_tokens(name)
    token_candidates = {
        player_id
        for player_id, candidate_tokens in token_lookup.items()
        if tokens_describe_same_name(source_tokens, candidate_tokens)
    }
    if len(token_candidates) == 1:
        player_id = next(iter(token_candidates))
        return player_id, master_names.get(player_id), "unique_full_name_match"
    if len(token_candidates) > 1:
        return None, None, "ambiguous_full_name"
    return None, None, "unmatched"


def standardise_source_file(
    source_path: Path,
    overrides: dict[str, tuple[str, str | None]],
    exact_lookup: dict[str, set[str]],
    token_lookup: dict[str, set[str]],
    master_names: dict[str, str],
    master_nationalities: dict[str, str],
) -> pd.DataFrame:
    """Convert one source CSV, despite its differing column spelling, to the common schema."""

    source = pd.read_csv(source_path)
    player_column = first_existing_column(source.columns, ["Name", "Player", "Player Name"])
    team_column = first_existing_column(source.columns, ["TeamName", "Team", "Team Name", "Franchise"])
    role_column = first_existing_column(source.columns, ["Role", "Player Role"])
    nationality_column = first_existing_column(source.columns, ["Nationality", "Country"])
    price_column = first_existing_column(
        source.columns,
        ["Winning Bid in Rs", "Winning Bid in INR", "Price in rs", "Price in Rs", "Price in $", "Price", "Winning Bid"],
    )
    if player_column is None or price_column is None:
        raise ValueError(
            f"{source_path.name} must contain a player column and price column. "
            f"Found: {list(source.columns)}"
        )

    result = pd.DataFrame(
        {
            "auction_year": infer_auction_year(source_path.name),
            "player_name_auction": source[player_column].astype("string").str.strip(),
            "team": source[team_column].astype("string").str.strip() if team_column else pd.NA,
            "role": source[role_column].astype("string").str.strip() if role_column else pd.NA,
            "nationality": source[nationality_column].astype("string").str.strip() if nationality_column else pd.NA,
            "price_original": source[price_column],
            "price_currency": infer_currency(price_column),
            "source_file": source_path.name,
        }
    )
    result = result.dropna(subset=["player_name_auction"]).copy()
    mappings = result["player_name_auction"].map(
        lambda name: map_player(name, overrides, exact_lookup, token_lookup, master_names)
    )
    result[["player_id", "player_name_canonical", "player_match_status"]] = pd.DataFrame(
        mappings.tolist(), index=result.index
    )
    # Keep the raw auction value for audit, then fill only a missing value from
    # the canonical player master. Unmatched players intentionally stay blank.
    result["nationality_source"] = result["nationality"]
    result["nationality"] = result["nationality"].fillna(result["player_id"].map(master_nationalities))
    result["nationality_match_status"] = np.select(
        [result["nationality_source"].notna(), result["nationality"].notna()],
        ["source", "player_master"],
        default="missing",
    )
    result["price_original"] = result["price_original"].map(parse_money)
    result["exchange_rate_to_inr"] = result.apply(
        lambda row: USD_TO_INR_BY_YEAR.get(int(row.auction_year), np.nan)
        if row.price_currency == "USD"
        else 1.0,
        axis=1,
    )
    result["price_inr"] = result["price_original"] * result["exchange_rate_to_inr"]
    return result


def build_auction_table(project_root: str | Path | None = None) -> pd.DataFrame:
    """Build cleaned auction records and a separate unresolved-player audit file."""

    root = Path(project_root) if project_root else project_root_from_file()
    raw_auction_directory = root / "data" / "raw" / "auction"
    source_files = sorted(raw_auction_directory.glob("*.csv"))
    if not source_files:
        raise FileNotFoundError(
            f"No auction CSV files found in {raw_auction_directory}. Add IPL_Auction_<YEAR>_Sold_Player.csv files."
        )

    master_path = root / "data" / "processed" / "player_master.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing {master_path}. Run player_master_etl.py first.")
    master = pd.read_csv(master_path, dtype="string").dropna(subset=["player_id"])
    overrides = load_name_overrides(root / "data" / "reference" / "player_name_overrides.csv")
    exact_lookup, token_lookup = build_player_lookup(master)
    master_names = dict(zip(master["player_id"], master["player"]))
    master_nationalities = (
        master.dropna(subset=["country"])
        .assign(country=lambda table: table["country"].str.strip())
        .loc[lambda table: table["country"].ne("")]
        .set_index("player_id")["country"]
        .to_dict()
    )

    cleaned = pd.concat(
        [
            standardise_source_file(
                path,
                overrides,
                exact_lookup,
                token_lookup,
                master_names,
                master_nationalities,
            )
            for path in source_files
        ],
        ignore_index=True,
    )
    if cleaned["price_inr"].isna().any():
        bad_files = sorted(cleaned.loc[cleaned["price_inr"].isna(), "source_file"].unique())
        raise ValueError(f"Could not parse some auction prices in: {', '.join(bad_files)}")

    year_median = cleaned.groupby("auction_year")["price_inr"].transform("median")
    cleaned["price_inr_year_median_ratio"] = (cleaned["price_inr"] / year_median).round(6)
    for column in OUTPUT_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = pd.NA
    cleaned = cleaned[OUTPUT_COLUMNS].sort_values(["auction_year", "player_name_auction"]).reset_index(drop=True)

    output_directory = root / "data" / "processed" / "auction"
    output_directory.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_directory / "auction_clean.csv", index=False)
    unresolved = cleaned.loc[cleaned["player_id"].isna()].copy()
    unresolved.to_csv(output_directory / "auction_unmatched_players.csv", index=False)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    arguments = parser.parse_args()
    auction = build_auction_table(arguments.project_root)
    print(f"Saved {len(auction):,} sold-player auction rows.")
    print(f"Unmatched player names: {auction['player_id'].isna().sum():,}")


if __name__ == "__main__":
    main()
