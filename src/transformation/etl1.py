"""Backward-compatible IPL wrapper for the canonical cricket_etl module.

New work should import from ``src.transformation.cricket_etl`` directly. This
file remains only so that older notebook cells using process_ipl_match or
process_all_ipl_matches do not silently break.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from src.transformation.cricket_etl import process_competition, process_match
except ModuleNotFoundError:  # Allows an old notebook to import this file directly.
    from cricket_etl import process_competition, process_match


def process_ipl_match(file_path: str | Path):
    """Process one IPL match through the canonical generic ETL."""

    return process_match(file_path, "IPL")


def process_all_ipl_matches(input_directory: str | Path, output_directory: str | Path):
    """Process the IPL archive and return the same three tables expected by old notebooks."""

    output_directory = Path(output_directory)
    process_competition(input_directory, output_directory, "IPL")
    return (
        pd.read_csv(output_directory / "matches.csv"),
        pd.read_csv(output_directory / "batting_match.csv"),
        pd.read_csv(output_directory / "bowling_match.csv"),
    )
