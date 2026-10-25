"""Create role-aware season-by-season IPL performance charts for players.

Examples:

    # One player (case-insensitive; a unique partial name also works).
    python -m src.visualization.create_player_visualizations --player "Virat Kohli"

    # Every player with IPL seasonal data. This can create hundreds of images.
    python -m src.visualization.create_player_visualizations --all-players

Charts are saved beneath ``data/processed/model/visualizations/players``.
They show IPL seasonal performance only, so comparisons remain like-for-like.
The panels are selected from the player's listed cricketing role: batters do
not receive an empty bowling panel, bowlers do not receive an empty batting
panel, and all-rounders receive both.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.transformation.cricket_etl import project_root_from_file


def file_safe_name(value: str) -> str:
    """Create a stable, readable file name component."""

    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def load_player_seasons(project_root: Path) -> pd.DataFrame:
    """Load the IPL seasonal feature table needed for historical charts."""

    path = project_root / "data" / "processed" / "ipl" / "player_season_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run feature_builder.py first.")
    table = pd.read_csv(path, dtype={"player_id": "string"})
    # Keep charts usable with the earlier feature-builder output too.
    if "season_end_year" not in table and "season" in table:
        table = table.rename(columns={"season": "season_end_year"})
    if "player_id" not in table:
        table["player_id"] = table["player"].astype("string").map(file_safe_name)
    required = {
        "player_id", "player", "season_end_year", "runs", "wickets", "batting_balls",
        "bowling_balls", "runs_conceded", "batting_matches", "bowling_matches",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} is missing chart columns: {missing}. Rebuild it with feature_builder.py.")
    for column in required.difference({"player_id", "player"}):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    # Older feature tables may contain the underlying totals but not the two
    # rate columns. Derive them here so historical charts remain backward-compatible.
    if "strike_rate" not in table:
        table["strike_rate"] = table["runs"].div(table["batting_balls"].replace(0, pd.NA)).mul(100)
    if "economy" not in table:
        table["economy"] = table["runs_conceded"].div(table["bowling_balls"].replace(0, pd.NA)).mul(6)
    table["strike_rate"] = pd.to_numeric(table["strike_rate"], errors="coerce")
    table["economy"] = pd.to_numeric(table["economy"], errors="coerce")
    return table.dropna(subset=["player_id", "player", "season_end_year"]).copy()


def normalise_player_name(value: object) -> str:
    """Return a conservative comparison key for a player display name."""

    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def classify_role(value: object, history: pd.DataFrame) -> str:
    """Classify the listed role, falling back to meaningful IPL involvement.

    Metadata takes precedence.  This matters for part-time bowlers such as
    Virat Kohli: one occasional spell should not turn a specialist batter's
    television-style profile into a bowling profile.
    """

    role = str(value or "").casefold()
    if "all" in role and "round" in role:
        return "All-rounder"
    if "wicket" in role or "keeper" in role:
        return "Wicketkeeper-batter"
    if "bowl" in role:
        return "Bowler"
    if "bat" in role or "order" in role:
        return "Batter"

    runs = pd.to_numeric(history["runs"], errors="coerce").fillna(0).sum()
    wickets = pd.to_numeric(history["wickets"], errors="coerce").fillna(0).sum()
    if runs > 0 and wickets > 0:
        return "All-rounder"
    if wickets > 0:
        return "Bowler"
    return "Batter"


def load_player_roles(project_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load role metadata from either current or legacy player-master paths."""

    candidates = [
        project_root / "data" / "processed" / "player_master.csv",
        project_root / "data" / "processed" / "ipl" / "player_master.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        master = pd.read_csv(path, dtype="string")
        if "playing_role" not in master:
            continue
        if "player_id" not in master and "cricsheet_id" in master:
            master = master.rename(columns={"cricsheet_id": "player_id"})

        by_id: dict[str, str] = {}
        by_name: dict[str, str] = {}
        for _, row in master.dropna(subset=["playing_role"]).iterrows():
            role = str(row["playing_role"]).strip()
            if not role:
                continue
            if "player_id" in master and pd.notna(row.get("player_id")):
                by_id[str(row["player_id"]).strip()] = role
            for name_column in ("player", "full_name", "unique_name", "name"):
                if name_column in master and pd.notna(row.get(name_column)):
                    by_name[normalise_player_name(row[name_column])] = role
        return by_id, by_name
    return {}, {}


def find_players(table: pd.DataFrame, requested_names: list[str] | None, all_players: bool) -> pd.DataFrame:
    """Resolve requested players safely, rejecting ambiguous search text."""

    players = table[["player_id", "player"]].drop_duplicates().sort_values("player")
    if all_players:
        return players
    if not requested_names:
        raise ValueError("Choose --player NAME or --all-players.")

    selected: list[pd.DataFrame] = []
    for name in requested_names:
        exact = players.loc[players["player"].str.casefold().eq(name.casefold())]
        matches = exact if not exact.empty else players.loc[
            players["player"].str.contains(name, case=False, regex=False, na=False)
        ]
        if len(matches) != 1:
            options = ", ".join(matches["player"].head(12).tolist()) or "none"
            raise ValueError(f"Player search {name!r} is ambiguous or missing. Matches: {options}")
        selected.append(matches)
    return pd.concat(selected, ignore_index=True).drop_duplicates("player_id")


def plot_player_history(
    history: pd.DataFrame,
    player_id: str,
    player: str,
    listed_role: str | None,
    output_directory: Path,
) -> tuple[Path, str]:
    """Plot only the performance dimensions appropriate to one player's role."""

    seasons = history.loc[history["player_id"].eq(player_id)].sort_values("season_end_year").copy()
    if seasons.empty:
        raise ValueError(f"No IPL seasonal history exists for {player}.")
    labels = seasons["season_end_year"].astype(int).astype(str).tolist()
    x_positions = list(range(len(seasons)))

    role = classify_role(listed_role, seasons)
    panel_count = 3 if role == "All-rounder" else 2
    figure, axes = plt.subplots(panel_count, 1, figsize=(11, 4.35 * panel_count), sharex=True)
    if panel_count == 1:
        axes = [axes]
    figure.suptitle(f"{player} — {role}\nIPL seasonal performance", fontsize=16, fontweight="bold")

    def plot_batting(axis: plt.Axes) -> None:
        axis.bar(x_positions, seasons["runs"].fillna(0), color="#1f77b4", label="Runs")
        axis.set_ylabel("Runs")
        rate_axis = axis.twinx()
        rate_axis.plot(x_positions, seasons["strike_rate"], color="#ff7f0e", marker="o", label="Strike rate")
        rate_axis.set_ylabel("Strike rate")
        axis.set_title("Batting output and strike rate")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(loc="upper left")
        rate_axis.legend(loc="upper right")

    def plot_bowling(axis: plt.Axes) -> None:
        axis.bar(x_positions, seasons["wickets"].fillna(0), color="#2ca02c", label="Wickets")
        axis.set_ylabel("Wickets")
        economy_axis = axis.twinx()
        economy_axis.plot(x_positions, seasons["economy"], color="#d62728", marker="o", label="Economy")
        economy_axis.set_ylabel("Economy")
        axis.set_title("Bowling output and economy")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(loc="upper left")
        economy_axis.legend(loc="upper right")

    def plot_batting_detail(axis: plt.Axes) -> None:
        width = 0.36
        fours = pd.to_numeric(seasons.get("fours", 0), errors="coerce").fillna(0)
        sixes = pd.to_numeric(seasons.get("sixes", 0), errors="coerce").fillna(0)
        axis.bar([position - width / 2 for position in x_positions], fours, width=width, color="#9467bd", label="Fours")
        axis.bar([position + width / 2 for position in x_positions], sixes, width=width, color="#e377c2", label="Sixes")
        axis.set_ylabel("Boundaries")
        appearances_axis = axis.twinx()
        appearances_axis.plot(x_positions, seasons["batting_matches"].fillna(0), color="#17becf", marker="o", label="Appearances")
        appearances_axis.set_ylabel("Batting appearances")
        axis.set_title("Boundary scoring and batting involvement")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(loc="upper left")
        appearances_axis.legend(loc="upper right")

    def plot_bowling_detail(axis: plt.Axes) -> None:
        overs = seasons["bowling_balls"].fillna(0).div(6)
        axis.bar(x_positions, overs, color="#8c564b", label="Overs bowled")
        axis.set_ylabel("Overs")
        appearances_axis = axis.twinx()
        appearances_axis.plot(x_positions, seasons["bowling_matches"].fillna(0), color="#bcbd22", marker="o", label="Appearances")
        appearances_axis.set_ylabel("Bowling appearances")
        axis.set_title("Overs bowled and bowling involvement")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(loc="upper left")
        appearances_axis.legend(loc="upper right")

    if role in {"Batter", "Wicketkeeper-batter"}:
        plot_batting(axes[0])
        plot_batting_detail(axes[1])
    elif role == "Bowler":
        plot_bowling(axes[0])
        plot_bowling_detail(axes[1])
    else:
        plot_batting(axes[0])
        plot_bowling(axes[1])
        involvement_axis = axes[2]
        width = 0.36
        involvement_axis.bar([position - width / 2 for position in x_positions], seasons["batting_matches"].fillna(0), width=width, color="#9467bd", label="Batting appearances")
        involvement_axis.bar([position + width / 2 for position in x_positions], seasons["bowling_matches"].fillna(0), width=width, color="#8c564b", label="Bowling appearances")
        involvement_axis.set_title("All-round involvement")
        involvement_axis.set_ylabel("Matches")
        involvement_axis.grid(axis="y", alpha=0.2)
        involvement_axis.legend(loc="upper left")

    axes[-1].set_xlabel("IPL season")
    axes[-1].set_xticks(x_positions, labels)

    figure.tight_layout(rect=(0, 0, 1, 0.97))
    path = output_directory / f"player_{file_safe_name(player)}_{player_id}_ipl_history.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path, role


def create_player_visualizations(
    project_root: str | Path | None = None,
    player_names: list[str] | None = None,
    all_players: bool = False,
    output_directory: str | Path | None = None,
) -> list[Path]:
    """Create one IPL historical-performance image for every selected player."""

    root = Path(project_root) if project_root else project_root_from_file()
    history = load_player_seasons(root)
    selected = find_players(history, player_names, all_players)
    roles_by_id, roles_by_name = load_player_roles(root)
    destination = Path(output_directory) if output_directory else root / "data" / "processed" / "model" / "visualizations" / "players"
    destination.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    index_rows: list[dict[str, object]] = []
    for _, player_row in selected.iterrows():
        player_id = str(player_row["player_id"])
        player = str(player_row["player"])
        listed_role = roles_by_id.get(player_id) or roles_by_name.get(normalise_player_name(player))
        chart_path, chart_role = plot_player_history(history, player_id, player, listed_role, destination)
        seasons = history.loc[history["player_id"].eq(player_id), "season_end_year"]
        created.append(chart_path)
        index_rows.append(
            {
                "player_id": player_id,
                "player": player,
                "role_profile": chart_role,
                "first_ipl_season": int(seasons.min()),
                "last_ipl_season": int(seasons.max()),
                "chart_file": chart_path.name,
            }
        )
    pd.DataFrame(index_rows).to_csv(destination / "player_chart_index.csv", index=False)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--player", action="append", dest="players", help="Player name; repeat for multiple players.")
    parser.add_argument("--all-players", action="store_true", help="Create a chart for every player with IPL history.")
    parser.add_argument("--output-directory", type=Path, default=None)
    arguments = parser.parse_args()
    if arguments.players and arguments.all_players:
        parser.error("Choose --player or --all-players, not both.")
    created = create_player_visualizations(
        project_root=arguments.project_root,
        player_names=arguments.players,
        all_players=arguments.all_players,
        output_directory=arguments.output_directory,
    )
    print(f"Saved {len(created):,} player chart(s).")


if __name__ == "__main__":
    main()
