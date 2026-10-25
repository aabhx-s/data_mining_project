"""Create report-ready charts from the forward IPL analysis outputs.

Examples:

    # Create the overall player chart and a Gujarat Titans shortlist chart.
    python -m src.visualization.create_model_visualizations --team "Gujarat Titans"

    # Create all available charts, including the team comparison when present.
    python -m src.visualization.create_model_visualizations

Run ``src.models.run_analysis`` first. Charts are saved beneath
``data/processed/model/visualizations`` by default.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.transformation.cricket_etl import project_root_from_file


RECOMMENDATION_COLOURS = {
    "Strong fit": "#198754",
    "Consider": "#d98e00",
    "Low fit": "#b02a37",
}


def file_safe_name(value: str) -> str:
    """Create a stable file name from a team name."""

    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def crore(value: float) -> str:
    """Format an INR amount in crore for compact chart labels."""

    return f"₹{value / 10_000_000:.1f} cr"


def read_required(model_directory: Path, filename: str) -> pd.DataFrame:
    """Read a required model output with an actionable error message."""

    path = model_directory / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the forward analysis before creating charts.")
    return pd.read_csv(path, dtype={"player_id": "string"})


def plot_value_vs_performance(
    fair_values: pd.DataFrame,
    predictions: pd.DataFrame,
    output_directory: Path,
) -> Path:
    """Show the trade-off between predicted batting output and fair auction value."""

    data = fair_values.merge(
        predictions[["player_id", "expected_runs", "expected_wickets"]],
        on="player_id",
        how="inner",
        validate="one_to_one",
    ).copy()
    if data.empty:
        raise ValueError("No player appears in both the fair-value and performance outputs.")
    data["fair_value_crore"] = data["predicted_fair_value_inr"] / 10_000_000

    figure, axis = plt.subplots(figsize=(11, 7))
    scatter = axis.scatter(
        data["fair_value_crore"],
        data["expected_runs"],
        c=data["expected_wickets"],
        s=70,
        cmap="viridis",
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
    )
    for _, row in data.nlargest(8, "expected_runs").iterrows():
        axis.annotate(
            row["player_name_canonical"],
            (row["fair_value_crore"], row["expected_runs"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    colour_bar = figure.colorbar(scatter, ax=axis)
    colour_bar.set_label("Expected wickets")
    axis.set_title("Expected IPL runs against estimated fair auction value")
    axis.set_xlabel("Estimated fair value (₹ crore)")
    axis.set_ylabel("Expected IPL runs")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = output_directory / "candidate_value_vs_predicted_runs.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def candidate_role_group(role: object) -> str:
    """Map detailed role metadata to the three auction-board sections."""

    text = "" if pd.isna(role) else str(role).casefold()
    if "all" in text and "round" in text:
        return "All-rounder"
    if "wicket" in text or "keeper" in text or "bat" in text or "order" in text:
        return "Batter / wicketkeeper"
    if "bowl" in text:
        return "Bowler"
    return "Other"


def plot_role_aware_candidate_board(
    fair_values: pd.DataFrame,
    predictions: pd.DataFrame,
    output_directory: Path,
    players_per_role: int = 8,
) -> Path:
    """Show a readable role-by-role heatmap of value, runs and wickets.

    The large all-player scatter plot is useful for broad exploration but gets
    crowded when every active cricketer is included.  This chart keeps the same
    three measures while showing a concise, television-style auction board.
    """

    data = fair_values.merge(
        predictions[["player_id", "expected_runs", "expected_wickets"]],
        on="player_id",
        how="inner",
        validate="one_to_one",
    ).copy()
    if data.empty:
        raise ValueError("No player appears in both the fair-value and performance outputs.")
    data["role_group"] = data["metadata_playing_role"].map(candidate_role_group)
    data["fair_value_crore"] = data["predicted_fair_value_inr"] / 10_000_000

    batters = data.loc[data["role_group"].eq("Batter / wicketkeeper")].nlargest(players_per_role, "expected_runs")
    bowlers = data.loc[data["role_group"].eq("Bowler")].nlargest(players_per_role, "expected_wickets")
    all_rounders = data.loc[data["role_group"].eq("All-rounder")].copy()
    if not all_rounders.empty:
        all_rounders["combined_rank"] = (
            all_rounders["expected_runs"].rank(pct=True)
            + all_rounders["expected_wickets"].rank(pct=True)
        )
        all_rounders = all_rounders.nlargest(players_per_role, "combined_rank")

    board = pd.concat([batters, bowlers, all_rounders], ignore_index=True)
    if board.empty:
        raise ValueError("No role-labelled candidates are available for the auction board.")

    metrics = ["fair_value_crore", "expected_runs", "expected_wickets"]
    # Colour is normalised within a metric. Cell labels retain the original
    # values, so different units can be compared in one compact board.
    colour_values = board[metrics].rank(pct=True).to_numpy()
    figure, axis = plt.subplots(figsize=(12, max(6, len(board) * 0.48)))
    image = axis.imshow(colour_values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    labels = (board["player_name_canonical"] + " — " + board["role_group"]).tolist()
    axis.set_yticks(range(len(board)), labels, fontsize=9)
    axis.set_xticks(range(3), ["Fair value\n(₹ crore)", "Expected\nruns", "Expected\nwickets"])
    axis.set_title("Role-aware auction candidate board")

    for row_number, (_, row) in enumerate(board.iterrows()):
        cell_labels = [
            f"₹{row.fair_value_crore:.1f} cr",
            f"{row.expected_runs:.0f}",
            f"{row.expected_wickets:.1f}",
        ]
        for column_number, label in enumerate(cell_labels):
            colour = "white" if colour_values[row_number, column_number] > 0.58 else "black"
            axis.text(column_number, row_number, label, ha="center", va="center", fontsize=9, color=colour)

    section_ends = [len(batters), len(batters) + len(bowlers)]
    for end in section_ends:
        if end and end < len(board):
            axis.axhline(end - 0.5, color="white", linewidth=2)
    colour_bar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    colour_bar.set_label("Higher standing within each metric")
    figure.tight_layout()
    path = output_directory / "role_aware_candidate_board.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_full_candidate_pool_hexbin(
    fair_values: pd.DataFrame,
    predictions: pd.DataFrame,
    output_directory: Path,
) -> Path:
    """Represent every candidate without hiding dense clusters of points.

    Each hexagon contains every player whose fair value and expected runs fall
    in that small range.  The left panel uses average expected wickets as the
    colour, matching the third variable in the scatter plot; the right panel
    shows how many players are represented by each hexagon.
    """

    data = fair_values.merge(
        predictions[["player_id", "expected_runs", "expected_wickets"]],
        on="player_id",
        how="inner",
        validate="one_to_one",
    ).copy()
    if data.empty:
        raise ValueError("No player appears in both the fair-value and performance outputs.")
    data["fair_value_crore"] = data["predicted_fair_value_inr"] / 10_000_000

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    wicket_hexagons = axes[0].hexbin(
        data["fair_value_crore"],
        data["expected_runs"],
        C=data["expected_wickets"],
        reduce_C_function=np.mean,
        gridsize=28,
        mincnt=1,
        cmap="viridis",
    )
    count_hexagons = axes[1].hexbin(
        data["fair_value_crore"],
        data["expected_runs"],
        gridsize=28,
        mincnt=1,
        cmap="magma",
    )

    for axis, title in zip(
        axes,
        [
            "All candidates: average expected wickets per value/performance range",
            "All candidates: number of players per value/performance range",
        ],
    ):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Estimated fair value (₹ crore)")
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("Expected IPL runs")
    figure.colorbar(wicket_hexagons, ax=axes[0], label="Average expected wickets")
    figure.colorbar(count_hexagons, ax=axes[1], label="Players in range")
    figure.suptitle("Full active-player candidate pool", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_directory / "full_candidate_pool_hexbin.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_team_shortlist(team_ranking: pd.DataFrame, team: str, top_n: int, output_directory: Path) -> Path:
    """Plot the most suitable candidates for one team."""

    top = team_ranking.nlargest(top_n, "team_fit_score").sort_values("team_fit_score")
    if top.empty:
        raise ValueError(f"No candidates are available for {team}.")

    figure, axis = plt.subplots(figsize=(11, max(5, top_n * 0.55)))
    colours = top["recommendation"].map(RECOMMENDATION_COLOURS).fillna("#6c757d")
    axis.barh(top["player_name_canonical"], top["team_fit_score"], color=colours)
    axis.axvline(60, color="#198754", linestyle="--", linewidth=1, label="Strong-fit threshold")
    axis.axvline(40, color="#d98e00", linestyle="--", linewidth=1, label="Consider threshold")
    for index, (_, row) in enumerate(top.iterrows()):
        axis.text(
            min(float(row["team_fit_score"]) + 1, 98),
            index,
            f"Fit {row['team_fit_score']:.1f} | Value {row['investment_score']:.1f} | {crore(float(row['predicted_fair_value_inr']))}",
            va="center",
            fontsize=9,
        )
    axis.set_xlim(0, 100)
    axis.set_title(f"{team}: top {len(top)} auction candidates")
    axis.set_xlabel("Team-fit score out of 100")
    axis.set_ylabel("Player")
    axis.legend(loc="lower right")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    path = output_directory / f"team_fit_{file_safe_name(team)}_top_{len(top)}.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_all_team_summary(recommendations: pd.DataFrame, output_directory: Path) -> Path:
    """Compare each team's highest-ranked candidate in one compact chart."""

    summary = (
        recommendations.sort_values(["team", "rank"])
        .groupby("team", as_index=False)
        .first()[["team", "player_name_canonical", "team_fit_score", "recommendation"]]
        .sort_values("team_fit_score")
    )
    if summary.empty:
        raise ValueError("The all-team recommendation file contains no rows.")

    figure, axis = plt.subplots(figsize=(11, max(5, len(summary) * 0.55)))
    colours = summary["recommendation"].map(RECOMMENDATION_COLOURS).fillna("#6c757d")
    axis.barh(summary["team"], summary["team_fit_score"], color=colours)
    for index, (_, row) in enumerate(summary.iterrows()):
        axis.text(
            min(float(row["team_fit_score"]) + 1, 98),
            index,
            f"{row['player_name_canonical']} ({row['team_fit_score']:.1f})",
            va="center",
            fontsize=9,
        )
    axis.set_xlim(0, 100)
    axis.set_title("Best-ranked candidate for each team profile")
    axis.set_xlabel("Team-fit score out of 100")
    axis.set_ylabel("Team")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    path = output_directory / "all_teams_best_candidate_summary.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def load_team_ranking(model_directory: Path, team: str) -> pd.DataFrame:
    """Find the team-specific ranking output, including names with punctuation."""

    expected = model_directory / f"team_fit_{file_safe_name(team)}.csv"
    if expected.exists():
        return pd.read_csv(expected, dtype={"player_id": "string"})
    matches = list(model_directory.glob("team_fit_*.csv"))
    for path in matches:
        table = pd.read_csv(path, usecols=["team"], nrows=1)
        if not table.empty and str(table.iloc[0, 0]).casefold() == team.casefold():
            return pd.read_csv(path, dtype={"player_id": "string"})
    raise FileNotFoundError(f"No team-fit CSV was found for {team!r} in {model_directory}.")


def create_visualizations(
    project_root: str | Path | None = None,
    team: str | None = None,
    top_n: int = 10,
    model_directory: str | Path | None = None,
) -> list[Path]:
    """Create all applicable charts and return their saved paths."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    root = Path(project_root) if project_root else project_root_from_file()
    source_directory = Path(model_directory) if model_directory else root / "data" / "processed" / "model"
    output_directory = source_directory / "visualizations"
    output_directory.mkdir(parents=True, exist_ok=True)

    fair_values = read_required(source_directory, "auction_candidate_fair_values.csv")
    predictions = read_required(source_directory, "upcoming_performance_predictions.csv")
    created = [
        plot_value_vs_performance(fair_values, predictions, output_directory),
        plot_full_candidate_pool_hexbin(fair_values, predictions, output_directory),
        plot_role_aware_candidate_board(fair_values, predictions, output_directory),
    ]

    if team:
        created.append(plot_team_shortlist(load_team_ranking(source_directory, team), team, top_n, output_directory))

    all_team_path = source_directory / "future_team_recommendations.csv"
    if all_team_path.exists():
        created.append(plot_all_team_summary(pd.read_csv(all_team_path), output_directory))
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--model-directory", type=Path, default=None)
    parser.add_argument("--team", default=None, help="Optional team name for a top-candidate chart.")
    parser.add_argument("--top-n", type=int, default=10)
    arguments = parser.parse_args()
    created = create_visualizations(
        project_root=arguments.project_root,
        model_directory=arguments.model_directory,
        team=arguments.team,
        top_n=arguments.top_n,
    )
    for path in created:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
