"""Build an offline, searchable visual report from the saved model outputs.

Run after the all-team forward analysis:

    python -m src.visualization.create_prediction_visualizations

Every forecast player is included in the browser report, not just the top N.
Use --player NAME for individual presentation PNGs, or --all-players to export
one PNG per player (potentially thousands of images). No models are retrained
and no source CSV, JSON or joblib file is changed or deserialised as executable
code. The report is written under model/visualizations/predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.transformation.cricket_etl import project_root_from_file
from src.visualization import create_model_visualizations as base
from src.visualization.prediction_report import write_report


ROLES = ["Batter", "Wicketkeeper-batter", "Bowler", "All-rounder", "Other"]
ROLE_COLOURS = {
    "Batter": "#2463a5", "Wicketkeeper-batter": "#7752a3",
    "Bowler": "#177965", "All-rounder": "#ad6d17", "Other": "#737c88",
}
BLUE, TEAL, GOLD, GREY = "#2463a5", "#177965", "#ad6d17", "#737c88"
PREDICTION_FIELDS = ["expected_runs", "expected_wickets", "expected_batting_matches", "expected_bowling_matches"]


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype={"player_id": "string"})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def numeric(table: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(table.get(column, pd.Series(np.nan, index=table.index)), errors="coerce")


def boolean(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.casefold().isin(["true", "1", "yes"])


def require(table: pd.DataFrame, columns: list[str], filename: str) -> None:
    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}. Regenerate the forward model outputs.")


def records(table: pd.DataFrame) -> list[dict[str, Any]]:
    # pandas' JSON encoder correctly converts NaN, pd.NA and NumPy scalars.
    return json.loads(table.to_json(orient="records", date_format="iso"))


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def disciplines(role: str) -> list[str]:
    if role in ("Batter", "Wicketkeeper-batter"):
        return ["runs"]
    if role == "Bowler":
        return ["wickets"]
    return ["runs", "wickets"]


def scenario_values(row: pd.Series | dict, measure: str) -> tuple[float, float | None, float | None]:
    centre = float(row[f"expected_{measure}"])
    lower = row.get(f"expected_{measure}_conservative")
    upper = row.get(f"expected_{measure}_upside")
    if finite(lower) and finite(upper) and 0 <= float(lower) <= centre <= float(upper):
        return centre, float(lower), float(upper)
    return centre, None, None


@dataclass
class OutputData:
    root: Path
    model: Path
    tables: dict[str, pd.DataFrame]
    metadata: dict[str, dict]
    players: pd.DataFrame
    teams: dict[str, pd.DataFrame]
    shortlists: dict[str, pd.DataFrame]
    history: pd.DataFrame
    forecast_year: int
    as_of_year: int
    warnings: list[str]


def load_outputs(project_root: Path, model_directory: Path | None = None) -> OutputData:
    model = model_directory or project_root / "data" / "processed" / "model"
    tables = {path.name: read_csv(path) for path in sorted(model.glob("*.csv"))}
    metadata = {}
    for path in sorted(model.glob("*.json")):
        with path.open(encoding="utf-8-sig") as handle:
            value = json.load(handle)
        metadata[path.name] = value if isinstance(value, dict) else {"value": value}
    for name in ("auction_candidate_fair_values.csv", "upcoming_performance_predictions.csv"):
        if name not in tables or tables[name].empty:
            raise FileNotFoundError(f"Missing or empty {model / name}. Run the forward analysis first.")
    fair = tables["auction_candidate_fair_values.csv"]
    forecast = tables["upcoming_performance_predictions.csv"]
    require(fair, ["player_id", "player_name_canonical", "predicted_fair_value_inr"], "auction_candidate_fair_values.csv")
    require(forecast, ["player_id", "predicted_ipl_season", "as_of_season", *PREDICTION_FIELDS], "upcoming_performance_predictions.csv")
    for name, frame in (("fair values", fair), ("forecasts", forecast)):
        if frame["player_id"].isna().any() or frame["player_id"].duplicated().any():
            raise ValueError(f"The {name} must contain one non-empty player_id per row.")
    years = numeric(forecast, "predicted_ipl_season").dropna().unique()
    cutoffs = numeric(forecast, "as_of_season").dropna().unique()
    if numeric(forecast, "predicted_ipl_season").isna().any() or numeric(forecast, "as_of_season").isna().any():
        raise ValueError("Every forecast row needs a valid forecast year and as-of year.")
    if len(years) != 1 or len(cutoffs) != 1 or years[0] <= cutoffs[0]:
        raise ValueError("The forecast CSV mixes seasons or has an invalid as-of/forecast year.")
    summary = metadata.get("future_analysis_summary.json", {})
    if summary.get("forecast_year") is not None and int(summary["forecast_year"]) != int(years[0]):
        raise ValueError("The analysis summary belongs to a different forecast year. Rerun the complete forward analysis.")
    if set(fair["player_id"]) != set(forecast["player_id"]):
        raise ValueError("Fair values and performance forecasts contain different player pools. Rerun the complete forward analysis.")
    if "forecast_auction_year" in fair and not numeric(fair, "forecast_auction_year").eq(years[0]).all():
        raise ValueError("The fair-value and performance CSVs refer to different forecast years.")
    for column in PREDICTION_FIELDS:
        values = numeric(forecast, column)
        if not np.isfinite(values).all() or values.lt(0).any():
            raise ValueError(f"{column} contains missing, negative or infinite forecasts.")
    if not np.isfinite(numeric(fair, "predicted_fair_value_inr")).all() or numeric(fair, "predicted_fair_value_inr").lt(0).any():
        raise ValueError("The fair-value CSV contains missing, negative or infinite values.")
    # Forecast fields are owned by the performance CSV, not an older team file.
    overlap = set(fair.columns).intersection(forecast.columns).difference({"player_id"})
    players = fair.drop(columns=sorted(overlap)).merge(forecast, on="player_id", validate="one_to_one")
    players["role_group"] = [
        base.candidate_role_group(row.get("metadata_playing_role"), row.get("role"))
        for _, row in players.iterrows()
    ]
    notes: list[str] = []
    for measure in ("runs", "wickets"):
        lo, hi = f"expected_{measure}_conservative", f"expected_{measure}_upside"
        if lo not in players or hi not in players:
            notes.append(f"{measure.title()} scenario ranges are absent. Expected values are shown without invented ranges.")
        else:
            complete = numeric(players, lo).notna() & numeric(players, hi).notna()
            invalid = complete & (
                numeric(players, lo).lt(0) | numeric(players, lo).gt(players[f"expected_{measure}"])
                | numeric(players, hi).lt(players[f"expected_{measure}"])
            )
            if invalid.any():
                raise ValueError(f"Invalid {measure} scenario ordering. Regenerate predictions rather than editing chart values.")
    selected_teams = metadata.get("future_analysis_summary.json", {}).get("team_profiles_used")
    teams: dict[str, pd.DataFrame] = {}
    shortlists: dict[str, pd.DataFrame] = {}
    for name, table in tables.items():
        if not (name.startswith("team_fit_") or name.startswith("team_role_shortlists_")) or table.empty:
            continue
        require(table, ["team", "player_id", "team_fit_score", "expected_runs", "expected_wickets"], name)
        names = table["team"].dropna().astype(str).str.strip().unique()
        if len(names) != 1:
            raise ValueError(f"{name} must belong to exactly one team.")
        team = str(names[0])
        if selected_teams is not None and team not in selected_teams:
            notes.append(f"Older output excluded from the active team report: {name}.")
            continue
        joined = table.merge(forecast, on="player_id", suffixes=("_team", "_current"), validate="many_to_one")
        if len(joined) != len(table):
            raise ValueError(f"{name} contains players outside the current forecast. Rerun the all-team analysis.")
        for field in ("expected_runs", "expected_wickets"):
            if not np.allclose(numeric(joined, field + "_team"), numeric(joined, field + "_current"), rtol=1e-5, atol=1e-4):
                raise ValueError(f"{name} is stale: its {field} does not match the current forecast. Rerun the all-team analysis first.")
        if name.startswith("team_fit_"):
            if table["player_id"].duplicated().any():
                raise ValueError(f"Duplicate player IDs in {name}.")
            teams[team] = table.sort_values("rank" if "rank" in table else "team_fit_score", ascending="rank" in table)
        else:
            shortlists[team] = table.copy()
    if selected_teams:
        for team in selected_teams:
            if team not in teams:
                notes.append(f"Missing team ranking: {team}. Rerun the all-team analysis to include it.")
    if teams:
        first = next(iter(teams.values())).set_index("player_id")
        for field in ("investment_score", "investment_label"):
            if field in first:
                for team, frame in teams.items():
                    if field not in frame:
                        raise ValueError(f"{team} is missing {field}. Rerun the all-team analysis.")
                    other = frame.set_index("player_id")[field]
                    shared = first.index.intersection(other.index)
                    left, right = first.loc[shared, field], other.loc[shared]
                    same = np.allclose(pd.to_numeric(left), pd.to_numeric(right), atol=1e-5) if field == "investment_score" else left.astype(str).equals(right.astype(str))
                    if not same:
                        raise ValueError("Team-independent investment scores differ across team CSVs. Rerun the all-team analysis.")
                players[field] = players["player_id"].map(first[field])
    recommendations = tables.get("future_team_recommendations.csv", pd.DataFrame())
    if not recommendations.empty:
        for team, group in recommendations.groupby("team"):
            if team not in teams:
                continue
            current = teams[team].set_index("player_id")
            if not set(group["player_id"]).issubset(current.index):
                raise ValueError("The combined recommendation CSV is stale. Rerun the all-team analysis.")
            for column in ("expected_runs", "expected_wickets", "team_fit_score", "investment_score"):
                if column in group and column in current:
                    if not np.allclose(numeric(group, column), pd.to_numeric(current.loc[group["player_id"], column]), rtol=1e-5, atol=1e-4):
                        raise ValueError(f"The combined recommendation CSV has outdated {column}. Rerun the all-team analysis.")
    history_path = project_root / "data" / "processed" / "ipl" / "player_season_features.csv"
    history = read_csv(history_path) if history_path.exists() else pd.DataFrame()
    if "season_end_year" not in history and "season" in history:
        history = history.rename(columns={"season": "season_end_year"})
    if not history.empty:
        require(history, ["player_id", "season_end_year", "runs", "wickets"], history_path.name)
        history = history.loc[numeric(history, "season_end_year").le(cutoffs[0])].copy()
    else:
        notes.append("IPL season history is not available here. Player forecasts are still shown, without fabricated historical bars.")
    return OutputData(project_root, model, tables, metadata, players, teams, shortlists, history, int(years[0]), int(cutoffs[0]), notes)


def finish(figure: plt.Figure, destination: Path, footer: str = "") -> Path:
    if footer:
        figure.text(0.02, 0.015, footer, fontsize=8, color="#55616e", ha="left", va="bottom", wrap=True)
    figure.tight_layout(rect=(0, 0.055 if footer else 0, 1, 0.94))
    figure.savefig(destination, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return destination


def horizontal(axis: plt.Axes, labels: list[str], values: pd.Series | np.ndarray, title: str, colour: str = BLUE, score: bool = False, annotate: bool = True) -> None:
    values = np.asarray(values, dtype=float)
    positions = np.arange(len(labels))
    axis.barh(positions, values, color=colour, height=0.62)
    axis.set_yticks(positions, labels, fontsize=9)
    axis.invert_yaxis()
    axis.set_title(title, loc="left", fontsize=11, pad=12)
    available = values[np.isfinite(values)]
    maximum = max(float(available.max()), 1.0) if len(available) else 1.0
    axis.set_xlim(0, 100 if score else maximum * 1.24)
    for i, value in enumerate(values):
        if np.isfinite(value) and annotate:
            axis.annotate(f"{value:.1f}" if score or maximum < 30 else f"{value:,.0f}", (value, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
        elif not np.isfinite(value):
            axis.text(0.02, i, "Not recorded", transform=axis.get_yaxis_transform(), va="center", fontsize=8)
    axis.grid(axis="x", alpha=0.17)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)


def plot_leaders(data: OutputData, destination: Path, measure: str, top_n: int) -> Path:
    allowed = [role for role in ROLES if measure in disciplines(role)]
    top = data.players.loc[data.players["role_group"].isin(allowed)].nlargest(top_n, f"expected_{measure}")
    figure, axis = plt.subplots(figsize=(12, max(5, len(top) * 0.43 + 2)))
    labels = (top["player_name_canonical"] + " — " + top["role_group"]).tolist()
    horizontal(axis, labels, top[f"expected_{measure}"], f"Expected {measure}", TEAL if measure == "wickets" else BLUE, annotate=False)
    for i, (_, row) in enumerate(top.iterrows()):
        centre, lower, upper = scenario_values(row, measure)
        if lower is not None:
            axis.errorbar(centre, i, xerr=[[centre - lower], [upper - centre]], fmt="none", color=GOLD, capsize=4, linewidth=1.5)
            axis.set_xlim(0, max(axis.get_xlim()[1], upper * 1.16))
        anchor = upper if upper is not None else centre
        axis.annotate(f"{centre:,.0f}" if measure == "runs" else f"{centre:.1f}", (anchor, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
    figure.suptitle(f"IPL {data.forecast_year}: top {len(top)} {measure} forecasts", fontsize=16, fontweight="bold")
    return finish(figure, destination, "Bars: expected totals. Whiskers, when available: conservative/upside scenarios, not confidence intervals. Conditional on appearing in IPL.")


def plot_pool(data: OutputData, destination: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    role_counts = data.players["role_group"].value_counts().reindex(ROLES, fill_value=0)
    horizontal(axes[0, 0], role_counts.index.tolist(), role_counts, "Forecast players by listed role")
    for axis, measure, colour in zip(axes[0, 1:], ["runs"], [BLUE]):
        axis.hist(numeric(data.players, f"expected_{measure}"), bins=35, color=colour, alpha=0.85)
        axis.set_title("Full-pool expected runs distribution", loc="left")
        axis.set_xlabel("Expected IPL runs")
        axis.set_ylabel("Players")
    axes[1, 0].hist(numeric(data.players, "expected_wickets"), bins=35, color=TEAL, alpha=0.85)
    axes[1, 0].set_title("Full-pool expected wickets distribution", loc="left")
    axes[1, 0].set_xlabel("Expected IPL wickets")
    axes[1, 0].set_ylabel("Players")
    fields = [("has_recent_ipl_history", "IPL"), ("has_recent_t20_history", "Other T20"), ("has_recent_odi_history", "ODI")]
    counts = [int(boolean(data.players[field]).sum()) if field in data.players else np.nan for field, _ in fields]
    horizontal(axes[1, 1], [label for _, label in fields], counts, "Recent performance evidence available")
    figure.suptitle(f"IPL {data.forecast_year}: all {len(data.players):,} forecast players", fontsize=16, fontweight="bold")
    return finish(figure, destination, "Evidence counts overlap: one player may have IPL, other T20 and ODI history. Unknown roles are shown as Other, not inferred as wicketkeepers.")


def plot_opportunities(data: OutputData, destination: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, field, label, colour in zip(axes, PREDICTION_FIELDS[2:], ["Batting", "Bowling"], [BLUE, TEAL]):
        axis.hist(numeric(data.players, field), bins=25, color=colour, alpha=0.85)
        axis.set_xlabel(f"Expected {label.lower()} appearances")
        axis.set_ylabel("Players")
        axis.set_title(f"{label} opportunity", loc="left")
    figure.suptitle(f"IPL {data.forecast_year}: expected playing opportunity", fontsize=15, fontweight="bold")
    return finish(figure, destination, "Batting appearances and bowling appearances are different quantities; neither is a probability of selection or an injury-availability forecast.")


def plot_value_distribution(data: OutputData, destination: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    values = numeric(data.players, "predicted_fair_value_inr") / 10_000_000
    axes[0].hist(values, bins=35, color=BLUE, alpha=0.85)
    axes[0].set_xlabel("Estimated fair value (₹ crore)")
    axes[0].set_ylabel("Players")
    axes[0].set_title("All-player fair-value distribution", loc="left")
    groups, labels = [], []
    for role in ROLES:
        subset = values.loc[data.players["role_group"].eq(role)].dropna()
        if len(subset):
            groups.append(subset.to_numpy())
            labels.append(role.replace("Wicketkeeper-batter", "Keeper-batter"))
    axes[1].boxplot(groups, showfliers=True)
    axes[1].set_xticks(range(1, len(labels) + 1), [label.replace("-batter", "\nbatter") for label in labels], fontsize=8)
    axes[1].set_ylabel("Estimated fair value (₹ crore)")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Median, spread and outliers by role", loc="left")
    figure.suptitle(f"IPL {data.forecast_year}: fair-value estimates", fontsize=15, fontweight="bold")
    return finish(figure, destination, "These are modelled fair values, not auction bids, sale guarantees, or totals that can be added into an optimised squad budget.")


def plot_team_dashboard(data: OutputData, team: str, destination: Path, top_n: int) -> Path:
    top = data.teams[team].head(top_n).copy()
    labels = top["player_name_canonical"].tolist()
    figure, axes = plt.subplots(2, 3, figsize=(21, max(8, len(top) * 0.72 + 2)))
    measures = [
        ("team_fit_score", "Team fit (0–100)", BLUE, True),
        ("investment_score", "Investment score (0–100)", TEAL, True),
        ("predicted_fair_value_inr", "Estimated fair value (₹ crore)", GREY, False),
        ("expected_runs", "Expected runs", BLUE, False),
        ("expected_wickets", "Expected wickets", TEAL, False),
    ]
    for axis, (field, title, colour, score) in zip(axes.flat, measures):
        if field not in top:
            axis.text(0.5, 0.5, f"{title} unavailable", ha="center", transform=axis.transAxes)
            axis.set_axis_off()
            continue
        values = numeric(top, field)
        if field.endswith("_inr"):
            values = values / 10_000_000
        horizontal(axis, labels, values, title, colour, score)
        if field == "team_fit_score":
            for threshold in (40, 60):
                axis.axvline(threshold, color=GREY, linestyle="--", linewidth=0.8)
    status = axes[1, 2]
    status.set_axis_off()
    status.set_title("Role and per-player budget checks", loc="left", pad=12)
    for i, (_, row) in enumerate(top.iterrows()):
        matched = bool(boolean(pd.Series([row.get("role_match")])).iloc[0])
        affordable = bool(boolean(pd.Series([row.get("within_budget")])).iloc[0])
        line = f"{row['player_name_canonical']}\n{'Role match' if matched else 'Role mismatch'}; {'within limit' if affordable else 'over limit'}"
        status.text(0, 1 - i / max(len(top), 1), line, transform=status.transAxes, fontsize=9, va="top", color="#263544")
    figure.suptitle(f"{team}: IPL {data.forecast_year} shortlist", fontsize=17, fontweight="bold")
    return finish(figure, destination, "Fit: >60 Strong fit; >40 to 60 Consider; ≤40 Low fit. Investment is a relative performance/price score, not financial ROI. Rankings are not a squad optimisation.")


def plot_role_shortlist(data: OutputData, team: str, destination: Path) -> Path:
    table = data.shortlists[team]
    require(table, ["shortlist_role", "role_rank", "player_name_canonical"], f"Role shortlist for {team}")
    figure, axes = plt.subplots(2, 2, figsize=(15, 9))
    for role, axis in zip(ROLES[:4], axes.flat):
        subset = table.loc[table["shortlist_role"].eq(role)].sort_values("role_rank")
        if subset.empty:
            axis.text(0.5, 0.5, "No candidates in this role", transform=axis.transAxes, ha="center")
            axis.set_title(role, loc="left")
            axis.set_axis_off()
            continue
        labels = []
        for _, row in subset.iterrows():
            labels.append(f"{row['player_name_canonical']}  (₹{float(row['predicted_fair_value_inr']) / 10_000_000:.1f} cr)")
        horizontal(axis, labels, numeric(subset, "team_fit_score"), role + ": team fit", ROLE_COLOURS[role], True)
    figure.suptitle(f"{team}: role-by-role market scan", fontsize=16, fontweight="bold")
    return finish(figure, destination, "These are separate role lists, not a recommended combined squad. Some roles may not match the team's chosen requirement; use the report's role/budget checks.")


def plot_investments(data: OutputData, destination: Path, top_n: int) -> Path | None:
    if "investment_score" not in data.players or numeric(data.players, "investment_score").notna().sum() == 0:
        return None
    top = data.players.nlargest(top_n, "investment_score")
    figure, axes = plt.subplots(1, 2, figsize=(15, max(5, len(top) * 0.46 + 2)))
    labels = (top["player_name_canonical"] + " — " + top["role_group"]).tolist()
    horizontal(axes[0], labels, top["investment_score"], "Investment score (0–100)", TEAL, True)
    horizontal(axes[1], labels, numeric(top, "predicted_fair_value_inr") / 10_000_000, "Estimated fair value (₹ crore)", GREY)
    figure.suptitle(f"IPL {data.forecast_year}: performance-for-price shortlist", fontsize=16, fontweight="bold")
    return finish(figure, destination, "A high score means relatively attractive predicted role-specific performance for the estimated price. It is not financial profit or proof of auction availability.")


def plot_comparisons(data: OutputData, destination: Path) -> Path | None:
    specs = [
        ("performance_model_comparison.csv", "model", "scaled_mae", "Performance model: scaled absolute error", 1),
        ("performance_calibration_comparison.csv", "forecast_method", "scaled_mae", "Direct totals versus calibrated forecast", 1),
        ("fair_value_model_comparison.csv", "model", "price_mae_inr", "Fair-value model: mean absolute error (₹ crore)", 10_000_000),
        ("fair_value_model_comparison.csv", "model", "ratio_mae", "Fair-value model: market-ratio error", 1),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(15, 8))
    plotted = False
    for axis, (filename, category, value, title, scale) in zip(axes.flat, specs):
        table = data.tables.get(filename, pd.DataFrame())
        if table.empty or category not in table or value not in table:
            axis.set_axis_off()
            axis.text(0.5, 0.5, f"{title}\nNot available in this run", ha="center", transform=axis.transAxes)
            continue
        plotted = True
        ordered = table.sort_values(value)
        vals = numeric(ordered, value) / scale
        bars = axis.bar(range(len(ordered)), vals, color=[TEAL] + [GREY] * (len(ordered) - 1))
        labels = ordered[category].astype(str).str.replace("_", " ").tolist()
        axis.set_xticks(range(len(ordered)), labels, fontsize=9)
        axis.set_ylim(0, max(float(vals.max()), 0.01) * 1.3)
        axis.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=4, fontsize=9)
        axis.set_title(title, loc="left", fontsize=11)
        axis.grid(axis="y", alpha=0.15)
    if not plotted:
        plt.close(figure)
        return None
    figure.suptitle("Model and calibration comparisons: lower error is better", fontsize=16, fontweight="bold")
    return finish(figure, destination, "These are the saved chronological evaluation/selection results. They are not a new untouched test, and no model is retrained by this report.")


def plot_validation(data: OutputData, destination: Path, price: bool = False) -> Path | None:
    filename = "fair_value_test_predictions.csv" if price else "performance_test_predictions.csv"
    table = data.tables.get(filename, pd.DataFrame())
    if table.empty:
        return None
    specs = [("price_inr", "predicted_fair_value_inr", "Auction price (₹ crore)", 10_000_000)] if price else [
        (f"next_{field}", f"predicted_next_{field}", field.replace("_", " ").title(), 1)
        for field in ("runs", "wickets", "batting_matches", "bowling_matches")
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5)) if price else plt.subplots(2, 2, figsize=(13, 9))
    for axis, (actual_col, predicted_col, title, scale) in zip(axes.flat, specs):
        require(table, [actual_col, predicted_col], filename)
        pair = pd.DataFrame({"actual": numeric(table, actual_col), "predicted": numeric(table, predicted_col)}).dropna() / scale
        if pair.empty:
            axis.set_axis_off()
            continue
        axis.scatter(pair["actual"], pair["predicted"], s=20, alpha=0.4, color=BLUE, linewidths=0)
        maximum = max(float(pair.max().max()), 1)
        axis.plot([0, maximum], [0, maximum], linestyle="--", color=GREY, linewidth=1)
        axis.set_xlim(0, maximum * 1.05)
        axis.set_ylim(0, maximum * 1.05)
        axis.set_xlabel("Actual")
        axis.set_ylabel("Predicted")
        error = np.mean(np.abs(pair["actual"] - pair["predicted"]))
        axis.set_title(f"{title} — average absolute error {error:.2f}", loc="left", fontsize=10)
    if price:
        errors = (numeric(table, "predicted_fair_value_inr") - numeric(table, "price_inr")) / 10_000_000
        axes[1].hist(errors.dropna(), bins=25, color=TEAL)
        axes[1].axvline(0, color=GREY, linestyle="--")
        axes[1].set_xlabel("Predicted minus actual price (₹ crore)")
        axes[1].set_ylabel("Player-auction rows")
        axes[1].set_title("Negative: underpricing; positive: overpricing", loc="left", fontsize=10)
    figure.suptitle("Historical auction-price evaluation" if price else "Historical IPL performance evaluation", fontsize=16, fontweight="bold")
    return finish(figure, destination, "All valid saved evaluation rows are included. On the scatter, points below the dashed line are underpredictions; points above it are overpredictions.")


def plot_features(data: OutputData, destination: Path) -> Path | None:
    table = data.tables.get("fair_value_feature_importance.csv", pd.DataFrame())
    if table.empty or not {"feature", "importance"}.issubset(table.columns):
        return None
    top = table.nlargest(15, "importance").copy()
    labels = top["feature"].astype(str).str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False).str.replace("_", " ", regex=False).tolist()
    figure, axis = plt.subplots(figsize=(13, 8))
    horizontal(axis, labels, numeric(top, "importance") * 100, "Relative model importance (%)", BLUE)
    figure.suptitle("What the fair-value model used most", fontsize=16, fontweight="bold")
    return finish(figure, destination, "Top 15 transformed inputs. Importance is a model diagnostic, not a causal explanation or a player-quality score. Missing-data indicators can also receive importance.")


def plot_audit(data: OutputData, destination: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    excluded = data.tables.get("auction_candidate_excluded_players.csv", pd.DataFrame())
    excluded_count = len(excluded) if "auction_candidate_excluded_players.csv" in data.tables else np.nan
    horizontal(axes[0, 0], ["Forecast / valued", "Not scored"], [len(data.players), excluded_count], "Candidate coverage", BLUE)
    country_column = "country_used_for_screen" if "country_used_for_screen" in data.players else "nationality"
    nations = data.players.get(country_column, pd.Series("Unknown", index=data.players.index)).fillna("Unknown").astype(str)
    nations = nations.mask(nations.str.strip().isin(["", "nan", "None"]), "Unknown")
    countries = nations.value_counts().head(10)
    title = "Top 10 screened countries in the forecast pool" if country_column == "country_used_for_screen" else "Top 10 nationalities in the forecast pool"
    horizontal(axes[0, 1], countries.index.tolist(), countries, title, GREY)
    training = data.tables.get("player_auction_training.csv", pd.DataFrame())
    if not training.empty and {"auction_year", "eligible_for_price_model"}.issubset(training):
        counted = training.assign(eligible=boolean(training["eligible_for_price_model"]))
        total = counted.groupby("auction_year").size()
        usable = counted.loc[counted["eligible"]].groupby("auction_year").size().reindex(total.index, fill_value=0)
        axes[1, 0].bar(total.index.astype(int).astype(str), usable, color=BLUE, label="Eligible")
        axes[1, 0].bar(total.index.astype(int).astype(str), total - usable, bottom=usable, color="#ced4da", label="Excluded")
        axes[1, 0].tick_params(axis="x", labelrotation=45)
        axes[1, 0].set_title("Auction training rows by year", loc="left")
        axes[1, 0].set_ylabel("Player-auction rows")
        axes[1, 0].legend()
    else:
        axes[1, 0].set_axis_off()
        axes[1, 0].text(0.5, 0.5, "Auction training audit unavailable", ha="center", transform=axes[1, 0].transAxes)
    excluded_training = data.tables.get("player_auction_excluded_rows.csv", pd.DataFrame())
    if not excluded_training.empty:
        counts = {}
        if "player_id" in excluded_training:
            counts["Missing player ID"] = int(excluded_training["player_id"].isna().sum())
        if "has_any_recent_performance" in excluded_training:
            counts["No recent supported history"] = int((~boolean(excluded_training["has_any_recent_performance"])).sum())
        if "nationality" in excluded_training:
            counts["Missing nationality"] = int(excluded_training["nationality"].isna().sum())
        if counts:
            horizontal(axes[1, 1], list(counts), list(counts.values()), "Signals in excluded training rows (can overlap)", GREY)
        else:
            axes[1, 1].set_axis_off()
    else:
        axes[1, 1].set_axis_off()
    figure.suptitle("Data coverage and training exclusions", fontsize=16, fontweight="bold")
    return finish(figure, destination, "Not-scored players are not automatically bad players or unavailable players. Coverage reflects the supported data window. Exclusion signals are not additive causes.")


def plot_eligibility(data: OutputData, destination: Path) -> Path | None:
    audit = data.tables.get("auction_candidate_eligibility.csv", pd.DataFrame())
    if audit.empty or "eligible_for_candidate_pool" not in audit:
        return None
    accepted = boolean(audit["eligible_for_candidate_pool"])
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    horizontal(axes[0, 0], ["Pass pool screen", "Excluded from pool"], [int(accepted.sum()), int((~accepted).sum())], "Candidate eligibility", BLUE)
    reasons = audit.loc[~accepted, "exclusion_reason"].fillna("Not recorded").str.split("; ").explode().value_counts()
    if not reasons.empty:
        labels = reasons.index.astype(str).str.replace("_", " ", regex=False).tolist()
        horizontal(axes[0, 1], labels, reasons.to_numpy(), "Exclusion reasons (can overlap)", GREY)
    else:
        axes[0, 1].set_axis_off()
        axes[0, 1].text(0.5, 0.5, "No exclusions in this snapshot", ha="center", transform=axes[0, 1].transAxes)
    status = audit.loc[accepted, "availability_status"].fillna("unknown").value_counts()
    if not status.empty:
        horizontal(axes[1, 0], status.index.tolist(), status.to_numpy(), "Availability notes for accepted players", TEAL)
    else:
        axes[1, 0].set_axis_off()
    counts = [int(numeric(audit.loc[accepted], "recent_ipl_matches").gt(0).sum()), int(numeric(audit.loc[accepted], "recent_international_matches").gt(0).sum())]
    horizontal(axes[1, 1], ["Recorded recent IPL", "Recorded recent international"], counts, "Qualifying evidence for accepted players", BLUE)
    figure.suptitle(f"IPL {data.forecast_year}: recent-history and availability screen", fontsize=16, fontweight="bold")
    return finish(figure, destination, "Activity evidence is not medical clearance or confirmed IPL registration. Unknown status is assumed available unless strict confirmation is enabled. A player can have both IPL and international history.")


def plot_player_forecast(data: OutputData, player_id: str, destination: Path) -> Path:
    row = data.players.set_index("player_id").loc[player_id]
    measures = disciplines(row["role_group"])
    figure, axes = plt.subplots(len(measures), 2, figsize=(14, 4.3 * len(measures) + 1.6), squeeze=False)
    seasons = data.history.loc[data.history["player_id"].eq(player_id)].sort_values("season_end_year") if not data.history.empty else pd.DataFrame()
    for (left, right), measure in zip(axes, measures):
        colour = TEAL if measure == "wickets" else BLUE
        years, actuals = [], []
        if not seasons.empty:
            valid = seasons.loc[numeric(seasons, measure).notna()]
            years = numeric(valid, "season_end_year").astype(int).tolist()
            actuals = numeric(valid, measure).tolist()
        positions = np.arange(len(years))
        if years:
            left.bar(positions, actuals, color="#bdc9d7", label="Actual IPL history")
        centre, lower, upper = scenario_values(row, measure)
        forecast_position = len(years) + (0.6 if years else 0)
        left.bar(forecast_position, centre, color=colour, width=0.65, label="Expected forecast")
        if not years:
            left.set_xlim(-1.3, 1.3)
        if lower is not None:
            left.errorbar(forecast_position, centre, yerr=[[centre - lower], [upper - centre]], fmt="none", color=GOLD, capsize=6, label="Scenario range")
        left.set_xticks([*positions, forecast_position], [*map(str, years), f"{data.forecast_year}\nForecast"], rotation=45 if len(years) > 10 else 0)
        left.set_title(f"{measure.title()}: actual seasons and forecast" if years else f"{measure.title()}: forecast (IPL history not supplied)", loc="left")
        left.set_ylim(bottom=0)
        left.legend(fontsize=8)
        appearance = "batting" if measure == "runs" else "bowling"
        matches = float(row[f"expected_{appearance}_matches"])
        rate = centre / matches if matches > 0 else None
        text = [f"Expected {measure}: {centre:,.0f}" if measure == "runs" else f"Expected wickets: {centre:.1f}"]
        decimals = 0 if measure == "runs" else 1
        text.append(f"Conservative / upside: {lower:,.{decimals}f} / {upper:,.{decimals}f}" if lower is not None else "Scenario range: not supplied")
        text.append(f"Expected {appearance} appearances: {matches:.1f}")
        text.append(f"Implied {measure} per appearance: {rate:.2f}" if rate is not None else "Per-appearance rate: unavailable")
        text.append(f"Estimated fair value: ₹{float(row['predicted_fair_value_inr']) / 10_000_000:.2f} crore")
        right.set_axis_off()
        for i, line in enumerate(text):
            right.text(0.03, 0.9 - i * 0.16, line, transform=right.transAxes, fontsize=12, color="#263544")
    figure.suptitle(f"{row['player_name_canonical']} — {row['role_group']}\nIPL {data.forecast_year} performance forecast", fontsize=17, fontweight="bold")
    return finish(figure, destination, "Forecasts are conditional on appearing in IPL. Scenario whiskers are not confidence intervals. No strike-rate/economy or wicketkeeping-performance forecast is produced by the saved model.")


def build_payload(data: OutputData, destination: Path, charts: list[dict], coverage: list[dict], top_n: int) -> dict:
    team_records = {}
    # Avoid repeating the entire player/forecast record ten times in the HTML.
    common = set(data.players.columns).difference({"player_id"})
    for team, ranking in data.teams.items():
        keep = [column for column in ranking if column not in common or column in {"player_id", "rank", "investment_score", "investment_label"}]
        team_records[team] = records(ranking[keep])
    history = data.history.copy()
    keep_history = [field for field in ("player_id", "season_end_year", "runs", "wickets", "batting_matches", "bowling_matches", "batting_balls", "bowling_balls", "runs_conceded", "strike_rate", "economy") if field in history]
    summary = data.metadata.get("future_analysis_summary.json", {})
    return {
        "year": data.forecast_year, "asOf": data.as_of_year,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "topN": top_n, "players": records(data.players), "teams": team_records,
        "shortlists": {team: records(table) for team, table in data.shortlists.items()},
        "history": records(history[keep_history]) if keep_history else [],
        "charts": charts, "coverage": coverage, "warnings": data.warnings,
        "metrics": {key: value for key, value in data.metadata.items() if key != "future_analysis_summary.json"},
        "configuration": {key: value for key, value in summary.items() if key != "team_recommendation_file"},
        "assumption": summary.get("candidate_pool_assumption", "All recently active candidates are treated as available. Retentions, releases and registrations are not predicted."),
    }


def create_prediction_visualizations(
    project_root: str | Path | None = None,
    model_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
    top_n: int = 15,
    player_names: list[str] | None = None,
    all_players: bool = False,
) -> tuple[Path, dict]:
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if player_names and all_players:
        raise ValueError("Choose player_names or all_players, not both.")
    root = Path(project_root) if project_root else project_root_from_file()
    data = load_outputs(root, Path(model_directory) if model_directory else None)
    destination = Path(output_directory) if output_directory else data.model / "visualizations" / "predictions"
    if destination.resolve() == data.model.resolve():
        raise ValueError("Use a visualisation subfolder, not the model source directory, for report output.")
    destination.mkdir(parents=True, exist_ok=True)
    charts: list[dict] = []

    def add(path: Path | None, title: str, section: str, sources: list[str], note: str, team: str = "") -> None:
        if path is not None:
            charts.append({"file": path.relative_to(destination).as_posix(), "title": title, "section": section, "sources": sources, "note": note, "team": team})

    fair = data.tables["auction_candidate_fair_values.csv"]
    forecast = data.tables["upcoming_performance_predictions.csv"]
    forecast_sources = ["auction_candidate_fair_values.csv", "upcoming_performance_predictions.csv"]
    with plt.rc_context({"font.family": "DejaVu Sans", "axes.titlesize": 11, "axes.labelsize": 10}):
        add(plot_pool(data, destination / "forecast_pool.png"), "The full forecast pool", "Overview", forecast_sources, "All players, their roles, forecast distribution and available recent evidence.")
        for measure in ("runs", "wickets"):
            add(plot_leaders(data, destination / f"predicted_{measure}_leaders.png", measure, top_n), f"Leading {measure} forecasts", "Forecasts", forecast_sources, f"Top {top_n} for readability; every player remains in the searchable explorer.")
        add(plot_opportunities(data, destination / "playing_opportunity.png"), "Expected batting and bowling opportunity", "Forecasts", ["upcoming_performance_predictions.csv"], "Separate appearances, not guaranteed selection.")
        add(plot_value_distribution(data, destination / "fair_value_distribution.png"), "Fair-value distribution", "Fair values", ["auction_candidate_fair_values.csv"], "Full pool and role-specific distributions in ₹ crore.")
        add(base.plot_value_vs_performance(fair, forecast, destination), "Value against predicted batting output", "Fair values", forecast_sources, "Every candidate is a point; colour represents predicted wickets.")
        add(base.plot_full_candidate_pool_hexbin(fair, forecast, destination), "Full-pool density and bowling output", "Fair values", forecast_sources, "All candidates are included, not only the labelled leaders.")
        add(base.plot_role_aware_candidate_board(fair, forecast, destination), "Role-aware auction board", "Forecasts", forecast_sources, "Separate batter, keeper-batter, bowler and all-rounder sections.")
        add(plot_investments(data, destination / "investment_leaders.png", top_n), "Performance for the estimated price", "Investment", [path.name for path in data.model.glob("team_fit_*.csv")], "Investment score is relative value, not financial return.")
        for team in sorted(data.teams):
            team_source = [path.name for path in data.model.glob("team_fit_*.csv") if path.stem.endswith(team.replace(" ", "_"))]
            add(plot_team_dashboard(data, team, destination / f"team_{base.file_safe_name(team)}.png", min(top_n, 10)), f"{team}: recommendations", "Teams", team_source, "Fit, investment, price, performance and role/budget checks for the ranked leaders.", team)
            if team in data.shortlists:
                shortlist_source = [path.name for path in data.model.glob("team_role_shortlists_*.csv") if path.stem.endswith(team.replace(" ", "_"))]
                add(plot_role_shortlist(data, team, destination / f"roles_{base.file_safe_name(team)}.png"), f"{team}: role shortlists", "Teams", shortlist_source, "A market scan for every role, not a combined squad-buying recommendation.", team)
        recommendations = data.tables.get("future_team_recommendations.csv", pd.DataFrame())
        if not recommendations.empty:
            add(base.plot_all_team_summary(recommendations, destination), "Each team's highest-ranked candidate", "Teams", ["future_team_recommendations.csv"], "Scores compare suitability to each team's own assumptions, not actual team strength.")
        add(plot_comparisons(data, destination / "model_comparisons.png"), "Model and calibration comparisons", "Evaluation", ["performance_model_comparison.csv", "performance_calibration_comparison.csv", "fair_value_model_comparison.csv", "performance_metrics.json", "fair_value_metrics.json"], "Saved evaluation errors; lower is better. Missing comparisons are explicitly marked.")
        add(plot_validation(data, destination / "performance_evaluation.png"), "Actual against forecast performance", "Evaluation", ["performance_test_predictions.csv", "performance_metrics.json"], "All saved held-out player-season rows, including zero outputs.")
        add(plot_validation(data, destination / "price_evaluation.png", price=True), "Actual against forecast auction price", "Evaluation", ["fair_value_test_predictions.csv", "fair_value_metrics.json"], "Price-error units are ₹ crore, not raw INR or percentages.")
        add(plot_features(data, destination / "price_feature_importance.png"), "Fair-value model inputs", "Diagnostics", ["fair_value_feature_importance.csv"], "Transformed-input importance is not a causal explanation.")
        add(plot_audit(data, destination / "data_coverage.png"), "Data coverage and exclusions", "Diagnostics", ["auction_candidate_excluded_players.csv", "player_auction_training.csv", "player_auction_excluded_rows.csv", "future_analysis_summary.json"], "Counts reconcile to the saved output files; missing history is not a low-performance label.")
        add(plot_eligibility(data, destination / "candidate_eligibility.png"), "Who passed the candidate-pool screen?", "Diagnostics", ["auction_candidate_eligibility.csv", "auction_candidate_excluded_players.csv", "future_analysis_summary.json"], "Recent IPL/international evidence, explicit exclusions and unknown availability are reported separately.")
        selected = data.players if all_players else data.players.iloc[:0]
        if player_names:
            found = []
            for text in player_names:
                exact = data.players.loc[data.players["player_name_canonical"].str.casefold().eq(text.casefold()) | data.players["player_id"].eq(text)]
                matches = exact if not exact.empty else data.players.loc[data.players["player_name_canonical"].str.contains(text, case=False, regex=False, na=False)]
                if len(matches) != 1:
                    raise ValueError(f"Player {text!r} is missing or ambiguous. Use an exact canonical name or player ID. Matches: {matches['player_name_canonical'].head(10).tolist()}")
                found.append(matches)
            selected = pd.concat(found, ignore_index=True).drop_duplicates("player_id")
        if not selected.empty:
            player_destination = destination / "players"
            player_destination.mkdir(exist_ok=True)
            for _, row in selected.iterrows():
                path = player_destination / f"{base.file_safe_name(row['player_name_canonical'])}_{base.file_safe_name(str(row['player_id']))}.png"
                add(plot_player_forecast(data, str(row["player_id"]), path), f"{row['player_name_canonical']}: player forecast", "Players", forecast_sources, "Role-aware actual history and forecast, with scenarios when supplied.")
    coverage: list[dict] = []
    for path in sorted(data.model.iterdir()):
        if not path.is_file() or path.suffix.casefold() not in {".csv", ".json", ".joblib"}:
            continue
        related = [chart["title"] for chart in charts if path.name in chart["sources"]]
        if path.name in {"auction_candidate_fair_values.csv", "upcoming_performance_predictions.csv"} or path.name.startswith("team_fit_") or path.name.startswith("team_role_shortlists_"):
            related.append("Searchable player/team explorer")
        purpose = "Saved runtime model, not chart data; never loaded by this report." if path.suffix == ".joblib" else "; ".join(related) or "Additional source file; available for inspection below."
        coverage.append({"file": path.name, "rows": len(data.tables[path.name]) if path.name in data.tables else None, "view": purpose, "href": Path(os.path.relpath(path, destination)).as_posix()})
    payload = build_payload(data, destination, charts, coverage, top_n)
    report_path = destination / "index.html"
    write_report(payload, report_path)
    manifest = {
        "forecast_year": data.forecast_year, "forecast_players": len(data.players), "teams": sorted(data.teams),
        "charts_created": len(charts), "source_files": coverage, "warnings": data.warnings,
        "report": report_path.name, "charts": charts,
    }
    (destination / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root_from_file())
    parser.add_argument("--model-directory", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=15)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--player", action="append", dest="players", help="Export a player PNG too; repeat for more players. Canonical name or player ID.")
    selection.add_argument("--all-players", action="store_true", help="Also export individual PNGs for all forecast players; may take a long time.")
    args = parser.parse_args()
    path, manifest = create_prediction_visualizations(args.project_root, args.model_directory, args.output_directory, args.top_n, args.players, args.all_players)
    print(f"Created {manifest['charts_created']} charts for {len(manifest['teams'])} teams.")
    print(f"All {manifest['forecast_players']:,} forecast players are searchable in the report.")
    for warning in manifest["warnings"]:
        print(f"Note: {warning}")
    print(f"Open in your browser: {path.resolve()}")


if __name__ == "__main__":
    main()
