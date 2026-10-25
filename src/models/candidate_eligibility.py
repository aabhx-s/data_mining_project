"""Auditable next-season candidate screening, separate from performance scoring.

Recent participation is evidence of cricket activity, not proof of fitness,
registration, IPL selection, release/retention, or legal eligibility. The default
rule uses IPL or the supported international ODI/T20I/World Cup archives. Other
domestic competitions can inform a forecast but do not qualify a candidate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


IPL_SOURCES = ("ipl",)
INTERNATIONAL_SOURCES = ("odi", "t20", "t20_wc")
DOMESTIC_SOURCES = ("bbl", "smat")
BLOCKED_STATUSES = {
    "unavailable", "ruled_out", "withdrawn", "suspended", "ineligible",
    "retired_from_ipl", "retired_from_all_cricket",
}
ALLOWED_STATUSES = {"available", "unknown", *BLOCKED_STATUSES}
AUDIT_COLUMNS = [
    "activity_window_start", "activity_window_end", "recent_ipl_matches",
    "recent_international_matches", "recent_domestic_matches",
    "recent_qualifying_matches", "last_ipl_match_date", "last_international_match_date",
    "last_domestic_match_date", "has_recent_qualifying_history",
    "availability_status", "availability_known_as_of", "availability_reason",
    "availability_source_url", "availability_evidence", "fitness_status",
    "ipl_registration_country", "country_used_for_screen", "country_evidence_basis",
    "country_rule_basis", "country_rule_reviewed_on", "country_rule_decision", "country_rule_reason",
    "eligible_for_candidate_pool", "eligibility_reason", "exclusion_reason",
]


def canonical_country(value: object) -> str | None:
    """Normalise explicit country labels, never infer a country from a name."""

    if pd.isna(value):
        return None
    text = " ".join(str(value).strip().split())
    key = re.sub(r"[^a-z]", "", text.casefold())
    if key in {"", "unknown", "notknown", "notavailable", "na", "none", "null", "nan"}:
        return None
    aliases = {
        "pak": "Pakistan", "pakistan": "Pakistan", "pakistani": "Pakistan",
        "islamicrepublicofpakistan": "Pakistan",
        "ban": "Bangladesh", "bgd": "Bangladesh", "bangladesh": "Bangladesh",
        "bangladeshi": "Bangladesh", "peoplesrepublicofbangladesh": "Bangladesh",
        "india": "India", "indian": "India", "ind": "India",
        "england": "England", "english": "England", "eng": "England",
        "uk": "United Kingdom", "british": "United Kingdom", "unitedkingdom": "United Kingdom",
    }
    if key in aliases:
        return aliases[key]
    # Dual/ambiguous metadata requires a documented registration-country note.
    # In particular, do not substring-match a birthplace inside another label.
    if any(character in text for character in "/|;,()&"):
        return None
    return text.title()


def load_country_policy(path: Path, forecast_year: int, required: bool = True) -> dict | None:
    """Load one season's documented project rule without inventing future bans."""

    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing {path}. Add the season-specific country-policy JSON supplied with the update before rerunning.")
        return None
    with path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("seasons"), dict):
        raise ValueError("Country policy must contain a seasons object keyed by forecast year.")
    rule = payload["seasons"].get(str(forecast_year))
    if rule is None:
        if required:
            raise ValueError(f"No country policy is recorded for IPL {forecast_year}. Review the rules and add that season; a 2027 exclusion is not automatically a permanent ban.")
        return None
    if not isinstance(rule, dict) or not isinstance(rule.get("excluded_countries"), list):
        raise ValueError("Each country-policy season needs an excluded_countries list.")
    if not all(isinstance(value, str) for value in rule["excluded_countries"]):
        raise ValueError("excluded_countries must contain country-name strings only.")
    countries = [canonical_country(value) for value in rule["excluded_countries"]]
    if any(country is None for country in countries):
        raise ValueError("Country policy contains an empty or ambiguous country label.")
    if not isinstance(rule.get("exclude_unknown_country"), bool):
        raise ValueError("Country policy needs an explicit true/false exclude_unknown_country setting.")
    if not isinstance(rule.get("basis"), str) or not rule["basis"].strip():
        raise ValueError("Country policy needs a non-empty basis describing its evidence or scenario assumption.")
    reviewed = pd.to_datetime(rule.get("reviewed_on"), errors="coerce")
    if pd.isna(reviewed) or reviewed > pd.Timestamp(year=forecast_year - 1, month=12, day=31):
        raise ValueError("Country-policy reviewed_on must be valid and no later than the declared forecast cutoff.")
    return {**rule, "forecast_year": forecast_year, "excluded_countries": sorted(set(countries))}


def participation_events(
    tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    directories: tuple[str, ...],
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Count recorded batting/bowling involvement once per player-match.

    Cricsheet match IDs are global, so World Cup duplicates across international
    archives and batting/bowling records must not count twice. Fielding-only and
    DNB appearances absent from these processed tables cannot be inferred.
    """

    frames = []
    for directory in directories:
        for table in tables.get(directory, ()):
            if table.empty:
                continue
            missing = {"player_id", "match_id", "date"}.difference(table.columns)
            if missing:
                raise ValueError(f"{directory} activity data is missing {sorted(missing)}")
            part = table[["player_id", "match_id", "date"]].copy()
            part["player_id"] = part["player_id"].astype("string")
            part["match_id"] = part["match_id"].astype("string")
            part["date"] = pd.to_datetime(part["date"], format="mixed", dayfirst=True, errors="coerce")
            if part["date"].isna().any():
                raise ValueError(f"{directory} activity data contains invalid dates.")
            in_window = part["date"].le(end_date)
            if start_date is not None:
                in_window &= part["date"].ge(start_date)
            frames.append(part.loc[in_window])
    if not frames:
        return pd.DataFrame(columns=["player_id", "match_id", "date"])
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["player_id", "match_id", "date"])
    return pd.concat(frames, ignore_index=True).dropna(subset=["player_id", "match_id"]).drop_duplicates(["player_id", "match_id"])


def load_availability(
    path: Path | None, forecast_year: int, cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Read the latest known, season-specific availability note for each player.

    International retirement is deliberately not an exclusion status: a player
    can retire from T20Is/ODIs and still play IPL. Never turn a current injury into
    a permanent exclusion; ruled_out applies only to the stated forecast year.
    """

    columns = ["player_id", "availability_status", "availability_known_as_of", "availability_reason", "availability_source_url", "ipl_registration_country"]
    if path is None or not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        source = pd.read_csv(path, dtype={"player_id": "string"})
    except pd.errors.EmptyDataError:
        raise ValueError(f"{path} is empty. Use a header-only availability template or remove the optional file.") from None
    required = {"player_id", "forecast_year", "availability_status", "known_as_of"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"{path} is missing availability fields: {sorted(missing)}")
    if source.empty:
        return pd.DataFrame(columns=columns)
    source = source.dropna(how="all").copy()
    if source["player_id"].isna().any() or source["player_id"].str.strip().eq("").any():
        raise ValueError("Each availability record needs a canonical player_id, not a guessed name.")
    source["player_id"] = source["player_id"].str.strip()
    source["forecast_year"] = pd.to_numeric(source["forecast_year"], errors="coerce")
    if source["forecast_year"].isna().any() or source["forecast_year"].mod(1).ne(0).any():
        raise ValueError("Each availability record needs an integer forecast_year.")
    source["known_as_of"] = pd.to_datetime(source["known_as_of"], format="mixed", dayfirst=True, errors="coerce")
    if source["known_as_of"].isna().any():
        raise ValueError("Each availability record needs a valid known_as_of date.")
    source["availability_status"] = source["availability_status"].astype("string").str.strip().str.casefold().str.replace("-", "_", regex=False).str.replace(" ", "_", regex=False)
    if source["availability_status"].isna().any() or not source["availability_status"].isin(ALLOWED_STATUSES).all():
        raise ValueError(f"Invalid availability status. Use one of {sorted(ALLOWED_STATUSES)}. International retirement alone must not be recorded as IPL retirement.")
    source = source.loc[source["forecast_year"].eq(forecast_year) & source["known_as_of"].le(cutoff)].copy()
    if source.duplicated(["player_id", "known_as_of"]).any():
        raise ValueError("Duplicate availability notes for the same player and date. Keep one unambiguous status per date.")
    source = source.sort_values("known_as_of").drop_duplicates("player_id", keep="last")
    for field in ("reason", "source_url"):
        if field not in source:
            source[field] = ""
    if "ipl_registration_country" not in source:
        source["ipl_registration_country"] = ""
    source = source.rename(columns={"known_as_of": "availability_known_as_of", "reason": "availability_reason", "source_url": "availability_source_url"})
    return source[columns]


def assess_candidate_eligibility(
    candidates: pd.DataFrame,
    tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    forecast_year: int,
    activity_years: int = 2,
    min_qualifying_matches: int = 1,
    availability_path: Path | None = None,
    require_confirmed_availability: bool = False,
    require_recent_ipl_or_international: bool = True,
    country_policy: dict | None = None,
) -> pd.DataFrame:
    """Return one explanatory eligibility row per candidate, never a fit score."""

    if activity_years < 1 or min_qualifying_matches < 1:
        raise ValueError("activity_years and min_qualifying_matches must be at least 1")
    if country_policy is not None and country_policy.get("forecast_year", forecast_year) != forecast_year:
        raise ValueError("The country policy belongs to a different forecast season.")
    cutoff = pd.Timestamp(year=forecast_year - 1, month=12, day=31)
    start = pd.Timestamp(year=forecast_year - activity_years, month=1, day=1)
    result = candidates[["player_id"]].copy()
    if result["player_id"].isna().any() or result["player_id"].duplicated().any():
        raise ValueError("Candidate eligibility needs one non-empty canonical player_id per candidate.")
    result["player_id"] = result["player_id"].astype("string")
    result["activity_window_start"] = start.date().isoformat()
    result["activity_window_end"] = cutoff.date().isoformat()
    events_by_group = {}
    for group, sources in (("ipl", IPL_SOURCES), ("international", INTERNATIONAL_SOURCES), ("domestic", DOMESTIC_SOURCES)):
        known_events = participation_events(tables, sources, None, cutoff)
        events = known_events.loc[known_events["date"].ge(start)].copy()
        events_by_group[group] = events
        counts = events.groupby("player_id")["match_id"].nunique()
        # Keep the latest known date even for excluded older players, so an
        # auditor can see why their history falls outside the qualification window.
        last_dates = known_events.groupby("player_id")["date"].max()
        result[f"recent_{group}_matches"] = result["player_id"].map(counts).fillna(0).astype(int)
        result[f"last_{group}_match_date"] = pd.to_datetime(result["player_id"].map(last_dates), errors="coerce").dt.strftime("%Y-%m-%d")
    qualifying_frames = [events_by_group[group] for group in ("ipl", "international") if not events_by_group[group].empty]
    qualifying = pd.concat(qualifying_frames, ignore_index=True).drop_duplicates(["player_id", "match_id"]) if qualifying_frames else pd.DataFrame(columns=["player_id", "match_id", "date"])
    counts = qualifying.groupby("player_id")["match_id"].nunique()
    result["recent_qualifying_matches"] = result["player_id"].map(counts).fillna(0).astype(int)
    result["has_recent_qualifying_history"] = result["recent_qualifying_matches"].ge(min_qualifying_matches)
    statuses = load_availability(availability_path, forecast_year, cutoff)
    result = result.merge(statuses, on="player_id", how="left", validate="one_to_one")
    result["availability_status"] = result["availability_status"].fillna("unknown")
    for field in ("availability_reason", "availability_source_url"):
        result[field] = result[field].fillna("")
    result["availability_known_as_of"] = pd.to_datetime(result["availability_known_as_of"], errors="coerce").dt.strftime("%Y-%m-%d")
    result["availability_evidence"] = result["availability_status"].map(
        lambda status: "No season-specific confirmation; assumed available" if status == "unknown" else "Season-specific status supplied in reference file"
    )
    result["fitness_status"] = "Not medically assessed by this model"
    country_context = candidates.copy()
    country_context["player_id"] = country_context["player_id"].astype("string")
    country_context = country_context.set_index("player_id")
    screened_countries, evidence_sources, decisions, country_reasons = [], [], [], []
    for _, row in result.iterrows():
        country = canonical_country(row.get("ipl_registration_country"))
        evidence = "Season-specific IPL registration-country reference" if country else "Country metadata, not verified IPL registration"
        if country is None and "ipl_registration_country" in country_context:
            country = canonical_country(country_context.loc[row["player_id"], "ipl_registration_country"])
            if country is not None:
                evidence = "IPL registration country supplied with the candidate record"
        if country is None:
            for field in ("nationality", "metadata_country", "country"):
                if field in country_context:
                    country = canonical_country(country_context.loc[row["player_id"], field])
                    if country is not None:
                        break
        if country is None:
            evidence = "Country unresolved; no nationality inferred from name or birthplace"
        decision, explanation = "not_configured", ""
        if country_policy is not None:
            decision = "allowed"
            if country in country_policy["excluded_countries"]:
                decision = "excluded"
                explanation = "project_country_exclusion_" + re.sub(r"[^a-z]+", "_", country.casefold()).strip("_")
            elif country is None and country_policy["exclude_unknown_country"]:
                decision = "review_required"
                explanation = "country_unresolved_review_required"
        screened_countries.append(country or "Unknown")
        evidence_sources.append(evidence)
        decisions.append(decision)
        country_reasons.append(explanation)
    result["country_used_for_screen"] = screened_countries
    result["country_evidence_basis"] = evidence_sources
    result["country_rule_basis"] = country_policy["basis"] if country_policy is not None else "not_configured"
    result["country_rule_reviewed_on"] = country_policy["reviewed_on"] if country_policy is not None else ""
    result["country_rule_decision"] = decisions
    result["country_rule_reason"] = country_reasons
    country_ok = ~result["country_rule_decision"].isin(["excluded", "review_required"])
    recent_ok = result["has_recent_qualifying_history"] if require_recent_ipl_or_international else pd.Series(True, index=result.index)
    status_ok = ~result["availability_status"].isin(BLOCKED_STATUSES)
    confirmed_ok = result["availability_status"].eq("available") if require_confirmed_availability else pd.Series(True, index=result.index)
    result["eligible_for_candidate_pool"] = recent_ok & status_ok & confirmed_ok & country_ok
    reasons = []
    for index, row in result.iterrows():
        problems = []
        if not recent_ok.loc[index]:
            problems.append("domestic_history_only" if row["recent_qualifying_matches"] == 0 and row["recent_domestic_matches"] > 0 else "no_recent_ipl_or_international_history" if row["recent_qualifying_matches"] == 0 else "insufficient_recent_qualifying_matches")
        if not status_ok.loc[index]:
            problems.append(f"availability_{row['availability_status']}")
        elif not confirmed_ok.loc[index]:
            problems.append("availability_not_confirmed")
        if not country_ok.loc[index]:
            problems.append(row["country_rule_reason"])
        reasons.append("; ".join(problems))
    result["exclusion_reason"] = reasons
    result["eligibility_reason"] = result.apply(
        lambda row: row["exclusion_reason"] if row["exclusion_reason"] else (
            "Recent IPL or supported international participation; availability " + row["availability_status"]
            if require_recent_ipl_or_international else "Legacy activity scope; availability " + row["availability_status"]
        ), axis=1,
    )
    return result[["player_id", *AUDIT_COLUMNS]]
