# Recent-history candidate pool

## Install

Merge the ZIP's `src`, `tests` and `docs` folders into the project root. Keep the
folder structure. It does not replace or delete your raw/processed input data.

| File | Action |
| --- | --- |
| `src/models/candidate_eligibility.py` | Add: eligibility and availability checks |
| `src/models/score_auction_candidates.py` | Replace: apply the screen before scoring candidates |
| `src/models/run_future_analysis.py` | Replace: propagate the screened IDs to forecasts and all teams |
| `src/models/run_analysis.py` | Replace: expose the screening options in the usual command |
| `src/models/score_team_fit.py` | Replace: retain the budget-chart fix and include eligibility evidence in role shortlists |
| `src/visualization/create_prediction_visualizations.py` | Replace: include the new eligibility audit chart |
| `src/visualization/prediction_report.py` | Replace: display the activity window, counts and availability status; retain the budget-chart fix |
| `src/visualization/create_model_visualizations.py` | Current companion version, including separate batter/keeper groups |

## Default rule for IPL 2027

A candidate must have at least one recorded batting/bowling participation in
either IPL or a supported international archive during 1 January 2025 through
31 December 2026. The sources currently supported for qualification are:

- IPL;
- ODI internationals;
- T20 internationals;
- Men's T20 World Cup.

Recent IPL **or** international history is sufficient; both are not required.
A new international player without IPL history can qualify. Someone whose IPL
history ends in 2024 and has no 2025–26 international involvement cannot.
BBL/SMAT-only activity no longer qualifies the player, although these
competitions remain supplementary forecast inputs for players who qualify.

This qualification window is separate from the four-season IPL feature window.
Older IPL performances may still inform the forecast of an otherwise qualified
player. Old players remain in historical training data; future eligibility must
not be applied retroactively to training labels.

## Run

After installing the files, rebuild the complete all-team outputs:

```powershell
python -m src.models.run_analysis --mode future --pool-year 2026 --forecast-year 2027 --team-needs data/reference/team_needs.csv --top-n 10
python -m src.visualization.create_prediction_visualizations --top-n 15
```

The first command rebuilds the candidate fair values, forecast pool and all-team
rankings/role shortlists using the new qualification rule. The second rebuilds
the report so old candidates are not left in its embedded snapshot. Reopen or
refresh `data/processed/model/visualizations/predictions/index.html` afterwards.
Historical player PNGs are not deleted: they are historical records, not the
current auction recommendation pool.

Options are available on the same future command:

```text
--activity-years 1             Require evidence during 2026 for a 2027 forecast
--activity-years 2             Default: evidence during 2025–26
--min-qualifying-matches 3     Require at least three distinct qualifying matches
```

The match threshold is combined across IPL and supported internationals.
Batting/bowling rows and duplicated T20 World Cup archives are counted once per
canonical player-match ID. Future matches cannot qualify a candidate.

## Activity does not prove fitness or registration

The project has cricket-performance tables, not a complete real-time injury,
retirement, registration, suspension or availability feed. The software cannot
honestly certify that every recently active player is fit to play next IPL.

Without an availability reference, qualified players carry an explicit
`availability_status = unknown` and are assumed available. A recorded next-season
availability note can exclude otherwise recently active players.

Create the optional file `data/reference/player_availability.csv` with this
header, then add only statuses you have evidence for:

```csv
player_id,forecast_year,availability_status,known_as_of,reason,source_url
```

Use the canonical `player_id` from `player_master.csv` or a model output, not a
name abbreviation. `forecast_year` is the season the note applies to. Use
`YYYY-MM-DD` for `known_as_of`. `reason` and `source_url` document your evidence;
the software does not independently verify these user-supplied assertions.

Statuses:

| Status | Treatment |
| --- | --- |
| `unknown` | Allowed by default, explicitly not confirmed |
| `available` | Allowed if the player also passes the recent-history rule |
| `retired_from_ipl` / `retired_from_all_cricket` | Excluded for the recorded forecast year |
| `ruled_out`, `withdrawn`, `unavailable`, `suspended`, `ineligible` | Excluded for the recorded forecast year |

Retirement from internationals alone is **not** IPL retirement. Do not use an
IPL-blocking status merely because someone stopped playing T20Is/ODIs. Similarly,
a past/current injury does not automatically mean the player is ruled out of the
next IPL season; only enter that season-specific status when supported.

Only records for the selected forecast year and known by the declared forecast
cutoff are used. The latest applicable dated note wins. Old-season injury notes
do not become permanent exclusions, and a conflicting duplicate note on the
same date is rejected. An `available` note never bypasses recent history.

To require a supplied `available` record for **every** admitted player, append:

```text
--require-confirmed-availability
```

Populate the status reference first. With no available records, this strict
option intentionally leaves no eligible pool and stops the run. It still does
not medically assess anyone or independently confirm IPL eligibility.

An alternate reference path can be supplied with `--availability-file PATH`.
If a specifically requested file is missing, the pipeline fails rather than
quietly ignoring it.

## Audit outputs

`data/processed/model/auction_candidate_eligibility.csv` records every screened
candidate, their recent counts, latest known IPL/international dates, recorded
status, qualification decision and exclusion reason. Excluded candidates are
also stored in `auction_candidate_excluded_players.csv`. Missing history is not
a claim that someone is retired, unfit or a bad player.

The eligibility chart and Sources & assumptions view explain the filter in the
report. Included players' qualification counts/status are also visible in the
player explorer and role shortlists. The current candidate pool no longer uses
all domestic-active metadata players as default.

Coverage limitation: these counts use the processed batting/bowling records.
Fielding-only or DNB-only participation not captured by those tables cannot be
inferred. Test-only international activity is not currently in the project's
supported input archives. Keep source data current; missing data is not evidence
that someone did not play.

Legacy historical-backtest scopes retain their old activity behaviour. The new
default applies to forward analysis; it does not rewrite past team scenarios.

If a run fails, do not treat an older existing HTML report as newly updated. A
zero-candidate run saves its exclusion audit and clears the candidate snapshot
instead of leaving an older non-empty pool masquerading as its result.

## Tests

```powershell
python -m unittest discover -s tests -p "test_candidate_eligibility.py" -v
python -m unittest discover -s tests -p "test_team_fit_budget.py" -v
python -m unittest discover -s tests -p "test_*visualizations.py" -v
```
