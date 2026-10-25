# Country filter for the IPL 2027 planning pool

## What was checked

Checked on 25 October 2026:

- Pakistani players have been effectively excluded from the IPL since 2009,
  with passport/registration-country exceptions. The registered cricketing
  country should not be inferred from birthplace, ancestry or a name.
- KKR's BCCI-directed release of Mustafizur Rahman concerned IPL 2026. Bangladesh
  players appeared on the official 2026 auction list. A blanket ban covering all
  Bangladeshi players for IPL 2027 was **not verified**. The player-specific 2026
  decision is not proof of a permanent country-wide rule.

Sources:

- https://sports.ndtv.com/cricket/who-are-the-last-pakistan-players-to-play-in-an-ipl-final-not-shoaib-akhtar-shahid-afridi-11211844
- https://indianexpress.com/article/sports/cricket/mustafizur-rahman-released-by-kkr-following-bcci-instruction-10453190/
- https://documents.iplt20.com/bcci/documents/1765197869375_TATA-IPL-2026-Auction-List-8.12.25.pdf

## Implemented scenario

At the user's request, **both Pakistan and Bangladesh are excluded from the
2027 forecast candidate pool as an explicit project planning assumption**. The
rule, date checked, source links and qualification are recorded in
`data/reference/candidate_country_policy.json`. It must not be described in the
project report as a verified permanent ban on both countries.

The filter is applied before candidate fair values, performance forecasts and
team recommendations. It does not remove historical performances or auction
training rows, or change player ability/fitness scores based on nationality.
Exclusions are recorded as country-policy decisions, separate from inactivity,
injury or unavailability.

## Install individual files

| File | Destination | Action |
| --- | --- | --- |
| `candidate_eligibility.py` | `src/models/` | Replace |
| `score_auction_candidates.py` | `src/models/` | Replace |
| `run_future_analysis.py` | `src/models/` | Replace |
| `candidate_country_policy.json` | `data/reference/` | Add |
| `create_prediction_visualizations.py` | `src/visualization/` | Replace |
| `prediction_report.py` | `src/visualization/` | Replace |

Do not place the JSON policy inside `src/models/`. Forward analysis checks that
the policy exists and includes the requested forecast year before training.

## Country evidence and exceptions

The screen normalises explicit labels such as Pakistan/Pakistani/PAK and
Bangladesh/Bangladeshi/BAN/BGD. It uses a dated, season-specific
`ipl_registration_country` reference if supplied; otherwise it uses country or
nationality metadata and explicitly labels that as not verified IPL registration.
Names and birthplaces are never used to infer nationality. Missing or ambiguous
country data is held for review under the supplied configuration.

If needed, add the optional `ipl_registration_country` column to the existing
`data/reference/player_availability.csv`. Use a canonical `player_id`, the
forecast year, a known-as-of date and a source/reason. A player with another
verified IPL-registration country must not be excluded solely for being born
in Pakistan or Bangladesh. No real player citizenship exceptions were invented
or prefilled in this update.

An `available` status alone does not bypass the country scenario. The supplied
registration country and the configured country policy determine that check.
The existing recent-history and next-season availability conditions also still
apply.

## Change the assumption without editing model code

Edit the selected season in `candidate_country_policy.json`:

- To retain the requested scenario, leave both countries in `excluded_countries`.
- To remove the unverified blanket Bangladesh assumption, change that list to
  `["Pakistan"]` and update the policy notes/review date.
- To change how unknown countries are handled, edit `exclude_unknown_country`.
- For 2028 or later, explicitly add and review the new season. The 2027 scenario
  is not automatically treated as a permanent rule for every future IPL.

Recheck actual BCCI auction-registration and player-availability information
before presenting a forecast pool as confirmed participation eligibility.

## Rebuild

```powershell
python -m src.models.run_analysis --mode future --pool-year 2026 --forecast-year 2027 --team-needs data/reference/team_needs.csv --top-n 10 --activity-years 2
python -m src.visualization.create_prediction_visualizations --top-n 15
```

Then reopen/refresh:

`data/processed/model/visualizations/predictions/index.html`

Inspect `auction_candidate_eligibility.csv` for `country_used_for_screen`,
`country_evidence_basis`, `country_rule_basis`, `country_rule_decision`,
`country_rule_reason`, and the overall exclusion reason. The report also shows
the configured scenario, the country used for screening and the revised pool.
Older already-generated CSVs/HTML are not changed just by downloading Python files.

## Regression checks

`tests/test_candidate_country_policy.py` checks country aliases, the requested
scenario, missing-country review, registration-country overrides, lack of
birthplace/name inference, editable Bangladesh exclusion and season boundaries.
The end-to-end eligibility test also verifies country-excluded players do not
reappear in the candidate values, forecasts, team rankings or role shortlists.
