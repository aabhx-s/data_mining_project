# Prediction visualisations

Copy the update's `src` files into the same locations in the project. This adds
`src/visualization/create_prediction_visualizations.py` and
`src/visualization/prediction_report.py`, and replaces
`src/visualization/create_model_visualizations.py`. Existing historical player
charts and model/training code are unchanged. No extra packages are needed
beyond pandas, NumPy and Matplotlib already used by the project.

## Generate the complete report

After generating the current all-team model outputs, run this from the project
root in VS Code:

```powershell
python -m src.visualization.create_prediction_visualizations --top-n 15
```

Open `data/processed/model/visualizations/predictions/index.html` in a browser.
You can double-click it in File Explorer, or use:

```powershell
Start-Process "data/processed/model/visualizations/predictions/index.html"
```

The report opens locally with no server, login, internet connection or CDN. Its
images are in the same output folder. Keep that folder together when sharing
the report. Links to original source files also require the original relative
model-folder layout. The embedded player and team explorers do not need those
source files after generation. Re-run the generator whenever the forecasts or
team outputs change; the HTML is a saved snapshot, not a live connection to CSVs.

## Included views

| Output | Visual view |
| --- | --- |
| Player performance predictions | Run/wicket leaders with supplied scenario ranges, full-pool distributions, expected batting/bowling opportunity, evidence coverage |
| Every individual forecast player | Searchable player explorer with role-appropriate output, actual IPL history when available, separate forecast bar, scenarios, fair value, implied per-appearance production and fit across all teams |
| Candidate fair values | Full-pool value/performance scatter, hexbin density chart and role-specific value distributions |
| All `team_fit_*.csv` outputs | A six-panel chart for each team; searchable full rankings with team fit, investment, price, role/budget checks and saved explanations |
| All `team_role_shortlists_*.csv` outputs | A four-section market-scan chart for each team and a saved-shortlist filter |
| Combined team recommendations | Best-ranked candidate for every current team profile |
| Investment scores | Top performance-for-price candidates and estimated prices; full scores in the player/team explorers |
| Model and calibration comparison CSVs | Lower-is-better error comparisons, with unavailable comparisons explicitly marked |
| Performance evaluation predictions | Actual versus predicted runs, wickets and separate appearance targets; every valid evaluation row is plotted |
| Auction-price evaluation predictions | Actual versus predicted price and signed-error distribution in INR crore |
| Model metrics JSON | Headline errors, model/target settings, all saved MAE/RMSE/R-squared values and recorded calibration settings |
| Fair-value feature importance | Largest transformed input importances; full source remains linked |
| Auction training and excluded rows | Eligibility by auction year, training-exclusion signals, active-candidate coverage and nationality counts |
| Forward analysis summary JSON | Forecast year, data cutoff, team coverage, recorded training settings and availability assumptions |
| Joblib model artifacts | Listed as saved runtime models, not misrepresented as chart data. They are never unpickled by the report generator. |

The Sources & assumptions section accounts for each top-level CSV, JSON and
joblib artifact in the model folder. New unrecognised files are listed rather
than silently described as visualised.

## All players versus top-N charts

`--top-n` limits leader charts only. The browser report always includes every
saved player and the complete current team rankings, not just ten or fifteen.
Team dashboard PNGs show up to ten ranked players so labels remain readable.

To additionally export selected player forecast PNGs:

```powershell
python -m src.visualization.create_prediction_visualizations --player "V Kohli" --player "Shubman Gill"
```

Use the canonical name in the CSV or a player ID. Ambiguous partial names are
rejected rather than guessed. To export an individual PNG for every forecast
player as well:

```powershell
python -m src.visualization.create_prediction_visualizations --all-players
```

That optional export can create thousands of images and take much longer than
the default searchable report. It is not required to view every player.

## Important interpretation

- The generator visualises saved results. It does not change predictions,
  retrain a model, increase forecasts or alter source CSVs/JSON/joblib files.
- Historical IPL data is separate from the forecast. History is filtered at
  the saved forecast cutoff and joined by player ID. Missing history is not
  filled with fictional zero seasons.
- Batters/keeper-batters get batting forecast panels; bowlers get bowling;
  all-rounders get both. A keeper label requires explicit role metadata.
  Other means role unavailable, not a guessed specialist role.
- The current model does not produce forecast strike rate, economy or
  wicketkeeping dismissals. The report does not invent them.
- Conservative/upside values are the saved scenarios, not confidence limits.
  Older files without these columns are supported with explicit missing-range
  notices. No ranges are invented from the chart code.
- Investment is a relative performance/price score, not financial ROI.
  Team fit is not a probability of success. The budget check is a per-player
  fair-value limit, not a full squad purse/overseas-rule optimisation.
- Role shortlists are market scans for multiple roles; some may be marked as
  mismatches to the team's current requirement. They are not a combined buying
  recommendation.
- Saved model comparisons use chronological evaluation/selection records;
  the report does not call them a new untouched final test.
- Older differently named charts are not deleted. The new report lists only
  the charts from its own run, so stale images are not silently mixed in.

## If the script reports stale team data

When team CSV forecasts disagree with the player prediction CSV, the generator
stops with an explanatory error. Rebuild consistent outputs first:

```powershell
python -m src.models.run_analysis --mode future --pool-year 2026 --forecast-year 2027 --team-needs data/reference/team_needs.csv --top-n 10
python -m src.visualization.create_prediction_visualizations
```

## Regression checks

```powershell
python -m unittest discover -s tests -p "test_*visualizations.py" -v
```

Tests cover role classification, all-player coverage, stale team/summary
rejection, scenario ordering, history cutoff, missing ranges, safe embedded JSON
and preservation of the original model output files.
