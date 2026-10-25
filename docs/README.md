# IPL player performance and auction-value ETL

This repository prepares cricket data for three later analytical tasks:

1. Predict a player's upcoming IPL performance.
2. Estimate a fair IPL auction value for a player who is sold.
3. Support a team-fit and investment recommendation.

The repository currently contains Cricsheet match archives for IPL, ODI, T20,
BBL, SMAT and Men's T20 World Cup. Auction source files must be copied into
`data/raw/auction/` before auction-value ETL can run.

## Setup

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Required source data

The Cricsheet JSON archives and the player metadata files already belong in
`data/raw/`. Copy the existing files named like
`IPL_Auction_2013_Sold_Player.csv` through `IPL_Auction_2026_Sold_Player.csv`
into `data/raw/auction/`. Do not rename their source columns; the auction ETL
normalises the supported variants documented in that folder's README.

## Run order

Run these commands from the repository root, in this exact order:

```bash
# 1. Rebuild match, batting and bowling tables with canonical player IDs.
python -m src.transformation.cricket_etl --all

# 2. Create the all-competition player master and IPL compatibility copy.
python -m src.transformation.player_master_etl

# 3. Create all IPL player-season features and a latest recent-form snapshot.
python -m src.transformation.feature_builder --as-of-season 2026

# 4. After putting the auction CSVs in data/raw/auction/, clean and map them.
python -m src.transformation.auction_etl

# 5. Build the model-ready player-auction input table using the prior 3 years.
python -m src.transformation.build_model_table --recent-years 3

# 6. Validate the processed cricket outputs.
python src/check.py
pytest -q
```

The four notebooks in `notebooks/` run steps 1–3 from Jupyter. The Python
modules are the source of truth; notebooks are intentionally small runners so
the pipeline is reproducible outside Jupyter too.

### Store whole-number fields cleanly

To update the actual files in `data/processed/` so counts, years and INR values
are stored as integers (rather than values such as `45.0`), run:

```bash
python -m src.transformation.normalise_processed_csvs
```

This modifies every current processed CSV in place. Rates, ratios and model
forecasts stay decimal because fractional values are meaningful there.

## Important data rules

- Use `player_id`, not a display name, for all joins. The ID comes from each
  Cricsheet file's `info.registry.people` mapping.
- Keep raw files unchanged. The generated outputs live under `data/processed/`.
- Resolve names in `data/processed/auction/auction_unmatched_players.csv` by
  adding rows to `data/reference/player_name_overrides.csv`, then rerun
  `auction_etl.py`.
- Auction files contain sold players only. A future price model estimates price
  conditional on a sale; it must not be described as a model of unsold-player
  probability.
- `build_model_table.py` uses matches before 1 January of each auction year,
  preventing performance from the later IPL season leaking into that auction's
  training row.

## Model stage

After the ETL commands succeed, install the modelling dependencies once:

```bash
python -m pip install -r requirements.txt
```

Then run these commands in order:

```bash
# 1. Predict next IPL-season performance, conditional on playing next season.
python -m src.models.train_performance_model --as-of-season 2026

# 2. Train and evaluate a fair-value model using the most recent two auction years as test data.
python -m src.models.train_fair_value_model --test-years 2

# 3. Score the latest sold-player pool for the next auction. The market median is
#    the latest observed auction median unless you supply --market-median-inr.
python -m src.models.score_auction_candidates --pool-year 2026 --auction-year 2027
```

The price model predicts a player's price relative to that auction year's
median sale price, then converts it to INR. This makes comparisons across
auction years more meaningful than modelling raw historical INR prices.

### Team-fit recommendations

Team fit cannot be inferred honestly from match data alone: each team must
state its role need, budget and the importance it assigns to value, batting,
bowling and predicted performance. Copy
`data/reference/team_needs_template.csv` to `data/reference/team_needs.csv`,
replace the illustrative row, then run:

```bash
python -m src.models.score_team_fit --team "Your Team Name"
```

The performance model keeps IPL statistics as its primary signal and adds
separate, pre-cutoff recent-form features from T20I, BBL, SMAT, T20 World Cup
and ODI data. The target remains next-season IPL output; non-IPL matches are
inputs, not mixed into the IPL runs or wickets being predicted.

`desired_role` can contain more than one acceptable capability with `|`, such
as `Batter|Wicketkeeper`. A wicketkeeper-batter is then eligible for either
need. The score is role-aware: specialist batters are compared on batting
forecast, specialist bowlers on bowling forecast, and all-round requirements
use both. The score also records the budget and role decision in separate
columns, so a player is not silently excluded because of incomplete metadata.

Recommendations use these fixed categories: **Strong fit** when the score is
above 60; **Consider** when it is above 40 and no higher than 60; otherwise
**Low fit**. A player above the stated fair-value budget is assigned score zero
and remains clearly marked `over budget` in the reason column.

The recommendation is an explainable ranking. It is not a claim to know a
team's unpublished squad plan or whether a player will be available at auction.

### Final forward analysis

After the ETL pipeline and feature builder have produced the latest processed
data, run the complete current forecast in one command:

```bash
python -m src.models.run_analysis --mode future --top-n 10
```

Use `--skip-team-fit` when you want player performance and fair-value outputs
only, or pass `--team-needs path/to/profiles.csv` to score a particular set of
forward-looking team requirements.

`run_analysis.py` is the recommended single entry point. Use `--mode future`
for upcoming-season predictions and `--mode historical --target-season YEAR`
for an earlier-season validation. The more specialised scripts remain available
for reuse inside the project but do not need to be run directly.

With data through the 2026 season and 2026 auction, this trains on all
available history and produces 2027 performance forecasts, fair values and,
when `data/reference/team_needs.csv` exists, team recommendations. Key files
are saved under `data/processed/model/`, including
`upcoming_performance_predictions.csv`, `auction_candidate_fair_values.csv`,
`future_team_recommendations.csv`, and `future_analysis_summary.json`.

### Visualisations

After the forward analysis finishes, create report-ready charts from its CSV
outputs:

```bash
# Overall player value/performance chart, Gujarat Titans shortlist, and all-team summary.
python -m src.visualization.create_model_visualizations --team "Gujarat Titans" --top-n 10
```

Charts are saved under `data/processed/model/visualizations/`.

### Historical validation example: IPL 2026

To recreate a recommendation as it could have been made before IPL 2026, while
keeping the current 2027 model outputs unchanged, first put an RCB profile in
`data/reference/team_needs.csv`, then run:

```bash
python -m src.models.run_historical_backtest --team "Royal Challengers Bengaluru" --target-season 2026 --player-search Iyer
```

The command trains only with IPL performance through 2025 and auction rows
through auction-year 2025. Its separate outputs are saved beneath
`data/processed/backtests/`. The 2026 source auction data contains sold players
only, so this checks whether a known candidate would have ranked well; it does
not predict which unsold candidates would have been selected.

For an acquisition backtest, use the season of the auction in which the player
actually became available. Do not add retained players to a later auction pool:
they were not available candidates in that later auction, so doing so would
invalidate the backtest.

### Compare multiple teams in one historical season

When `team_needs.csv` contains one documented profile per team, run one shared
historical snapshot instead of retraining the models for every team separately:

```bash
python -m src.models.run_multi_team_backtest --target-season 2026 --top-n 5
```

The command saves each full team ranking and an
`all_teams_top_recommendations.csv` comparison table under
`data/processed/backtests/ipl_2026_all_teams/`.

Two transparent retrospective profile drafts are included for validation:

- `data/reference/team_needs_2026_backtest_draft.csv` compares RCB, CSK, MI
  and KKR in the 2026 auction backtest.
- `data/reference/team_needs_2025_srh_ishan_backtest_draft.csv` is a separate
  2025 SRH/Ishan Kishan case because he was not available in the 2026 auction.

These profiles are backtest hypotheses, not claims about private franchise
strategy. Their notes name the real acquisition used for outcome comparison.

## Main generated outputs

| File | Meaning |
|---|---|
| `data/processed/<competition>/matches.csv` | Match metadata, toss and outcome fields |
| `data/processed/<competition>/batting_match.csv` | Player-match batting statistics keyed by `player_id` |
| `data/processed/<competition>/bowling_match.csv` | Player-match bowling statistics keyed by `player_id` |
| `data/processed/player_master.csv` | Canonical player IDs, aliases and available metadata |
| `data/processed/ipl/player_season_features.csv` | All historical IPL player-season features |
| `data/processed/ipl/recent_form_features.csv` | Latest leakage-safe IPL form snapshot |
| `data/processed/auction/auction_clean.csv` | Standardised sold-player auction records |
| `data/processed/model/player_auction_training.csv` | Pre-auction cricket features joined to auction labels |
| `data/processed/model/upcoming_performance_predictions.csv` | Conditional next-season IPL runs, wickets and match forecasts, with IPL/T20/ODI form-availability fields |
| `data/processed/model/fair_value_metrics.json` | Chronological fair-value evaluation metrics |
| `data/processed/model/auction_candidate_fair_values.csv` | Fair-value estimates for a recent sold-player pool |
| `data/processed/model/team_fit_<team>.csv` | Transparent team-specific candidate ranking |
