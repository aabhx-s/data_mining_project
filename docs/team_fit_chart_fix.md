# Explain budget-blocked player fit

The previous score calculation deliberately set `team_fit_score` to zero when
the player's estimated price exceeded a team's per-player fair-value limit.
That rule remains unchanged, including the final ranking, recommendation
thresholds and affordable-first sorting.

The scorer now saves `pre_budget_fit_score` before applying that hard gate, and
saves `max_fair_value_inr` alongside each team/player result. These fields are
also carried into the role-shortlist files. The pre-budget score is the
existing weighted role/performance/value score, not a newly fitted model or a
pure skill score with every price effect removed.

The player report's across-team chart now shows this saved pre-budget score,
with amber bars for players outside the stated price limit. A separate column
shows the price-limit status, the configured limit and the final score. Legacy
blocked rows without a pre-budget score are explicitly labelled not recorded.
Missing potential scores are never replaced with investment scores or invented
by the chart.

From the project root, rescore the current saved predictions for every team:

```powershell
Import-Csv "data/reference/team_needs.csv" | ForEach-Object {
    python -m src.models.score_team_fit --team $_.team --team-needs data/reference/team_needs.csv
    if ($LASTEXITCODE -ne 0) { throw "Scoring failed for $($_.team)" }
}
python -m src.visualization.create_prediction_visualizations --top-n 15
```

Then reopen or refresh:

`data/processed/model/visualizations/predictions/index.html`

No performance-model retraining is required when the forecasts and team-needs
profiles are otherwise unchanged. If you changed the team-needs assumptions
since the previous run, regenerate the full all-team analysis as well so its
combined recommendation summary remains consistent.

Tests:

```powershell
python -m unittest discover -s tests -p "test_team_fit_budget.py" -v
python -m unittest discover -s tests -p "test_*visualizations.py" -v
```
