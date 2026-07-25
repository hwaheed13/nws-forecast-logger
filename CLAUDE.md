# CLAUDE.md — Working notes for this repo

## THE BOTTOM LINE / THE MOAT / THE LEG-UP

**Always frame work in terms of the moat.** Before any change, ask:
- Does this build, preserve, or erode the leg-up?
- Anyone can read NWS, AccuWeather, HRRR for free. **Our only edge is predicting where these public forecasts are wrong** (cap-break days, sea-breeze days, overnight peaks, urban heat island, stall regimes).
- A model that just outputs HRRR has zero moat (anyone has HRRR). A model that learns HRRR's *systematic errors* over 4 years and corrects them has the moat.

**Concretely for v16 (unified residual, both cities as of 2026-07-25):**
- Target: `actual_high − HRRR_max` (residual). Forces model to learn HRRR's errors, not regurgitate it.
- Inference: `final = HRRR + v16_residual_prediction`.
- CV estimate of moat: `improvement_vs_hrrr_alone` in `model_metadata_v16.json`.
- **PRODUCTION measurement of moat: `python production_report.py`** — compares the
  nightly-exported `predictions_ledger.csv` (what the model actually predicted)
  against NWS/AccuWeather day-before, all scored vs OFFICIAL CLI actuals.
  CV numbers estimate; the ledger measures. If ml_mae isn't clearly below
  nws_mae there, the moat does not exist in production, whatever CV says.
- If `improvement_vs_hrrr_alone` drops to ~0, the moat is gone — investigate immediately.

**Ground truth (audit 2026-07-25 — do not regress this):**
- `actual_high` in `*multiyear_atmospheric.csv` is overwritten with OFFICIAL
  CLI highs from the IEM CLI archive by `backfill_official_actuals.py`
  (runs in the daily retrain). The original Open-Meteo grid temps (preserved
  in `grid_actual_high`) disagreed with the settle value by a MEAN of 2.1°F —
  all pre-2026-07-25 CV metrics were measured against the wrong answer.
- v16 CV is **date-grouped** TimeSeriesSplit: intraday snapshot rows of one
  date must never straddle train/test (same-date rows share the label).

**Anti-patterns to avoid:**
- Predicting `actual_high` directly with HRRR as a feature → model collapses to "HRRR + tiny offset" (no moat).
- Adding leaky features (e.g. archive `obs_latest_temp` set to noon temp ≈ daily peak) → CV looks great, production fails.
- Shipping inference and training changes in the same PR → produces 122°F bugs (PR #42 → PR #43 hotfix).
- Training against non-settle-source actuals (the grid-temp bug above).
- Letting a city run without an HRRR anchor: LAX silently skipped v16 for
  ~3 months ("Only 27 HRRR-anchored rows") and served the legacy no-moat
  direct model. `backfill_multimodel_history.py --include-hrrr` fills the
  anchor (NaN-only; never overwrites live captures).
- Dropping log-era rows from the v16 pool (fixed 2026-07-26): the multiyear
  twin of each log-era date was discarded as an "overlap duplicate," taking
  mm_hrrr_max with it, so the HRRR-anchored pool filter dropped EVERY log-era
  row — nws_last coverage in the pool was 0/1295 and the production model had
  never trained on a single day of collected NWS/AccuWeather/obs data.
  train_v2 now ENRICHES log-era rows from overlapping multiyear rows (fill
  mm_*/atm_*/intra_* where NaN) before excluding them. Training logs print
  "Enriched log-era rows…"; if pool nws_last coverage reads 0% again, this
  regressed.
- FEATURE_COLS_V16 carried 7 duplicate names (186 listed, 179 unique) until
  2026-07-26 — the dedupe lives at the definition in model_config.py.

**When debugging a prediction issue, always check:**
1. Is metadata showing `v16_unified_residual` (post-PR #42) or `v16_unified` (legacy DIRECT)? Inference branches on this.
2. Does the regressor's bias output look reasonable (typically -3 to +3°F)?
3. Is HRRR populated? Without it, residual model can't anchor.
4. Did the sanity clamp fire? (inference logs "🚨 SANITY CLAMP" when the
   prediction is >15°F from every reference — means wiring is broken.)

## Architecture cheat-sheet

- `train_models.py` — trains v1 → v16 cascade. v16 = unified residual (the moat).
- `prediction_writer.py` — inference. v16 inference path detects DIRECT vs RESIDUAL via `_v16_is_residual()` reading metadata.
- `model_metadata_v16.json` — source of truth for which architecture is loaded.
- `coverage_report.json` — gates regression detection. `_record_coverage` uses MAX-WINS so v16's smaller features_df doesn't clobber v15's view.
- `backfill_bl_925.py` + `backfill_multimodel_history.py` — populate 4yr atmospheric/multi-model history that the moat features depend on.
- `backfill_official_actuals.py` — official CLI ground truth (see above).
- `export_predictions_ledger.py` → `*predictions_ledger.csv` → `production_report.py` — the production moat measurement loop.

## Workflow scheduling

- `nightly-lightweight-retrain.yml` — 05:00 UTC, model-only refresh + ledger
  export; no coverage-report ownership.
- `retrain-model.yml` — heavy retrain at 05:15 UTC (15 min after nightly, on
  purpose). Both share concurrency group `model-training` (no-cancel) because
  they write the SAME model files — they used to fire at the same hour and
  race on git push, silently discarding the loser's models.
- All main workflows have a `notify-failure` job that opens/comments a GitHub
  issue labeled `workflow-failure`. Keep that when editing workflows — 26
  `continue-on-error` steps mean red runs are otherwise invisible.
- Append-only log CSVs use `merge=union` in `.gitattributes` so concurrent
  workflow appends merge instead of one side being dropped.

## Recent firefighting

2026-07-25 (full audit — "1 year of data" review):
- Ground truth was wrong (grid temps, mean 2.1°F off) → `backfill_official_actuals.py`.
- LAX had NO historical HRRR → v16 skipped nightly for ~3 months, stale
  legacy direct model from 05-02 → HRRR backfill (168 → 1581 rows) + retrain.
- v16 CV leaked same-date intraday rows across folds → date-grouped CV.
- Production predictions were unlogged since 2025-12-18 (snapshot logger died);
  on the last measurable 67 days (Oct–Dec 2025) ML LOST to NWS day-before by
  ~0.18°F MAE → predictions ledger + production_report.py.
- Kalshi WIN/MISS scoring could false-WIN via bucket-label low-edge shortcut →
  center-temp strict same-bucket scoring (`_score_bucket`).
- `backfill_multimodel_history.py` used downtown-LA coords for "lax" → now
  reads `city_config` (KLAX airport).

2026-05-02:
- PR #41: removed leaky archive `obs_latest_temp` (was within 1°F of actual_high on 37% of training rows).
- PR #42: 3-phase moat — residual v16 + BL safeguard 4yr coverage + workflow backfills.
- PR #43: HOTFIX for 122°F production bug — inference now self-describes via metadata.
- PR #44: reset LAX v14 baseline (truthful drop after PR #41 removed leakage).
