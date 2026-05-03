# CLAUDE.md — Working notes for this repo

## THE BOTTOM LINE / THE MOAT / THE LEG-UP

**Always frame work in terms of the moat.** Before any change, ask:
- Does this build, preserve, or erode the leg-up?
- Anyone can read NWS, AccuWeather, HRRR for free. **Our only edge is predicting where these public forecasts are wrong** (cap-break days, sea-breeze days, overnight peaks, urban heat island, stall regimes).
- A model that just outputs HRRR has zero moat (anyone has HRRR). A model that learns HRRR's *systematic errors* over 4 years and corrects them has the moat.

**Concretely for v16 (current architecture as of 2026-05-03):**
- Target: `actual_high − HRRR_max` (residual). Forces model to learn HRRR's errors, not regurgitate it.
- Inference: `final = HRRR + v16_residual_prediction`.
- CV proof of moat: `improvement_vs_hrrr_alone` in `model_metadata_v16.json`. Currently **+0.85°F** (NYC) — v16 reduces HRRR's untreated MAE from 1.94°F → 1.09°F. That gap IS the moat.
- If `improvement_vs_hrrr_alone` ever drops to ~0, the moat is gone — investigate immediately.

**Anti-patterns to avoid:**
- Predicting `actual_high` directly with HRRR as a feature → model collapses to "HRRR + tiny offset" (no moat).
- Adding leaky features (e.g. archive `obs_latest_temp` set to noon temp ≈ daily peak) → CV looks great, production fails.
- Shipping inference and training changes in the same PR → produces 122°F bugs (PR #42 → PR #43 hotfix).

**When debugging a prediction issue, always check:**
1. Is metadata showing `v16_unified_residual` (post-PR #42) or `v16_unified` (legacy DIRECT)? Inference branches on this.
2. Does the regressor's bias output look reasonable (typically -3 to +3°F)?
3. Is HRRR populated? Without it, residual model can't anchor.

## Architecture cheat-sheet

- `train_models.py` — trains v1 → v16 cascade. v16 = unified residual (the moat).
- `prediction_writer.py` — inference. v16 inference path detects DIRECT vs RESIDUAL via `_v16_is_residual()` reading metadata.
- `model_metadata_v16.json` — source of truth for which architecture is loaded.
- `coverage_report.json` — gates regression detection. `_record_coverage` uses MAX-WINS so v16's smaller features_df doesn't clobber v15's view.
- `backfill_bl_925.py` + `backfill_multimodel_history.py` — populate 4yr atmospheric/multi-model history that the moat features depend on.

## Workflow scheduling

- `retrain-model.yml` — heavy retrain at 6am ET (`cron: "0 10 * * *"`). Currently takes 4-7hrs because backfills are non-incremental. **Don't move earlier without first making backfills incremental** — moving the schedule alone gives no measurable accuracy benefit because training pool is 18k+ rows and adding 5 hours of obs adds <0.1% data.
- `nightly-lightweight-retrain.yml` — 1am ET, model-only refresh, no coverage report ownership.

## Recent firefighting (2026-05-02)

- PR #41: removed leaky archive `obs_latest_temp` (was within 1°F of actual_high on 37% of training rows).
- PR #42: 3-phase moat — residual v16 + BL safeguard 4yr coverage + workflow backfills.
- PR #43: HOTFIX for 122°F production bug — inference now self-describes via metadata.
- PR #44: reset LAX v14 baseline (truthful drop after PR #41 removed leakage).
