# Gut-Renovation Blueprint — nws-forecast-logger

*Drafted 2026-07-26 after the "1 year of data" audit. Status: PROPOSAL — awaiting
approval before any implementation.*

## Why renovate at all

The audit fixed the correctness layer (ground truth, training pool, CV, scoring,
measurement). But every fix had to be threaded through structure that actively
resists change:

- A **479KB single-file inference engine** (`prediction_writer.py`) doing 8 jobs,
  with NYC/LAX logic duplicated in ~30 places. Three of this audit's bugs lived
  in seams inside it.
- **Sixteen generations of models (v1–v16) retrained nightly**, 4.5 CPU-hours a
  day, when production uses exactly one model family (v16 + quantile head) plus
  a KNN blend. v1–v15 are archaeology that still gates the workflows.
- A **split-brain data layer**: half the truth in git CSVs, half in Supabase,
  reconciled by 9 backfill scripts. Both worst audit bugs (wrong actuals, empty
  training pool) were seam failures between these two halves.
- An **accreted decision layer**: residual model + KNN blend + physical caps +
  disagreement gate + confidence caps, stacked in the order they were invented,
  several validated against the pre-audit (wrong) actuals.
- **10 workflows** where ~4 would do; a UI that gives the misleading "latest"
  WIN/MISS equal billing with the real (canonical) one.

None of this blocks tomorrow's predictions. All of it taxes every future change.

## Target architecture

```
nwslogger/                       # one Python package, city-parameterized
  config.py                      # city_config (kept, already good)
  data/
    sources.py                   # NWS, AccuWeather, Open-Meteo, IEM, Synoptic,
                                 # Kalshi clients (thin, retrying, typed dicts)
    store.py                     # ONE write path. Supabase = live store,
                                 # git CSVs = daily immutable export of it.
    truth.py                     # official CLI actuals — the only actuals API
  features/
    build.py                     # the feature matrix builder (from train_models)
    live.py                      # intraday obs/atm snapshot features
  model/
    train.py                     # v16 residual + quantile head ONLY
    predict.py                   # anchor + residual + quantile bucket probs
    blend.py                     # KNN blend (kept only if it survives re-validation)
    guards.py                    # physical cap/floor, sanity clamp (kept)
  score/
    settle.py                    # canonical/latest scoring vs official actuals
    report.py                    # production_report (the moat scoreboard)
  jobs/                          # entrypoints the workflows call
    collect.py  predict.py  settle.py  retrain.py  export.py
```

**Data flow becomes one sentence:** sources → Supabase (live) → nightly export
to git (audit trail) → trainer reads the same store the scorer scores — no
second copy of the truth anywhere.

## What dies

| Today | Fate |
|---|---|
| v1–v15 training paths, their pkls/metadata (~60 files) | **Deleted.** One retrain: v16 + quantiles. Frees ~4 CPU-hours/night. |
| `prediction_writer.py` (479KB) | **Dismantled** into the package above; behavior-frozen port, no logic changes during the move. |
| 5 of 9 backfill scripts (one-time migrations already run) | Archived to `docs/postmortems/`, deleted from root. |
| `dsm-endpoint-test.yml`, v13/synoptic one-shot workflows | Deleted (manual-only runbook entry instead). |
| Root-level postmortem .md files, `api.py.save`, dead SQL files | Archived/deleted. |
| Committed pkls for v1–v15 (both cities) | Deleted; git history keeps them if ever needed. |
| The nested `nws-forecast-logger/` clone (~6GB) | Deleted after your confirmation. |

## What stays (proven load-bearing)

- v16 residual design, d1 anchor, date-grouped CV, quantile head + conformal
  offsets, official-actuals truth pipe, predictions ledger + production report,
  physical guards, `city_config.py`, `merge=union` log CSVs, failure alerting.
- The KNN blend stays **only if** it beats the plain residual path when
  re-validated against official actuals (its 588-day backtest predates the
  ground-truth fix — currently unproven).

## Workflow consolidation (10 → 4)

| New workflow | Replaces | Schedule |
|---|---|---|
| `collect.yml` | forecast-frequent, logger, synoptic-backfill | every 10 min daytime / hourly night |
| `predict.yml` | overnight-d1 + the predict half of forecast-frequent | hourly + on-demand |
| `settle-and-retrain.yml` | nightly-lightweight + heavy retrain (one job: settle → truth backfill → train v16+quantiles → ledger export → commit) | 05:00 UTC, ~40 min total |
| `report.yml` | (new) posts weekly production_report to a GitHub issue | weekly |

## Decision layer, redesigned (the one true prediction path)

```
anchor      = live HRRR (or NBM fallback)
residual    = v16(features)                    # trained vs d1 anchor
center      = anchor + residual
center      = blend(center, KNN)               # ONLY if re-validation passes
center      = guards(center)                   # physical cap/floor, sanity clamp
dist        = quantile_head(features) + conformal, median-shifted to center
bucket_probs= CDF(dist) over Kalshi buckets
bet_signal  = edge(bucket_probs, kalshi_prices) # bet ONLY when model prob −
                                                # market prob > threshold
```

Every stage logs its input/output so a bad day is diagnosable in one read.
The bet signal is the piece that turns calibration into money: no more
"prediction of the day," but "is any bucket mispriced enough to bet."

## UI redesign (final phase, on top of clean data)

- **Headline = canonical.** Canonical WIN/MISS and running canonical-vs-NWS MAE
  are the scoreboard; "latest" demoted to an intraday health strip.
- One **production scoreboard page** rendered from the ledger (30/90-day ML vs
  NWS vs Accu, per city) — the moat, public and honest.
- A **today page**: current distribution (the quantile CDF as a fan chart),
  bucket probabilities vs Kalshi prices, edge highlighted, guards state.
- Static site stays on Pages; data comes from the nightly export + a small
  JSON the predict job writes. No framework needed; current vanilla JS is fine.

## Migration order (production never goes dark)

1. **Freeze & baseline** (½ day): tag current main; capture golden outputs
   (given a fixed feature snapshot → identical predictions) as regression tests.
2. **Carve out `score/` + `data/truth.py`** (1 day): move scoring/actuals with
   golden tests; workflows still call old entrypoints via shims.
3. **Carve out `model/`** (1–2 days): port v16+quantiles train/predict; retire
   v1–v15 from workflows the same day (biggest instant win: 4.5h → ~40min).
4. **Carve out `features/` + `data/`** (2–3 days): the big one; golden tests
   make it mechanical. Delete duplicated NYC/LAX branches as they collapse.
5. **Re-validate the blend** against official actuals; keep or delete.
6. **Workflow consolidation** (½ day) + repo hygiene purge.
7. **UI rebuild** (2–3 days) on the clean exports.
8. **Bet-signal layer** (1–2 days): edge threshold vs Kalshi prices in the
   ledger, so the report can eventually show simulated P&L, not just MAE.

Each phase ships independently; any phase can stop without stranding the system.

## Risks / honest caveats

- The port must be behavior-frozen (golden tests) — the temptation to "improve
  while moving" is how 122°F bugs happen. Improvements come after the move.
- The moat is real but thin (+0.32°F NYC / +0.14°F LAX vs day-before HRRR, CV).
  Renovation makes the system trustworthy and cheap to evolve; it does not by
  itself widen the moat. The ledger decides that.
- LAX forecast history is 5 months; its numbers will lag NYC's for a while.
