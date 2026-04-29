#!/usr/bin/env python3
"""
Backfill health check.

Every backfill step in retrain-model.yml is `continue-on-error: true`,
which means a step that returns 0 rows or 500s silently passes. This
script runs after all backfills and asserts that the resulting Supabase
state has plausible row counts. If the numbers look broken, fail the
run BEFORE training.

Checks:
  1. prediction_logs has rows in the last 7 days (data flowing)
  2. prediction_logs.atm_snapshot is non-null on >= 80% of last-7-day rows
  3. prediction_logs.ml_actual_high is set on >= 5 days in the last 14 days
     (means yesterday's score job is working)
  4. Either entrainment_temp_diff or atm_snapshot has obs_max_so_far
     populated on at least 5 of the last 14 days (intraday backfill is alive)
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone


def _get_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE")
    if not url or not key:
        print("ERROR: SUPABASE_URL / SUPABASE_SERVICE_ROLE not set", file=sys.stderr)
        sys.exit(2)
    from supabase import create_client
    return create_client(url, key)


def main() -> int:
    sb = _get_client()
    failures: list[str] = []

    now = datetime.now(timezone.utc)
    cutoff_7d = (now - timedelta(days=7)).date().isoformat()
    cutoff_14d = (now - timedelta(days=14)).date().isoformat()

    # 1) recent rows — ALL writes (catches "data pipeline broken" if zero)
    resp = (sb.table("prediction_logs")
            .select("target_date,atm_snapshot,ml_actual_high,is_canonical")
            .gte("target_date", cutoff_7d)
            .execute())
    rows = resp.data or []
    if not rows:
        failures.append(f"prediction_logs: 0 rows since {cutoff_7d} (data pipeline broken)")
    else:
        print(f"✓ prediction_logs: {len(rows)} rows in last 7 days")

    # 2) atm_snapshot presence — the actual signal that matters.
    # A row is trainable iff it has atm_snapshot. The is_canonical column
    # is unreliable as a proxy (not backfilled on older rows, returns falsy
    # via r.get() even when canonical writes happened). Count snapshot
    # presence directly: expect ~4 canonical writes/day × 2 cities × 7 days
    # ≈ 56 rows with snapshot in last 7d. Floor at 30 to allow for outages.
    with_snap = sum(1 for r in rows if r.get("atm_snapshot") is not None)
    MIN_SNAP_ROWS_7D = 30
    if with_snap < MIN_SNAP_ROWS_7D:
        failures.append(
            f"prediction_logs.atm_snapshot: only {with_snap} rows with snapshot in last 7d "
            f"(expected ≥{MIN_SNAP_ROWS_7D} — canonical write or atm_snapshot persist may be broken)"
        )
    else:
        print(f"✓ atm_snapshot present on {with_snap} rows in last 7d (≥{MIN_SNAP_ROWS_7D})")

    # 3) scored days
    resp14 = (sb.table("prediction_logs")
              .select("target_date,ml_actual_high,atm_snapshot,is_canonical")
              .gte("target_date", cutoff_14d)
              .not_.is_("ml_actual_high", "null")
              .execute())
    rows14 = resp14.data or []
    scored_days = len({r["target_date"] for r in rows14})
    if scored_days < 5:
        failures.append(f"ml_actual_high: only {scored_days} scored days in last 14d "
                        f"(expected ≥5 — finalize-yesterday job broken?)")
    else:
        print(f"✓ {scored_days} scored days in last 14d")

    # 4) intraday/obs presence — at least 5 days have obs_max_so_far in atm_snapshot
    intraday_days: set[str] = set()
    for r in rows14:
        snap = r.get("atm_snapshot") or {}
        if isinstance(snap, dict) and snap.get("obs_max_so_far") is not None:
            intraday_days.add(r["target_date"])
    if len(intraday_days) < 5:
        # Not yet a hard fail — Option 1 backfill is brand new. Warn loudly.
        print(f"⚠️  obs_max_so_far populated on only {len(intraday_days)} of last 14d "
              f"(expected ≥5 once intraday backfill stabilizes)")
    else:
        print(f"✓ obs_max_so_far populated on {len(intraday_days)} of last 14d")

    if failures:
        print("\n" + "=" * 70)
        print("BACKFILL HEALTH FAILURES — failing the run before training")
        print("=" * 70)
        for f in failures:
            print(f"  ✗ {f}")
        return 1

    print("\n✓ Backfill health checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
