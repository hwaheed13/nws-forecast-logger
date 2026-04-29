#!/usr/bin/env python3
"""
Supabase schema preflight check.

Asserts that every column the inference + training code expects is
actually present in the prediction_logs table. Run as the FIRST step
of every workflow that writes to or reads from prediction_logs.

This guard exists because PR #29 was a 4-month-old bug caused by
inference reading a column (`actual_high`) that never existed in the
schema — caught only after I started auditing. Schema drift is now
fail-fast.

Exit codes:
  0  all required columns present
  1  one or more required columns missing
  2  cannot connect to Supabase (env/network problem)
"""
from __future__ import annotations

import os
import sys

REQUIRED_COLUMNS = {
    # Outcome columns the trainer keys off
    "ml_actual_high",      # PR #29 — what we use instead of nonexistent actual_high
    "ml_bucket",
    "ml_f",                # winning bucket forecast in degrees-F
    "target_date",
    "is_canonical",
    "atm_snapshot",        # JSONB blob of inference-time features
    "nws_last",            # PR #29 — required for v15 autoreg lookup
    "timestamp",
    # v13 BL safeguard top-level columns
    "entrainment_temp_diff",
    "marine_containment",
    "inland_strength",
    # v15 morning/autoreg derived columns (added in earlier migrations)
    "forecast_revision",
    "cap_violation_925",
    "yesterday_signed_miss",
    "rolling_3day_bias",
}


def _get_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE")
    if not url or not key:
        print("ERROR: SUPABASE_URL / SUPABASE_SERVICE_ROLE not set", file=sys.stderr)
        sys.exit(2)
    try:
        from supabase import create_client
    except ImportError:
        print("ERROR: supabase-py not installed", file=sys.stderr)
        sys.exit(2)
    return create_client(url, key)


def main() -> int:
    client = _get_client()
    # Fetch one row, list its column keys.
    try:
        resp = client.table("prediction_logs").select("*").limit(1).execute()
    except Exception as e:
        print(f"ERROR querying prediction_logs: {e}", file=sys.stderr)
        return 2
    rows = resp.data or []
    if not rows:
        print("WARN: prediction_logs is empty — cannot verify columns", file=sys.stderr)
        return 0
    present = set(rows[0].keys())
    missing = sorted(REQUIRED_COLUMNS - present)
    if missing:
        print("=" * 70)
        print("SUPABASE SCHEMA DRIFT DETECTED — failing the run")
        print("=" * 70)
        for col in missing:
            print(f"  ✗ prediction_logs.{col} is missing")
        print(
            "\nAdd the missing columns via ALTER TABLE before continuing.\n"
            "If the column was intentionally renamed, update REQUIRED_COLUMNS\n"
            "in verify_supabase_schema.py to match.\n"
        )
        return 1
    print(f"✓ prediction_logs has all {len(REQUIRED_COLUMNS)} required columns "
          f"({len(present)} total).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
