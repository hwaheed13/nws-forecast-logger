#!/usr/bin/env python3
"""
backfill_v13_features.py — Backfill v13 BL safeguard features into Supabase prediction_logs.

The 3 new features are derived from existing columns:
  entrainment_temp_diff = atm_925mb_temp_mean - obs_latest_temp
  marine_containment = obs_kjfk_vs_knyc / atm_bl_height_max
  inland_strength = mean(obs_kteb_temp, obs_kcdw_temp, obs_ksmq_temp) - mm_mean

Backfill strategy:
  1. Load all prediction_logs rows from Supabase for NYC (full history)
  2. For each row, compute the 3 features from existing columns
  3. Upsert the 3 new columns back to Supabase
  4. Log coverage and test cases (April 12 cap day, April 15 BL spike day)

Usage:
    python backfill_v13_features.py            # NYC only
    python backfill_v13_features.py --city lax # LAX only
    python backfill_v13_features.py --dry-run  # Show what would be computed
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime

import numpy as np
from supabase import create_client

def _float_or_none(v):
    """Safe float conversion."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def backfill_v13_features(city_key: str = "nyc", dry_run: bool = False):
    """
    Backfill v13 BL safeguard features into prediction_logs.
    """
    # Load Supabase credentials
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE environment variables")
        return False

    sb = create_client(supabase_url, supabase_key)
    print(f"\n{'='*60}")
    print(f"Backfill v13 BL Safeguard Features — {city_key.upper()}")
    print(f"{'='*60}")

    # Fetch all prediction_logs for this city. Supabase caps .select() at 1000
    # rows per request by default, so we paginate via .range().
    print(f"\n📥 Loading prediction_logs from Supabase for {city_key}...")
    rows = []
    page_size = 1000
    offset = 0
    try:
        while True:
            response = (
                sb.table("prediction_logs")
                .select("*")
                .eq("city", city_key)
                .order("id")
                .range(offset, offset + page_size - 1)
                .execute()
            )
            batch = response.data or []
            rows.extend(batch)
            print(f"  Fetched {len(batch)} rows (total so far: {len(rows)})")
            if len(batch) < page_size:
                break
            offset += page_size
        print(f"  Loaded {len(rows)} rows total")
    except Exception as e:
        print(f"❌ Failed to load prediction_logs: {e}")
        return False

    if not rows:
        print(f"⚠️ No rows found for city={city_key}")
        return False

    # Compute v13 features for each row
    updates = []
    test_cases = {}  # Track specific dates for verification

    entrainment_populated = 0
    marine_populated = 0
    inland_populated = 0

    import json as _json

    def _src(row, key):
        """Read a feature from either top-level column or atm_snapshot JSONB.
        prediction_logs stores most obs/atm features inside the atm_snapshot
        JSONB column, not as top-level columns — so plain row.get(key) returns
        None. Fall back to parsing atm_snapshot."""
        v = row.get(key)
        if v is not None:
            return v
        snap = row.get("atm_snapshot")
        if isinstance(snap, str):
            try:
                snap = _json.loads(snap)
            except Exception:
                snap = None
        if isinstance(snap, dict):
            return snap.get(key)
        return None

    for row in rows:
        row_id = row.get("id")
        target_date = row.get("target_date")

        # Extract source columns (check top-level AND atm_snapshot JSONB)
        atm_925mb_mean = _float_or_none(_src(row, "atm_925mb_temp_mean"))
        obs_latest_temp = _float_or_none(_src(row, "obs_latest_temp"))
        obs_kjfk_vs_knyc = _float_or_none(_src(row, "obs_kjfk_vs_knyc"))
        atm_bl_height_max = _float_or_none(_src(row, "atm_bl_height_max"))
        obs_kteb_temp = _float_or_none(_src(row, "obs_kteb_temp"))
        obs_kcdw_temp = _float_or_none(_src(row, "obs_kcdw_temp"))
        obs_ksmq_temp = _float_or_none(_src(row, "obs_ksmq_temp"))
        mm_mean = _float_or_none(_src(row, "mm_mean"))

        # Compute entrainment_temp_diff
        entrainment_temp_diff = None
        if atm_925mb_mean is not None and obs_latest_temp is not None:
            entrainment_temp_diff = round(atm_925mb_mean - obs_latest_temp, 1)
            entrainment_populated += 1

        # Compute marine_containment
        marine_containment = None
        if obs_kjfk_vs_knyc is not None and atm_bl_height_max is not None and atm_bl_height_max > 0:
            marine_containment = round(obs_kjfk_vs_knyc / atm_bl_height_max, 6)
            marine_populated += 1

        # Compute inland_strength
        inland_strength = None
        inland_temps = [t for t in [obs_kteb_temp, obs_kcdw_temp, obs_ksmq_temp] if t is not None]
        if inland_temps and mm_mean is not None:
            inland_mean = sum(inland_temps) / len(inland_temps)
            inland_strength = round(inland_mean - mm_mean, 1)
            inland_populated += 1

        # Prepare update — only include keys with real values so we don't
        # overwrite existing populated values with None. (Live prediction_writer
        # runs now write these top-level; backfill should augment, not wipe.)
        update = {"id": row_id}
        if entrainment_temp_diff is not None:
            update["entrainment_temp_diff"] = entrainment_temp_diff
        if marine_containment is not None:
            update["marine_containment"] = marine_containment
        if inland_strength is not None:
            update["inland_strength"] = inland_strength
        # Skip rows where nothing was computed — avoids a pointless update that
        # touches updated_at and burns Supabase quota.
        if len(update) > 1:
            updates.append(update)

        # Track test cases
        if target_date in ["2026-04-12", "2026-04-15"]:
            test_cases[target_date] = {
                "entrainment_temp_diff": entrainment_temp_diff,
                "marine_containment": marine_containment,
                "inland_strength": inland_strength,
                "sources": {
                    "atm_925mb_mean": atm_925mb_mean,
                    "obs_latest_temp": obs_latest_temp,
                    "obs_kjfk_vs_knyc": obs_kjfk_vs_knyc,
                    "atm_bl_height_max": atm_bl_height_max,
                    "inland_mean": sum(inland_temps) / len(inland_temps) if inland_temps else None,
                    "mm_mean": mm_mean,
                }
            }

    print(f"\n  entrainment_temp_diff: {entrainment_populated}/{len(rows)} rows will be populated")
    print(f"  marine_containment:    {marine_populated}/{len(rows)} rows will be populated")
    print(f"  inland_strength:       {inland_populated}/{len(rows)} rows will be populated")

    # Show test cases (April 12 cap day, April 15 BL spike day)
    print(f"\n📋 Test Cases Verification:")
    for date_str, data in sorted(test_cases.items()):
        print(f"\n  {date_str}:")
        print(f"    entrainment_temp_diff = {data['entrainment_temp_diff']} (925mb={data['sources']['atm_925mb_mean']} - obs={data['sources']['obs_latest_temp']})")
        print(f"    marine_containment = {data['marine_containment']} (jfk_vs_knyc={data['sources']['obs_kjfk_vs_knyc']} / bl_max={data['sources']['atm_bl_height_max']})")
        print(f"    inland_strength = {data['inland_strength']} (inland_mean={data['sources']['inland_mean']} - mm_mean={data['sources']['mm_mean']})")

    if dry_run:
        print(f"\n🏜️  DRY RUN MODE — no changes committed. Re-run without --dry-run to commit.")
        return True

    # Upsert updates to Supabase
    print(f"\n📤 Upserting {len(updates)} rows to Supabase...")
    upserted = 0
    failed = 0

    for i, update in enumerate(updates):
        try:
            sb.table("prediction_logs").update(update).eq("id", update["id"]).execute()
            upserted += 1
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(updates)} rows upserted...")
        except Exception as e:
            failed += 1
            if failed <= 5:  # Log first 5 failures
                row_id = update["id"]
                print(f"  ⚠️ Failed to upsert id={row_id}: {e}")

    print(f"\n✅ Backfill complete: {upserted} rows upserted, {failed} rows failed")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill v13 BL safeguard features")
    parser.add_argument("--city", default="nyc", help="City key (nyc, lax)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be computed without committing")
    args = parser.parse_args()

    success = backfill_v13_features(city_key=args.city, dry_run=args.dry_run)
    sys.exit(0 if success else 1)
