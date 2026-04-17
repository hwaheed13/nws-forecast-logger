#!/usr/bin/env python3
"""
Backfill obs_heating_rate and obs_cloud_cover into prediction_logs atm_snapshot.

These fields were computed but only written on stable-cycle refreshes (after 2pm),
not on canonical writes (morning). This backfill ensures they're available for all
historical rows, enabling proper heating window analysis.
"""

import os
import json
import math
from datetime import datetime, timedelta
import numpy as np
from supabase import create_client

# ─────────────────────────────────────────────────────────────────────
# Initialize Supabase
# ─────────────────────────────────────────────────────────────────────
sb_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
sb_key = os.environ.get("SUPABASE_SERVICE_ROLE", "")

if not sb_url or not sb_key:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE environment variables")

sb = create_client(sb_url, sb_key)

# ─────────────────────────────────────────────────────────────────────
# Helper: Convert sky condition text to cloud cover percentage
# ─────────────────────────────────────────────────────────────────────
def _sky_to_cloud_cover(sky_str: str) -> float | None:
    """Convert NWS sky condition text to cloud cover percentage."""
    if not sky_str:
        return None
    sky = sky_str.lower().strip()
    mapping = {
        "skc": 0,      # sky clear
        "clr": 0,      # clear
        "few": 25,     # 1/8 to 2/8 coverage
        "sct": 50,     # 3/8 to 4/8 coverage (scattered)
        "bkn": 75,     # 5/8 to 7/8 coverage (broken)
        "ovc": 100,    # 8/8 coverage (overcast)
        "vv": None,    # vertical visibility (obscured)
    }
    for key, val in mapping.items():
        if key in sky:
            return val
    return None


# ─────────────────────────────────────────────────────────────────────
# Fetch observations for a given date and city
# ─────────────────────────────────────────────────────────────────────
def fetch_observations(target_date_iso: str, city: str) -> list[dict]:
    """Fetch all NWS observations for a given date from Supabase."""
    try:
        result = sb.table("nws_observations").select("*").eq(
            "observation_date", target_date_iso
        ).order("observed_at", desc=False).execute()
        return result.data if result.data else []
    except Exception as e:
        print(f"  ⚠️ Error fetching observations for {target_date_iso}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────
# Compute heating rate from observations
# ─────────────────────────────────────────────────────────────────────
def compute_heating_rate(obs_rows: list[dict]) -> float | None:
    """Compute heating rate (°F/hr) over last 3 hours of observations."""
    if not obs_rows:
        return None

    valid_obs = [r for r in obs_rows if r.get("temp_f") is not None]
    if len(valid_obs) < 2:
        return None

    # Use last ~3 hours (4 observations)
    recent = valid_obs[-4:] if len(valid_obs) >= 4 else valid_obs

    try:
        temps = [r["temp_f"] for r in recent]
        t0_str = recent[0]["observed_at"].replace("Z", "+00:00")
        t0 = datetime.fromisoformat(t0_str)

        hours_list = []
        for r in recent:
            ti = datetime.fromisoformat(r["observed_at"].replace("Z", "+00:00"))
            hours_list.append((ti - t0).total_seconds() / 3600.0)

        if hours_list[-1] > 0:
            rate = (temps[-1] - temps[0]) / hours_list[-1]
            return round(rate, 2)
    except Exception:
        pass

    return None


# ─────────────────────────────────────────────────────────────────────
# Compute cloud cover from observations
# ─────────────────────────────────────────────────────────────────────
def compute_cloud_cover(obs_rows: list[dict]) -> float | None:
    """Compute cloud cover (%) from latest observation."""
    if not obs_rows:
        return None

    # Use latest observation
    latest = obs_rows[-1] if obs_rows else None
    if not latest:
        return None

    sky_condition = latest.get("sky_condition")
    if not sky_condition:
        return None

    return _sky_to_cloud_cover(sky_condition)


# ─────────────────────────────────────────────────────────────────────
# Backfill a single prediction_logs row
# ─────────────────────────────────────────────────────────────────────
def backfill_row(row_id: str, target_date: str, city: str, atm_snapshot: dict) -> bool:
    """
    Compute and update obs_heating_rate and obs_cloud_cover in a single row.
    Returns True if the row was updated, False otherwise.
    """
    # Fetch observations for this date
    obs_rows = fetch_observations(target_date, city)
    if not obs_rows:
        return False

    # Compute features
    heating_rate = compute_heating_rate(obs_rows)
    cloud_cover = compute_cloud_cover(obs_rows)

    # Check if we actually need to update
    current_heating = atm_snapshot.get("obs_snap_heating_rate")
    current_cloud = atm_snapshot.get("obs_snap_cloud_cover")

    needs_update = False

    if current_heating is None and heating_rate is not None:
        atm_snapshot["obs_snap_heating_rate"] = heating_rate
        needs_update = True

    if current_cloud is None and cloud_cover is not None:
        atm_snapshot["obs_snap_cloud_cover"] = cloud_cover
        needs_update = True

    if not needs_update:
        return False

    # Update Supabase
    try:
        sb.table("prediction_logs").update({
            "atm_snapshot": json.dumps(atm_snapshot)
        }).eq("id", row_id).execute()
        return True
    except Exception as e:
        print(f"    ⚠️ Update failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# Main backfill
# ─────────────────────────────────────────────────────────────────────
def backfill_all(city: str = "nyc", limit: int = None):
    """Backfill observation features for all prediction_logs rows."""
    print(f"\n🔄 Backfilling obs_heating_rate and obs_cloud_cover for {city.upper()}...")

    # Fetch all prediction_logs for this city
    try:
        query = sb.table("prediction_logs").select("id, target_date, atm_snapshot").eq("city", city)
        if limit:
            query = query.limit(limit)
        result = query.execute()
        rows = result.data if result.data else []
    except Exception as e:
        print(f"❌ Error fetching prediction_logs: {e}")
        print(f"   Error type: {type(e).__name__}")
        if hasattr(e, 'response'):
            print(f"   Response: {e.response}")
        return

    if not rows:
        print(f"  No rows found for {city}")
        return

    print(f"  Found {len(rows)} rows to check")

    updated = 0
    skipped = 0
    errors = 0

    for i, row in enumerate(rows, 1):
        row_id = row["id"]
        target_date = row["target_date"]
        atm_snap_raw = row.get("atm_snapshot")

        # Parse snapshot
        try:
            if isinstance(atm_snap_raw, str):
                atm_snap = json.loads(atm_snap_raw)
            elif isinstance(atm_snap_raw, dict):
                atm_snap = atm_snap_raw
            else:
                atm_snap = {}
        except json.JSONDecodeError:
            atm_snap = {}

        # Backfill
        if backfill_row(row_id, target_date, city, atm_snap):
            updated += 1
            status = "✓"
        else:
            skipped += 1
            status = "—"

        if i % 50 == 0 or i == 1:
            print(f"  {status} [{i}/{len(rows)}] {target_date}")

    print(f"\n  ✅ Updated: {updated}")
    print(f"  ⏭️  Skipped: {skipped} (already populated)")
    print(f"  ❌ Errors: {errors}")


# ─────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("OBSERVATION FEATURES BACKFILL")
    print("=" * 70)

    backfill_all("nyc", limit=500)
    backfill_all("lax", limit=500)

    print("\n✅ Backfill complete!")
