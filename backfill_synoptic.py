#!/usr/bin/env python3
"""
backfill_synoptic.py — Backfill Synoptic station features into atm_snapshot for historical rows.

Uses the Synoptic Data timeseries API (1-year history on enterprise tier) to reconstruct
what the per-station and aggregate features would have been at each historical prediction
cycle. Writes the results back into the atm_snapshot JSONB column in Supabase.

This is the unlock: instead of waiting months for live Synoptic data to accumulate,
we can immediately populate obs_kjfk_temp, obs_kjfk_vs_knyc, obs_coastal_vs_inland,
obs_synoptic_mean etc. for every historical training row. The model can then learn
from actual cap-day fingerprints on its first v9 training run.

Usage:
    SYNOPTIC_TOKEN=xxx SUPABASE_URL=xxx SUPABASE_SERVICE_ROLE_KEY=xxx python backfill_synoptic.py
    python backfill_synoptic.py --dry-run          # print what would be written, no DB writes
    python backfill_synoptic.py --days 30          # only backfill last 30 days
    python backfill_synoptic.py --date 2026-04-12  # single specific date

Designed to run as a one-off GitHub Actions workflow job:
    - Queries Synoptic timeseries for each day in the training set
    - For each day, picks the "noon cycle" observation window (10am-2pm local)
    - Computes the same features as get_synoptic_obs_features() + named stations
    - Merges into existing atm_snapshot JSONB (does NOT overwrite other keys)
    - Skips rows that already have obs_kjfk_temp populated (idempotent)
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.parse
from datetime import datetime, date, timedelta, timezone
from typing import Optional

import numpy as np

# ── Load .env if present (allows running outside a pre-configured shell) ──────
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path, override=False)  # don't override real env vars
        print(f"  📄 Loaded credentials from {_env_path}")
except ImportError:
    pass  # python-dotenv not installed; rely on shell env vars

# ── Config ────────────────────────────────────────────────────────────────────
SYNOPTIC_BASE = "https://api.synopticdata.com/v2"
# Stations to query: KNYC + the four airport ASOS stations
BACKFILL_STIDS = "KNYC,KJFK,KLGA,KEWR,KTEB"
# Borough mesonet stations also tracked in the aggregate
NYSM_STIDS = "MANH,BRON,QUEE,BKLN,STAT"
CENTRAL_PARK_LAT = 40.7834
CENTRAL_PARK_LON = -73.965
RADIUS_MILES = 10.0
# The "noon window" for computing midday station readings (local EDT hour range)
NOON_WINDOW_START_EDT = 10   # 10 AM
NOON_WINDOW_END_EDT = 14     # 2 PM
# Rate limit: Synoptic free/enterprise has 1000 calls/day on free, more on enterprise
# We do 2 API calls per day (named + radius); sleep between days to be safe.
SLEEP_BETWEEN_DAYS = 1.0     # seconds; reduce if you have high rate limits
# ──────────────────────────────────────────────────────────────────────────────


def _token() -> Optional[str]:
    return os.environ.get("SYNOPTIC_TOKEN", "").strip() or None


def _supabase_client():
    from supabase import create_client
    url = os.environ["SUPABASE_URL"]
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def fetch_timeseries(
    stids: str,
    date_str: str,          # YYYY-MM-DD local date
    start_hour_edt: int,    # local EDT hour (0-23)
    end_hour_edt: int,
) -> dict[str, list[tuple[datetime, float]]]:
    """
    Query Synoptic timeseries for named stations on a specific date window.
    Returns dict: stid → list of (utc_datetime, temp_f) tuples.

    EDT = UTC-4 (April–October).  We add 4h to convert EDT to UTC for the API.
    """
    token = _token()
    if not token:
        return {}

    # Convert local EDT to UTC for API (EDT = UTC-4)
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    # Build start/end in UTC by adding 4h to the EDT hours
    start_utc = date_obj + timedelta(hours=start_hour_edt + 4)
    end_utc   = date_obj + timedelta(hours=end_hour_edt + 4)

    start_str = start_utc.strftime("%Y%m%d%H%M")
    end_str   = end_utc.strftime("%Y%m%d%H%M")

    params = {
        "token":  token,
        "stid":   stids,
        "start":  start_str,
        "end":    end_str,
        "vars":   "air_temp",
        "units":  "english",
        "output": "json",
    }
    qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    url = f"{SYNOPTIC_BASE}/stations/timeseries?{qs}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
        if data.get("SUMMARY", {}).get("RESPONSE_CODE") != 1:
            msg = data.get("SUMMARY", {}).get("RESPONSE_MESSAGE", "unknown")
            print(f"    ⚠️  Synoptic timeseries error: {msg}")
            return {}
        result: dict[str, list] = {}
        for stn in data.get("STATION", []):
            stid = stn.get("STID", "").upper()
            obs  = stn.get("OBSERVATIONS", {})
            times = obs.get("date_time", [])
            vals  = obs.get("air_temp_set_1", [])
            pairs = []
            for t_str, v in zip(times, vals):
                if v is None:
                    continue
                try:
                    dt = datetime.fromisoformat(t_str.replace("Z", "+00:00"))
                    pairs.append((dt, float(v)))
                except Exception:
                    pass
            if pairs:
                result[stid] = pairs
        return result
    except Exception as e:
        print(f"    ⚠️  Synoptic timeseries fetch failed: {e}")
        return {}


def fetch_radius_timeseries(
    lat: float,
    lon: float,
    radius_miles: float,
    date_str: str,
    start_hour_edt: int,
    end_hour_edt: int,
) -> dict[str, list[tuple[datetime, float]]]:
    """
    Query timeseries for all stations within radius on a specific date window.
    Returns dict: stid → list of (utc_datetime, temp_f).
    """
    token = _token()
    if not token:
        return {}

    date_obj = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_utc = date_obj + timedelta(hours=start_hour_edt + 4)
    end_utc   = date_obj + timedelta(hours=end_hour_edt + 4)
    start_str = start_utc.strftime("%Y%m%d%H%M")
    end_str   = end_utc.strftime("%Y%m%d%H%M")

    params = {
        "token":  token,
        "radius": f"{lat},{lon},{radius_miles}",
        "limit":  "30",
        "start":  start_str,
        "end":    end_str,
        "vars":   "air_temp",
        "units":  "english",
        "output": "json",
    }
    qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    url = f"{SYNOPTIC_BASE}/stations/timeseries?{qs}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
        if data.get("SUMMARY", {}).get("RESPONSE_CODE") != 1:
            print(f"    ⚠️  Synoptic radius error: {data.get('SUMMARY',{}).get('RESPONSE_MESSAGE','?')}")
            return {}
        result: dict[str, list] = {}
        for stn in data.get("STATION", []):
            stid = stn.get("STID", "").upper()
            obs  = stn.get("OBSERVATIONS", {})
            times = obs.get("date_time", [])
            vals  = obs.get("air_temp_set_1", [])
            pairs = []
            for t_str, v in zip(times, vals):
                if v is None: continue
                try:
                    dt = datetime.fromisoformat(t_str.replace("Z", "+00:00"))
                    pairs.append((dt, float(v)))
                except Exception:
                    pass
            if pairs:
                result[stid] = pairs
        return result
    except Exception as e:
        print(f"    ⚠️  Synoptic radius fetch failed: {e}")
        return {}


def _mean_temp(pairs: list[tuple[datetime, float]]) -> Optional[float]:
    """Mean temperature across all readings in the window."""
    if not pairs:
        return None
    return round(sum(v for _, v in pairs) / len(pairs), 1)


def compute_features_for_day(date_str: str, nws_high: Optional[float] = None) -> dict:
    """
    Fetch Synoptic data for a day and compute the same features as
    get_synoptic_obs_features() + the v9 named station features.

    Uses the noon window (NOON_WINDOW_START_EDT to NOON_WINDOW_END_EDT)
    as a representative midday snapshot.

    Returns dict of obs_* feature keys → values (NaN if unavailable).
    """
    nan = float("nan")
    result = {
        # Aggregate
        "obs_synoptic_mean":   nan, "obs_synoptic_min":    nan,
        "obs_synoptic_max":    nan, "obs_synoptic_spread": nan,
        "obs_synoptic_vs_nws": nan, "obs_synoptic_count":  nan,
        # Named stations
        "obs_kjfk_temp": nan, "obs_klga_temp": nan,
        "obs_kewr_temp": nan, "obs_kteb_temp": nan, "obs_knyc_temp": nan,
        # Cross-station diffs
        "obs_kjfk_vs_knyc":      nan, "obs_klga_vs_knyc":      nan,
        "obs_kewr_vs_knyc":      nan, "obs_airport_spread":     nan,
        "obs_coastal_vs_inland": nan,
        # NYSM borough aggregate
        "obs_nysm_mean": nan, "obs_nysm_min": nan, "obs_nysm_max": nan,
        "obs_nysm_spread": nan, "obs_nysm_vs_nws": nan, "obs_nysm_count": nan,
    }

    print(f"  Fetching Synoptic timeseries for {date_str} "
          f"(window: {NOON_WINDOW_START_EDT}am-{NOON_WINDOW_END_EDT}pm EDT)...")

    # ── Call 1: named ASOS + NYSM stations ────────────────────────────
    all_stids = BACKFILL_STIDS + "," + NYSM_STIDS
    named_ts = fetch_timeseries(
        all_stids, date_str, NOON_WINDOW_START_EDT, NOON_WINDOW_END_EDT
    )
    time.sleep(0.3)

    # ── Call 2: radius-based aggregate ────────────────────────────────
    radius_ts = fetch_radius_timeseries(
        CENTRAL_PARK_LAT, CENTRAL_PARK_LON, RADIUS_MILES,
        date_str, NOON_WINDOW_START_EDT, NOON_WINDOW_END_EDT,
    )
    time.sleep(0.3)

    # ── Named station readings (mean of all readings in window) ───────
    _NYSM_STIDS = {"MANH", "BRON", "QUEE", "BKLN", "STAT"}
    knyc_t = _mean_temp(named_ts.get("KNYC", []))
    kjfk_t = _mean_temp(named_ts.get("KJFK", []))
    klga_t = _mean_temp(named_ts.get("KLGA", []))
    kewr_t = _mean_temp(named_ts.get("KEWR", []))
    kteb_t = _mean_temp(named_ts.get("KTEB", []))

    if knyc_t is not None: result["obs_knyc_temp"] = knyc_t
    if kjfk_t is not None: result["obs_kjfk_temp"] = kjfk_t
    if klga_t is not None: result["obs_klga_temp"] = klga_t
    if kewr_t is not None: result["obs_kewr_temp"] = kewr_t
    if kteb_t is not None: result["obs_kteb_temp"] = kteb_t

    # Cross-station diffs (anchored at KNYC)
    if knyc_t is not None:
        if kjfk_t is not None: result["obs_kjfk_vs_knyc"] = round(kjfk_t - knyc_t, 1)
        if klga_t is not None: result["obs_klga_vs_knyc"] = round(klga_t - knyc_t, 1)
        if kewr_t is not None: result["obs_kewr_vs_knyc"] = round(kewr_t - knyc_t, 1)

    airport_readings = [t for t in [kjfk_t, klga_t, kewr_t, kteb_t] if t is not None]
    if len(airport_readings) >= 2:
        result["obs_airport_spread"] = round(max(airport_readings) - min(airport_readings), 1)

    coastal = [t for t in [kjfk_t, klga_t] if t is not None]
    inland  = [t for t in [kewr_t, kteb_t] if t is not None]
    if coastal and inland:
        result["obs_coastal_vs_inland"] = round(
            sum(coastal)/len(coastal) - sum(inland)/len(inland), 1
        )

    # ── NYSM borough aggregate ────────────────────────────────────────
    borough_temps = []
    for stid, pairs in named_ts.items():
        if stid in _NYSM_STIDS:
            t = _mean_temp(pairs)
            if t is not None:
                borough_temps.append(t)
    if borough_temps:
        b_mean = sum(borough_temps) / len(borough_temps)
        result["obs_nysm_mean"]   = round(b_mean, 1)
        result["obs_nysm_min"]    = round(min(borough_temps), 1)
        result["obs_nysm_max"]    = round(max(borough_temps), 1)
        result["obs_nysm_spread"] = round(max(borough_temps) - min(borough_temps), 1)
        result["obs_nysm_count"]  = float(len(borough_temps))
        if nws_high is not None:
            result["obs_nysm_vs_nws"] = round(b_mean - nws_high, 1)

    # ── Radius aggregate ──────────────────────────────────────────────
    all_temps = []
    for pairs in radius_ts.values():
        t = _mean_temp(pairs)
        if t is not None:
            all_temps.append(t)

    if all_temps:
        mean_t = sum(all_temps) / len(all_temps)
        result["obs_synoptic_mean"]   = round(mean_t, 1)
        result["obs_synoptic_min"]    = round(min(all_temps), 1)
        result["obs_synoptic_max"]    = round(max(all_temps), 1)
        result["obs_synoptic_spread"] = round(max(all_temps) - min(all_temps), 1)
        result["obs_synoptic_count"]  = float(len(all_temps))
        if nws_high is not None:
            result["obs_synoptic_vs_nws"] = round(mean_t - nws_high, 1)

    # Summary
    pop = sum(1 for v in result.values() if not (isinstance(v, float) and np.isnan(v)))
    print(f"    ✓ {pop}/{len(result)} features populated  "
          f"KJFK={result.get('obs_kjfk_temp')}  "
          f"KNYC={result.get('obs_knyc_temp')}  "
          f"KJFK-KNYC={result.get('obs_kjfk_vs_knyc')}  "
          f"coastal-inland={result.get('obs_coastal_vs_inland')}")

    return result


def backfill(
    dry_run: bool = False,
    days: Optional[int] = None,
    target_date: Optional[str] = None,
):
    """
    Main backfill loop.  Fetches all rows from daily_forecasts where
    atm_snapshot is missing obs_kjfk_temp, then backfills from Synoptic.
    """
    token = _token()
    if not token:
        print("❌ SYNOPTIC_TOKEN not set — aborting")
        sys.exit(1)

    client = _supabase_client()
    print(f"✅ Synoptic token present, Supabase connected")
    print(f"   dry_run={dry_run}  days={days}  target_date={target_date}\n")

    # ── Fetch rows to backfill ────────────────────────────────────────
    if target_date:
        rows_resp = (
            client.table("daily_forecasts")
            .select("id, forecast_date, atm_snapshot, nws_last")
            .eq("forecast_date", target_date)
            .execute()
        )
    else:
        rows_resp = (
            client.table("daily_forecasts")
            .select("id, forecast_date, atm_snapshot, nws_last")
            .order("forecast_date", desc=True)
            .limit(days * 10 if days else 2000)   # some days have multiple rows
            .execute()
        )

    all_rows = rows_resp.data or []
    print(f"Fetched {len(all_rows)} rows from daily_forecasts\n")

    # Group by date, pick the row with most atm_snapshot data (canonical row)
    by_date: dict[str, dict] = {}
    for row in all_rows:
        d = row.get("forecast_date", "")
        if not d:
            continue
        existing = by_date.get(d)
        if existing is None:
            by_date[d] = row
        else:
            # prefer row with larger atm_snapshot
            new_snap = row.get("atm_snapshot") or {}
            ex_snap  = existing.get("atm_snapshot") or {}
            if isinstance(new_snap, str): new_snap = json.loads(new_snap)
            if isinstance(ex_snap,  str): ex_snap  = json.loads(ex_snap)
            if len(new_snap) > len(ex_snap):
                by_date[d] = row

    # Filter to dates we need to backfill
    dates_to_process = []
    already_filled   = 0
    for d, row in sorted(by_date.items(), reverse=True):
        snap = row.get("atm_snapshot") or {}
        if isinstance(snap, str):
            try: snap = json.loads(snap)
            except: snap = {}
        # Skip if already has kjfk data (idempotent)
        if snap.get("obs_kjfk_temp") is not None:
            already_filled += 1
            continue
        dates_to_process.append((d, row))

    # Apply --days filter
    if days is not None:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        dates_to_process = [(d, r) for d, r in dates_to_process if d >= cutoff]

    print(f"Dates to backfill: {len(dates_to_process)}  "
          f"(already filled: {already_filled})\n")

    if not dates_to_process:
        print("Nothing to backfill — all rows already have Synoptic station data.")
        return

    ok_count = 0
    err_count = 0

    for d, row in dates_to_process:
        print(f"{'[DRY RUN] ' if dry_run else ''}Processing {d}...")
        nws_high = None
        try:
            nws_high = float(row.get("nws_last") or 0) or None
        except Exception:
            pass

        feats = compute_features_for_day(d, nws_high=nws_high)

        # Only write keys that are actually populated (not NaN)
        feats_to_write = {
            k: v for k, v in feats.items()
            if not (isinstance(v, float) and np.isnan(v))
        }

        if not feats_to_write:
            print(f"    ⚠️  No valid features for {d}, skipping\n")
            err_count += 1
        elif dry_run:
            print(f"    [DRY RUN] Would write {len(feats_to_write)} keys: "
                  f"{list(feats_to_write.keys())[:5]}...\n")
            ok_count += 1
        else:
            # Merge into existing atm_snapshot (don't overwrite other keys)
            snap = row.get("atm_snapshot") or {}
            if isinstance(snap, str):
                try: snap = json.loads(snap)
                except: snap = {}
            snap.update(feats_to_write)

            try:
                client.table("daily_forecasts").update(
                    {"atm_snapshot": snap}
                ).eq("id", row["id"]).execute()
                print(f"    ✓ Written to row id={row['id']}\n")
                ok_count += 1
            except Exception as e:
                print(f"    ❌ Write failed for {d}: {e}\n")
                err_count += 1

        time.sleep(SLEEP_BETWEEN_DAYS)

    print(f"\n{'='*60}")
    print(f"Backfill complete: {ok_count} OK, {err_count} errors")
    print(f"Next step: retrain v9 with `python train_models.py --v9`")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill Synoptic named-station features into atm_snapshot"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without touching Supabase")
    parser.add_argument("--days", type=int, default=None,
                        help="Only backfill last N days (default: all available)")
    parser.add_argument("--date", type=str, default=None, dest="target_date",
                        help="Backfill a single specific date (YYYY-MM-DD)")
    args = parser.parse_args()
    backfill(dry_run=args.dry_run, days=args.days, target_date=args.target_date)


if __name__ == "__main__":
    main()
