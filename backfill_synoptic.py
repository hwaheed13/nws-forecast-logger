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
    Main backfill loop.  Fetches all rows from prediction_logs where
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
    # Table: prediction_logs  |  date col: target_date  |  nws col: nws_d0
    # Supabase has a hard 1000-row default limit per request, so paginate.
    PAGE = 1000
    all_rows: list[dict] = []

    if target_date:
        rows_resp = (
            client.table("prediction_logs")
            .select("id, target_date, atm_snapshot, nws_d0")
            .eq("target_date", target_date)
            .execute()
        )
        all_rows = rows_resp.data or []
    else:
        offset = 0
        while True:
            page_resp = (
                client.table("prediction_logs")
                .select("id, target_date, atm_snapshot, nws_d0")
                .order("target_date", desc=True)
                .range(offset, offset + PAGE - 1)
                .execute()
            )
            page = page_resp.data or []
            all_rows.extend(page)
            if len(page) < PAGE:
                break  # last page
            offset += PAGE

    print(f"Fetched {len(all_rows)} rows from prediction_logs\n")

    # Group by date, pick the row with most atm_snapshot data (canonical row)
    by_date: dict[str, dict] = {}
    for row in all_rows:
        d = row.get("target_date", "")
        if not d:
            continue
        existing = by_date.get(d)
        if existing is None:
            by_date[d] = row
        else:
            # prefer row with larger atm_snapshot
            new_snap = row.get("atm_snapshot") or {}
            ex_snap  = existing.get("atm_snapshot") or {}
            if isinstance(new_snap, str):
                try: new_snap = json.loads(new_snap)
                except: new_snap = {}
            if isinstance(ex_snap, str):
                try: ex_snap = json.loads(ex_snap)
                except: ex_snap = {}
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
            nws_high = float(row.get("nws_d0") or 0) or None
        except Exception:
            pass

        feats = compute_features_for_day(d, nws_high=nws_high)

        # Filter out NaN/inf — catches both Python float and np.float64
        import math
        feats_to_write = {}
        for k, v in feats.items():
            try:
                fv = float(v)
                if math.isfinite(fv):
                    feats_to_write[k] = round(fv, 4)
            except (TypeError, ValueError):
                feats_to_write[k] = v  # keep non-numeric values (strings, ints)

        if not feats_to_write:
            print(f"    ⚠️  No valid features for {d}, skipping\n")
            err_count += 1
        elif dry_run:
            print(f"    [DRY RUN] Would write {len(feats_to_write)} keys: "
                  f"{list(feats_to_write.keys())[:5]}...\n")
            ok_count += 1
        else:
            # Merge into existing atm_snapshot — scrub the entire dict for
            # NaN/inf before writing (existing snap may also contain bad values)
            snap = row.get("atm_snapshot") or {}
            if isinstance(snap, str):
                try: snap = json.loads(snap)
                except: snap = {}
            snap.update(feats_to_write)

            # Deep-scrub: replace any remaining non-finite floats with None
            def _scrub(obj):
                if isinstance(obj, dict):
                    return {k: _scrub(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_scrub(v) for v in obj]
                try:
                    fv = float(obj)
                    return None if not math.isfinite(fv) else obj
                except (TypeError, ValueError):
                    return obj
            snap = _scrub(snap)

            try:
                client.table("prediction_logs").update(
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


def _scrub_snap(obj):
    """Recursively replace NaN/inf with None so Supabase JSONB accepts it."""
    import math
    if isinstance(obj, dict):
        return {k: _scrub_snap(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_snap(v) for v in obj]
    try:
        fv = float(obj)
        return None if not math.isfinite(fv) else obj
    except (TypeError, ValueError):
        return obj


def csv_backfill(
    csv_path: str = "multiyear_atmospheric.csv",
    nws_log_path: str = "nws_forecast_log.csv",
    city: str = "nyc",
    dry_run: bool = False,
    limit: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sleep_sec: float = 1.5,
):
    """
    THE MOAT UNLOCK.

    Reads historical dates from multiyear_atmospheric.csv (2022-01-01 → today),
    fetches Synoptic timeseries for KJFK/KLGA/KEWR/KTEB/KNYC/MANH + radius for
    each date, then upserts minimal prediction_logs rows so train_models.py's
    _load_prediction_logs_with_snapshots() picks them up for v9/v10 training.

    This transforms v9 from "10 days of KJFK signal" → "3+ years of actual
    marine cap fingerprints" — the full historical moat.

    Why this matters:
      On a cap day: KJFK=50°F, KNYC=56°F → obs_kjfk_vs_knyc=-6 → strong signal
      On a clear day: KJFK=72°F, KNYC=71°F → diff≈0 → no suppression signal
      With 3 years of data, the model learns exactly what the gradient means.

    Rate limits: Synoptic free tier = 1000 calls/day. Each date = 2 calls
    (named STIDs + radius). For 1562 dates: ~3124 calls.
    Use --limit 400 to run in 3-day batches under the free tier.
    Enterprise tier (unlimited) can run the full set in one shot.

    Upserted prediction_logs row schema (minimum required by _load_prediction_logs):
      target_date, city, lead_used, ml_actual_high, nws_last, atm_snapshot

    Idempotent: skips dates that already have obs_kjfk_temp in prediction_logs.

    Usage:
        # Full historical backfill (enterprise token):
        python backfill_synoptic.py --csv-backfill

        # Incremental batches under free tier (run 3 days in a row):
        python backfill_synoptic.py --csv-backfill --limit 400

        # Specific date range:
        python backfill_synoptic.py --csv-backfill --start 2024-06-01 --end 2024-09-30

        # Dry run to verify:
        python backfill_synoptic.py --csv-backfill --dry-run --limit 5
    """
    import math, csv as csv_mod

    token = _token()
    if not token:
        print("❌ SYNOPTIC_TOKEN not set — aborting")
        sys.exit(1)

    client = _supabase_client()
    print(f"✅ Synoptic token present, Supabase connected")
    print(f"   city={city}  dry_run={dry_run}  limit={limit}  "
          f"start={start_date}  end={end_date}\n")

    # ── Load multiyear CSV → dates + actual highs ─────────────────────
    if not os.path.exists(csv_path):
        print(f"❌ {csv_path} not found — run from the repo root directory")
        sys.exit(1)

    import pandas as pd
    multi = pd.read_csv(csv_path)
    multi["target_date"] = pd.to_datetime(multi["target_date"]).dt.strftime("%Y-%m-%d")
    # Filter to the correct city
    if "city" in multi.columns:
        multi = multi[multi["city"] == city].copy()

    # Date range filter
    if start_date:
        multi = multi[multi["target_date"] >= start_date]
    if end_date:
        multi = multi[multi["target_date"] <= end_date]

    # Build actual_high lookup
    actual_map: dict[str, float] = {}
    for _, row in multi.iterrows():
        d = str(row["target_date"])
        ah = row.get("actual_high")
        if ah is not None and not (isinstance(ah, float) and math.isnan(ah)):
            actual_map[d] = float(ah)

    print(f"CSV: {len(multi)} rows  ({multi['target_date'].min()} → {multi['target_date'].max()})")
    print(f"     {len(actual_map)} dates with actual_high\n")

    # ── Load NWS forecast log → nws_last per date ─────────────────────
    nws_map: dict[str, float] = {}
    if os.path.exists(nws_log_path):
        nws_df = pd.read_csv(nws_log_path)
        forecasts = nws_df[nws_df["forecast_or_actual"] == "forecast"].copy()
        forecasts["timestamp"] = pd.to_datetime(forecasts["timestamp"])
        forecasts["target_date"] = pd.to_datetime(forecasts["target_date"]).dt.strftime("%Y-%m-%d")
        # Last D0 forecast per date (closest to day-of) = what NWS said that morning
        last_f = (forecasts.sort_values("timestamp")
                            .groupby("target_date")["predicted_high"]
                            .last())
        for d, v in last_f.items():
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                nws_map[str(d)] = float(v)
        print(f"NWS log: {len(nws_map)} D0 forecasts\n")

    # ── Check which dates already have Synoptic data in prediction_logs ─
    PAGE = 1000
    existing_kjfk: set[str] = set()
    offset = 0
    while True:
        resp = (
            client.table("prediction_logs")
            .select("target_date, atm_snapshot")
            .eq("city", city)
            .range(offset, offset + PAGE - 1)
            .execute()
        )
        page = resp.data or []
        for row in page:
            d = str(row.get("target_date", ""))[:10]
            snap = row.get("atm_snapshot") or {}
            if isinstance(snap, str):
                try: snap = json.loads(snap)
                except: snap = {}
            if snap.get("obs_kjfk_temp") is not None:
                existing_kjfk.add(d)
        if len(page) < PAGE:
            break
        offset += PAGE

    print(f"Supabase: {len(existing_kjfk)} dates already have obs_kjfk_temp — will skip\n")

    # ── Build ordered work list ────────────────────────────────────────
    # Process oldest-first so the model learns cap season in order
    all_dates = sorted(actual_map.keys())
    todo = [d for d in all_dates if d not in existing_kjfk]

    print(f"Dates to process: {len(todo)}  (skipping {len(all_dates)-len(todo)} already done)")
    if limit:
        todo = todo[:limit]
        print(f"  → Limiting to {limit} dates for this run (rate-limit safety)\n")
    else:
        print()

    if not todo:
        print("Nothing to backfill — all CSV dates already have Synoptic station data.")
        return

    # ── Determine STIDs by city ────────────────────────────────────────
    if city.lower() == "lax":
        named_stids_str = "KLAX,KSMO,KBUR,KVNY,KCQT"
        nysm_stids_str  = ""
        radius_lat, radius_lon = 33.9425, -118.4081  # LAX
    else:
        named_stids_str = "KNYC,KJFK,KLGA,KEWR,KTEB"
        nysm_stids_str  = "MANH,BRON,QUEE,BKLN,STAT"
        radius_lat, radius_lon = CENTRAL_PARK_LAT, CENTRAL_PARK_LON

    all_stids_str = named_stids_str
    if nysm_stids_str:
        all_stids_str += "," + nysm_stids_str

    ok_count  = 0
    err_count = 0

    for idx, d in enumerate(todo, 1):
        actual_high = actual_map.get(d)
        nws_last    = nws_map.get(d)

        print(f"[{idx}/{len(todo)}] {d}  actual={actual_high}°F  nws_last={nws_last}°F")

        # ── Fetch Synoptic timeseries ──────────────────────────────────
        named_ts = fetch_timeseries(all_stids_str, d,
                                    NOON_WINDOW_START_EDT, NOON_WINDOW_END_EDT)
        time.sleep(0.3)
        radius_ts = fetch_radius_timeseries(
            radius_lat, radius_lon, RADIUS_MILES, d,
            NOON_WINDOW_START_EDT, NOON_WINDOW_END_EDT,
        )
        time.sleep(0.3)

        feats = compute_features_for_day(d, nws_high=nws_last)

        # ── Scrub NaN/inf ──────────────────────────────────────────────
        snap: dict = {}
        for k, v in feats.items():
            try:
                fv = float(v)
                if math.isfinite(fv):
                    snap[k] = round(fv, 4)
            except (TypeError, ValueError):
                snap[k] = v

        if not snap or snap.get("obs_kjfk_temp") is None:
            print(f"    ⚠️  No KJFK data for {d} — Synoptic may not have records "
                  f"this far back. Skipping.\n")
            err_count += 1
            time.sleep(sleep_sec)
            continue

        snap = _scrub_snap(snap)

        if dry_run:
            print(f"    [DRY RUN] Would upsert: KJFK={snap.get('obs_kjfk_temp')}°F  "
                  f"KJFK-KNYC={snap.get('obs_kjfk_vs_knyc')}  "
                  f"coastal-inland={snap.get('obs_coastal_vs_inland')}\n")
            ok_count += 1
            time.sleep(sleep_sec)
            continue

        # ── Upsert into prediction_logs ────────────────────────────────
        # Use upsert with on_conflict="target_date,city,lead_used" so we
        # don't duplicate rows. For dates that already have a row, we merge
        # the atm_snapshot rather than replacing it wholesale.
        try:
            # Check if row exists
            existing_resp = (
                client.table("prediction_logs")
                .select("id, atm_snapshot")
                .eq("target_date", d)
                .eq("city", city)
                .in_("lead_used", ["today_for_today", "D0"])
                .limit(1)
                .execute()
            )
            existing = (existing_resp.data or [])

            if existing:
                # Row exists — merge snapshot
                row_id = existing[0]["id"]
                old_snap = existing[0].get("atm_snapshot") or {}
                if isinstance(old_snap, str):
                    try: old_snap = json.loads(old_snap)
                    except: old_snap = {}
                old_snap.update(snap)
                merged = _scrub_snap(old_snap)
                update_payload: dict = {"atm_snapshot": merged}
                if actual_high is not None:
                    update_payload["ml_actual_high"] = actual_high
                client.table("prediction_logs").update(update_payload).eq("id", row_id).execute()
                print(f"    ✓ Updated row id={row_id}  KJFK={snap.get('obs_kjfk_temp')}°F  "
                      f"KJFK-KNYC={snap.get('obs_kjfk_vs_knyc'):+.1f}  "
                      f"coastal-inland={snap.get('obs_coastal_vs_inland', 'NaN')}\n"
                      if snap.get('obs_kjfk_vs_knyc') is not None else
                      f"    ✓ Updated row id={row_id}  KJFK={snap.get('obs_kjfk_temp')}°F\n")
            else:
                # No row — insert minimal row with enough fields for training
                insert_payload = {
                    "target_date":    d,
                    "city":           city,
                    "lead_used":      "today_for_today",
                    "atm_snapshot":   snap,
                    "ml_actual_high": actual_high,
                }
                if nws_last is not None:
                    insert_payload["nws_last"] = nws_last
                    insert_payload["nws_d0"]   = nws_last
                client.table("prediction_logs").insert(insert_payload).execute()
                print(f"    ✓ Inserted new row  KJFK={snap.get('obs_kjfk_temp')}°F  "
                      f"KJFK-KNYC={snap.get('obs_kjfk_vs_knyc')}\n")

            ok_count += 1

        except Exception as e:
            print(f"    ❌ Supabase write failed for {d}: {e}\n")
            err_count += 1

        time.sleep(sleep_sec)

    print(f"\n{'='*60}")
    print(f"CSV Backfill complete: {ok_count} OK, {err_count} errors / no-data")
    if ok_count > 0:
        print(f"\n🎯 Next: retrain v9 with `python train_models.py --v9`")
        print(f"   v9 now has {ok_count} historical Synoptic rows to learn from")
        print(f"   (was: ~10 rows → now: potentially {ok_count}+ years of cap-day signal)")


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

    # ── CSV backfill mode — the historical moat unlock ─────────────────
    parser.add_argument("--csv-backfill", action="store_true",
                        help="Backfill Synoptic features for all dates in multiyear_atmospheric.csv. "
                             "Creates/updates prediction_logs rows so v9/v10 training has full history.")
    parser.add_argument("--csv-path", type=str, default="multiyear_atmospheric.csv",
                        help="Path to multiyear_atmospheric.csv (default: ./multiyear_atmospheric.csv)")
    parser.add_argument("--nws-log", type=str, default="nws_forecast_log.csv",
                        help="Path to nws_forecast_log.csv for D0 forecast lookup")
    parser.add_argument("--city", type=str, default="nyc",
                        help="City key: nyc (default) or lax")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max dates to process per run (rate-limit safety — "
                             "free tier: use 400; enterprise: omit for all)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date filter YYYY-MM-DD (csv-backfill mode)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date filter YYYY-MM-DD (csv-backfill mode)")
    parser.add_argument("--sleep", type=float, default=1.5,
                        help="Seconds to sleep between dates (default: 1.5 — "
                             "reduce to 0.5 on enterprise tier for faster runs)")

    args = parser.parse_args()

    if args.csv_backfill:
        csv_backfill(
            csv_path=args.csv_path,
            nws_log_path=args.nws_log,
            city=args.city,
            dry_run=args.dry_run,
            limit=args.limit,
            start_date=args.start,
            end_date=args.end,
            sleep_sec=args.sleep,
        )
    else:
        backfill(dry_run=args.dry_run, days=args.days, target_date=args.target_date)


if __name__ == "__main__":
    main()
