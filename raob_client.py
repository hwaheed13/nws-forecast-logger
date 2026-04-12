# raob_client.py — Upper-air radiosonde (balloon) sounding data for NYC area
#
# WHY THIS MATTERS:
#   The 925mb cap signal in the ML model was previously sourced ONLY from GFS/Open-Meteo
#   forecast model output. On a cap day, GFS at 13km resolution can show 925mb at 55°F
#   while the actual atmosphere (measured by a weather balloon) shows 48°F — the model
#   gets fooled by its own atmospheric input and flips the prediction.
#
#   Radiosonde soundings from OKX (Upton, NY — NWS New York upper-air station) provide
#   ACTUAL OBSERVED 925mb and 850mb temperatures from the morning (12Z = 8 AM EDT) and
#   evening (00Z = 8 PM EDT) balloon launches. These are ground truth, not model output.
#
# DATA SOURCE:
#   Iowa State University Mesonet — https://mesonet.agron.iastate.edu/json/raob.py
#   Station: OKX (Upton, NY) — 40.9°N, 72.9°W — ~50 miles from Central Park
#   Sounding times: 12Z (8 AM EDT / 7 AM EST) and 00Z (8 PM EDT / 7 PM EST)
#
# FEATURES PRODUCED:
#   raob_925mb_temp       — Observed 925mb temperature (°F) from most recent sounding
#   raob_850mb_temp       — Observed 850mb temperature (°F) from most recent sounding
#   raob_sounding_hour    — Hour (UTC) of the sounding used (0 or 12)
#   raob_925mb_gfs_diff   — GFS-forecast 925mb minus observed (positive = GFS too warm)
#   raob_925mb_hrrr_diff  — HRRR-forecast 925mb minus observed (positive = HRRR too warm)
#
# TRAINING NOTE:
#   Radiosonde archive is available going back decades via Iowa State. The training
#   backfill script (backfill_new_columns.py) should call fetch_raob_for_date() for each
#   historical training row to populate these features retroactively.

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timedelta
from typing import Optional

import numpy as np


# Iowa State Mesonet RAOB JSON endpoint
# Docs: https://mesonet.agron.iastate.edu/json/raob.py?help
IASTATE_RAOB_URL = "https://mesonet.agron.iastate.edu/json/raob.py"

# OKX = Upton, NY (NWS New York upper-air station)
# Closest radiosonde station to Central Park. 12Z sounding = 8 AM EDT.
OKX_STATION = "OKX"

# Target pressure levels (hPa)
LEVELS_HPA = [925, 850, 700, 500]


def _c_to_f(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return c * 9.0 / 5.0 + 32.0


def _get_json(url: str, retries: int = 3, delay: float = 1.5) -> dict:
    """Fetch JSON from Iowa State Mesonet with retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "nws-forecast-logger/1.0 (weather research)",
                },
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise RuntimeError(f"RAOB request failed after {retries} attempts: {e}") from e


def fetch_raob_for_date(
    target_date: str,
    station: str = OKX_STATION,
    prefer_12z: bool = True,
) -> dict:
    """
    Fetch radiosonde upper-air sounding for a target date.

    For day-of predictions, uses the 12Z (8 AM EDT) sounding which gives
    the morning atmospheric profile — the most relevant cap signal before
    afternoon heating begins.

    Args:
        target_date: 'YYYY-MM-DD' string (local date)
        station: ICAO station ID (default 'OKX' for Upton, NY)
        prefer_12z: If True, prefer the 12Z sounding (morning); else prefer 00Z (previous evening)

    Returns:
        dict with raob_925mb_temp, raob_850mb_temp, raob_sounding_hour, raob_valid
        All temperatures in °F. Returns NaN values if sounding unavailable.
    """
    empty = {
        "raob_925mb_temp": np.nan,
        "raob_850mb_temp": np.nan,
        "raob_700mb_temp": np.nan,
        "raob_sounding_hour": np.nan,
        "raob_valid": 0,
    }

    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        return empty

    # Iowa State API uses UTC datetime for the sounding time.
    # 12Z on target_date = 8 AM EDT = morning profile (what we want).
    # 00Z on target_date = 8 PM EDT previous evening = overnight profile.
    if prefer_12z:
        sounding_hours_utc = [
            (dt, 12),   # 12Z on target date (8 AM EDT) — primary
            (dt, 0),    # 00Z on target date (midnight EDT) — fallback
            (dt - timedelta(days=1), 12),  # Prior day 12Z — last resort
        ]
    else:
        sounding_hours_utc = [
            (dt, 0),    # 00Z (midnight EDT) — previous evening primary
            (dt, 12),   # 12Z (8 AM EDT) — fallback
        ]

    for sounding_dt, hour_utc in sounding_hours_utc:
        # Iowa State format: YYYY-MM-DDTHH:00
        runtime_str = f"{sounding_dt.strftime('%Y-%m-%d')}T{hour_utc:02d}:00"
        url = (
            f"{IASTATE_RAOB_URL}"
            f"?station={station}"
            f"&runtime={runtime_str}"
            f"&fmt=json"
        )

        try:
            data = _get_json(url)
        except Exception as e:
            print(f"  ⚠️ RAOB fetch failed for {runtime_str}: {e}")
            continue

        # Iowa State returns {"profiles": [{"station": ..., "levels": [...]}]}
        profiles = data.get("profiles", [])
        if not profiles:
            continue

        profile = profiles[0]
        levels = profile.get("levels", [])
        if not levels:
            continue

        # Extract temperatures at target pressure levels
        temps = {}
        for level_data in levels:
            pres = level_data.get("pres")  # hPa
            tmpc = level_data.get("tmpc")  # °C
            if pres is None or tmpc is None:
                continue
            pres = float(pres)
            tmpc = float(tmpc)
            # Match to nearest target level (within ±15 hPa)
            for target_level in LEVELS_HPA:
                if abs(pres - target_level) <= 15:
                    # Only store if not already stored (first = closest match)
                    if target_level not in temps:
                        temps[target_level] = _c_to_f(tmpc)

        if 925 not in temps and 850 not in temps:
            # No useful pressure level data, try next sounding time
            continue

        result = {
            "raob_925mb_temp": temps.get(925, np.nan),
            "raob_850mb_temp": temps.get(850, np.nan),
            "raob_700mb_temp": temps.get(700, np.nan),
            "raob_sounding_hour": float(hour_utc),
            "raob_valid": 1,
        }

        level_str = ", ".join(
            f"{lvl}mb={temps[lvl]:.1f}°F" for lvl in LEVELS_HPA if lvl in temps
        )
        print(f"  🎈 RAOB {station} {runtime_str}Z: {level_str}")
        return result

    print(f"  ⚠️ RAOB {station}: no valid sounding for {target_date}")
    return empty


def compute_raob_vs_model_diffs(
    raob_features: dict,
    gfs_925mb_mean: Optional[float] = None,
    hrrr_925mb_mean: Optional[float] = None,
) -> dict:
    """
    Compute differences between observed sounding and model 925mb forecasts.

    These diffs are the core signal: when GFS/HRRR show a warm 925mb but the
    radiosonde shows it's actually cold, models are missing the cap. Negative
    diff = models too warm (cap stronger than they think).

    Args:
        raob_features: dict from fetch_raob_for_date()
        gfs_925mb_mean: GFS-forecast 925mb mean (°F) from Open-Meteo
        hrrr_925mb_mean: HRRR-forecast 925mb mean (°F) from Open-Meteo

    Returns:
        dict with raob_925mb_gfs_diff, raob_925mb_hrrr_diff
    """
    raob_925 = raob_features.get("raob_925mb_temp")
    result = {}

    if raob_925 is not None and not np.isnan(raob_925):
        if gfs_925mb_mean is not None and not np.isnan(gfs_925mb_mean):
            # Positive = GFS too warm vs observed (GFS missing the cap)
            result["raob_925mb_gfs_diff"] = gfs_925mb_mean - raob_925
        else:
            result["raob_925mb_gfs_diff"] = np.nan

        if hrrr_925mb_mean is not None and not np.isnan(hrrr_925mb_mean):
            # Positive = HRRR too warm vs observed
            result["raob_925mb_hrrr_diff"] = hrrr_925mb_mean - raob_925
        else:
            result["raob_925mb_hrrr_diff"] = np.nan
    else:
        result["raob_925mb_gfs_diff"] = np.nan
        result["raob_925mb_hrrr_diff"] = np.nan

    return result


def get_raob_features(
    target_date: str,
    station: str = OKX_STATION,
    gfs_925mb_mean: Optional[float] = None,
    hrrr_925mb_mean: Optional[float] = None,
) -> dict:
    """
    Convenience wrapper: fetch sounding and compute model comparison diffs.

    Returns all raob_* features ready to merge into the ML feature dict.
    """
    raob = fetch_raob_for_date(target_date, station)
    diffs = compute_raob_vs_model_diffs(raob, gfs_925mb_mean, hrrr_925mb_mean)
    return {**raob, **diffs}


# ═════════════════════════════════════════════════════════════════════════
# CLI for testing
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Radiosonde sounding fetch test")
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--station", default=OKX_STATION, help="Upper-air station ID")
    parser.add_argument("--gfs-925", type=float, default=None, help="GFS 925mb mean for diff calc")
    parser.add_argument("--hrrr-925", type=float, default=None, help="HRRR 925mb mean for diff calc")
    args = parser.parse_args()

    from datetime import date as dt_date
    target = args.date or dt_date.today().isoformat()

    print(f"Fetching radiosonde sounding for {target} from station {args.station}...")
    features = get_raob_features(
        target,
        station=args.station,
        gfs_925mb_mean=args.gfs_925,
        hrrr_925mb_mean=args.hrrr_925,
    )
    print(json.dumps(
        {k: round(v, 2) if isinstance(v, float) and not np.isnan(v) else
            (None if isinstance(v, float) and np.isnan(v) else v)
         for k, v in features.items()},
        indent=2,
    ))
