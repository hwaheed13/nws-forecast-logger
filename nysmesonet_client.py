# nysmesonet_client.py — Fetch real-time observations from the NY State Mesonet
# (operated by SUNY Albany). ~126 stations across New York State with hyper-dense
# coverage in NYC boroughs.
#
# NYC-area NY Mesonet stations:
#   BKLN — Brooklyn (Prospect Park area)
#   QUEE — Queens (Flushing Meadows)
#   STAT — Staten Island
#   BRON — Bronx (Lehman College)
#   MANH — Manhattan (City College)
#   NWRK — Newark (near KEWR)
#
# Data source priority:
#   1. Direct NYSM CSV endpoint (nysmesonet.nysed.gov) — fails DNS on GitHub Actions
#   2. Iowa Environmental Mesonet (IEM) API — free, no key, aggregates NYSM
#   3. Return NaN gracefully
#
# These stations report every 5 minutes, are QC'd by SUNY, and capture the
# urban heat island gradient across the five boroughs. When Brooklyn is 3°F
# colder than Manhattan at 11am while NWS says 66°F — that's your signal.

import json
import math
import urllib.request
from typing import Optional

NYSM_BASE = "https://nysmesonet.nysed.gov"

# NYC-area NY Mesonet station IDs — sorted by proximity to Central Park
NYC_NYSM_STATIONS = ["MANH", "BRON", "QUEE", "BKLN", "STAT"]

# Primary: public CSV endpoint (no key needed, but DNS fails on GitHub Actions)
NYSM_LATEST_URL = "https://nysmesonet.nysed.gov/data/csv/latest/nysm.csv"

# IEM fallback: Iowa Environmental Mesonet aggregates NYSM data, free, no key
# Returns current observations for all NYSM stations in JSON
IEM_NYSM_URL = "https://mesonet.agron.iastate.edu/api/1/currents.json?network=NYSM"


def fetch_nysm_latest_csv() -> dict[str, dict]:
    """
    Fetch the latest NY Mesonet observations CSV (updated every 5 min).
    Returns dict keyed by station ID with observation values.
    """
    req = urllib.request.Request(
        NYSM_LATEST_URL,
        headers={"User-Agent": "nws-forecast-logger/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        print(f"  ⚠️ NYSM CSV fetch failed: {e}")
        return {}

    lines = raw.strip().splitlines()
    if len(lines) < 2:
        return {}

    headers = [h.strip().lower() for h in lines[0].split(",")]
    result = {}
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < len(headers):
            continue
        row = {headers[i]: parts[i].strip() for i in range(len(headers))}
        stid = row.get("stid", "").strip().upper()
        if stid:
            result[stid] = row
    return result


def fetch_nysm_via_iem(target_stations: list[str]) -> dict[str, float]:
    """
    Fetch NYC borough temperatures from Iowa Environmental Mesonet (IEM).
    IEM mirrors NYSM data and is accessible from GitHub Actions runners.

    Returns dict of {STID: temp_f} for whichever target stations have data.
    """
    req = urllib.request.Request(
        IEM_NYSM_URL,
        headers={"User-Agent": "nws-forecast-logger/1.0", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  ⚠️ IEM NYSM fetch failed: {e}")
        return {}

    # IEM currents.json response:
    # {"data": [{"station": "BKLN", "tmpf": 54.0, "tmpc": 12.2, ...}, ...]}
    stations_data = data.get("data", [])
    if not stations_data:
        print("  ⚠️ IEM NYSM: empty data response")
        return {}

    target_set = {s.upper() for s in target_stations}
    result = {}
    for stn in stations_data:
        stid = (stn.get("station") or stn.get("stid") or "").upper()
        if stid not in target_set:
            continue

        # Try Fahrenheit first, then Celsius → convert
        t = None
        tmpf = stn.get("tmpf")
        if tmpf is not None:
            try:
                tf = float(tmpf)
                if not math.isnan(tf) and -60 < tf < 130:
                    t = round(tf, 1)
            except (TypeError, ValueError):
                pass

        if t is None:
            tmpc = stn.get("tmpc")
            if tmpc is not None:
                try:
                    tc = float(tmpc)
                    if not math.isnan(tc) and -50 < tc < 55:
                        t = round(tc * 9 / 5 + 32, 1)
                except (TypeError, ValueError):
                    pass

        if t is not None:
            result[stid] = t
            print(f"  🌡️ NYSM {stid} (IEM): {t:.1f}°F")

    return result


def _parse_temp_f(row: dict) -> Optional[float]:
    """
    Extract temperature in °F from a NY Mesonet CSV row.
    NY Mesonet reports temperature as 'ta' (air temp in °C).
    """
    for col in ["ta", "tair", "air_temp", "temp_air"]:
        val = row.get(col)
        if val and val not in ("", "NaN", "nan", "M", "missing"):
            try:
                temp_c = float(val)
                return round(temp_c * 9 / 5 + 32, 1)  # C → F
            except ValueError:
                pass
    return None


def get_nysm_obs_features(
    stations: list[str] = None,
    nws_last: Optional[float] = None,
) -> dict:
    """
    Fetch NY Mesonet observations for NYC-area stations and return features.

    Returns dict with:
        obs_nysm_mean        — mean temp across NYC borough stations (°F)
        obs_nysm_min         — coldest borough station (°F)
        obs_nysm_max         — warmest borough station (°F)
        obs_nysm_spread      — max - min across borough stations
        obs_nysm_vs_nws      — obs_nysm_mean - nws_last
        obs_nysm_count       — number of valid stations

    Tries NYSM direct CSV first, then IEM as fallback.
    """
    import numpy as np

    nan_result = {
        "obs_nysm_mean": np.nan,
        "obs_nysm_min": np.nan,
        "obs_nysm_max": np.nan,
        "obs_nysm_spread": np.nan,
        "obs_nysm_vs_nws": np.nan,
        "obs_nysm_count": np.nan,
    }

    target_stations = stations or NYC_NYSM_STATIONS

    # ── Primary: direct NYSM CSV ──────────────────────────────────────────────
    all_obs = fetch_nysm_latest_csv()
    temps = []

    if all_obs:
        for stid in target_stations:
            row = all_obs.get(stid)
            if row is None:
                continue
            t = _parse_temp_f(row)
            if t is not None:
                temps.append(t)
                print(f"  🌡️ NYSM {stid}: {t:.1f}°F")

        if not temps:
            avail = list(all_obs.keys())[:10]
            print(f"  ⚠️ NYSM CSV: no temps for {target_stations}. Sample stations: {avail}")

    # ── Fallback: Iowa Environmental Mesonet (IEM) ────────────────────────────
    if not temps:
        print("  ℹ️  Trying IEM fallback for NYSM borough data...")
        iem_obs = fetch_nysm_via_iem(target_stations)
        temps = list(iem_obs.values())

    if not temps:
        return nan_result

    mean_t = sum(temps) / len(temps)
    nan_result["obs_nysm_mean"]   = round(mean_t, 1)
    nan_result["obs_nysm_min"]    = round(min(temps), 1)
    nan_result["obs_nysm_max"]    = round(max(temps), 1)
    nan_result["obs_nysm_spread"] = round(max(temps) - min(temps), 1)
    nan_result["obs_nysm_count"]  = float(len(temps))
    if nws_last is not None:
        nan_result["obs_nysm_vs_nws"] = round(mean_t - nws_last, 1)

    vs_str = f"  vs NWS={nan_result['obs_nysm_vs_nws']:+.1f}°F" if nws_last else ""
    print(f"  🗺️ NYSM: {len(temps)} boroughs — "
          f"min={nan_result['obs_nysm_min']:.1f}  "
          f"mean={mean_t:.1f}  "
          f"max={nan_result['obs_nysm_max']:.1f}°F{vs_str}")

    return nan_result


if __name__ == "__main__":
    print("Fetching NY Mesonet latest observations (direct CSV)...")
    all_obs = fetch_nysm_latest_csv()
    if all_obs:
        print(f"Total stations in dataset: {len(all_obs)}")
        for stid in NYC_NYSM_STATIONS:
            row = all_obs.get(stid)
            if row:
                t = _parse_temp_f(row)
                ts = row.get("time", row.get("datetime", "?"))
                print(f"  {stid:8s}  temp={t}°F  time={ts}")
            else:
                print(f"  {stid:8s}  not in dataset")
    else:
        print("Direct CSV failed — trying IEM fallback...")
        result = get_nysm_obs_features()
        print(result)
