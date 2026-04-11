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
# Public data endpoint — NO API KEY REQUIRED.
# Data is available freely at nysmesonet.org under their open data policy.
#
# These stations report every 5 minutes, are QC'd by SUNY, and capture the
# urban heat island gradient across the five boroughs. When Brooklyn is 3°F
# colder than Manhattan at 11am while NWS says 66°F — that's your signal.

import json
import urllib.request
from typing import Optional

NYSM_BASE = "https://nysmesonet.nysed.gov"

# NYC-area NY Mesonet station IDs — sorted by proximity to Central Park
NYC_NYSM_STATIONS = ["MANH", "BRON", "QUEE", "BKLN", "STAT"]

# Fallback: public CSV endpoint (no key needed)
NYSM_LATEST_URL = "https://nysmesonet.nysed.gov/data/csv/latest/nysm.csv"


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
        print(f"  ⚠️ NY Mesonet CSV fetch failed: {e}")
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


def _parse_temp_f(row: dict) -> Optional[float]:
    """
    Extract temperature in °F from a NY Mesonet row.
    NY Mesonet reports temperature as 'ta' (air temp in °C).
    """
    # Try common column names NY Mesonet uses
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

    No API key required — uses public CSV endpoint.
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

    all_obs = fetch_nysm_latest_csv()
    if not all_obs:
        # DNS failure on GitHub Actions — fall back to Synoptic API borough query
        print("  ℹ️  NYSM CSV unavailable — trying Synoptic borough fallback")
        try:
            from synoptic_client import get_nysm_via_synoptic
            return get_nysm_via_synoptic(nws_last=nws_last)
        except Exception as _fb_e:
            print(f"  ⚠️ Synoptic NYSM fallback failed: {_fb_e}")
            return nan_result

    temps = []
    for stid in target_stations:
        row = all_obs.get(stid)
        if row is None:
            continue
        t = _parse_temp_f(row)
        if t is not None:
            temps.append(t)
            print(f"  🌡️ NYSM {stid}: {t:.1f}°F")

    if not temps:
        # Data came back but none of our stations had valid temps — print debug
        avail = list(all_obs.keys())[:10]
        print(f"  ⚠️ NYSM: no temps for {target_stations}. Available stations sample: {avail}")
        return nan_result

    mean_t = sum(temps) / len(temps)
    nan_result["obs_nysm_mean"] = round(mean_t, 1)
    nan_result["obs_nysm_min"] = round(min(temps), 1)
    nan_result["obs_nysm_max"] = round(max(temps), 1)
    nan_result["obs_nysm_spread"] = round(max(temps) - min(temps), 1)
    nan_result["obs_nysm_count"] = float(len(temps))
    if nws_last is not None:
        nan_result["obs_nysm_vs_nws"] = round(mean_t - nws_last, 1)
        print(f"  🗺️ NYSM: {len(temps)} boroughs — "
              f"min={nan_result['obs_nysm_min']:.1f}  "
              f"mean={mean_t:.1f}  "
              f"max={nan_result['obs_nysm_max']:.1f}°F  "
              f"vs NWS={nan_result['obs_nysm_vs_nws']:+.1f}°F")

    return nan_result


if __name__ == "__main__":
    print("Fetching NY Mesonet latest observations...")
    all_obs = fetch_nysm_latest_csv()
    if all_obs:
        print(f"Total stations in dataset: {len(all_obs)}")
        print(f"\nNYC-area stations:\n")
        for stid in NYC_NYSM_STATIONS + ["NWRK", "ISLIP", "YONK"]:
            row = all_obs.get(stid)
            if row:
                t = _parse_temp_f(row)
                # Print a few key columns
                ts = row.get("time", row.get("datetime", "?"))
                print(f"  {stid:8s}  temp={t}°F  time={ts}")
            else:
                print(f"  {stid:8s}  not in dataset")
        print(f"\nAll available columns: {list(list(all_obs.values())[0].keys())[:20]}")
    else:
        print("No data returned.")
