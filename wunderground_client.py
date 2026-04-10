# wunderground_client.py — Fetch real-time personal weather station data
# from Weather Underground (The Weather Company) for hyper-local NYC signals.
#
# WU has thousands of citizen PWS stations near Central Park. Unlike NWS KNYC
# (Central Park official), these stations often catch sea-breeze incursions and
# local cold pools 30-60 min before they register at the official station.
#
# Setup:
#   1. Create a free account at wunderground.com/member/registration
#   2. Generate an API key at wunderground.com/member/api-keys
#   3. Set WU_API_KEY in GitHub Secrets
#   4. Optionally set WU_STATION_IDS as comma-separated station IDs (e.g. "KNYNEWYO123,KNYNEWYORK456")
#      If not set, we auto-discover nearby stations from the geocode endpoint.
#
# Without WU_API_KEY the module degrades gracefully: all features return NaN.

import os
import json
import time
import urllib.request
import urllib.parse
from typing import Optional

WU_BASE = "https://api.weather.com/v2/pws"

# Known reliable Central Park-area WU stations as fallback if geocode discovery fails.
# Browse wunderground.com/wundermap?lat=40.783&lon=-73.965 to find more.
NYC_FALLBACK_STATIONS = [
    "KNYNEWYORK715",   # Upper West Side
    "KNYNEWYORK1108",  # Central Park area
    "KNYNEWYORK308",   # Upper East Side
]


def _wu_key() -> Optional[str]:
    return os.environ.get("WU_API_KEY", "").strip() or None


def fetch_nearby_stations(lat: float, lon: float, limit: int = 5) -> list[dict]:
    """
    Discover nearby WU PWS stations via geocode. Returns list of station dicts
    with 'stationID', 'neighborhood', 'lat', 'lon'.
    Free tier supports this endpoint.
    """
    key = _wu_key()
    if not key:
        return []

    url = (f"{WU_BASE}/observations/nearby"
           f"?geocode={lat},{lon}"
           f"&limit={limit}"
           f"&format=json&units=e"
           f"&apiKey={urllib.parse.quote(key)}")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("observations", [])
    except Exception as e:
        print(f"  ⚠️ WU nearby stations fetch failed: {e}")
        return []


def fetch_station_obs(station_id: str) -> Optional[dict]:
    """
    Fetch current observation for a single WU station.
    Returns obs dict with 'imperial' sub-dict containing 'temp', 'windSpeed', etc.
    """
    key = _wu_key()
    if not key:
        return None

    url = (f"{WU_BASE}/observations/current"
           f"?stationId={urllib.parse.quote(station_id)}"
           f"&format=json&units=e&numericPrecision=decimal"
           f"&apiKey={urllib.parse.quote(key)}")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        obs_list = data.get("observations", [])
        return obs_list[0] if obs_list else None
    except Exception as e:
        print(f"  ⚠️ WU obs fetch for {station_id}: {e}")
        return None


def get_wu_obs_features(
    lat: float = 40.7834,
    lon: float = -73.965,
    nws_last: Optional[float] = None,
) -> dict:
    """
    Fetch latest WU PWS data for Central Park area stations and return
    observation features.

    Station IDs: reads WU_STATION_IDS env var (comma-separated) first.
    If not set, auto-discovers nearby stations via geocode.

    Returns dict with:
        obs_ambient_temp      — mean temp across stations (°F)
        obs_ambient_vs_nws    — obs_ambient_temp - nws_last
        obs_ambient_spread    — max - min temp across stations
        obs_ambient_count     — number of stations with valid readings

    All NaN if WU_API_KEY is missing or no stations return data.
    """
    import numpy as np

    nan_result = {
        "obs_ambient_temp": np.nan,
        "obs_ambient_vs_nws": np.nan,
        "obs_ambient_spread": np.nan,
        "obs_ambient_count": np.nan,
    }

    if not _wu_key():
        return nan_result

    # Determine which stations to query
    station_ids_env = os.environ.get("WU_STATION_IDS", "").strip()
    if station_ids_env:
        station_ids = [s.strip() for s in station_ids_env.split(",") if s.strip()]
    else:
        # Auto-discover nearby stations
        nearby = fetch_nearby_stations(lat, lon, limit=5)
        if nearby:
            station_ids = [obs["stationID"] for obs in nearby if obs.get("stationID")]
            print(f"  🔍 WU auto-discovered {len(station_ids)} nearby stations: {station_ids}")
        else:
            station_ids = NYC_FALLBACK_STATIONS
            print(f"  📍 WU using fallback station list: {station_ids}")

    temps = []
    for i, sid in enumerate(station_ids):
        if i > 0:
            time.sleep(0.5)  # gentle rate limiting
        obs = fetch_station_obs(sid)
        if obs is None:
            continue
        imperial = obs.get("imperial", {})
        temp_f = imperial.get("temp")
        if temp_f is not None:
            try:
                temps.append(float(temp_f))
                print(f"  🌡️ WU {sid}: {float(temp_f):.1f}°F")
            except (ValueError, TypeError):
                pass

    if not temps:
        return nan_result

    mean_temp = sum(temps) / len(temps)
    nan_result["obs_ambient_temp"] = round(mean_temp, 1)
    nan_result["obs_ambient_count"] = float(len(temps))
    nan_result["obs_ambient_spread"] = round(max(temps) - min(temps), 1) if len(temps) > 1 else 0.0
    if nws_last is not None:
        nan_result["obs_ambient_vs_nws"] = round(mean_temp - nws_last, 1)
        print(f"  🗺️ WU mean: {mean_temp:.1f}°F vs NWS {nws_last}°F "
              f"→ {nan_result['obs_ambient_vs_nws']:+.1f}°F ({len(temps)} stations)")

    return nan_result


if __name__ == "__main__":
    # List nearby stations — run this once to find good station IDs for WU_STATION_IDS
    import sys
    lat, lon = 40.7834, -73.965
    print(f"Discovering WU stations near Central Park ({lat}, {lon})...")
    stations = fetch_nearby_stations(lat, lon, limit=10)
    if stations:
        print(f"\nFound {len(stations)} nearby stations:\n")
        for s in stations:
            imp = s.get("imperial", {})
            print(f"  {s.get('stationID'):20s}  {s.get('neighborhood','?'):30s}  "
                  f"temp={imp.get('temp','?')}°F  "
                  f"lat={s.get('lat','?'):.4f}  lon={s.get('lon','?'):.4f}")
        print(f"\nSet WU_STATION_IDS to your preferred comma-separated IDs above.")
    else:
        print("No stations found — check WU_API_KEY is set correctly.")
