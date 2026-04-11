# synoptic_client.py — Fetch real-time station observations from Synoptic Data API
# (formerly MesoWest). Aggregates data from 100+ networks: ASOS, AWOS, NY Mesonet,
# personal weather stations, road sensors, buoys — all in one API call.
#
# Within 10 miles of Central Park this typically returns 15-25 stations including:
#   - KNYC (Central Park NWS), KJFK, KLGA, KEWR, KTEB (airport ASOS)
#   - NY Mesonet sites: BKLN (Brooklyn), QUEE (Queens), STAT (Staten Island)
#   - CUNY/Columbia campus sensors
#   - Coast Guard and maritime stations
#
# Free tier: 1000 API calls/day — more than enough at 30-min prediction frequency.
# Sign up at: synopticdata.com → get a Public API token (takes ~2 minutes)
# Set: SYNOPTIC_TOKEN in GitHub Secrets
#
# Without SYNOPTIC_TOKEN all features return NaN gracefully.

import os
import json
import urllib.request
import urllib.parse
from typing import Optional

SYNOPTIC_BASE = "https://api.synopticdata.com/v2"


def _token() -> Optional[str]:
    return os.environ.get("SYNOPTIC_TOKEN", "").strip() or None


def fetch_nearby_obs(
    lat: float,
    lon: float,
    radius_miles: float = 10.0,
    limit: int = 20,
    within_minutes: int = 90,
) -> list[dict]:
    """
    Fetch current observations from all stations within radius_miles of lat/lon.
    Returns list of station obs dicts with 'STID', 'NAME', 'OBSERVATIONS' etc.
    within_minutes: only return obs from the last N minutes (default 90).
    """
    token = _token()
    if not token:
        return []

    params = {
        "token": token,
        "radius": f"{lat},{lon},{radius_miles}",
        "vars": "air_temp,wind_speed,wind_gust,wind_direction,relative_humidity,dew_point_temperature",
        "units": "english",
        "within": str(within_minutes),
        "limit": str(limit),
        "obtimezone": "local",
        "output": "json",
    }
    qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    url = f"{SYNOPTIC_BASE}/stations/nearesttime?{qs}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        summary = data.get("SUMMARY", {})
        if summary.get("RESPONSE_CODE") != 1:
            print(f"  ⚠️ Synoptic API: {summary.get('RESPONSE_MESSAGE', 'unknown error')}")
            return []
        return data.get("STATION", [])
    except Exception as e:
        print(f"  ⚠️ Synoptic fetch failed: {e}")
        return []


# NY Mesonet borough station IDs as they appear in the Synoptic radius results
_NYSM_BOROUGH_STIDS = {"BKLN", "QUEE", "STAT", "BRON", "MANH"}


def get_synoptic_obs_features(
    lat: float = 40.7834,
    lon: float = -73.965,
    nws_last: Optional[float] = None,
    radius_miles: float = 10.0,
) -> dict:
    """
    Fetch nearby station obs via Synoptic and return aggregated features.

    Returns dict with:
        obs_synoptic_mean        — mean temp across nearby stations (°F)
        obs_synoptic_min         — coldest station reading (°F) — best cold-bias signal
        obs_synoptic_max         — warmest station reading (°F)
        obs_synoptic_spread      — max - min across stations
        obs_synoptic_vs_nws      — obs_synoptic_mean - nws_last
        obs_synoptic_count       — number of valid stations
        obs_nysm_mean            — mean temp across NYSM borough stations found (°F)
        obs_nysm_min/max/spread  — borough min/max/spread
        obs_nysm_vs_nws          — obs_nysm_mean - nws_last
        obs_nysm_count           — number of borough stations found

    All NaN if SYNOPTIC_TOKEN is not set.
    """
    import numpy as np

    nan_result = {
        "obs_synoptic_mean": np.nan,
        "obs_synoptic_min": np.nan,
        "obs_synoptic_max": np.nan,
        "obs_synoptic_spread": np.nan,
        "obs_synoptic_vs_nws": np.nan,
        "obs_synoptic_count": np.nan,
        # Borough subset (from NYSM stations found in the radius)
        "obs_nysm_mean": np.nan,
        "obs_nysm_min": np.nan,
        "obs_nysm_max": np.nan,
        "obs_nysm_spread": np.nan,
        "obs_nysm_vs_nws": np.nan,
        "obs_nysm_count": np.nan,
    }

    if not _token():
        return nan_result

    stations = fetch_nearby_obs(lat, lon, radius_miles=radius_miles)
    if not stations:
        return nan_result

    temps = []
    borough_temps = []
    for stn in stations:
        stid = stn.get("STID", "?")
        obs = stn.get("OBSERVATIONS", {})
        temp_val = obs.get("air_temp_value_1", {})
        # Synoptic returns {"value": 62.1, "date_time": "..."} per variable
        if isinstance(temp_val, dict):
            t = temp_val.get("value")
        else:
            t = temp_val
        if t is not None:
            try:
                tf = float(t)
                temps.append(tf)
                if stid.upper() in _NYSM_BOROUGH_STIDS:
                    borough_temps.append(tf)
                    print(f"  🏙️ NYSM borough via Synoptic — {stid}: {tf:.1f}°F")
            except (ValueError, TypeError):
                pass

    if not temps:
        return nan_result

    # Debug: log all returned STIDs so we can identify NYSM naming in Synoptic
    all_stids = [s.get("STID", "?") for s in stations]
    print(f"  🔍 Synoptic radius STIDs: {all_stids}")

    mean_t = sum(temps) / len(temps)
    nan_result["obs_synoptic_mean"] = round(mean_t, 1)
    nan_result["obs_synoptic_min"] = round(min(temps), 1)
    nan_result["obs_synoptic_max"] = round(max(temps), 1)
    nan_result["obs_synoptic_spread"] = round(max(temps) - min(temps), 1)
    nan_result["obs_synoptic_count"] = float(len(temps))
    if nws_last is not None:
        nan_result["obs_synoptic_vs_nws"] = round(mean_t - nws_last, 1)

    vs_str = f"  vs NWS={nan_result['obs_synoptic_vs_nws']:+.1f}°F" if nws_last else ""
    print(f"  🗺️ Synoptic: {len(temps)} stations — "
          f"min={nan_result['obs_synoptic_min']:.1f}°F  "
          f"mean={mean_t:.1f}°F  "
          f"max={nan_result['obs_synoptic_max']:.1f}°F{vs_str}")

    # Borough subset
    if borough_temps:
        b_mean = sum(borough_temps) / len(borough_temps)
        nan_result["obs_nysm_mean"]   = round(b_mean, 1)
        nan_result["obs_nysm_min"]    = round(min(borough_temps), 1)
        nan_result["obs_nysm_max"]    = round(max(borough_temps), 1)
        nan_result["obs_nysm_spread"] = round(max(borough_temps) - min(borough_temps), 1)
        nan_result["obs_nysm_count"]  = float(len(borough_temps))
        if nws_last is not None:
            nan_result["obs_nysm_vs_nws"] = round(b_mean - nws_last, 1)
        vs_b = f"  vs NWS={nan_result['obs_nysm_vs_nws']:+.1f}°F" if nws_last else ""
        print(f"  🏙️ Boroughs (NYSM via Synoptic): {len(borough_temps)} stations — "
              f"min={nan_result['obs_nysm_min']:.1f}°F  "
              f"mean={b_mean:.1f}°F  "
              f"max={nan_result['obs_nysm_max']:.1f}°F{vs_b}")

    return nan_result


def get_nysm_via_synoptic(nws_last: Optional[float] = None) -> dict:
    """
    Fetch NYC borough temperatures via Synoptic API using specific NY Mesonet station IDs.
    This is a fallback for when the nysmesonet.nysed.gov endpoint is unreachable
    (e.g. DNS failure on GitHub Actions runners).

    Uses the same Synoptic token already set for the radius-based fetch.
    Borough stations: MANH (Manhattan), BRON (Bronx), QUEE (Queens), BKLN (Brooklyn), STAT (Staten Island).

    Returns dict with obs_nysm_* keys (same schema as nysmesonet_client.get_nysm_obs_features).
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

    token = _token()
    if not token:
        return nan_result

    # The five NYC borough NY Mesonet stations available via Synoptic
    stids = "MANH,BRON,QUEE,BKLN,STAT"
    params = {
        "token": token,
        "stid": stids,
        "vars": "air_temp",
        "units": "english",
        "within": "90",
        "output": "json",
    }
    qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    url = f"{SYNOPTIC_BASE}/stations/nearesttime?{qs}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        summary = data.get("SUMMARY", {})
        if summary.get("RESPONSE_CODE") != 1:
            print(f"  ⚠️ Synoptic NYSM: {summary.get('RESPONSE_MESSAGE', 'unknown error')}")
            return nan_result
        stations = data.get("STATION", [])
    except Exception as e:
        print(f"  ⚠️ Synoptic NYSM fetch failed: {e}")
        return nan_result

    temps = []
    for stn in stations:
        obs = stn.get("OBSERVATIONS", {})
        temp_val = obs.get("air_temp_value_1", {})
        t = temp_val.get("value") if isinstance(temp_val, dict) else temp_val
        if t is not None:
            try:
                temps.append(float(t))
            except (ValueError, TypeError):
                pass

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
    print(f"  🏙️ NYSM via Synoptic: {len(temps)} boroughs — "
          f"min={nan_result['obs_nysm_min']:.1f}°F  "
          f"mean={mean_t:.1f}°F  "
          f"max={nan_result['obs_nysm_max']:.1f}°F{vs_str}")

    return nan_result


if __name__ == "__main__":
    # Test: list nearby stations and their current temps
    print("Fetching Synoptic stations near Central Park...")
    stns = fetch_nearby_obs(40.7834, -73.965, radius_miles=10, limit=25)
    if stns:
        print(f"\n{len(stns)} stations found:\n")
        for s in sorted(stns, key=lambda x: x.get("DISTANCE", 999)):
            obs = s.get("OBSERVATIONS", {})
            t = obs.get("air_temp_value_1", {})
            temp = t.get("value") if isinstance(t, dict) else t
            dist = s.get("DISTANCE", "?")
            print(f"  {s.get('STID','?'):10s}  {s.get('NAME','?'):35s}  "
                  f"temp={temp}°F  dist={dist}mi")
    else:
        print("No stations — check SYNOPTIC_TOKEN")
