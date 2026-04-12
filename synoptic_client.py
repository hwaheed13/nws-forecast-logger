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
        # Include air_temp_high_24_hour so COOPNYC's 8am observer-verified daily
        # high/low is returned — useful for cross-checking training labels.
        "vars": "air_temp,wind_speed,wind_gust,wind_direction,relative_humidity,"
                "dew_point_temperature,air_temp_high_24_hour,air_temp_low_24_hour",
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

# Named ASOS stations we track individually as ML features.
# These give far more signal than the aggregate alone:
#   KJFK: coastal Queens/Jamaica Bay — first to feel sea breeze, coldest on cap days
#   KLGA: north Queens/East River — intermediate marine exposure
#   KEWR: Newark, NJ — slightly inland and west, warmer on marine cap days
#   KTEB: Teterboro, NJ — most inland, warmest on marine cap days
#   KNYC: Central Park — our target; stored to anchor all the cross-station diffs
#
# On a marine cap day: KJFK < KLGA < KNYC < KEWR < KTEB
# On a normal warm day: spread is small, all tracking similarly
_NAMED_ASOS_STIDS = {"KNYC", "KJFK", "KLGA", "KEWR", "KTEB"}

# COOPNYC is the cooperative observer at Central Park (human-read max/min thermometer).
# At ~8 AM each day it reports air_temp_high_24_hour = yesterday's official high.
# This is what NWS uses for official records and is independent of the ASOS automation.
_COOPNYC_STID = "COOPNYC"


def get_synoptic_obs_features(
    lat: float = 40.7834,
    lon: float = -73.965,
    nws_last: Optional[float] = None,
    radius_miles: float = 10.0,
) -> dict:
    """
    Fetch nearby station obs via Synoptic and return aggregated + per-station features.

    Aggregate features:
        obs_synoptic_mean        — mean temp across all nearby stations (°F)
        obs_synoptic_min         — coldest station reading (°F)
        obs_synoptic_max         — warmest station reading (°F)
        obs_synoptic_spread      — max - min across all stations
        obs_synoptic_vs_nws      — obs_synoptic_mean - nws_last
        obs_synoptic_count       — number of valid stations

    Borough subset (NYSM stations):
        obs_nysm_mean/min/max/spread/vs_nws/count

    Named ASOS stations (individually — key for marine cap detection):
        obs_kjfk_temp            — JFK Airport temp (°F); coastal, coldest on cap days
        obs_klga_temp            — LaGuardia temp (°F)
        obs_kewr_temp            — Newark temp (°F); inland NJ, warmer on cap days
        obs_kteb_temp            — Teterboro temp (°F); most inland
        obs_knyc_temp            — Central Park via Synoptic (°F); cross-check
        obs_kjfk_vs_knyc         — KJFK - KNYC: negative = sea breeze/marine cap signal
        obs_klga_vs_knyc         — KLGA - KNYC
        obs_kewr_vs_knyc         — KEWR - KNYC: positive = NJ warmer = no marine cap
        obs_airport_spread       — max(airports) - min(airports): high = localized cap
        obs_coastal_vs_inland    — mean(KJFK,KLGA) - mean(KEWR,KTEB): strong marine signal

    April 12 example: KJFK=50°F, KNYC=52°F → obs_kjfk_vs_knyc=-2°F (cap confirmed)
    All NaN if SYNOPTIC_TOKEN is not set.
    """
    import numpy as np

    nan_result = {
        # Network aggregate
        "obs_synoptic_mean": np.nan,
        "obs_synoptic_min": np.nan,
        "obs_synoptic_max": np.nan,
        "obs_synoptic_spread": np.nan,
        "obs_synoptic_vs_nws": np.nan,
        "obs_synoptic_count": np.nan,
        # Borough subset (NYSM)
        "obs_nysm_mean": np.nan,
        "obs_nysm_min": np.nan,
        "obs_nysm_max": np.nan,
        "obs_nysm_spread": np.nan,
        "obs_nysm_vs_nws": np.nan,
        "obs_nysm_count": np.nan,
        # Named ASOS stations (individual readings)
        "obs_kjfk_temp": np.nan,
        "obs_klga_temp": np.nan,
        "obs_kewr_temp": np.nan,
        "obs_kteb_temp": np.nan,
        "obs_knyc_temp": np.nan,
        # Cross-station diffs (marine cap signals)
        "obs_kjfk_vs_knyc": np.nan,
        "obs_klga_vs_knyc": np.nan,
        "obs_kewr_vs_knyc": np.nan,
        "obs_airport_spread": np.nan,
        "obs_coastal_vs_inland": np.nan,
        # COOPNYC cooperative observer: official prior-day high/low (at ~8am each day)
        "obs_coopnyc_24h_high": np.nan,
        "obs_coopnyc_24h_low":  np.nan,
    }

    if not _token():
        return nan_result

    stations = fetch_nearby_obs(lat, lon, radius_miles=radius_miles)
    if not stations:
        return nan_result

    temps = []
    borough_temps = []
    named_temps: dict = {}      # stid.upper() → temp_f
    named_obs_at: dict = {}     # stid.upper() → ISO datetime string of observation

    for stn in stations:
        stid = stn.get("STID", "?").upper()
        obs = stn.get("OBSERVATIONS", {})
        temp_val = obs.get("air_temp_value_1", {})
        # Synoptic returns {"value": 62.1, "date_time": "2026-04-12T14:54:00-0400"} per variable
        if isinstance(temp_val, dict):
            t = temp_val.get("value")
            obs_dt = temp_val.get("date_time")  # ISO timestamp of this reading
        else:
            t = temp_val
            obs_dt = None
        if t is not None:
            try:
                tf = float(t)
                temps.append(tf)
                if stid in _NYSM_BOROUGH_STIDS:
                    borough_temps.append(tf)
                    print(f"  🏙️ NYSM borough via Synoptic — {stid}: {tf:.1f}°F")
                if stid in _NAMED_ASOS_STIDS:
                    named_temps[stid] = tf
                    if obs_dt:
                        named_obs_at[stid] = obs_dt
                    print(f"  ✈️  {stid}: {tf:.1f}°F"
                          + (f"  (obs {obs_dt})" if obs_dt else ""))
                # COOPNYC: extract the 24h high/low (observer-verified, reported at 8am)
                if stid == _COOPNYC_STID:
                    hi_val = obs.get("air_temp_high_24_hour_value_1", {})
                    lo_val = obs.get("air_temp_low_24_hour_value_1", {})
                    hi = hi_val.get("value") if isinstance(hi_val, dict) else hi_val
                    lo = lo_val.get("value") if isinstance(lo_val, dict) else lo_val
                    if hi is not None:
                        try:
                            nan_result["obs_coopnyc_24h_high"] = round(float(hi), 1)
                            print(f"  📋 COOPNYC observer: 24h_high={float(hi):.1f}°F  "
                                  f"24h_low={float(lo):.1f}°F" if lo else
                                  f"  📋 COOPNYC observer: 24h_high={float(hi):.1f}°F")
                        except (ValueError, TypeError):
                            pass
                    if lo is not None:
                        try:
                            nan_result["obs_coopnyc_24h_low"] = round(float(lo), 1)
                        except (ValueError, TypeError):
                            pass
            except (ValueError, TypeError):
                pass

    if not temps:
        return nan_result

    # ── Network aggregate ─────────────────────────────────────────────
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

    # ── Borough subset (NYSM) ─────────────────────────────────────────
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

    # ── Named ASOS station features ───────────────────────────────────
    knyc = named_temps.get("KNYC")
    kjfk = named_temps.get("KJFK")
    klga = named_temps.get("KLGA")
    kewr = named_temps.get("KEWR")
    kteb = named_temps.get("KTEB")

    if knyc is not None: nan_result["obs_knyc_temp"] = round(knyc, 1)
    if kjfk is not None: nan_result["obs_kjfk_temp"] = round(kjfk, 1)
    if klga is not None: nan_result["obs_klga_temp"] = round(klga, 1)
    if kewr is not None: nan_result["obs_kewr_temp"] = round(kewr, 1)
    if kteb is not None: nan_result["obs_kteb_temp"] = round(kteb, 1)

    # Observation timestamps — lets dashboard show "KNYC: 52°F (47 min ago)"
    # KNYC is hourly ASOS; NYSM updates every 5 min; these tell you which you can trust
    for stid_key, feat_key in [
        ("KNYC", "obs_knyc_obs_at"),
        ("KJFK", "obs_kjfk_obs_at"),
        ("KLGA", "obs_klga_obs_at"),
        ("KEWR", "obs_kewr_obs_at"),
        ("KTEB", "obs_kteb_obs_at"),
    ]:
        if stid_key in named_obs_at:
            nan_result[feat_key] = named_obs_at[stid_key]  # ISO string, not float

    # Cross-station diffs anchored at KNYC (our prediction target)
    if knyc is not None:
        if kjfk is not None: nan_result["obs_kjfk_vs_knyc"] = round(kjfk - knyc, 1)
        if klga is not None: nan_result["obs_klga_vs_knyc"] = round(klga - knyc, 1)
        if kewr is not None: nan_result["obs_kewr_vs_knyc"] = round(kewr - knyc, 1)

    # Airport spread: how uniform is the cap across all airports?
    airport_readings = [t for t in [kjfk, klga, kewr, kteb] if t is not None]
    if len(airport_readings) >= 2:
        nan_result["obs_airport_spread"] = round(max(airport_readings) - min(airport_readings), 1)

    # Coastal (KJFK, KLGA) vs inland (KEWR, KTEB) mean diff
    coastal = [t for t in [kjfk, klga] if t is not None]
    inland  = [t for t in [kewr, kteb] if t is not None]
    if coastal and inland:
        coastal_mean = sum(coastal) / len(coastal)
        inland_mean  = sum(inland)  / len(inland)
        nan_result["obs_coastal_vs_inland"] = round(coastal_mean - inland_mean, 1)
        # Negative = coastal colder = marine influence present
        sign = "🌊 coastal colder" if coastal_mean < inland_mean else "inland colder"
        print(f"  🌡️  Coastal({','.join(k for k,v in named_temps.items() if k in ('KJFK','KLGA'))})="
              f"{coastal_mean:.1f}°F  "
              f"Inland({','.join(k for k,v in named_temps.items() if k in ('KEWR','KTEB'))})="
              f"{inland_mean:.1f}°F  "
              f"diff={nan_result['obs_coastal_vs_inland']:+.1f}°F  {sign}")

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
