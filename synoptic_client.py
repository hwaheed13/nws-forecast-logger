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

# Named stations we track individually as ML features — city-specific
#
# NYC stations:
#   KJFK: coastal Queens/Jamaica Bay — first to feel sea breeze, coldest on cap days
#   KLGA: north Queens/East River — intermediate marine exposure
#   KEWR: Newark, NJ — slightly inland and west, warmer on marine cap days
#   KTEB: Teterboro, NJ — most inland, warmest on marine cap days
#   KNYC: Central Park ASOS — our target; anchors all cross-station diffs (hourly)
#   MANH: NY Mesonet Manhattan (~Columbia/125th St) — 5-min updates, fills KNYC gap
# On a marine cap day: KJFK < KLGA < KNYC ≈ MANH < KEWR < KTEB
#
# LAX stations:
#   KLAX: coastal airport — first to feel marine layer, coldest
#   KSMO: Santa Monica airport — coastal, intermediate
#   KBUR: Burbank airport — inland San Fernando Valley, WARMEST on marine layer days
#   KVNY: Van Nuys airport — inland
#   KCQT: USC Campus Downtown LA — official NWS/Kalshi reference point
# On a marine layer day: KLAX < KSMO < (offshore/cap) < KBUR ≈ KVNY
# (opposite geometry from NYC: inland KBUR is WARMER when marine layer pins coast)
#
_NAMED_ASOS_STIDS_NYC = {"KNYC", "KJFK", "KLGA", "KEWR", "KTEB", "MANH"}
_NAMED_ASOS_STIDS_LAX = {"KLAX", "KSMO", "KBUR", "KVNY", "KCQT"}
# Default for backward compat (legacy calls without city param)
_NAMED_ASOS_STIDS = _NAMED_ASOS_STIDS_NYC

# COOPNYC is the cooperative observer at Central Park (human-read max/min thermometer).
# At ~8 AM each day it reports air_temp_high_24_hour = yesterday's official high.
# This is what NWS uses for official records and is independent of the ASOS automation.
_COOPNYC_STID = "COOPNYC"


def _fetch_stids_direct(stids: list[str]) -> list[dict]:
    """
    Fetch current observations for specific station IDs via Synoptic nearesttime API.
    Used as a supplement to radius-based fetch to capture named stations (KJFK, KEWR)
    that fall just outside the default 10-mile radius but are critical ML features.

    KJFK is 13.9 miles from Central Park; KEWR is 12.7 miles — both outside radius=10
    but fetched by the historical backfill via STID. This direct fetch closes that
    training-inference mismatch so the live system always sees the same features as training.
    """
    token = _token()
    if not token or not stids:
        return []

    params = {
        "token": token,
        "stid": ",".join(stids),
        "vars": "air_temp,wind_speed,wind_gust,wind_direction,relative_humidity,"
                "dew_point_temperature",
        "units": "english",
        "within": "90",
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
            print(f"  ⚠️ Synoptic STID fetch: {summary.get('RESPONSE_MESSAGE', 'unknown error')}")
            return []
        return data.get("STATION", [])
    except Exception as e:
        print(f"  ⚠️ Synoptic STID fetch failed: {e}")
        return []


def get_synoptic_obs_features(
    lat: float = 40.7834,
    lon: float = -73.965,
    nws_last: Optional[float] = None,
    radius_miles: float = 10.0,
    city: str = "nyc",
) -> dict:
    """
    Fetch nearby station obs via Synoptic and return aggregated + per-station features.

    city: "nyc" (default) or "lax" — determines which named stations to track

    Aggregate features (both cities):
        obs_synoptic_mean        — mean temp across all nearby stations (°F)
        obs_synoptic_min         — coldest station reading (°F)
        obs_synoptic_max         — warmest station reading (°F)
        obs_synoptic_spread      — max - min across all stations
        obs_synoptic_vs_nws      — obs_synoptic_mean - nws_last
        obs_synoptic_count       — number of valid stations

    Borough subset (NYC only, NYSM stations):
        obs_nysm_mean/min/max/spread/vs_nws/count

    NYC named ASOS stations (city="nyc"):
        obs_kjfk_temp            — JFK Airport temp (°F); coastal, coldest on cap days
        obs_klga_temp            — LaGuardia temp (°F)
        obs_kewr_temp            — Newark temp (°F); inland NJ, warmer on cap days
        obs_kteb_temp            — Teterboro temp (°F); most inland
        obs_knyc_temp            — Central Park via Synoptic (°F); cross-check
        obs_manh_temp            — Manhattan Mesonet (5-min updates)
        obs_kjfk_vs_knyc, obs_klga_vs_knyc, obs_kewr_vs_knyc — cross-station diffs
        obs_airport_spread       — max-min across 4 airports
        obs_coastal_vs_inland    — mean(KJFK,KLGA) - mean(KEWR,KTEB): marine signal

    LAX named ASOS stations (city="lax"):
        obs_lax_temp             — LAX Airport temp (coastal reference)
        obs_smo_temp             — Santa Monica temp (coastal)
        obs_bur_temp             — Burbank temp (inland, warmest on marine layer)
        obs_vny_temp             — Van Nuys temp (inland)
        obs_cqt_temp             — USC Campus (official NWS/Kalshi reference)
        obs_bur_vs_lax           — KBUR - KLAX: positive = marine layer (opposite from NYC)
        obs_coastal_vs_inland_lax — mean(KLAX,KSMO) - mean(KBUR,KVNY): negative = marine layer
        obs_airport_spread_lax   — max - min across all 4 airports

    April 12 NYC example: KJFK=50°F, KNYC=52°F → obs_kjfk_vs_knyc=-2°F (cap confirmed)
    LAX marine layer: KLAX=65°F, KBUR=75°F → obs_bur_vs_lax=+10°F (layer active)

    All NaN if SYNOPTIC_TOKEN is not set.
    """
    import numpy as np

    nan_result = {
        # Network aggregate (all cities)
        "obs_synoptic_mean": np.nan,
        "obs_synoptic_min": np.nan,
        "obs_synoptic_max": np.nan,
        "obs_synoptic_spread": np.nan,
        "obs_synoptic_vs_nws": np.nan,
        "obs_synoptic_count": np.nan,
        # Borough subset (NYC only, NYSM)
        "obs_nysm_mean": np.nan,
        "obs_nysm_min": np.nan,
        "obs_nysm_max": np.nan,
        "obs_nysm_spread": np.nan,
        "obs_nysm_vs_nws": np.nan,
        "obs_nysm_count": np.nan,
        # NYC named ASOS stations (individual readings)
        "obs_kjfk_temp": np.nan,
        "obs_klga_temp": np.nan,
        "obs_kewr_temp": np.nan,
        "obs_kteb_temp": np.nan,
        "obs_knyc_temp": np.nan,
        # NYC Manhattan Mesonet — 5-min updates, near-Central Park fill-in
        "obs_manh_temp": np.nan,
        "obs_manh_vs_knyc": np.nan,   # MANH - KNYC: shows intraday cap signal at 5-min res
        # NYC cross-station diffs (marine cap signals)
        "obs_kjfk_vs_knyc": np.nan,
        "obs_klga_vs_knyc": np.nan,
        "obs_kewr_vs_knyc": np.nan,
        "obs_airport_spread": np.nan,
        "obs_coastal_vs_inland": np.nan,
        # LAX named ASOS stations (individual readings)
        "obs_lax_temp": np.nan,         # KLAX — coastal airport, reference like KJFK for NYC
        "obs_smo_temp": np.nan,         # KSMO — Santa Monica, coastal
        "obs_bur_temp": np.nan,         # KBUR — Burbank, inland, warmest on marine layer days
        "obs_vny_temp": np.nan,         # KVNY — Van Nuys, inland
        "obs_cqt_temp": np.nan,         # KCQT — USC Downtown, official NWS/Kalshi reference
        # LAX cross-station diffs (marine layer signals)
        "obs_bur_vs_lax": np.nan,       # KBUR - KLAX: positive = marine layer active
        "obs_coastal_vs_inland_lax": np.nan,  # mean(KLAX,KSMO) - mean(KBUR,KVNY): negative = layer
        "obs_airport_spread_lax": np.nan,     # max - min across all 4 airports
        # COOPNYC cooperative observer: official prior-day high/low (at ~8am each day)
        "obs_coopnyc_24h_high": np.nan,
        "obs_coopnyc_24h_low":  np.nan,
    }

    if not _token():
        return nan_result

    stations = fetch_nearby_obs(lat, lon, radius_miles=radius_miles)
    if not stations:
        return nan_result

    # Determine which named stations to track based on city
    named_stids = _NAMED_ASOS_STIDS_LAX if city.lower() == "lax" else _NAMED_ASOS_STIDS_NYC

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
                # Only accumulate borough temps for NYC
                if city.lower() != "lax" and stid in _NYSM_BOROUGH_STIDS:
                    borough_temps.append(tf)
                    print(f"  🏙️ NYSM borough via Synoptic — {stid}: {tf:.1f}°F")
                if stid in named_stids:
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

    # ── Supplement: direct STID fetch for named stations outside the radius ────
    # KJFK (13.9mi) and KEWR (12.7mi) are beyond the default 10-mile radius so they
    # never appear in the radius results. The nightly backfill fetches them by STID
    # directly, so historical training rows have their values — but without this
    # secondary fetch, the live inference would always see KJFK=NaN (training-inference
    # mismatch). We fix it by fetching any missing named stations by STID explicitly.
    missing_named = [s for s in named_stids if s not in named_temps]
    if missing_named:
        print(f"  🔍 Fetching {len(missing_named)} named station(s) by STID "
              f"(outside radius): {', '.join(sorted(missing_named))}")
        direct_stns = _fetch_stids_direct(list(missing_named))
        for stn in direct_stns:
            stid = stn.get("STID", "?").upper()
            obs = stn.get("OBSERVATIONS", {})
            temp_val = obs.get("air_temp_value_1", {})
            if isinstance(temp_val, dict):
                t = temp_val.get("value")
                obs_dt = temp_val.get("date_time")
            else:
                t = temp_val
                obs_dt = None
            if t is not None and stid in named_stids and stid not in named_temps:
                try:
                    tf = float(t)
                    named_temps[stid] = tf
                    if obs_dt:
                        named_obs_at[stid] = obs_dt
                    print(f"  ✈️  {stid}: {tf:.1f}°F  (obs {obs_dt})  [direct STID fetch]")
                except (ValueError, TypeError):
                    pass

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

    # ── Named station features — city-specific ─────────────────────────
    if city.lower() == "lax":
        # LAX stations
        lax = named_temps.get("KLAX")
        smo = named_temps.get("KSMO")
        bur = named_temps.get("KBUR")
        vny = named_temps.get("KVNY")
        cqt = named_temps.get("KCQT")

        if lax is not None: nan_result["obs_lax_temp"] = round(lax, 1)
        if smo is not None: nan_result["obs_smo_temp"] = round(smo, 1)
        if bur is not None: nan_result["obs_bur_temp"] = round(bur, 1)
        if vny is not None: nan_result["obs_vny_temp"] = round(vny, 1)
        if cqt is not None: nan_result["obs_cqt_temp"] = round(cqt, 1)

        # Observation timestamps
        for stid_key, feat_key in [
            ("KLAX", "obs_lax_obs_at"),
            ("KSMO", "obs_smo_obs_at"),
            ("KBUR", "obs_bur_obs_at"),
            ("KVNY", "obs_vny_obs_at"),
            ("KCQT", "obs_cqt_obs_at"),
        ]:
            if stid_key in named_obs_at:
                nan_result[feat_key] = named_obs_at[stid_key]  # ISO string, not float

        # Marine layer signal: KBUR (inland) > KLAX (coastal) when marine layer is active
        # obs_bur_vs_lax positive = marine layer pinning coast
        if lax is not None and bur is not None:
            nan_result["obs_bur_vs_lax"] = round(bur - lax, 1)

        # Coastal (KLAX, KSMO) vs inland (KBUR, KVNY) mean diff
        # Negative = coastal colder = marine layer present
        coastal = [t for t in [lax, smo] if t is not None]
        inland  = [t for t in [bur, vny] if t is not None]
        if coastal and inland:
            coastal_mean = sum(coastal) / len(coastal)
            inland_mean  = sum(inland)  / len(inland)
            nan_result["obs_coastal_vs_inland_lax"] = round(coastal_mean - inland_mean, 1)
            sign = "🌊 marine layer" if coastal_mean < inland_mean else "clear offshore"
            print(f"  🌡️  LAX coastal({','.join(k for k,v in named_temps.items() if k in ('KLAX','KSMO'))})="
                  f"{coastal_mean:.1f}°F  "
                  f"inland({','.join(k for k,v in named_temps.items() if k in ('KBUR','KVNY'))})="
                  f"{inland_mean:.1f}°F  "
                  f"diff={nan_result['obs_coastal_vs_inland_lax']:+.1f}°F  {sign}")

        # Airport spread across all 4 LAX regional stations
        airport_readings = [t for t in [lax, smo, bur, vny] if t is not None]
        if len(airport_readings) >= 2:
            nan_result["obs_airport_spread_lax"] = round(max(airport_readings) - min(airport_readings), 1)
    else:
        # NYC stations (default)
        knyc = named_temps.get("KNYC")
        kjfk = named_temps.get("KJFK")
        klga = named_temps.get("KLGA")
        kewr = named_temps.get("KEWR")
        kteb = named_temps.get("KTEB")
        manh = named_temps.get("MANH")   # NY Mesonet Manhattan — 5-min, near Central Park

        if knyc is not None: nan_result["obs_knyc_temp"] = round(knyc, 1)
        if kjfk is not None: nan_result["obs_kjfk_temp"] = round(kjfk, 1)
        if klga is not None: nan_result["obs_klga_temp"] = round(klga, 1)
        if kewr is not None: nan_result["obs_kewr_temp"] = round(kewr, 1)
        if kteb is not None: nan_result["obs_kteb_temp"] = round(kteb, 1)
        if manh is not None:
            nan_result["obs_manh_temp"] = round(manh, 1)
            print(f"  🏙️ MANH (Manhattan Mesonet, 5-min): {manh:.1f}°F")

        # Observation timestamps — lets dashboard show "KNYC: 52°F (47 min ago)"
        # MANH updates every 5 min; KNYC/airports are hourly at :51
        for stid_key, feat_key in [
            ("KNYC", "obs_knyc_obs_at"),
            ("KJFK", "obs_kjfk_obs_at"),
            ("KLGA", "obs_klga_obs_at"),
            ("KEWR", "obs_kewr_obs_at"),
            ("KTEB", "obs_kteb_obs_at"),
            ("MANH", "obs_manh_obs_at"),
        ]:
            if stid_key in named_obs_at:
                nan_result[feat_key] = named_obs_at[stid_key]  # ISO string, not float

        # Cross-station diffs anchored at KNYC (our prediction target)
        if knyc is not None:
            if kjfk is not None: nan_result["obs_kjfk_vs_knyc"] = round(kjfk - knyc, 1)
            if klga is not None: nan_result["obs_klga_vs_knyc"] = round(klga - knyc, 1)
            if kewr is not None: nan_result["obs_kewr_vs_knyc"] = round(kewr - knyc, 1)
            if manh is not None: nan_result["obs_manh_vs_knyc"] = round(manh - knyc, 1)

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
