# city_config.py — single source of truth for per-city constants
# Used by nws_auto_logger.py, accuweather_logger.py, prediction_writer.py, train_models.py

CITIES = {
    "nyc": {
        "label": "New York City",
        "short_label": "NYC",
        "timezone": "America/New_York",
        "nws_api_endpoint": "https://api.weather.gov/points/40.7834,-73.965",
        "obs_station": "KNYC",
        "cli_site": "NWS",
        "cli_issuedby": "NYC",
        "accu_location_key_env": "ACCU_LOCATION_KEY",
        "kalshi_series": "KXHIGHNY",
        "kalshi_url": "https://kalshi.com/markets/kxhighny/highest-temperature-in-nyc",
        "nws_csv": "nws_forecast_log.csv",
        "accu_csv": "accuweather_log.csv",
        "model_prefix": "",        # backward compat: temp_model.pkl
        "has_dsm": True,
        # Open-Meteo coordinates (same as NWS API endpoint)
        "open_meteo_lat": 40.7834,
        "open_meteo_lon": -73.965,
        # Regional supplemental NWS stations — fetched alongside primary KNYC
        # JFK often 3-5°F colder on sea-breeze days; LGA captures Queens/Bronx air mass
        "regional_obs_stations": ["KJFK", "KLGA"],
        # Synoptic radius for NYC — expanded from default 10mi to 15mi so the
        # sweep reliably captures MANH (Manhattan Mesonet at Columbia/125th,
        # ~2.5mi from Central Park) even when it drops off the radius list due
        # to Synoptic API quirks.  Direct STID fallback also exists as backup.
        "synoptic_radius_miles": 15.0,
    },
    "lax": {
        "label": "Los Angeles",
        "short_label": "LAX",
        "timezone": "America/Los_Angeles",
        "nws_api_endpoint": "https://api.weather.gov/points/33.94,-118.39",
        "obs_station": "KLAX",
        "cli_site": "LOX",
        "cli_issuedby": "LAX",
        "accu_location_key_env": "ACCU_LOCATION_KEY_LAX",
        "kalshi_series": "KXHIGHLAX",
        "kalshi_url": "https://kalshi.com/markets/kxhighlax/highest-temperature-in-los-angeles",
        "nws_csv": "lax_nws_forecast_log.csv",
        "accu_csv": "lax_accuweather_log.csv",
        "model_prefix": "lax_",    # lax_temp_model.pkl
        "has_dsm": False,
        # Open-Meteo coordinates (same as NWS API endpoint — KLAX airport)
        "open_meteo_lat": 33.94,
        "open_meteo_lon": -118.39,
        # Synoptic query coordinates — use KCQT (USC Downtown, 34.02°N 118.29°W) rather
        # than KLAX airport. KCQT is the NWS/Kalshi reference point for the official LA
        # high, and the urban core around USC has far better Synoptic station density
        # than the airport perimeter. Radius expanded to 8mi for urban LA spread.
        "synoptic_lat": 34.022,
        "synoptic_lon": -118.291,
        "synoptic_radius_miles": 8.0,
        # Regional supplemental NWS stations — fetched alongside primary KLAX
        # KBUR (Burbank) is in the San Fernando Valley and runs 8-15°F hotter on sunny days.
        # A large KBUR-KLAX gap means marine layer is pinned to the coast (inland heating uncapped).
        # KCQT (USC Downtown) is near the NWS/Kalshi reference point for the official LA high.
        "regional_obs_stations": ["KBUR", "KCQT"],
    },
}

DEFAULT_CITY = "nyc"

# Dynamic agency cutoff times by season (hour in local timezone)
# Before this hour: NWS/AccuWeather/obs triggers active (can trigger ML recompute)
# After this hour: These triggers silenced, only atmospheric triggers live
# Rationale: solar heating peak shifts with season length
#   Spring (Mar-May):    3 PM — warming extends into afternoon
#   Summer (Jun-Aug):    4 PM — peak heating 3-4 PM, especially late spring/early summer
#   Fall (Sep-Oct):      3 PM — heating still extends into afternoon
#   Winter (Nov-Feb):    2 PM — heating stops early, evenings get cold
SEASONAL_AGENCY_CUTOFF = {
    1: 14,   # Jan - 2 PM
    2: 14,   # Feb - 2 PM
    3: 15,   # Mar - 3 PM (spring starts)
    4: 15,   # Apr - 3 PM
    5: 16,   # May - 4 PM (late spring peak warming)
    6: 16,   # Jun - 4 PM (summer peak warming)
    7: 16,   # Jul - 4 PM (summer peak warming)
    8: 16,   # Aug - 4 PM (summer peak warming)
    9: 15,   # Sep - 3 PM (fall starts)
    10: 15,  # Oct - 3 PM
    11: 14,  # Nov - 2 PM (fall ends, winter starts)
    12: 14,  # Dec - 2 PM
}


def get_city_config(city_key: str) -> dict:
    """Return config dict for the given city key. Raises KeyError if not found."""
    key = city_key.strip().lower()
    if key not in CITIES:
        raise KeyError(f"Unknown city '{key}'. Available: {', '.join(CITIES.keys())}")
    return CITIES[key]


def get_seasonal_agency_cutoff(month: int) -> int:
    """Return the agency cutoff hour for a given month (1-12)."""
    return SEASONAL_AGENCY_CUTOFF.get(month, 14)  # Default to 2 PM if undefined
