# open_meteo_client.py — Unified client for Open-Meteo APIs (free, no API key)
# Provides historical archive, ensemble forecasts, and multi-model comparisons
# for atmospheric feature extraction used by the ML bucket classifier.

from __future__ import annotations

import json
import math
import time
import urllib.request
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ── API base URLs ─────────────────────────────────────────────────────────
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# ── Hourly variables requested ────────────────────────────────────────────
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]


# ═════════════════════════════════════════════════════════════════════════
# HTTP helpers
# ═════════════════════════════════════════════════════════════════════════

def _get_json(url: str, retries: int = 3, delay: float = 1.0) -> dict:
    """Fetch JSON from a URL with retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise RuntimeError(f"Open-Meteo request failed after {retries} attempts: {e}") from e


# ═════════════════════════════════════════════════════════════════════════
# Data fetching
# ═════════════════════════════════════════════════════════════════════════

def fetch_historical_hourly(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """
    Fetch historical hourly weather data from the Open-Meteo archive API.

    Args:
        lat, lon: coordinates
        start_date, end_date: 'YYYY-MM-DD' strings
        timezone: IANA timezone for local time alignment

    Returns:
        DataFrame with 'time' column (datetime) and all hourly variables in °F.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "daily": ",".join(DAILY_VARS),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": timezone,
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{ARCHIVE_URL}?{qs}"
    data = _get_json(url)

    if "error" in data:
        raise RuntimeError(f"Open-Meteo archive error: {data.get('reason', data['error'])}")

    # Parse hourly data
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    df = pd.DataFrame({"time": pd.to_datetime(times)})
    for var in HOURLY_VARS:
        df[var] = hourly.get(var, [None] * len(times))

    # Parse daily data (separate DataFrame, merged by date)
    daily = data.get("daily", {})
    daily_times = daily.get("time", [])
    if daily_times:
        daily_df = pd.DataFrame({"date": pd.to_datetime(daily_times).date})
        for var in DAILY_VARS:
            daily_df[var] = daily.get(var, [None] * len(daily_times))
        # Attach daily data via date column
        df["_date"] = df["time"].dt.date
        df = df.merge(daily_df, left_on="_date", right_on="date", how="left")
        df.drop(columns=["_date", "date"], inplace=True, errors="ignore")

    return df


def fetch_ensemble_forecast(
    lat: float,
    lon: float,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """
    Fetch 51-member ECMWF ensemble forecast for temperature_2m.
    Returns a DataFrame with columns: time, member_0 ... member_50.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "models": "ecmwf_ifs025",
        "temperature_unit": "fahrenheit",
        "timezone": timezone,
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{ENSEMBLE_URL}?{qs}"
    data = _get_json(url)

    if "error" in data:
        raise RuntimeError(f"Open-Meteo ensemble error: {data.get('reason', data['error'])}")

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    df = pd.DataFrame({"time": pd.to_datetime(times)})

    # Ensemble members come as temperature_2m_member0, temperature_2m_member1, etc.
    member_cols = []
    for key, values in hourly.items():
        if key.startswith("temperature_2m_member"):
            member_num = key.replace("temperature_2m_member", "")
            col_name = f"member_{member_num}"
            df[col_name] = values
            member_cols.append(col_name)

    # If no member columns found, try temperature_2m directly
    if not member_cols and "temperature_2m" in hourly:
        df["member_0"] = hourly["temperature_2m"]
        member_cols = ["member_0"]

    return df


def fetch_multimodel_forecast(
    lat: float,
    lon: float,
    timezone: str = "America/New_York",
) -> dict:
    """
    Fetch daily max temperature from multiple models (ECMWF, GFS, ICON, GEM).
    Returns dict like:
        {"2026-03-07": {"ecmwf": 52.1, "gfs": 51.3, "icon": 50.8, "gem": 51.0}, ...}
    """
    models = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"]
    model_short = {"ecmwf_ifs025": "ecmwf", "gfs_seamless": "gfs",
                   "icon_seamless": "icon", "gem_seamless": "gem"}

    result: dict[str, dict[str, float]] = {}

    for model in models:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "models": model,
            "temperature_unit": "fahrenheit",
            "timezone": timezone,
        }
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{FORECAST_URL}?{qs}"

        try:
            data = _get_json(url)
            if "error" in data:
                print(f"  ⚠️ {model}: {data.get('reason', 'error')}")
                continue

            daily = data.get("daily", {})
            dates = daily.get("time", [])
            maxes = daily.get("temperature_2m_max", [])

            for d, t in zip(dates, maxes):
                if t is not None:
                    result.setdefault(d, {})[model_short[model]] = float(t)
        except Exception as e:
            print(f"  ⚠️ {model} fetch failed: {e}")
            continue

        # Brief pause between model requests
        time.sleep(0.3)

    return result


# ═════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═════════════════════════════════════════════════════════════════════════

def extract_daily_atmospheric(hourly_df: pd.DataFrame, target_date: str) -> dict:
    """
    Extract atmospheric features for a single date from hourly data.

    Args:
        hourly_df: DataFrame from fetch_historical_hourly() with 'time' column
        target_date: 'YYYY-MM-DD' string

    Returns:
        dict of 15 atmospheric features, or empty dict if no data for date.
    """
    target = pd.Timestamp(target_date)
    day_mask = hourly_df["time"].dt.date == target.date()
    day = hourly_df[day_mask]

    if day.empty:
        return {}

    features = {}

    # Wind
    wind = day["wind_speed_10m"].dropna()
    features["atm_wind_max"] = float(wind.max()) if len(wind) > 0 else np.nan
    features["atm_wind_mean"] = float(wind.mean()) if len(wind) > 0 else np.nan

    # Wind direction — cyclical encoding (mean direction via sin/cos)
    wdir = day["wind_direction_10m"].dropna()
    if len(wdir) > 0:
        rad = np.deg2rad(wdir.values)
        features["atm_wind_dir_sin"] = float(np.mean(np.sin(rad)))
        features["atm_wind_dir_cos"] = float(np.mean(np.cos(rad)))
    else:
        features["atm_wind_dir_sin"] = 0.0
        features["atm_wind_dir_cos"] = 0.0

    # Humidity
    hum = day["relative_humidity_2m"].dropna()
    features["atm_humidity_mean"] = float(hum.mean()) if len(hum) > 0 else np.nan
    features["atm_humidity_min"] = float(hum.min()) if len(hum) > 0 else np.nan

    # Dewpoint
    dew = day["dewpoint_2m"].dropna()
    features["atm_dewpoint_mean"] = float(dew.mean()) if len(dew) > 0 else np.nan

    # Pressure
    pres = day["surface_pressure"].dropna()
    features["atm_pressure_mean"] = float(pres.mean()) if len(pres) > 0 else np.nan
    features["atm_pressure_change"] = float(pres.iloc[-1] - pres.iloc[0]) if len(pres) > 1 else 0.0

    # Cloud cover
    cloud = day["cloud_cover"].dropna()
    features["atm_cloud_cover_mean"] = float(cloud.mean()) if len(cloud) > 0 else np.nan
    features["atm_cloud_cover_max"] = float(cloud.max()) if len(cloud) > 0 else np.nan

    # Precipitation
    precip = day["precipitation"].dropna()
    features["atm_precip_total"] = float(precip.sum()) if len(precip) > 0 else 0.0

    # Temperature range (from hourly, more granular than daily min/max)
    temp = day["temperature_2m"].dropna()
    if len(temp) > 0:
        features["atm_temp_range"] = float(temp.max() - temp.min())
        # Overnight minimum (midnight to 8am)
        overnight = day[(day["time"].dt.hour >= 0) & (day["time"].dt.hour < 8)]["temperature_2m"].dropna()
        features["atm_overnight_min"] = float(overnight.min()) if len(overnight) > 0 else float(temp.min())
        # Morning temp at ~6am
        morning = day[day["time"].dt.hour == 6]["temperature_2m"].dropna()
        features["atm_morning_temp_6am"] = float(morning.iloc[0]) if len(morning) > 0 else np.nan
    else:
        features["atm_temp_range"] = np.nan
        features["atm_overnight_min"] = np.nan
        features["atm_morning_temp_6am"] = np.nan

    return features


def extract_ensemble_features(ensemble_df: pd.DataFrame, target_date: str) -> dict:
    """
    Extract ensemble uncertainty features for a single date.

    The ensemble gives us 51 model runs — the spread across members
    is a direct measure of forecast uncertainty for that day.

    Returns dict of 5 features, or empty dict if no data.
    """
    target = pd.Timestamp(target_date)
    day_mask = ensemble_df["time"].dt.date == target.date()
    day = ensemble_df[day_mask]

    if day.empty:
        return {}

    # Get all member columns
    member_cols = [c for c in day.columns if c.startswith("member_")]
    if not member_cols:
        return {}

    # For each member, compute the daily max temperature
    member_maxes = []
    for col in member_cols:
        vals = day[col].dropna()
        if len(vals) > 0:
            member_maxes.append(float(vals.max()))

    if len(member_maxes) < 2:
        return {}

    arr = np.array(member_maxes)
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))

    # Skewness
    std_val = float(arr.std())
    if std_val > 0:
        skew = float(((arr - arr.mean()) ** 3).mean() / (std_val ** 3))
    else:
        skew = 0.0

    return {
        "ens_spread": float(arr.max() - arr.min()),
        "ens_std": std_val,
        "ens_iqr": q75 - q25,
        "ens_mean": float(arr.mean()),
        "ens_skew": skew,
    }


def extract_multimodel_features(multimodel_data: dict, target_date: str) -> dict:
    """
    Extract cross-model spread features for a single date.

    Different weather models (ECMWF, GFS, ICON, GEM) disagree on the forecast.
    The spread is a signal for forecast uncertainty.

    Returns dict of 4 features, or empty dict if insufficient data.
    """
    day_data = multimodel_data.get(target_date, {})
    if len(day_data) < 2:
        return {}

    values = list(day_data.values())
    arr = np.array(values)

    features = {
        "mm_spread": float(arr.max() - arr.min()),
        "mm_std": float(arr.std()),
        "mm_mean": float(arr.mean()),
    }

    # ECMWF vs GFS difference (when both available)
    ecmwf = day_data.get("ecmwf")
    gfs = day_data.get("gfs")
    if ecmwf is not None and gfs is not None:
        features["mm_ecmwf_gfs_diff"] = ecmwf - gfs
    else:
        features["mm_ecmwf_gfs_diff"] = 0.0

    return features


# ═════════════════════════════════════════════════════════════════════════
# Convenience: get all features for a date
# ═════════════════════════════════════════════════════════════════════════

def get_atmospheric_features_historical(
    lat: float,
    lon: float,
    target_date: str,
    timezone: str = "America/New_York",
    hourly_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Get all atmospheric features for a historical date (from archive API).

    If hourly_df is provided, uses it instead of fetching (for batch backfill).
    Ensemble/multimodel features are NOT available for historical dates — returns NaN.
    """
    if hourly_df is None:
        hourly_df = fetch_historical_hourly(lat, lon, target_date, target_date, timezone)

    features = extract_daily_atmospheric(hourly_df, target_date)

    # Ensemble and multimodel not available in archive — fill with NaN
    # HistGradientBoosting handles NaN natively
    for col in ["ens_spread", "ens_std", "ens_iqr", "ens_mean", "ens_skew"]:
        features[col] = np.nan
    for col in ["mm_spread", "mm_std", "mm_mean", "mm_ecmwf_gfs_diff"]:
        features[col] = np.nan

    features["target_date"] = target_date
    return features


def get_atmospheric_features_live(
    lat: float,
    lon: float,
    target_date: str,
    timezone: str = "America/New_York",
) -> dict:
    """
    Get all atmospheric features for today/tomorrow (from forecast + ensemble APIs).
    Uses forecast data for atmospheric context and live ensemble for uncertainty.
    """
    # Atmospheric features from the standard forecast API
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": timezone,
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{FORECAST_URL}?{qs}"
    data = _get_json(url)

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if times:
        forecast_df = pd.DataFrame({"time": pd.to_datetime(times)})
        for var in HOURLY_VARS:
            forecast_df[var] = hourly.get(var, [None] * len(times))
        features = extract_daily_atmospheric(forecast_df, target_date)
    else:
        features = {}

    # Ensemble features
    try:
        ens_df = fetch_ensemble_forecast(lat, lon, timezone)
        ens_features = extract_ensemble_features(ens_df, target_date)
        features.update(ens_features)
    except Exception as e:
        print(f"  ⚠️ Ensemble fetch failed: {e}")
        for col in ["ens_spread", "ens_std", "ens_iqr", "ens_mean", "ens_skew"]:
            features[col] = np.nan

    # Multi-model features
    try:
        mm_data = fetch_multimodel_forecast(lat, lon, timezone)
        mm_features = extract_multimodel_features(mm_data, target_date)
        features.update(mm_features)
    except Exception as e:
        print(f"  ⚠️ Multi-model fetch failed: {e}")
        for col in ["mm_spread", "mm_std", "mm_mean", "mm_ecmwf_gfs_diff"]:
            features[col] = np.nan

    features["target_date"] = target_date
    return features


# ═════════════════════════════════════════════════════════════════════════
# CLI for testing
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Open-Meteo client test")
    parser.add_argument("--lat", type=float, default=40.7834, help="Latitude")
    parser.add_argument("--lon", type=float, default=-73.965, help="Longitude")
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--mode", choices=["historical", "live", "ensemble", "multimodel"],
                        default="live", help="Which API to test")
    args = parser.parse_args()

    from datetime import date as dt_date
    target = args.date or dt_date.today().isoformat()

    if args.mode == "historical":
        print(f"Fetching historical data for {target}...")
        features = get_atmospheric_features_historical(args.lat, args.lon, target)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v
                          for k, v in features.items()}, indent=2))

    elif args.mode == "live":
        print(f"Fetching live features for {target}...")
        features = get_atmospheric_features_live(args.lat, args.lon, target)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v
                          for k, v in features.items()}, indent=2))

    elif args.mode == "ensemble":
        print(f"Fetching ensemble forecast...")
        df = fetch_ensemble_forecast(args.lat, args.lon)
        print(f"Got {len(df)} rows, {len([c for c in df.columns if c.startswith('member_')])} members")
        features = extract_ensemble_features(df, target)
        print(json.dumps({k: round(v, 3) for k, v in features.items()}, indent=2))

    elif args.mode == "multimodel":
        print(f"Fetching multi-model forecasts...")
        mm = fetch_multimodel_forecast(args.lat, args.lon)
        for d, models in sorted(mm.items()):
            print(f"  {d}: {models}")
        features = extract_multimodel_features(mm, target)
        print(json.dumps({k: round(v, 3) for k, v in features.items()}, indent=2))
