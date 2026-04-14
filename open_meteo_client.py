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
    "boundary_layer_height",  # PBL height (m) — available in BOTH archive and forecast APIs
]

# Pressure-level and additional hourly variables for live forecast API
# NOTE: these are NOT available in the archive API (returns None) — only used
# for live inference. HistGradientBoosting handles NaN for historical rows.
HOURLY_PRESSURE_LEVEL_VARS = [
    "temperature_850hPa",   # Warm air advection aloft — key for temp overshoot detection
    "temperature_925hPa",   # Near-surface warm advection (925hPa more relevant than 850 for NYC)
    "shortwave_radiation",  # Solar irradiance (W/m²) — drives afternoon heating
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
    all_hourly_vars = HOURLY_VARS + HOURLY_PRESSURE_LEVEL_VARS
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(all_hourly_vars),
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
    for var in all_hourly_vars:
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
    Fetch daily max temperature from multiple models.
    Models ranked by 90-day accuracy (wethr.net):
      #1 HRRR, #2 NBM, #3 GEM HRDPS, then ECMWF, GFS, ICON, GEM global.
    Returns dict like:
        {"2026-03-07": {"ecmwf": 52.1, "gfs": 51.3, "hrrr": 54.0, "nbm": 53.1, ...}, ...}
    """
    # NBM (National Blend of Models) and GEM HRDPS added — top accuracy per wethr.net rankings.
    # NWS point forecast is a lagged post-processing of GFS; NBM blends 50+ models and updates
    # more frequently, making it far more useful for nowcasting.
    models = [
        "ecmwf_ifs025",         # ECMWF global deterministic
        "gfs_seamless",          # GFS — the model NWS point forecasts lag behind
        "icon_seamless",         # ICON (German DWD)
        "gem_seamless",          # GEM global (Canadian CMC)
        "ncep_hrrr_conus",       # HRRR — #1 accuracy, runs hourly, best boundary layer
        "nbm_conus",             # NBM — blends 50+ models, faster than NWS, top-3 accuracy
        "gem_hrdps_continental", # GEM HRDPS — Canadian high-res, top-5 accuracy
    ]
    model_short = {
        "ecmwf_ifs025": "ecmwf",
        "gfs_seamless": "gfs",
        "icon_seamless": "icon",
        "gem_seamless": "gem",
        "ncep_hrrr_conus": "hrrr",
        "nbm_conus": "nbm",
        "gem_hrdps_continental": "gem_hrdps",
    }

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
        dict of 35+ features (21 atmospheric incl 925mb+solar + 10 intraday curve + overnight),
        or empty dict if no data for date.
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

    # 850mb temperature — warm air advection aloft detection
    # Available from forecast API (live inference) but NOT from archive API.
    # Daytime hours (10am-6pm local) to capture synoptic-scale warm advection.
    if "temperature_850hPa" in day.columns:
        daytime_850 = day[
            (day["time"].dt.hour >= 10) & (day["time"].dt.hour <= 18)
        ]["temperature_850hPa"].dropna()
        features["atm_850mb_temp_max"] = float(daytime_850.max()) if len(daytime_850) > 0 else np.nan
        features["atm_850mb_temp_mean"] = float(daytime_850.mean()) if len(daytime_850) > 0 else np.nan
    else:
        features["atm_850mb_temp_max"] = np.nan
        features["atm_850mb_temp_mean"] = np.nan

    # 925mb temperature — more relevant than 850mb for near-surface NYC boundary layer
    # Also only available from forecast API (not archive).
    if "temperature_925hPa" in day.columns:
        daytime_925 = day[
            (day["time"].dt.hour >= 10) & (day["time"].dt.hour <= 18)
        ]["temperature_925hPa"].dropna()
        features["atm_925mb_temp_max"] = float(daytime_925.max()) if len(daytime_925) > 0 else np.nan
        features["atm_925mb_temp_mean"] = float(daytime_925.mean()) if len(daytime_925) > 0 else np.nan
    else:
        features["atm_925mb_temp_max"] = np.nan
        features["atm_925mb_temp_mean"] = np.nan

    # Solar irradiance — peak solar drives afternoon heating; low solar = temperature cap
    # Available from forecast API only. Archive API returns None for shortwave_radiation.
    if "shortwave_radiation" in day.columns:
        # Midday solar potential (10am-2pm) — peak heating window
        midday = day[
            (day["time"].dt.hour >= 10) & (day["time"].dt.hour <= 14)
        ]["shortwave_radiation"].dropna()
        features["atm_solar_radiation_peak"] = float(midday.max()) if len(midday) > 0 else np.nan
        features["atm_solar_radiation_mean"] = float(midday.mean()) if len(midday) > 0 else np.nan
    else:
        features["atm_solar_radiation_peak"] = np.nan
        features["atm_solar_radiation_mean"] = np.nan

    # Planetary boundary layer height — available in BOTH archive and forecast APIs.
    # Deep BL (>2000m) combined with strong solar radiation is the physical mechanism
    # behind radiation-driven temperature spikes that overshoot standard forecast models.
    # Peak heating window (10am-4pm) captures the BL as it reaches maximum daily depth.
    if "boundary_layer_height" in day.columns:
        bl = day[
            (day["time"].dt.hour >= 10) & (day["time"].dt.hour <= 16)
        ]["boundary_layer_height"].dropna()
        features["atm_bl_height_max"] = float(bl.max()) if len(bl) > 0 else np.nan
        features["atm_bl_height_mean"] = float(bl.mean()) if len(bl) > 0 else np.nan
    else:
        features["atm_bl_height_max"] = np.nan
        features["atm_bl_height_mean"] = np.nan

    # Temperature range (from hourly, more granular than daily min/max)
    temp = day["temperature_2m"].dropna()
    if len(temp) > 0:
        features["atm_temp_range"] = float(temp.max() - temp.min())
        # Overnight minimum (midnight to 8am)
        overnight = day[(day["time"].dt.hour >= 0) & (day["time"].dt.hour < 8)]["temperature_2m"].dropna()
        overnight_min = float(overnight.min()) if len(overnight) > 0 else float(temp.min())
        features["atm_overnight_min"] = overnight_min
        # Morning temp at ~6am
        morning = day[day["time"].dt.hour == 6]["temperature_2m"].dropna()
        features["atm_morning_temp_6am"] = float(morning.iloc[0]) if len(morning) > 0 else np.nan

        # Midnight temp (12am) — overnight carryover detection
        midnight = day[day["time"].dt.hour == 0]["temperature_2m"].dropna()
        features["midnight_temp"] = float(midnight.iloc[0]) if len(midnight) > 0 else np.nan

        # ── Intraday temperature curve features (10 features) ──────────
        # These capture the SHAPE of the daily heating curve — crucial for
        # predicting whether the actual high will overshoot the forecast.

        # Temperatures at key hours
        def _temp_at_hour(h: int) -> float:
            vals = day[day["time"].dt.hour == h]["temperature_2m"].dropna()
            return float(vals.iloc[0]) if len(vals) > 0 else np.nan

        temp_9am = _temp_at_hour(9)
        temp_noon = _temp_at_hour(12)
        temp_3pm = _temp_at_hour(15)
        temp_5pm = _temp_at_hour(17)

        features["intra_temp_9am"] = temp_9am
        features["intra_temp_noon"] = temp_noon
        features["intra_temp_3pm"] = temp_3pm
        features["intra_temp_5pm"] = temp_5pm

        # Heating rates (°F per hour)
        if not np.isnan(temp_9am) and not np.isnan(temp_noon):
            features["intra_heating_rate_am"] = (temp_noon - temp_9am) / 3.0
        else:
            features["intra_heating_rate_am"] = np.nan

        if not np.isnan(temp_noon) and not np.isnan(temp_3pm):
            features["intra_heating_rate_pm"] = (temp_3pm - temp_noon) / 3.0
        else:
            features["intra_heating_rate_pm"] = np.nan

        # Peak hour: hour when max temperature occurred
        temp_with_hours = day[["time", "temperature_2m"]].dropna(subset=["temperature_2m"])
        if len(temp_with_hours) > 0:
            peak_idx = temp_with_hours["temperature_2m"].idxmax()
            features["intra_peak_hour"] = float(temp_with_hours.loc[peak_idx, "time"].hour)
        else:
            features["intra_peak_hour"] = np.nan

        # Late heating: positive = still warming after 3pm (midnight push signal)
        if not np.isnan(temp_5pm) and not np.isnan(temp_3pm):
            features["intra_late_heating"] = temp_5pm - temp_3pm
        else:
            features["intra_late_heating"] = np.nan

        # Rise from overnight: how much warming by 9am
        if not np.isnan(temp_9am):
            features["intra_rise_from_overnight"] = temp_9am - overnight_min
        else:
            features["intra_rise_from_overnight"] = np.nan

        # High vs noon: how much additional heating after noon
        if not np.isnan(temp_noon):
            features["intra_high_vs_noon"] = float(temp.max()) - temp_noon
        else:
            features["intra_high_vs_noon"] = np.nan

    else:
        features["atm_temp_range"] = np.nan
        features["atm_overnight_min"] = np.nan
        features["atm_morning_temp_6am"] = np.nan
        # 850mb features also NaN when no data
        if "atm_850mb_temp_max" not in features:
            features["atm_850mb_temp_max"] = np.nan
            features["atm_850mb_temp_mean"] = np.nan
        features["midnight_temp"] = np.nan
        # Intraday curve features — all NaN when no temp data
        features["intra_temp_9am"] = np.nan
        features["intra_temp_noon"] = np.nan
        features["intra_temp_3pm"] = np.nan
        features["intra_temp_5pm"] = np.nan
        features["intra_heating_rate_am"] = np.nan
        features["intra_heating_rate_pm"] = np.nan
        features["intra_peak_hour"] = np.nan
        features["intra_late_heating"] = np.nan
        features["intra_rise_from_overnight"] = np.nan
        features["intra_high_vs_noon"] = np.nan

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
    hrrr = day_data.get("hrrr")
    if ecmwf is not None and gfs is not None:
        features["mm_ecmwf_gfs_diff"] = ecmwf - gfs
    else:
        features["mm_ecmwf_gfs_diff"] = np.nan

    # ECMWF (European HRES) — top-tier global model, #2 accuracy behind HRRR for regional.
    # Store explicit forecast value for dashboard display and ensemble analysis.
    features["mm_ecmwf_max"] = float(ecmwf) if ecmwf is not None else np.nan

    # HRRR features — HRRR has a known boundary-layer warm bias that sophisticated
    # traders exploit. When HRRR diverges from ECMWF, it's a strong uncertainty signal.
    features["mm_hrrr_max"] = float(hrrr) if hrrr is not None else np.nan
    if hrrr is not None and ecmwf is not None:
        features["mm_hrrr_ecmwf_diff"] = float(hrrr - ecmwf)
    else:
        features["mm_hrrr_ecmwf_diff"] = np.nan
    if hrrr is not None and gfs is not None:
        features["mm_hrrr_gfs_diff"] = float(hrrr - gfs)
    else:
        features["mm_hrrr_gfs_diff"] = np.nan

    # ICON (German DWD) and GEM (Canadian CMC) — individual model predictions.
    # Previously fetched but silently dropped. Now exposed as independent signals.
    icon = day_data.get("icon")
    gem = day_data.get("gem")
    features["mm_icon_max"] = float(icon) if icon is not None else np.nan
    features["mm_gem_max"] = float(gem) if gem is not None else np.nan
    features["mm_icon_gfs_diff"] = float(icon - gfs) if (icon is not None and gfs is not None) else np.nan
    features["mm_gem_ecmwf_diff"] = float(gem - ecmwf) if (gem is not None and ecmwf is not None) else np.nan

    # NBM (National Blend of Models) — top-3 accuracy per wethr.net 90-day rankings.
    # Blends 50+ models and updates more frequently than NWS point forecasts.
    # When NBM diverges from HRRR, it's a strong signal of forecast instability.
    nbm = day_data.get("nbm")
    features["mm_nbm_max"] = float(nbm) if nbm is not None else np.nan
    features["mm_nbm_hrrr_diff"] = float(nbm - hrrr) if (nbm is not None and hrrr is not None) else np.nan
    features["mm_nbm_gfs_diff"] = float(nbm - gfs) if (nbm is not None and gfs is not None) else np.nan
    features["mm_nbm_ecmwf_diff"] = float(nbm - ecmwf) if (nbm is not None and ecmwf is not None) else np.nan

    # GEM HRDPS (Canadian High-Resolution Deterministic Prediction System) — top-5 accuracy.
    # ~2.5km grid resolution, superior boundary layer physics for mesoscale events.
    gem_hrdps = day_data.get("gem_hrdps")
    features["mm_gem_hrdps_max"] = float(gem_hrdps) if gem_hrdps is not None else np.nan
    features["mm_gem_hrdps_hrrr_diff"] = float(gem_hrdps - hrrr) if (gem_hrdps is not None and hrrr is not None) else np.nan

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
    for col in ["mm_spread", "mm_std", "mm_mean", "mm_ecmwf_gfs_diff",
                "mm_hrrr_max", "mm_hrrr_ecmwf_diff", "mm_hrrr_gfs_diff",
                "mm_icon_max", "mm_gem_max", "mm_icon_gfs_diff", "mm_gem_ecmwf_diff",
                "mm_nbm_max", "mm_nbm_hrrr_diff", "mm_nbm_gfs_diff", "mm_nbm_ecmwf_diff",
                "mm_gem_hrdps_max", "mm_gem_hrdps_hrrr_diff"]:
        features[col] = np.nan
    # HRRR-specific 925mb and radiosonde not available in archive
    for col in ["atm_925mb_hrrr_max", "atm_925mb_hrrr_mean",
                "atm_850mb_hrrr_max", "atm_850mb_hrrr_mean",
                "atm_925mb_gfs_hrrr_diff",
                "raob_925mb_temp", "raob_850mb_temp",
                "raob_925mb_gfs_diff", "raob_925mb_hrrr_diff"]:
        features[col] = np.nan

    features["target_date"] = target_date
    return features


def extract_observation_proxy_features(
    hourly_df: pd.DataFrame,
    target_date: str,
    as_of_hour: int = 12,
    nws_last: float = None,
    intra_features: dict = None,
) -> dict:
    """
    Compute observation proxy features from Open-Meteo archive hourly data.

    For training: the archive IS real observations, so we simulate what the ML
    model would see at a specific hour during live inference. This teaches the
    model to use partial-day observations to predict the final daily high.

    Args:
        hourly_df: DataFrame with hourly archive data (temperature_2m, wind, etc.)
        target_date: 'YYYY-MM-DD' string
        as_of_hour: Simulate observations up to this local hour (default: noon)
        nws_last: Latest NWS forecast value (for obs_temp_vs_forecast_max). NaN for multi-year.
        intra_features: Dict of intraday forecast features (for obs_vs_intra_forecast).
                       For training on archive data, this is the same source → delta ≈ 0.

    Returns dict with all 12 OBSERVATION_COLS features.
    """
    from model_config import OBSERVATION_COLS

    nan_result = {col: np.nan for col in OBSERVATION_COLS}

    target = pd.Timestamp(target_date)
    day_mask = hourly_df["time"].dt.date == target.date()
    day = hourly_df[day_mask].copy()

    if day.empty or "temperature_2m" not in day.columns:
        return nan_result

    # Filter to hours up to as_of_hour (simulating partial-day observations)
    day_partial = day[day["time"].dt.hour <= as_of_hour]
    temp = day_partial["temperature_2m"].dropna()

    if len(temp) == 0:
        return nan_result

    features = {}

    # Latest observation (the last reading up to as_of_hour)
    latest_idx = temp.index[-1]
    features["obs_latest_temp"] = float(temp.iloc[-1])
    features["obs_latest_hour"] = float(day_partial.loc[latest_idx, "time"].hour)

    # Running daily max up to as_of_hour
    features["obs_max_so_far"] = float(temp.max())

    # 6-hour max: max of last 6 hours of observations
    recent_6h = day_partial[day_partial["time"].dt.hour > (as_of_hour - 6)]
    recent_6h_temp = recent_6h["temperature_2m"].dropna()
    features["obs_6hr_max"] = float(recent_6h_temp.max()) if len(recent_6h_temp) > 0 else features["obs_max_so_far"]

    # Delta vs intraday forecast at same hour
    # For training on archive data, both come from the same source → delta ≈ 0
    # The model learns that non-zero deltas during live inference signal forecast error
    if intra_features:
        intra_map = {
            9: "intra_temp_9am", 10: "intra_temp_9am",
            11: "intra_temp_noon", 12: "intra_temp_noon",
            13: "intra_temp_3pm", 14: "intra_temp_3pm", 15: "intra_temp_3pm",
            16: "intra_temp_5pm", 17: "intra_temp_5pm",
        }
        obs_hour = int(features["obs_latest_hour"])
        intra_key = intra_map.get(obs_hour)
        if intra_key and intra_key in intra_features:
            intra_val = intra_features[intra_key]
            if intra_val is not None and not (isinstance(intra_val, float) and np.isnan(intra_val)):
                features["obs_vs_intra_forecast"] = round(
                    features["obs_latest_temp"] - intra_val, 1
                )
            else:
                features["obs_vs_intra_forecast"] = np.nan
        else:
            features["obs_vs_intra_forecast"] = np.nan
    else:
        features["obs_vs_intra_forecast"] = np.nan

    # Wind speed and gust
    if "wind_speed_10m" in day_partial.columns:
        wind = day_partial.loc[latest_idx, "wind_speed_10m"]
        features["obs_wind_speed"] = float(wind) if pd.notna(wind) else np.nan
    else:
        features["obs_wind_speed"] = np.nan

    if "wind_gusts_10m" in day_partial.columns:
        gust = day_partial.loc[latest_idx, "wind_gusts_10m"]
        features["obs_wind_gust"] = float(gust) if pd.notna(gust) else np.nan
    else:
        features["obs_wind_gust"] = np.nan

    # Wind direction (circular encoding)
    if "wind_direction_10m" in day_partial.columns:
        wdir = day_partial.loc[latest_idx, "wind_direction_10m"]
        if pd.notna(wdir):
            features["obs_wind_dir_sin"] = round(float(np.sin(np.deg2rad(wdir))), 4)
            features["obs_wind_dir_cos"] = round(float(np.cos(np.deg2rad(wdir))), 4)
        else:
            features["obs_wind_dir_sin"] = np.nan
            features["obs_wind_dir_cos"] = np.nan
    else:
        features["obs_wind_dir_sin"] = np.nan
        features["obs_wind_dir_cos"] = np.nan

    # Cloud cover (direct from archive — already numeric %)
    if "cloud_cover" in day_partial.columns:
        cloud = day_partial.loc[latest_idx, "cloud_cover"]
        features["obs_cloud_cover"] = round(float(cloud) / 100.0, 2) if pd.notna(cloud) else np.nan
    else:
        features["obs_cloud_cover"] = np.nan

    # Heating rate: slope over last 3 hours
    if len(temp) >= 2:
        recent_3h = day_partial[day_partial["time"].dt.hour > (as_of_hour - 3)]
        recent_temp = recent_3h["temperature_2m"].dropna()
        if len(recent_temp) >= 2:
            hours_span = (recent_3h["time"].iloc[-1] - recent_3h["time"].iloc[0]).total_seconds() / 3600.0
            if hours_span > 0:
                features["obs_heating_rate"] = round(
                    float(recent_temp.iloc[-1] - recent_temp.iloc[0]) / hours_span, 2
                )
            else:
                features["obs_heating_rate"] = np.nan
        else:
            features["obs_heating_rate"] = np.nan
    else:
        features["obs_heating_rate"] = np.nan

    # Obs max vs NWS forecast
    if nws_last is not None and not (isinstance(nws_last, float) and np.isnan(nws_last)):
        features["obs_temp_vs_forecast_max"] = round(features["obs_max_so_far"] - nws_last, 1)
    else:
        features["obs_temp_vs_forecast_max"] = np.nan

    # Fill any missing keys
    for col in OBSERVATION_COLS:
        if col not in features:
            features[col] = np.nan

    return features


def fetch_hrrr_925mb_live(
    lat: float,
    lon: float,
    target_date: str,
    timezone: str = "America/New_York",
) -> dict:
    """
    Fetch HRRR-specific 925mb and 850mb temperatures for a target date.

    HRRR runs every hour with ~3km grid resolution and superior boundary layer physics
    vs GFS (13km). The GFS-derived 925mb in the standard forecast API can miss caps
    that HRRR resolves. On a cap day, GFS may show 925mb at 55°F while HRRR and actual
    radiosonde show 48°F — that delta is the missed signal.

    Returns features:
        atm_925mb_hrrr_max, atm_925mb_hrrr_mean  (daytime 10am-6pm)
        atm_850mb_hrrr_max, atm_850mb_hrrr_mean
        atm_925mb_gfs_hrrr_diff  (set later when both are available)
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_925hPa,temperature_850hPa",
        "models": "ncep_hrrr_conus",
        "temperature_unit": "fahrenheit",
        "timezone": timezone,
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{FORECAST_URL}?{qs}"

    try:
        data = _get_json(url)
        if "error" in data:
            raise RuntimeError(f"HRRR 925mb error: {data.get('reason', data['error'])}")

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return {"atm_925mb_hrrr_max": np.nan, "atm_925mb_hrrr_mean": np.nan,
                    "atm_850mb_hrrr_max": np.nan, "atm_850mb_hrrr_mean": np.nan}

        df = pd.DataFrame({"time": pd.to_datetime(times)})
        df["temperature_925hPa"] = hourly.get("temperature_925hPa", [None] * len(times))
        df["temperature_850hPa"] = hourly.get("temperature_850hPa", [None] * len(times))

        target = pd.Timestamp(target_date)
        day_mask = df["time"].dt.date == target.date()
        day = df[day_mask]

        # Daytime hours (10am-6pm) — peak heating window, same window as standard 925mb
        daytime = day[(day["time"].dt.hour >= 10) & (day["time"].dt.hour <= 18)]

        features = {}
        t925 = daytime["temperature_925hPa"].dropna()
        features["atm_925mb_hrrr_max"] = float(t925.max()) if len(t925) > 0 else np.nan
        features["atm_925mb_hrrr_mean"] = float(t925.mean()) if len(t925) > 0 else np.nan

        t850 = daytime["temperature_850hPa"].dropna()
        features["atm_850mb_hrrr_max"] = float(t850.max()) if len(t850) > 0 else np.nan
        features["atm_850mb_hrrr_mean"] = float(t850.mean()) if len(t850) > 0 else np.nan

        return features

    except Exception as e:
        print(f"  ⚠️ HRRR 925mb fetch failed: {e}")
        return {"atm_925mb_hrrr_max": np.nan, "atm_925mb_hrrr_mean": np.nan,
                "atm_850mb_hrrr_max": np.nan, "atm_850mb_hrrr_mean": np.nan}


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
    # Atmospheric features from the standard forecast API (including pressure levels)
    all_hourly_vars = HOURLY_VARS + HOURLY_PRESSURE_LEVEL_VARS
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(all_hourly_vars),
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
        for var in all_hourly_vars:
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

    # Multi-model features (ECMWF, GFS, ICON, GEM, HRRR, NBM, GEM HRDPS)
    try:
        mm_data = fetch_multimodel_forecast(lat, lon, timezone)
        mm_features = extract_multimodel_features(mm_data, target_date)
        features.update(mm_features)
    except Exception as e:
        print(f"  ⚠️ Multi-model fetch failed: {e}")
        for col in ["mm_spread", "mm_std", "mm_mean", "mm_ecmwf_gfs_diff",
                    "mm_hrrr_max", "mm_hrrr_ecmwf_diff", "mm_hrrr_gfs_diff",
                    "mm_nbm_max", "mm_nbm_hrrr_diff", "mm_nbm_gfs_diff", "mm_nbm_ecmwf_diff",
                    "mm_gem_hrdps_max", "mm_gem_hrdps_hrrr_diff"]:
            features[col] = np.nan

    # HRRR-specific 925mb — better boundary layer resolution than GFS-derived 925mb.
    # The GFS 925mb can miss caps that HRRR resolves at 3km vs 13km grid spacing.
    # Adds atm_925mb_hrrr_max, atm_925mb_hrrr_mean, atm_850mb_hrrr_max, atm_850mb_hrrr_mean.
    hrrr_925mb = fetch_hrrr_925mb_live(lat, lon, target_date, timezone)
    features.update(hrrr_925mb)

    # GFS vs HRRR 925mb diff — when this is large, GFS is missing the cap signal.
    # Negative = HRRR sees cooler air aloft than GFS (cap stronger than GFS thinks).
    gfs_925 = features.get("atm_925mb_temp_mean")
    hrrr_925 = features.get("atm_925mb_hrrr_mean")
    if gfs_925 is not None and hrrr_925 is not None and not np.isnan(gfs_925) and not np.isnan(hrrr_925):
        features["atm_925mb_gfs_hrrr_diff"] = gfs_925 - hrrr_925
    else:
        features["atm_925mb_gfs_hrrr_diff"] = np.nan

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
