# model_config.py — single source of truth for ML model feature columns
# Imported by train_models.py, predict.py, and api.py

import math

FEATURE_COLS = [
    # NWS forecast statistics
    "nws_first", "nws_last", "nws_max", "nws_min", "nws_mean",
    "nws_spread", "nws_std", "nws_trend", "nws_count",
    "forecast_velocity", "forecast_acceleration",
    # AccuWeather forecast statistics (parity with NWS)
    "accu_first", "accu_last", "accu_max", "accu_min", "accu_mean",
    "accu_spread", "accu_std", "accu_trend", "accu_count",
    # Cross-source features
    "nws_accu_spread", "nws_accu_mean_diff",
    # Temporal features (cyclical + categorical)
    "day_of_year_sin", "day_of_year_cos",
    "month", "is_summer", "is_winter",
    # Rolling bias from prior completed days
    "rolling_bias_7d", "rolling_bias_21d",
    # Data availability flag
    "has_accu_data",
]

# AccuWeather columns that fall back to NWS equivalents when missing
ACCU_NWS_FALLBACK = {
    "accu_first": "nws_first",
    "accu_last": "nws_last",
    "accu_max": "nws_max",
    "accu_min": "nws_min",
    "accu_mean": "nws_mean",
}


def norm_cdf(x, mu, sigma):
    """Gaussian CDF using math.erf (no scipy dependency)."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def derive_bucket_probabilities(predicted_temp, residual_std, spread=8):
    """
    Derive Kalshi bucket probabilities from a Gaussian centered on predicted_temp.

    Each bucket is a 1-degree range [low, low+1). Returns dict like {"42-43": 0.27, ...}
    with probabilities > 0.001, covering ±spread degrees around the prediction.
    """
    center = int(round(predicted_temp))
    buckets = {}
    for low in range(center - spread, center + spread + 1):
        high = low + 1
        p = norm_cdf(high, predicted_temp, residual_std) - \
            norm_cdf(low, predicted_temp, residual_std)
        if p > 0.001:
            buckets[f"{low}-{high}"] = round(p, 4)
    return buckets
