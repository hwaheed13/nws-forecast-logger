# predict.py
import sys
import json
import pickle
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

from model_config import FEATURE_COLS, ACCU_NWS_FALLBACK, derive_bucket_probabilities

warnings.filterwarnings('ignore')


def load_models():
    """Load trained temperature model and bucket info from disk."""
    with open('temp_model.pkl', 'rb') as f:
        temp_model = pickle.load(f)
    with open('bucket_model.pkl', 'rb') as f:
        bucket_info = pickle.load(f)
    return temp_model, bucket_info


def prepare_features(raw_features):
    """Convert raw features from JS/caller to model input DataFrame."""
    m = int(raw_features.get('month', 1))

    # Compute day-of-year cyclical features from target_date if available
    target_date = raw_features.get('target_date', '')
    if target_date:
        try:
            doy = datetime.strptime(target_date, '%Y-%m-%d').timetuple().tm_yday
        except ValueError:
            doy = datetime.now().timetuple().tm_yday
    else:
        doy = datetime.now().timetuple().tm_yday

    features = {
        # NWS forecast statistics
        'nws_first': raw_features.get('nws_first', np.nan),
        'nws_last': raw_features.get('nws_last', np.nan),
        'nws_max': raw_features.get('nws_max', np.nan),
        'nws_min': raw_features.get('nws_min', np.nan),
        'nws_mean': raw_features.get('nws_mean', np.nan),
        'nws_spread': raw_features.get('nws_spread', 0),
        'nws_std': raw_features.get('nws_std', 0),
        'nws_trend': raw_features.get('nws_trend', 0),
        'nws_count': raw_features.get('nws_count', 1),
        'forecast_velocity': raw_features.get('forecast_velocity', 0),
        'forecast_acceleration': raw_features.get('forecast_acceleration', 0),

        # AccuWeather forecast statistics
        'accu_first': raw_features.get('accu_first', np.nan),
        'accu_last': raw_features.get('accu_last', np.nan),
        'accu_max': raw_features.get('accu_max', np.nan),
        'accu_min': raw_features.get('accu_min', np.nan),
        'accu_mean': raw_features.get('accu_mean', np.nan),
        'accu_spread': raw_features.get('accu_spread', 0),
        'accu_std': raw_features.get('accu_std', 0),
        'accu_trend': raw_features.get('accu_trend', 0),
        'accu_count': raw_features.get('accu_count', 0),

        # Cross-source features
        'nws_accu_spread': raw_features.get('nws_accu_spread', 0),
        'nws_accu_mean_diff': raw_features.get('nws_accu_mean_diff', 0),

        # Temporal features
        'day_of_year_sin': np.sin(2 * np.pi * doy / 365),
        'day_of_year_cos': np.cos(2 * np.pi * doy / 365),
        'month': m,
        'is_summer': int(m in [6, 7, 8]),
        'is_winter': int(m in [12, 1, 2]),

        # Rolling bias (must be provided by caller or default to 0)
        'rolling_bias_7d': raw_features.get('rolling_bias_7d', 0),
        'rolling_bias_21d': raw_features.get('rolling_bias_21d', 0),

        # Data availability flag
        'has_accu_data': raw_features.get('has_accu_data', int(raw_features.get('accu_last') is not None)),
    }

    # Create DataFrame with correct column order
    X = pd.DataFrame([features])[FEATURE_COLS]

    # Fill NaN AccuWeather values with NWS equivalents
    for accu_col, nws_col in ACCU_NWS_FALLBACK.items():
        if pd.isna(X.loc[0, accu_col]):
            X.loc[0, accu_col] = X.loc[0, nws_col]

    return X


def main():
    # Parse input from Node.js
    raw_features = json.loads(sys.argv[1])

    # Load models
    temp_model, bucket_info = load_models()

    # Prepare features
    X = prepare_features(raw_features)

    # Model predicts bias (actual - best_base); base = AccuWeather if available, else NWS
    predicted_bias = float(temp_model.predict(X)[0])
    accu_last = raw_features.get('accu_last')
    if accu_last is not None:
        base = float(accu_last)
    else:
        base = float(raw_features.get('nws_last', raw_features.get('nws_mean', 0)))
    temperature = base + predicted_bias

    # Bucket probabilities from Gaussian derivation
    if isinstance(bucket_info, dict) and 'residual_std' in bucket_info:
        residual_std = bucket_info['residual_std']
    else:
        # Backward compat: if bucket_model.pkl is an old-style classifier
        residual_std = 2.0

    bucket_dict = derive_bucket_probabilities(temperature, residual_std)
    best_bucket = max(bucket_dict, key=bucket_dict.get)
    confidence = bucket_dict[best_bucket]

    result = {
        'temperature': round(temperature, 1),
        'residual_std': round(residual_std, 2),
        'bucket_probabilities': bucket_dict,
        'best_bucket': best_bucket,
        'confidence': round(confidence, 4),
        'should_bet': confidence > 0.15,
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
