# predict.py — ML inference for temperature and bucket prediction
# Called from Node.js via: python predict.py '{"nws_last": 50, ...}'
# Returns JSON with temperature, bucket probabilities, and classifier predictions.

import sys
import json
import pickle
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

from model_config import (
    FEATURE_COLS, FEATURE_COLS_V2,
    ACCU_NWS_FALLBACK,
    derive_bucket_probabilities,
)

warnings.filterwarnings('ignore')


def load_models(prefix=""):
    """Load trained temperature model and bucket info from disk."""
    with open(f'{prefix}temp_model.pkl', 'rb') as f:
        temp_model = pickle.load(f)
    with open(f'{prefix}bucket_model.pkl', 'rb') as f:
        bucket_info = pickle.load(f)
    return temp_model, bucket_info


def load_v2_models(prefix=""):
    """Load v2 models if available. Returns (regressor, bucket_info, classifier) or Nones."""
    try:
        with open(f'{prefix}temp_model_v2.pkl', 'rb') as f:
            v2_regressor = pickle.load(f)
        with open(f'{prefix}bucket_model_v2.pkl', 'rb') as f:
            v2_bucket_info = pickle.load(f)
        from train_classifier import BucketClassifier
        v2_classifier = BucketClassifier.load(f'{prefix}bucket_classifier.pkl')
        return v2_regressor, v2_bucket_info, v2_classifier
    except (FileNotFoundError, Exception):
        return None, None, None


def prepare_features(raw_features, feature_cols=None):
    """Convert raw features from JS/caller to model input DataFrame."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS

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

    # Add atmospheric features if present in raw_features
    from model_config import FEATURE_COLS_V2
    for col in FEATURE_COLS_V2:
        if col not in features and col in raw_features:
            features[col] = raw_features[col]
        elif col not in features:
            features[col] = np.nan

    # Create DataFrame with correct column order
    X = pd.DataFrame([features])[feature_cols]

    # Fill NaN AccuWeather values with NWS equivalents
    for accu_col, nws_col in ACCU_NWS_FALLBACK.items():
        if accu_col in X.columns and nws_col in X.columns:
            if pd.isna(X.loc[0, accu_col]):
                X.loc[0, accu_col] = X.loc[0, nws_col]

    return X


def main():
    # Parse input from Node.js
    raw_features = json.loads(sys.argv[1])

    # Load v1 models (always available)
    temp_model, bucket_info = load_models()

    # Prepare features
    X = prepare_features(raw_features, feature_cols=FEATURE_COLS)

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
        'version': 'v1',
    }

    # Try v2 models
    v2_regressor, v2_bucket_info, v2_classifier = load_v2_models()
    if v2_regressor is not None and v2_classifier is not None:
        try:
            X_v2 = prepare_features(raw_features, feature_cols=FEATURE_COLS_V2)
            v2_bias = float(v2_regressor.predict(X_v2)[0])
            v2_temp = base + v2_bias

            bucket_probs = v2_classifier.predict_bucket_probs(
                features=raw_features,
                center_temp=v2_temp,
                accu_last=accu_last,
                nws_last=raw_features.get('nws_last'),
                n_candidates=11,
            )

            if bucket_probs:
                v2_best = bucket_probs[0]
                result['temperature'] = round(v2_temp, 1)
                result['best_bucket'] = v2_best['bucket']
                result['confidence'] = v2_best['probability']
                result['bucket_probabilities'] = {
                    bp['bucket']: bp['probability'] for bp in bucket_probs
                }
                result['should_bet'] = v2_best['probability'] > 0.15
                result['version'] = 'v2_atm_classifier'

                if v2_bucket_info and 'residual_std' in v2_bucket_info:
                    result['residual_std'] = round(v2_bucket_info['residual_std'], 2)
        except Exception as e:
            # Fall back to v1 results (already in result dict)
            result['v2_error'] = str(e)

    # Compute bet signal if market_probs provided in input
    market_probs = raw_features.get('market_probs', {})
    if market_probs and result.get('bucket_probabilities'):
        # Map our 1°F ML buckets → Kalshi's actual bucket structure
        ml_probs = result['bucket_probabilities']
        kalshi_aligned = {}
        for kalshi_label in market_probs:
            parts = kalshi_label.split("-")
            if len(parts) != 2:
                continue
            try:
                lo, hi = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            agg = sum(ml_probs.get(f"{t}-{t+1}", 0) for t in range(lo, hi + 1))
            kalshi_aligned[kalshi_label] = round(agg, 4)

        if kalshi_aligned:
            best_kalshi = max(kalshi_aligned, key=kalshi_aligned.get)
            ml_conf = kalshi_aligned[best_kalshi]
            result['best_bucket'] = best_kalshi
            result['confidence'] = ml_conf
            result['kalshi_aligned_probs'] = kalshi_aligned

            market_prob = market_probs.get(best_kalshi, 0)
            edge = ml_conf - market_prob

            if ml_conf >= 0.55 and edge >= 0.10:
                signal = "STRONG_BET"
            elif ml_conf >= 0.40 and edge >= 0.05:
                signal = "BET"
            elif ml_conf >= 0.30:
                signal = "LEAN"
            else:
                signal = "SKIP"

            result['bet_signal'] = signal
            result['ml_edge'] = round(edge, 4)
            result['market_prob'] = round(market_prob, 4)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
