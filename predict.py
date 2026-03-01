# predict.py
import sys
import json
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def load_models():
    """Load trained models from disk"""
    with open('temp_model.pkl', 'rb') as f:
        temp_model = pickle.load(f)
    with open('bucket_model.pkl', 'rb') as f:
        bucket_model = pickle.load(f)
    return temp_model, bucket_model

def prepare_features(raw_features):
    """Convert raw features from JS to model input"""
    # Expected feature columns (must match training)
    feature_cols = [
        'nws_first', 'nws_last', 'nws_max', 'nws_min', 'nws_mean',
        'nws_spread', 'nws_std', 'nws_trend', 'nws_count',
        'forecast_velocity', 'forecast_acceleration',
        'accu_last', 'nws_accu_spread',
        'month', 'is_summer', 'is_winter'
    ]
    
    # Build feature vector
    features = {
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
        'accu_last': raw_features.get('accu_last', raw_features.get('nws_last', np.nan)),
        'nws_accu_spread': raw_features.get('nws_accu_spread', 0),
        'month': raw_features.get('month', 1),
        'is_summer': raw_features.get('month', 1) in [6, 7, 8],
        'is_winter': raw_features.get('month', 1) in [12, 1, 2],
    }
    
    # Create DataFrame with correct column order
    X = pd.DataFrame([features])[feature_cols]
    
    # Fill NaN values
    X['accu_last'].fillna(X['nws_last'], inplace=True)
    X['nws_accu_spread'].fillna(0, inplace=True)
    
    return X

def main():
    # Parse input from Node.js
    raw_features = json.loads(sys.argv[1])
    
    # Load models
    temp_model, bucket_model = load_models()
    
    # Prepare features
    X = prepare_features(raw_features)
    
    # Make predictions
    temperature = temp_model.predict(X)[0]
    bucket_probs = bucket_model.predict_proba(X)[0]
    bucket_classes = bucket_model.classes_
    
    # Format bucket probabilities
    bucket_dict = {
        cls: float(prob) 
        for cls, prob in zip(bucket_classes, bucket_probs)
    }
    
    # Find most likely bucket
    best_bucket = max(bucket_dict, key=bucket_dict.get)
    
    # Calculate confidence (based on probability spread)
    confidence = float(max(bucket_probs))
    
    # Output result
    result = {
        'temperature': round(float(temperature), 1),
        'bucket_probabilities': bucket_dict,
        'best_bucket': best_bucket,
        'confidence': round(confidence, 2),
        'should_bet': confidence > 0.6
    }
    
    print(json.dumps(result))

if __name__ == "__main__":
    main()
