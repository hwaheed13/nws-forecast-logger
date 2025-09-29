# train_models.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, accuracy_score
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = [
    'nws_first', 'nws_last', 'nws_max', 'nws_min', 'nws_mean',
    'nws_spread', 'nws_std', 'nws_trend', 'nws_count',
    'forecast_velocity', 'forecast_acceleration',
    'accu_last', 'nws_accu_spread',
    'month', 'is_summer', 'is_winter'
]

class NYCTemperatureModelTrainer:
    def __init__(self):
        self.nws_df = None
        self.accu_df = None
        self.features_df = None
        self.temp_model = None
        self.bucket_model = None

    def load_data(self):
        """Load NWS and AccuWeather CSV files"""
        print("Loading CSV files...")
        self.nws_df = pd.read_csv('nws_forecast_log.csv')

        try:
            self.accu_df = pd.read_csv('accuweather_log.csv')
        except FileNotFoundError:
            print("AccuWeather file not found, proceeding with NWS only")
            self.accu_df = pd.DataFrame()

        # Convert timestamps
        self.nws_df['timestamp'] = pd.to_datetime(self.nws_df['timestamp'], errors='coerce')
        if not self.accu_df.empty:
            self.accu_df['timestamp'] = pd.to_datetime(self.accu_df['timestamp'], errors='coerce')

        # Drop rows with invalid timestamps
        self.nws_df = self.nws_df.dropna(subset=['timestamp'])
        if not self.accu_df.empty:
            self.accu_df = self.accu_df.dropna(subset=['timestamp'])

        print(f"Loaded {len(self.nws_df)} NWS rows, {len(self.accu_df)} AccuWeather rows")

    def extract_features_for_date(self, target_date):
        """Extract all features for a single date"""
        # Get NWS forecasts for this date
        nws_forecasts = self.nws_df[
            (self.nws_df['target_date'] == target_date) &
            (self.nws_df['forecast_or_actual'] == 'forecast')
        ].copy()

        # Get AccuWeather forecasts
        if not self.accu_df.empty:
            accu_forecasts = self.accu_df[
                (self.accu_df['target_date'] == target_date) &
                (self.accu_df['forecast_or_actual'] == 'forecast')
            ].copy()
        else:
            accu_forecasts = pd.DataFrame()

        # Get actual
        actual_row = self.nws_df[
            (self.nws_df['cli_date'] == target_date) &
            (self.nws_df['forecast_or_actual'] == 'actual')
        ]

        if actual_row.empty or nws_forecasts.empty:
            return None

        actual_high = float(actual_row.iloc[0]['actual_high'])

        # Sort forecasts by time
        nws_forecasts = nws_forecasts.sort_values('timestamp')
        if not accu_forecasts.empty:
            accu_forecasts = accu_forecasts.sort_values('timestamp')

        # NWS features
        nws_values = nws_forecasts['predicted_high'].astype(float).values

        features = {
            'nws_first': nws_values[0] if len(nws_values) > 0 else np.nan,
            'nws_last': nws_values[-1] if len(nws_values) > 0 else np.nan,
            'nws_max': nws_values.max() if len(nws_values) > 0 else np.nan,
            'nws_min': nws_values.min() if len(nws_values) > 0 else np.nan,
            'nws_mean': nws_values.mean() if len(nws_values) > 0 else np.nan,
            'nws_median': np.median(nws_values) if len(nws_values) > 0 else np.nan,

            'nws_count': len(nws_values),
            'nws_spread': nws_values.max() - nws_values.min() if len(nws_values) > 0 else 0,
            'nws_std': nws_values.std() if len(nws_values) > 1 else 0,
            'nws_trend': nws_values[-1] - nws_values[0] if len(nws_values) > 1 else 0,

            'forecast_velocity': self.calculate_velocity(nws_values),
            'forecast_acceleration': self.calculate_acceleration(nws_values),

            'accu_last': float(accu_forecasts.iloc[-1]['predicted_high']) if not accu_forecasts.empty else np.nan,
            'accu_count': len(accu_forecasts),

            'nws_accu_spread': np.nan,

            'month': int(target_date.split('-')[1]),
            'day': int(target_date.split('-')[2]),
            'is_weekend': int(datetime.strptime(target_date, '%Y-%m-%d').weekday() >= 5),
            'is_summer': int(int(target_date.split('-')[1]) in [6, 7, 8]),
            'is_winter': int(int(target_date.split('-')[1]) in [12, 1, 2]),

            'actual_high': actual_high,
            'winning_bucket': f"{int(actual_high)}-{int(actual_high)+1}",
            'target_date': target_date
        }

        if not np.isnan(features['nws_last']) and not np.isnan(features['accu_last']):
            features['nws_accu_spread'] = abs(features['nws_last'] - features['accu_last'])

        return features

    def calculate_velocity(self, values):
        """Calculate rate of change in forecasts"""
        if len(values) < 2:
            return 0
        diffs = np.diff(values)
        return float(diffs.mean())

    def calculate_acceleration(self, values):
        """Calculate acceleration (change in velocity)"""
        if len(values) < 3:
            return 0
        diffs = np.diff(values)
        acc = np.diff(diffs)
        return float(acc.mean()) if len(acc) > 0 else 0

    def build_feature_matrix(self):
        """Build complete feature matrix from all dates"""
        print("\nExtracting features for all dates...")

        dates_with_actuals = self.nws_df[
            self.nws_df['forecast_or_actual'] == 'actual'
        ]['cli_date'].unique()

        features_list = []
        for date in sorted(dates_with_actuals):
            feat = self.extract_features_for_date(date)
            if feat:
                features_list.append(feat)

        self.features_df = pd.DataFrame(features_list)
        print(f"Built features for {len(self.features_df)} days")

        # Fill NaN values
        if 'accu_last' in self.features_df.columns:
            self.features_df['accu_last'] = self.features_df['accu_last'].fillna(self.features_df['nws_last'])
        self.features_df['nws_accu_spread'] = self.features_df['nws_accu_spread'].fillna(0)

        return self.features_df

    def train_temperature_model(self):
        """Train model to predict actual temperature"""
        print("\nTraining temperature prediction model...")

        X = self.features_df[FEATURE_COLS]
        y = self.features_df['actual_high']

        tscv = TimeSeriesSplit(n_splits=3)
        mae_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mae_scores.append(mean_absolute_error(y_test, pred))

        print(f"Cross-validation MAE: {np.mean(mae_scores):.2f}°F")

        self.temp_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        self.temp_model.fit(X, y)
        return self.temp_model

    def train_bucket_model(self):
        """Train model to predict Kalshi bucket"""
        print("\nTraining Kalshi bucket prediction model...")

        X = self.features_df[FEATURE_COLS]
        y = self.features_df['winning_bucket']

        tscv = TimeSeriesSplit(n_splits=3)
        acc_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                random_state=42
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc_scores.append(accuracy_score(y_test, pred))

        print(f"Cross-validation accuracy: {np.mean(acc_scores):.1%}")

        self.bucket_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42
        )
        self.bucket_model.fit(X, y)
        return self.bucket_model

    def save_models(self):
        """Save trained models and metadata"""
        print("\nSaving models...")

        with open('temp_model.pkl', 'wb') as f:
            pickle.dump(self.temp_model, f)
        with open('bucket_model.pkl', 'wb') as f:
            pickle.dump(self.bucket_model, f)

        metadata = {
            'trained_on': datetime.now().isoformat(),
            'num_days': int(len(self.features_df)),
            'date_range': {
                'start': str(self.features_df['target_date'].min()),
                'end': str(self.features_df['target_date'].max())
            },
            'feature_columns': FEATURE_COLS,
        }

        # Compute simple in-sample metrics for quick sanity check
        temp_pred_all = self.temp_model.predict(self.features_df[FEATURE_COLS])
        bucket_pred_all = self.bucket_model.predict(self.features_df[FEATURE_COLS])
        metadata['model_performance'] = {
            'temperature_mae': float(mean_absolute_error(self.features_df['actual_high'], temp_pred_all)),
            'bucket_accuracy': float(accuracy_score(self.features_df['winning_bucket'], bucket_pred_all))
        }

        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print("Models saved successfully!")
        print(f"Temperature MAE: {metadata['model_performance']['temperature_mae']:.2f}°F")
        print(f"Bucket accuracy: {metadata['model_performance']['bucket_accuracy']:.1%}")

    def run(self):
        """Main training pipeline"""
        print("="*50)
        print("NYC Temperature Model Training")
        print("="*50)

        self.load_data()
        self.build_feature_matrix()
        self.train_temperature_model()
        self.train_bucket_model()
        self.save_models()
        print("\n✅ Training complete!")

if __name__ == "__main__":
    trainer = NYCTemperatureModelTrainer()
    trainer.run()
