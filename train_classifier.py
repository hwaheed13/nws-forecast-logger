# train_classifier.py — Direct bucket classification model for Kalshi
#
# Instead of predicting temperature and mapping to a bucket (regression),
# this model directly predicts P(bucket wins) for each candidate bucket.
#
# Approach: Neighborhood Binary Classification
#   For each training day, generate 7 candidate buckets around the point estimate.
#   Each candidate becomes a training row with label: 1 if winning, 0 if not.
#   This multiplies training data 7x while keeping binary classification clean.
#
# The classifier learns:
#   - When atmospheric conditions make forecasts unreliable
#   - Which bucket boundary crossings are likely given wind/humidity/pressure
#   - How ensemble spread correlates with bucket uncertainty
#   - Seasonal patterns in forecast errors
#
# Usage:
#   Called by train_models.py during v2 training pipeline.
#   Can also be run standalone for testing.

from __future__ import annotations

import json
import math
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

from model_config import (
    FEATURE_COLS_V2,
    BUCKET_POSITION_COLS,
    temp_to_bucket_label,
    get_candidate_buckets,
)

warnings.filterwarnings("ignore")

N_CANDIDATE_BUCKETS = 7  # 3 on each side + center = 7 candidates per day


def build_classification_dataset(
    features_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build the classification training dataset from the regression feature matrix.

    IMPORTANT: Only pass rows that have real forecast data (NWS or AccuWeather).
    Rows without forecasts (historical backfill) used actual_high as center,
    which leaked the answer into position features and inflated accuracy from
    ~46% to ~82%. Training on forecast-only rows gives honest accuracy.

    For each day:
      1. Use the best available forecast (AccuWeather or NWS) as center
      2. Generate N_CANDIDATE_BUCKETS candidate buckets around that center
      3. Label the winning bucket as 1, all others as 0
      4. Add bucket-position features for each candidate

    Returns:
        X: DataFrame with feature_cols + BUCKET_POSITION_COLS
        y: Series of 0/1 labels
        day_ids: Series with day index (for grouped cross-validation)
    """
    all_rows = []
    all_labels = []
    all_day_ids = []
    skipped = 0

    for idx, row in features_df.iterrows():
        actual_high = row["actual_high"]
        winning_bucket = temp_to_bucket_label(actual_high)

        # Center prediction: use best available forecast
        accu_last = row.get("accu_last")
        nws_last = row.get("nws_last")
        if pd.notna(accu_last):
            center = float(accu_last)
        elif pd.notna(nws_last):
            center = float(nws_last)
        else:
            # No forecast data — skip this row.
            # Using actual_high as center leaks the answer.
            skipped += 1
            continue

        # Also get the regression prediction if available
        regression_pred = row.get("_regression_pred")
        if pd.notna(regression_pred) if regression_pred is not None else False:
            center = float(regression_pred)

        # Generate candidate buckets
        candidates = get_candidate_buckets(center, n_neighbors=N_CANDIDATE_BUCKETS // 2)

        # Skip if winning bucket isn't reachable (extreme forecast miss)
        if winning_bucket not in candidates:
            skipped += 1
            continue

        for bucket_label in candidates:
            parts = bucket_label.split("-")
            bucket_lo = int(parts[0])
            bucket_center = bucket_lo + 0.5

            feature_row = {}
            for col in feature_cols:
                feature_row[col] = row.get(col, np.nan)

            # Bucket-position features (these vary per candidate)
            feature_row["bucket_center"] = bucket_center
            feature_row["dist_from_prediction"] = abs(center - bucket_center)
            feature_row["dist_from_accu"] = (
                abs(float(accu_last) - bucket_center) if pd.notna(accu_last) else
                abs(center - bucket_center)
            )
            feature_row["dist_from_nws"] = (
                abs(float(nws_last) - bucket_center) if pd.notna(nws_last) else
                abs(center - bucket_center)
            )

            all_rows.append(feature_row)
            all_labels.append(1 if bucket_label == winning_bucket else 0)
            all_day_ids.append(idx)

    if skipped > 0:
        print(f"  ⚠️ Skipped {skipped} days (no forecast data or winning bucket unreachable)")

    X = pd.DataFrame(all_rows)
    y = pd.Series(all_labels, name="label")
    day_ids = pd.Series(all_day_ids, name="day_id")

    return X, y, day_ids


class BucketClassifier:
    """
    Direct bucket classification model for Kalshi temperature markets.

    Trains a binary classifier: P(candidate bucket wins | features + bucket position).
    At inference, scores each candidate bucket and normalizes to get probabilities.
    """

    def __init__(self):
        self.model: HistGradientBoostingClassifier | None = None
        self.feature_cols: list[str] = []
        self.cv_bucket_accuracy: float = 0.0
        self.cv_log_loss: float = 0.0
        self.training_stats: dict = {}

    def train(
        self,
        features_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> None:
        """
        Train the bucket classifier.

        Args:
            features_df: DataFrame from train_models.py with all features + actual_high
            feature_cols: list of feature columns (default: FEATURE_COLS_V2)
        """
        if feature_cols is None:
            # Use whatever v2 columns are available in the data
            feature_cols = [c for c in FEATURE_COLS_V2 if c in features_df.columns]

        self.feature_cols = feature_cols
        all_cols = feature_cols + BUCKET_POSITION_COLS

        print(f"\n{'─'*50}")
        print(f"Training bucket classifier")
        print(f"{'─'*50}")
        print(f"  Feature cols: {len(feature_cols)} base + {len(BUCKET_POSITION_COLS)} position = {len(all_cols)} total")
        print(f"  Training days: {len(features_df)}")

        # Build classification dataset
        X, y, day_ids = build_classification_dataset(features_df, feature_cols)
        print(f"  Classification rows: {len(X)} ({len(X) // len(features_df)} per day)")
        print(f"  Positive labels: {y.sum()} ({y.mean():.1%})")

        # Cross-validation with TimeSeriesSplit on DAYS (not rows)
        unique_days = sorted(day_ids.unique())
        n_days = len(unique_days)
        tscv = TimeSeriesSplit(n_splits=5)

        cv_accuracies = []
        cv_losses = []

        for fold_idx, (train_day_idx, test_day_idx) in enumerate(tscv.split(unique_days)):
            train_days = set(np.array(unique_days)[train_day_idx])
            test_days = set(np.array(unique_days)[test_day_idx])

            train_mask = day_ids.isin(train_days)
            test_mask = day_ids.isin(test_days)

            X_train = X[train_mask][all_cols]
            X_test = X[test_mask][all_cols]
            y_train = y[train_mask]
            y_test = y[test_mask]

            model = HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=3,
                learning_rate=0.05,
                min_samples_leaf=10,
                l2_regularization=2.0,
                max_leaf_nodes=15,
                class_weight={0: 1.0, 1: float(N_CANDIDATE_BUCKETS - 1)},
                random_state=42,
            )
            model.fit(X_train, y_train)

            # Evaluate: for each test day, pick the bucket with highest predicted P
            test_day_list = sorted(test_days)
            correct = 0
            total = 0
            fold_losses = []

            for day_id in test_day_list:
                day_mask_d = (day_ids == day_id) & test_mask
                X_day = X[day_mask_d][all_cols]
                y_day = y[day_mask_d]

                if len(X_day) == 0:
                    continue

                proba = model.predict_proba(X_day)[:, 1]  # P(wins)
                predicted_idx = np.argmax(proba)
                actual_winner = y_day.values.argmax() if y_day.sum() > 0 else -1

                if predicted_idx == actual_winner:
                    correct += 1
                total += 1

                # Log loss for this day
                try:
                    fold_losses.append(log_loss(y_day, proba, labels=[0, 1]))
                except Exception:
                    pass

            fold_acc = correct / total if total > 0 else 0.0
            fold_loss = np.mean(fold_losses) if fold_losses else float("inf")
            cv_accuracies.append(fold_acc)
            cv_losses.append(fold_loss)
            print(f"    Fold {fold_idx+1}: bucket accuracy={fold_acc:.1%} ({correct}/{total}), "
                  f"log_loss={fold_loss:.3f}")

        self.cv_bucket_accuracy = float(np.mean(cv_accuracies))
        self.cv_log_loss = float(np.mean(cv_losses))
        print(f"  CV Bucket Accuracy: {self.cv_bucket_accuracy:.1%}")
        print(f"  CV Log Loss: {self.cv_log_loss:.3f}")

        # Train final model on all data
        self.model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=10,
            l2_regularization=2.0,
            max_leaf_nodes=15,
            class_weight={0: 1.0, 1: float(N_CANDIDATE_BUCKETS - 1)},
            random_state=42,
        )
        self.model.fit(X[all_cols], y)

        # In-sample accuracy
        correct_insample = 0
        total_insample = 0
        for day_id in unique_days:
            day_mask_d = day_ids == day_id
            X_day = X[day_mask_d][all_cols]
            y_day = y[day_mask_d]
            if len(X_day) == 0:
                continue
            proba = self.model.predict_proba(X_day)[:, 1]
            predicted_idx = np.argmax(proba)
            actual_winner = y_day.values.argmax() if y_day.sum() > 0 else -1
            if predicted_idx == actual_winner:
                correct_insample += 1
            total_insample += 1

        insample_acc = correct_insample / total_insample if total_insample > 0 else 0.0
        print(f"  In-sample Bucket Accuracy: {insample_acc:.1%}")

        self.training_stats = {
            "cv_bucket_accuracy": round(self.cv_bucket_accuracy, 4),
            "cv_log_loss": round(self.cv_log_loss, 4),
            "insample_bucket_accuracy": round(insample_acc, 4),
            "num_training_days": len(features_df),
            "num_classification_rows": len(X),
            "n_candidate_buckets": N_CANDIDATE_BUCKETS,
            "feature_cols_used": len(all_cols),
            "positive_rate": round(float(y.mean()), 4),
        }

    def predict_bucket_probs(
        self,
        features: dict,
        center_temp: float,
        accu_last: float | None = None,
        nws_last: float | None = None,
        n_candidates: int | None = None,
    ) -> list[dict]:
        """
        Predict bucket probabilities for a given feature set.

        Args:
            features: dict of v2 features for the day
            center_temp: regression model's temperature prediction (used as center)
            accu_last: latest AccuWeather forecast (optional)
            nws_last: latest NWS forecast (optional)
            n_candidates: override for number of candidate buckets

        Returns:
            List of {bucket, probability} dicts, sorted by probability descending.
        """
        if self.model is None:
            return []

        n = n_candidates or N_CANDIDATE_BUCKETS
        candidates = get_candidate_buckets(center_temp, n_neighbors=n // 2)
        all_cols = self.feature_cols + BUCKET_POSITION_COLS

        rows = []
        for bucket_label in candidates:
            parts = bucket_label.split("-")
            bucket_center = int(parts[0]) + 0.5

            row = {}
            for col in self.feature_cols:
                row[col] = features.get(col, np.nan)

            row["bucket_center"] = bucket_center
            row["dist_from_prediction"] = abs(center_temp - bucket_center)
            row["dist_from_accu"] = (
                abs(accu_last - bucket_center) if accu_last is not None else
                abs(center_temp - bucket_center)
            )
            row["dist_from_nws"] = (
                abs(nws_last - bucket_center) if nws_last is not None else
                abs(center_temp - bucket_center)
            )
            rows.append(row)

        X = pd.DataFrame(rows)[all_cols]
        probas = self.model.predict_proba(X)[:, 1]

        # Normalize to sum to 1
        total = probas.sum()
        if total > 0:
            probas = probas / total

        results = []
        for bucket_label, prob in zip(candidates, probas):
            results.append({
                "bucket": bucket_label,
                "probability": round(float(prob), 4),
            })

        results.sort(key=lambda x: x["probability"], reverse=True)
        return results

    def save(self, path: str) -> None:
        """Save the trained classifier and metadata."""
        save_data = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "training_stats": self.training_stats,
            "n_candidate_buckets": N_CANDIDATE_BUCKETS,
            "version": "v2_bucket_classifier",
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
        print(f"  Saved bucket classifier to {path}")

    @classmethod
    def load(cls, path: str) -> "BucketClassifier":
        """Load a trained classifier from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        obj = cls()
        obj.model = data["model"]
        obj.feature_cols = data["feature_cols"]
        obj.training_stats = data.get("training_stats", {})
        return obj


# ═══════════════════════════════════════════════════════════════════════
# Standalone training (for testing)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train bucket classifier (standalone)")
    parser.add_argument("--city", default="nyc", help="City key")
    args = parser.parse_args()

    from city_config import get_city_config
    from train_models import NYCTemperatureModelTrainer

    # Use the existing training pipeline to build the feature matrix
    trainer = NYCTemperatureModelTrainer(city_key=args.city)
    trainer.load_data()
    trainer.build_feature_matrix()

    # Train the classifier
    classifier = BucketClassifier()
    classifier.train(trainer.features_df)

    prefix = trainer.model_prefix
    classifier.save(f"{prefix}bucket_classifier.pkl")

    print("\nTraining stats:")
    print(json.dumps(classifier.training_stats, indent=2))
