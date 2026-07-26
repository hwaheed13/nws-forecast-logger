"""Golden tests — bucket-probability math in model_config (renovation Phase 1)."""
import math

import model_config as mc


class TestTempToBucketLabel:
    def test_rounds_to_settle_integer(self):
        assert mc.temp_to_bucket_label(75.6) == "76-77"
        assert mc.temp_to_bucket_label(75.4) == "75-76"
        assert mc.temp_to_bucket_label(75.0) == "75-76"


class TestDeriveBucketProbabilities:
    def test_gaussian_centered_and_normalized(self):
        probs = mc.derive_bucket_probabilities(74.0, residual_std=2.0)
        # Argmax bucket is the one containing the center.
        assert max(probs, key=probs.get) == "74-75"
        assert 0.9 < sum(probs.values()) <= 1.001
        # Symmetric distribution → symmetric neighbors.
        assert abs(probs["73-74"] - probs["75-76"]) < 1e-6

    def test_zero_sigma_degenerates_to_point_mass(self):
        probs = mc.derive_bucket_probabilities(74.0, residual_std=0)
        assert probs["74-75"] == 1.0


class TestQuantileBucketProbs:
    QUANTILES = {0.05: 71.2, 0.1: 71.9, 0.25: 72.9, 0.5: 74.0,
                 0.75: 75.0, 0.9: 75.7, 0.95: 76.1}

    def test_argmax_is_median_bucket(self):
        probs = mc.quantile_bucket_probs(self.QUANTILES)
        assert max(probs, key=probs.get) == "74-75"

    def test_probs_form_valid_distribution(self):
        probs = mc.quantile_bucket_probs(self.QUANTILES)
        assert all(0 < p <= 1 for p in probs.values())
        assert 0.9 < sum(probs.values()) <= 1.001

    def test_skew_shows_up_in_tails(self):
        # Fat warm tail: q95 far above median, q05 close below.
        skewed = {0.05: 73.0, 0.5: 74.0, 0.95: 80.0}
        probs = mc.quantile_bucket_probs(skewed)
        warm = sum(p for b, p in probs.items() if int(b.split("-")[0]) >= 76)
        cold = sum(p for b, p in probs.items() if int(b.split("-")[0]) <= 72)
        assert warm > cold

    def test_quantile_crossing_is_repaired(self):
        crossed = {0.05: 71.0, 0.5: 74.0, 0.9: 73.5, 0.95: 74.5}  # q90 < q50
        probs = mc.quantile_bucket_probs(crossed)
        assert probs  # monotone repair keeps it a valid distribution
        assert all(p >= 0 for p in probs.values())

    def test_empty_input(self):
        assert mc.quantile_bucket_probs({}) == {}


class TestNormCdf:
    def test_reference_values(self):
        assert abs(mc.norm_cdf(0, 0, 1) - 0.5) < 1e-9
        assert abs(mc.norm_cdf(1.96, 0, 1) - 0.975) < 1e-3

    def test_degenerate_sigma(self):
        assert mc.norm_cdf(1, 0, 0) == 1.0
        assert mc.norm_cdf(-1, 0, 0) == 0.0
