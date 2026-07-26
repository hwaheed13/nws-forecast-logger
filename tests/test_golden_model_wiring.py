"""Golden tests — model wiring invariants (renovation Phase 1).

These run against the REAL committed artifacts in the repo root, so they catch
the historical failure classes: metadata/pkl architecture mismatch (the 122°F
bug), feature-list duplication, missing quantile head, blend mistuning.
"""
import json
import os
import pickle

import pytest

import model_config as mc


class TestFeatureColumns:
    def test_v16_features_are_unique(self):
        cols = mc.FEATURE_COLS_V16
        assert len(cols) == len(set(cols)), "FEATURE_COLS_V16 has duplicates again"

    def test_v16_lax_matches_nyc_superset(self):
        assert list(mc.FEATURE_COLS_V16_LAX) == list(mc.FEATURE_COLS_V16)


@pytest.mark.parametrize("prefix", ["", "lax_"])
class TestV16Artifacts:
    def test_metadata_declares_residual(self, prefix):
        with open(f"{prefix}model_metadata_v16.json") as f:
            meta = json.load(f)
        assert "residual" in meta["version"], (
            "v16 metadata no longer residual — inference will treat the pkl "
            "as DIRECT and stop anchoring on HRRR"
        )
        assert meta["v16_regression"]["improvement_vs_hrrr_alone"] > 0, (
            "the moat metric went non-positive — investigate before shipping"
        )

    def test_feature_cols_pkl_matches_metadata(self, prefix):
        with open(f"{prefix}bcp_v16_feature_cols.pkl", "rb") as f:
            cols = pickle.load(f)
        with open(f"{prefix}model_metadata_v16.json") as f:
            meta = json.load(f)
        assert cols == meta["feature_columns_v16"], (
            "saved feature-cols pkl and metadata disagree — inference matrix "
            "will be built against the wrong column order"
        )

    def test_quantile_head_present_and_calibrated(self, prefix):
        path = f"{prefix}bcp_v16_quantiles.pkl"
        assert os.path.exists(path), "quantile head pkl missing"
        with open(path, "rb") as f:
            pack = pickle.load(f)
        assert set(pack["models"]) == set(pack["quantiles"])
        assert "conformal_offsets" in pack, "conformal offsets missing"
        with open(f"{prefix}model_metadata_v16.json") as f:
            meta = json.load(f)
        cov = meta.get("v16_quantiles", {}).get("holdout_coverage_conformal", {})
        if cov:  # calibration within ±7pts of nominal at the tails
            assert abs(cov["0.1"] - 0.10) < 0.07, f"q10 miscalibrated: {cov['0.1']}"
            assert abs(cov["0.9"] - 0.90) < 0.07, f"q90 miscalibrated: {cov['0.9']}"


class TestBlendConfig:
    def test_scale_stays_at_validated_optimum(self):
        import predictor_blend as pb
        # Re-validated 2026-07-26 on official actuals: scale 10 was WORSE than
        # HRRR alone; 5-6 optimal. Changing this requires re-running
        # predictor_blend_research.py and updating this golden with the number.
        assert pb.BLEND_SCALE == 6.0

    def test_weight_formula_bounds(self):
        import predictor_blend as pb
        # weight_knn = max(MIN, 1 - std/scale), clamped to [MIN, 1]
        assert pb.MIN_KNN_WEIGHT == 0.10


class TestInferenceArchitectureDetection:
    def test_loaded_v16_detects_residual(self):
        import prediction_writer as pw
        pw._load_v16_models()
        assert pw._v16_is_residual() is True, (
            "inference thinks v16 is DIRECT — the 122°F failure precondition"
        )

    def test_quantile_pack_loads(self):
        import prediction_writer as pw
        pw._load_v16_models()
        pack = pw._v16_quantiles()
        assert pack is not None and "conformal_offsets" in pack
