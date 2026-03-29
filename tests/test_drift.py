"""Tests for src/drift.py - PSI-based drift detection."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.drift import (
    DriftReport,
    FeatureDrift,
    PSI_THRESHOLD_MAJOR,
    PSI_THRESHOLD_MINOR,
    compute_categorical_psi,
    compute_numeric_psi,
    detect_drift,
    load_reference_distributions,
    resolve_reference_path,
    save_reference_distributions,
    _get_drift_level,
)


class TestNumericPSI:
    """Tests for numeric PSI computation."""

    def test_identical_distributions_near_zero(self):
        """Identical distributions should have PSI close to 0."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        psi, _, _, _ = compute_numeric_psi(data, data)

        assert psi < 0.01

    def test_shifted_distribution_detects_drift(self):
        """Shifted distribution should have high PSI."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(3, 1, 1000)  # Mean shifted by 3 std

        psi, _, _, _ = compute_numeric_psi(expected, actual)

        assert psi > PSI_THRESHOLD_MAJOR

    def test_empty_arrays_return_zero(self):
        """Empty arrays should return PSI of 0."""
        psi, _, _, _ = compute_numeric_psi(np.array([]), np.array([]))
        assert psi == 0.0

    def test_handles_nan_values(self):
        """NaN values should be ignored."""
        expected = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        actual = np.array([1.0, 2.0, 3.0, np.nan, 5.0])

        psi, _, _, _ = compute_numeric_psi(expected, actual)

        # Should compute without error
        assert psi >= 0

    def test_returns_bin_info(self):
        """Should return bin edges and percentages."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        psi, bin_edges, exp_pct, act_pct = compute_numeric_psi(data, data, n_bins=5)

        assert len(bin_edges) == 6  # n_bins + 1
        assert len(exp_pct) == 5
        assert len(act_pct) == 5


class TestCategoricalPSI:
    """Tests for categorical PSI computation."""

    def test_identical_distributions_near_zero(self):
        """Identical distributions should have PSI close to 0."""
        data = pd.Series(["a", "b", "c"] * 100)

        psi, _, _ = compute_categorical_psi(data, data)

        assert psi < 0.01

    def test_new_category_detected(self):
        """New category in actual should contribute to drift."""
        expected = pd.Series(["a", "b", "c"] * 100)
        actual = pd.Series(["a", "b", "d"] * 100)  # "d" is new, "c" is gone

        psi, exp_pct, act_pct = compute_categorical_psi(expected, actual)

        assert psi > 0
        assert "d" in act_pct
        assert act_pct["d"] > exp_pct.get("d", 0)

    def test_handles_missing_values(self):
        """NaN values should be treated as __MISSING__ category."""
        expected = pd.Series(["a", "b", None, "a", "b"])
        actual = pd.Series(["a", None, None, "a", "b"])

        psi, exp_pct, act_pct = compute_categorical_psi(expected, actual)

        assert "__MISSING__" in exp_pct
        assert "__MISSING__" in act_pct
        assert psi >= 0

    def test_returns_proportions(self):
        """Should return category proportions."""
        expected = pd.Series(["a", "a", "b"])
        actual = pd.Series(["a", "b", "b"])

        psi, exp_pct, act_pct = compute_categorical_psi(expected, actual)

        assert "a" in exp_pct
        assert "b" in exp_pct


class TestDriftLevel:
    """Tests for drift level classification."""

    def test_no_drift(self):
        assert _get_drift_level(0.05) == "none"

    def test_minor_drift(self):
        assert _get_drift_level(0.15) == "minor"

    def test_major_drift(self):
        assert _get_drift_level(0.30) == "major"

    def test_threshold_boundaries(self):
        assert _get_drift_level(PSI_THRESHOLD_MINOR - 0.001) == "none"
        assert _get_drift_level(PSI_THRESHOLD_MINOR) == "minor"
        assert _get_drift_level(PSI_THRESHOLD_MAJOR - 0.001) == "minor"
        assert _get_drift_level(PSI_THRESHOLD_MAJOR) == "major"


class TestReferenceDistributions:
    """Tests for saving and loading reference distributions."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Reference distributions should survive save/load."""
        output_path = tmp_path / "ref_dist.json"

        df = pd.DataFrame(
            {
                "age": [25, 30, 35, 40, 45],
                "balance": [1000, 2000, 3000, 4000, 5000],
                "job": ["admin", "technician", "admin", "services", "admin"],
                "education": [
                    "primary",
                    "secondary",
                    "tertiary",
                    "secondary",
                    "primary",
                ],
            }
        )

        save_reference_distributions(
            df,
            output_path,
            numeric_features=["age", "balance"],
            categorical_features=["job", "education"],
        )

        loaded = load_reference_distributions(output_path)

        assert "numeric_features" in loaded
        assert "categorical_features" in loaded
        assert "age" in loaded["numeric_features"]
        assert "job" in loaded["categorical_features"]

    def test_load_nonexistent_raises(self, tmp_path):
        """Loading non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_reference_distributions(tmp_path / "nonexistent.json")


class TestReferenceResolution:
    """Tests for registry-aware reference path resolution."""

    def test_uses_production_registry_reference_when_available(
        self, tmp_path, monkeypatch
    ):
        """The production model reference should take precedence over legacy fallback."""
        reference_path = tmp_path / "models" / "v1.0.1" / "reference_distributions.json"
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text("{}")

        monkeypatch.setattr(
            "src.registry.get_production_model",
            lambda: SimpleNamespace(
                version="1.0.1",
                reference_distributions_path=str(reference_path),
            ),
        )

        resolved_path, model_version = resolve_reference_path()

        assert resolved_path == reference_path.resolve()
        assert model_version == "1.0.1"

    def test_falls_back_to_legacy_reference_when_registry_is_missing(
        self, tmp_path, monkeypatch
    ):
        """Legacy reference should be used when no production registry entry exists."""
        legacy_path = tmp_path / "reference_distributions.json"
        legacy_path.write_text("{}")

        monkeypatch.setattr("src.registry.get_production_model", lambda: None)
        monkeypatch.setattr(
            "src.drift.LEGACY_REFERENCE_DISTRIBUTIONS_PATH",
            legacy_path,
        )

        resolved_path, model_version = resolve_reference_path()

        assert resolved_path == legacy_path.resolve()
        assert model_version is None

    def test_explicit_reference_overrides_registry(self, tmp_path, monkeypatch):
        """An explicit operator-provided reference must bypass the registry lookup."""
        explicit_path = tmp_path / "custom_reference.json"
        explicit_path.write_text("{}")

        monkeypatch.setattr(
            "src.registry.get_production_model",
            lambda: SimpleNamespace(
                version="9.9.9",
                reference_distributions_path="/does/not/matter.json",
            ),
        )

        resolved_path, model_version = resolve_reference_path(explicit_path)

        assert resolved_path == explicit_path.resolve()
        assert model_version is None

    def test_raises_clear_error_when_no_reference_exists(self, monkeypatch, tmp_path):
        """Missing registry and legacy references should raise a clear error."""
        monkeypatch.setattr("src.registry.get_production_model", lambda: None)
        monkeypatch.setattr(
            "src.drift.LEGACY_REFERENCE_DISTRIBUTIONS_PATH",
            tmp_path / "missing_reference.json",
        )

        with pytest.raises(FileNotFoundError, match="No reference distributions found"):
            resolve_reference_path()


class TestDetectDrift:
    """Tests for the main detect_drift function."""

    def test_detects_no_drift_on_same_data(self, tmp_path):
        """Same data should show no drift."""
        ref_path = tmp_path / "ref_dist.json"

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "age": np.random.normal(35, 10, 200),
                "balance": np.random.normal(5000, 2000, 200),
                "job": np.random.choice(["admin", "tech", "services"], 200),
            }
        )

        save_reference_distributions(
            df,
            ref_path,
            numeric_features=["age", "balance"],
            categorical_features=["job"],
        )

        report = detect_drift(df, ref_path)

        assert report.overall_status == "healthy"
        assert len(report.major_drift_features) == 0
        assert len(report.minor_drift_features) == 0

    def test_detects_major_drift(self, tmp_path):
        """Shifted data should show major drift."""
        ref_path = tmp_path / "ref_dist.json"

        np.random.seed(42)
        df_train = pd.DataFrame(
            {
                "age": np.random.normal(35, 10, 200),
            }
        )
        df_inference = pd.DataFrame(
            {
                "age": np.random.normal(60, 10, 200),  # Major shift
            }
        )

        save_reference_distributions(
            df_train,
            ref_path,
            numeric_features=["age"],
            categorical_features=[],
        )

        report = detect_drift(df_inference, ref_path)

        assert report.overall_status == "critical"
        assert "age" in report.major_drift_features

    def test_sample_size_too_small_raises(self, tmp_path):
        """Should raise ValueError when sample size is too small."""
        ref_path = tmp_path / "ref_dist.json"

        df = pd.DataFrame({"age": [25, 30, 35]})
        save_reference_distributions(
            df, ref_path, numeric_features=["age"], categorical_features=[]
        )

        small_df = pd.DataFrame({"age": [25, 30]})

        with pytest.raises(ValueError, match="Sample size"):
            detect_drift(small_df, ref_path, min_sample_size=100)

    def test_report_contains_all_features(self, tmp_path):
        """Report should contain all checked features."""
        ref_path = tmp_path / "ref_dist.json"

        df = pd.DataFrame(
            {
                "age": np.random.normal(35, 10, 200),
                "balance": np.random.normal(5000, 2000, 200),
                "job": np.random.choice(["admin", "tech"], 200),
            }
        )

        save_reference_distributions(
            df,
            ref_path,
            numeric_features=["age", "balance"],
            categorical_features=["job"],
        )

        report = detect_drift(df, ref_path)

        feature_names = [f.feature_name for f in report.features]
        assert "age" in feature_names
        assert "balance" in feature_names
        assert "job" in feature_names

    def test_report_keeps_model_version_when_provided(self, tmp_path):
        """The drift report should expose the production model version used."""
        ref_path = tmp_path / "ref_dist.json"

        df = pd.DataFrame(
            {
                "age": np.random.normal(35, 10, 200),
            }
        )
        save_reference_distributions(
            df,
            ref_path,
            numeric_features=["age"],
            categorical_features=[],
        )

        report = detect_drift(df, ref_path, model_version="1.0.7")

        assert report.model_version == "1.0.7"


class TestDriftReport:
    """Tests for DriftReport dataclass."""

    def test_to_dict(self):
        """Should serialize to dict properly."""
        report = DriftReport(
            timestamp="2024-01-01T00:00:00Z",
            model_version="1.0.0",
            sample_size=100,
            features=[
                FeatureDrift(
                    feature_name="age",
                    feature_type="numeric",
                    psi=0.05,
                    drift_level="none",
                )
            ],
            overall_status="healthy",
            major_drift_features=[],
            minor_drift_features=[],
        )

        d = report.to_dict()

        assert d["timestamp"] == "2024-01-01T00:00:00Z"
        assert d["model_version"] == "1.0.0"
        assert len(d["features"]) == 1
        assert d["features"][0]["feature_name"] == "age"


class TestEdgeCases:
    """Edge case tests."""

    def test_missing_column_in_current(self, tmp_path):
        """Should handle missing columns gracefully."""
        ref_path = tmp_path / "ref_dist.json"

        df_train = pd.DataFrame(
            {
                "age": [25, 30, 35, 40] * 50,
                "balance": [1000, 2000, 3000, 4000] * 50,
            }
        )
        df_inference = pd.DataFrame(
            {
                "age": [25, 30, 35, 40] * 50,
                # balance is missing
            }
        )

        save_reference_distributions(
            df_train,
            ref_path,
            numeric_features=["age", "balance"],
            categorical_features=[],
        )

        # Should not crash, just skip missing column
        report = detect_drift(df_inference, ref_path)

        feature_names = [f.feature_name for f in report.features]
        assert "age" in feature_names
        assert "balance" not in feature_names  # Skipped

    def test_all_nan_column(self, tmp_path):
        """Should handle all-NaN columns."""
        ref_path = tmp_path / "ref_dist.json"

        df_train = pd.DataFrame(
            {
                "age": [25, 30, 35, 40] * 50,
            }
        )
        df_inference = pd.DataFrame(
            {
                "age": [np.nan] * 200,
            }
        )

        save_reference_distributions(
            df_train,
            ref_path,
            numeric_features=["age"],
            categorical_features=[],
        )

        # Should not crash
        report = detect_drift(df_inference, ref_path)
        assert report is not None
