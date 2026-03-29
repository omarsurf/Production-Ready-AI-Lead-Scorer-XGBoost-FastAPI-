"""Tests for src/metrics.py - Precision@K scorer and business metrics."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.metrics import (
    get_tuning_scorers,
    make_precision_at_k_scorer,
    precision_at_k_proba,
    precision_at_k_score,
)


class TestPrecisionAtKScore:
    """Tests for precision_at_k_score function."""

    def test_perfect_ranking_captures_all_positives(self):
        """Perfect ranking should give P@K = 1.0 when K captures all positives."""
        y_true = np.array([1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])

        # Top 40% = 2 samples, both are positive
        result = precision_at_k_score(y_true, y_score, k_fraction=0.4)
        assert result == 1.0

    def test_worst_ranking_gives_zero(self):
        """Worst ranking (positives at bottom) should give P@K = 0."""
        y_true = np.array([1, 1, 0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.7, 0.8, 0.9])  # Positives ranked last

        # Top 40% = 2 samples, both are negative
        result = precision_at_k_score(y_true, y_score, k_fraction=0.4)
        assert result == 0.0

    def test_small_dataset_minimum_k_is_one(self):
        """K should be at least 1 even for tiny datasets."""
        y_true = np.array([1, 0])
        y_score = np.array([0.9, 0.1])

        # 10% of 2 = 0.2 -> rounds to 1
        result = precision_at_k_score(y_true, y_score, k_fraction=0.1)
        assert result == 1.0

    def test_empty_array_returns_zero(self):
        """Empty arrays should return 0.0 without crashing."""
        y_true = np.array([])
        y_score = np.array([])

        result = precision_at_k_score(y_true, y_score, k_fraction=0.1)
        assert result == 0.0

    def test_all_positive_class(self):
        """All positive samples should give P@K = 1.0 for any K."""
        y_true = np.array([1, 1, 1, 1, 1])
        y_score = np.array([0.5, 0.6, 0.4, 0.7, 0.3])

        result = precision_at_k_score(y_true, y_score, k_fraction=0.2)
        assert result == 1.0

    def test_all_negative_class(self):
        """All negative samples should give P@K = 0.0 for any K."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_score = np.array([0.5, 0.6, 0.4, 0.7, 0.3])

        result = precision_at_k_score(y_true, y_score, k_fraction=0.2)
        assert result == 0.0

    def test_tied_scores_deterministic(self):
        """Tied scores should produce deterministic results."""
        y_true = np.array([1, 0, 1, 0])
        y_score = np.array([0.5, 0.5, 0.5, 0.5])  # All tied

        # Run multiple times to verify determinism
        results = [
            precision_at_k_score(y_true, y_score, k_fraction=0.5) for _ in range(5)
        ]
        assert len(set(results)) == 1  # All results should be the same


class TestPrecisionAtKProba:
    """Tests for precision_at_k_proba (predict_proba compatibility)."""

    def test_handles_2d_proba(self):
        """Should extract positive class from 2D predict_proba output."""
        y_true = np.array([1, 1, 0, 0])
        y_proba = np.array([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3], [0.8, 0.2]])

        result = precision_at_k_proba(y_true, y_proba, k_fraction=0.5)
        assert result == 1.0  # Top 2 have highest positive proba

    def test_handles_1d_proba(self):
        """Should work with 1D probability array."""
        y_true = np.array([1, 1, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.3, 0.2])

        result = precision_at_k_proba(y_true, y_proba, k_fraction=0.5)
        assert result == 1.0


class TestMakePrecisionAtKScorer:
    """Tests for sklearn scorer compatibility."""

    def test_scorer_with_cross_val_score(self):
        """Scorer should work with sklearn cross_val_score."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        scorer = make_precision_at_k_scorer(0.2)
        model = LogisticRegression(random_state=42)

        scores = cross_val_score(model, X, y, cv=3, scoring=scorer)

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_scorer_with_pipeline(self):
        """Scorer should work with sklearn Pipeline."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=42)),
            ]
        )
        scorer = make_precision_at_k_scorer(0.3)

        scores = cross_val_score(pipeline, X, y, cv=2, scoring=scorer)

        assert len(scores) == 2
        assert all(0 <= s <= 1 for s in scores)

    def test_different_k_fractions(self):
        """Different k_fractions should produce different scorers."""
        scorer_10 = make_precision_at_k_scorer(0.1)
        scorer_50 = make_precision_at_k_scorer(0.5)

        # They should be different objects
        assert scorer_10 is not scorer_50


class TestGetTuningScorers:
    """Tests for get_tuning_scorers helper."""

    def test_returns_dict_and_refit_metric(self):
        """Should return scoring dict and refit metric name."""
        scoring, refit = get_tuning_scorers(0.1)

        assert isinstance(scoring, dict)
        assert "roc_auc" in scoring
        assert "precision_at_10" in scoring
        assert refit == "precision_at_10"

    def test_different_k_values(self):
        """Should generate correct metric names for different k values."""
        scoring_10, refit_10 = get_tuning_scorers(0.10)
        scoring_20, refit_20 = get_tuning_scorers(0.20)

        assert "precision_at_10" in scoring_10
        assert refit_10 == "precision_at_10"

        assert "precision_at_20" in scoring_20
        assert refit_20 == "precision_at_20"


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_single_sample(self):
        """Should handle single sample gracefully."""
        y_true = np.array([1])
        y_score = np.array([0.5])

        result = precision_at_k_score(y_true, y_score, k_fraction=0.5)
        assert result == 1.0

    def test_k_fraction_one(self):
        """k_fraction=1.0 should consider all samples."""
        y_true = np.array([1, 0, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        result = precision_at_k_score(y_true, y_score, k_fraction=1.0)
        assert result == 0.4  # 2/5 positives

    def test_k_fraction_very_small(self):
        """Very small k_fraction should still work (k=1)."""
        y_true = np.array([0, 0, 0, 0, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.9])

        result = precision_at_k_score(y_true, y_score, k_fraction=0.001)
        assert result == 1.0  # Top 1 sample is positive
