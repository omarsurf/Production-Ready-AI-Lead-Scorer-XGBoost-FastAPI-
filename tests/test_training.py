"""
Tests pour le module d'entraînement reproductible.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.training import evaluate_model, train_model


class DummyModel:
    """Modèle minimal pour tester l'évaluation sans entraîner XGBoost."""

    def __init__(self, scores):
        self._scores = np.asarray(scores, dtype=float)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        scores = self._scores[: len(X)]
        return np.column_stack([1 - scores, scores])


def test_training_evaluate_model_handles_single_class_target():
    """Le rapport d'entraînement doit gérer proprement ROC-AUC non défini."""
    model = DummyModel([0.1, 0.2, 0.3])
    X_test = pd.DataFrame({"feature": [1, 2, 3]})
    y_test = pd.Series([0, 0, 0])

    report = evaluate_model(model, X_test, y_test)

    assert report["classification_metrics"]["roc_auc"] is None
    assert report["classification_metrics"]["accuracy"] == 1.0
    assert report["classification_metrics"]["precision"] == 0.0
    assert report["classification_metrics"]["recall"] == 0.0
    assert report["classification_metrics"]["f1_score"] == 0.0
    assert report["warnings"] == [
        "ROC-AUC non defini: une seule classe est presente dans y_true."
    ]


def test_train_model_writes_strict_json_for_single_class_eval(tmp_path, monkeypatch):
    """Le rapport JSON d'entraînement doit écrire null au lieu de NaN."""
    X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
    X_test = pd.DataFrame({"feature": [5, 6, 7]})
    y_train = pd.Series([0, 1, 0, 1])
    y_test = pd.Series([0, 0, 0])
    model_output = tmp_path / "model.joblib"
    metrics_output = tmp_path / "training_metrics.json"

    monkeypatch.setattr(
        "src.training.load_training_frame",
        lambda input_path: pd.DataFrame({"placeholder": [1]}),
    )
    monkeypatch.setattr(
        "src.training.prepare_training_split",
        lambda df, test_size, random_state: (X_train, X_test, y_train, y_test),
    )
    monkeypatch.setattr(
        "src.training.build_pipeline",
        lambda scale_pos_weight, classifier_params=None, random_state=42: DummyModel(
            [0.2, 0.1, 0.3]
        ),
    )
    monkeypatch.setattr("src.training.joblib.dump", lambda model, path: None)

    result = train_model(
        input_path=Path("ignored.csv"),
        model_output=model_output,
        metrics_output=metrics_output,
        save_model_metadata=False,
    )

    payload = json.loads(metrics_output.read_text())

    assert result["classification_metrics"]["roc_auc"] is None
    assert result["warnings"] == [
        "ROC-AUC non defini: une seule classe est presente dans y_true."
    ]
    assert payload["classification_metrics"]["roc_auc"] is None
    assert payload["warnings"] == [
        "ROC-AUC non defini: une seule classe est presente dans y_true."
    ]
    assert "NaN" not in metrics_output.read_text()
