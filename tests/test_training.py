"""
Tests pour le module d'entraînement reproductible.
"""

import json
from pathlib import Path
from types import SimpleNamespace

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


def test_train_model_passes_flattened_metrics_to_registry(tmp_path, monkeypatch):
    """L'entraînement doit transmettre des métriques aplaties au registry."""
    X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
    X_test = pd.DataFrame({"feature": [5, 6, 7, 8]})
    y_train = pd.Series([0, 1, 0, 1])
    y_test = pd.Series([0, 1, 1, 0])
    model_output = tmp_path / "model.joblib"
    metrics_output = tmp_path / "training_metrics.json"
    captured_registry_call = {}

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
            [0.9, 0.8, 0.2, 0.1]
        ),
    )
    monkeypatch.setattr("src.training.joblib.dump", lambda model, path: None)
    monkeypatch.setattr(
        "src.training.save_metadata",
        lambda metadata: tmp_path / "metadata.json",
    )
    monkeypatch.setattr(
        "src.training.save_input_schema",
        lambda: tmp_path / "input_schema.json",
    )

    def fake_register_model(**kwargs):
        captured_registry_call.update(kwargs)
        return SimpleNamespace(version="1.0.0")

    monkeypatch.setattr("src.registry.register_model", fake_register_model)

    train_model(
        input_path=Path("ignored.csv"),
        model_output=model_output,
        metrics_output=metrics_output,
        save_model_metadata=True,
    )

    assert "metrics" in captured_registry_call
    assert "roc_auc" in captured_registry_call["metrics"]
    assert "precision_at_10" in captured_registry_call["metrics"]
    assert "uplift_at_10" in captured_registry_call["metrics"]


def test_train_model_continues_when_registry_registration_fails(tmp_path, monkeypatch):
    """Un échec du registry ne doit pas faire échouer l'entraînement."""
    X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
    X_test = pd.DataFrame({"feature": [5, 6, 7]})
    y_train = pd.Series([0, 1, 0, 1])
    y_test = pd.Series([0, 1, 1])
    model_output = tmp_path / "model.joblib"

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
            [0.8, 0.7, 0.1]
        ),
    )
    monkeypatch.setattr("src.training.joblib.dump", lambda model, path: None)
    monkeypatch.setattr(
        "src.training.save_metadata",
        lambda metadata: tmp_path / "metadata.json",
    )
    monkeypatch.setattr(
        "src.training.save_input_schema",
        lambda: tmp_path / "input_schema.json",
    )
    monkeypatch.setattr(
        "src.registry.register_model",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("registry unavailable")),
    )

    result = train_model(
        input_path=Path("ignored.csv"),
        model_output=model_output,
        save_model_metadata=True,
    )

    assert result["classification_metrics"]["roc_auc"] is not None
    assert result["model_output"] == str(model_output)
