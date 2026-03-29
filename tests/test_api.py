"""
Tests API pour le service FastAPI.
"""

import shutil

import pytest
from fastapi.testclient import TestClient

import app.model_loader as model_loader_module
from app.main import app
from src.config import MODEL_PATH
import src.inference as inference_module
from tests.test_inference import SAMPLE_LEAD


@pytest.fixture(autouse=True)
def reset_model_loader_state(monkeypatch):
    """Réinitialise le chargeur API et le cache d'inférence entre tests."""
    monkeypatch.setattr(model_loader_module, "_loaded_model_path", None)
    monkeypatch.setattr(inference_module, "_model_cache", {})


def test_health_reports_loaded_legacy_model():
    """Le health check doit refléter le fallback legacy quand le registry est absent."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True
    assert payload["model_path"].endswith("models/tuned_xgb_pipeline.joblib")


def test_health_reports_versioned_model_path_when_registry_is_present(
    tmp_path, monkeypatch
):
    """Le health check doit exposer le vrai chemin versionné servi par l'API."""
    versioned_model_path = tmp_path / "models" / "v1.0.2" / "model.joblib"
    versioned_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODEL_PATH, versioned_model_path)

    monkeypatch.setattr(
        "src.registry.get_production_model_path",
        lambda: versioned_model_path,
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True
    assert payload["model_path"].endswith("models/v1.0.2/model.joblib")


def test_predict_returns_score_priority_and_label():
    """Le scoring unitaire renvoie le contrat métier attendu."""
    with TestClient(app) as client:
        response = client.post("/predict", json=SAMPLE_LEAD)

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"score", "predicted_label", "priority"}
    assert 0 <= payload["score"] <= 1
    assert payload["predicted_label"] in [0, 1]
    assert payload["priority"] in ["high", "medium", "low"]


def test_predict_rejects_invalid_education_value():
    """Une catégorie hors vocabulaire doit produire une 422 exploitable."""
    invalid_payload = {**SAMPLE_LEAD, "education": "doctorate"}

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"][0]["loc"] == ["body", "education"]
    assert "doctorate" in payload["detail"][0]["msg"]


def test_predict_batch_preserves_traceability_and_ranking():
    """Le batch doit garder l'index d'origine et le lead_id optionnel."""
    leads = [
        {**SAMPLE_LEAD, "lead_id": "lead-low", "balance": 100, "age": 25},
        {**SAMPLE_LEAD, "lead_id": "lead-high", "balance": 4000, "age": 60},
        {**SAMPLE_LEAD, "lead_id": "lead-mid", "balance": 900, "age": 40},
    ]

    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"leads": leads})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 3
    assert len(payload["results"]) == 3

    scores = [item["score"] for item in payload["results"]]
    ranks = [item["priority_rank"] for item in payload["results"]]
    input_indexes = {item["input_index"] for item in payload["results"]}
    lead_ids = {item["lead_id"] for item in payload["results"]}

    assert scores == sorted(scores, reverse=True)
    assert ranks == [1, 2, 3]
    assert input_indexes == {0, 1, 2}
    assert lead_ids == {"lead-low", "lead-mid", "lead-high"}


def test_predict_batch_normalizes_integer_lead_id_to_string():
    """Les lead_id numériques doivent être normalisés en string dans la réponse."""
    leads = [{**SAMPLE_LEAD, "lead_id": 12345}]

    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"leads": leads})

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["lead_id"] == "12345"
