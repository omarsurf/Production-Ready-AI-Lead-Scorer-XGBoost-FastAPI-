"""
Tests API pour le service FastAPI.
"""

from fastapi.testclient import TestClient

from app.main import app
from tests.test_inference import SAMPLE_LEAD


def test_health_reports_loaded_model():
    """Le health check doit refléter un modèle chargé au démarrage."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True
    assert payload["model_path"].endswith("models/tuned_xgb_pipeline.joblib")


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
