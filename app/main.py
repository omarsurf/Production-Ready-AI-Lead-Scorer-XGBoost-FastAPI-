"""
API FastAPI pour le scoring de leads.

Endpoints:
- GET /health : État du service
- POST /predict : Score un lead unique
- POST /predict/batch : Score plusieurs leads avec ranking
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any, List

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.model_loader import (
    get_model,
    get_model_path,
    is_model_loaded,
    load_model_at_startup,
)
from app.schemas import (
    HealthResponse,
    LeadBatchInput,
    LeadBatchResult,
    LeadBatchScoreResponse,
    LeadInput,
    LeadScoreResponse,
)
from src.inference import score_leads
from src.logging_config import (
    log_batch_prediction,
    log_prediction,
    log_validation_error,
    setup_logging,
)
from src.schema import SchemaValidationError

# Configurer le logging (JSON en production, texte en dev)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
JSON_LOGS = os.getenv("JSON_LOGS", "true").lower() == "true"
setup_logging(level=LOG_LEVEL, json_format=JSON_LOGS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage."""
    load_model_at_startup()
    yield


app = FastAPI(
    title="AI Lead Scoring API",
    description="API de priorisation de leads pour optimiser l'effort commercial.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """
    Vérifie l'état du service et du modèle.

    Returns:
        Statut du service et disponibilité du modèle.
    """
    return HealthResponse(
        status="healthy" if is_model_loaded() else "degraded",
        model_loaded=is_model_loaded(),
        model_path=get_model_path(),
    )


@app.post(
    "/predict",
    response_model=LeadScoreResponse,
    response_model_exclude_none=True,
)
def predict(lead: LeadInput):
    """
    Score un lead unique.

    Args:
        lead: Données du lead à scorer.

    Returns:
        Score, label prédit et niveau de priorité.
    """
    start_time = time.time()

    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    scored = _score_or_422(pd.DataFrame([lead.model_dump(exclude_none=True)]), model)
    top_result = scored.iloc[0]

    score = round(float(top_result["score"]), 4)
    priority = str(top_result["priority"])
    latency_ms = (time.time() - start_time) * 1000

    # Log structuré
    log_prediction(
        lead_id=lead.lead_id if hasattr(lead, "lead_id") else None,
        score=score,
        priority=priority,
        latency_ms=latency_ms,
    )

    return LeadScoreResponse(
        score=score,
        predicted_label=int(top_result["predicted_label"]),
        priority=priority,
    )


@app.post(
    "/predict/batch",
    response_model=LeadBatchScoreResponse,
    response_model_exclude_none=True,
)
def predict_batch(batch: LeadBatchInput):
    """
    Score plusieurs leads avec ranking.

    Args:
        batch: Liste de leads à scorer.

    Returns:
        Résultats triés par score décroissant.
    """
    start_time = time.time()

    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    # Convertir en DataFrame avec trace de la position d'origine
    leads_data = []
    for index, lead in enumerate(batch.leads):
        payload = lead.model_dump(exclude_none=True)
        payload["input_index"] = index
        leads_data.append(payload)

    df = pd.DataFrame(leads_data)
    scored = _score_or_422(df, model)

    results: List[LeadBatchResult] = []
    for _, row in scored.iterrows():
        results.append(
            LeadBatchResult(
                input_index=int(row["input_index"]),
                lead_id=_serialize_optional(row.get("lead_id")),
                score=round(float(row["score"]), 4),
                predicted_label=int(row["predicted_label"]),
                priority=str(row["priority"]),
                priority_rank=int(row["priority_rank"]),
            )
        )

    high_count = sum(1 for r in results if r.priority == "high")
    latency_ms = (time.time() - start_time) * 1000

    # Log structuré
    log_batch_prediction(
        total_leads=len(results),
        high_priority_count=high_count,
        latency_ms=latency_ms,
    )

    return LeadBatchScoreResponse(
        results=results,
        total=len(results),
        high_priority_count=high_count,
    )


def _score_or_422(df: pd.DataFrame, model) -> pd.DataFrame:
    """Transforme les erreurs d'entrée/modèle en réponse HTTP exploitable."""
    try:
        return score_leads(df, model=model)
    except (SchemaValidationError, ValueError) as exc:
        log_validation_error(error_type=type(exc).__name__, details=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _serialize_optional(value: Any) -> Any:
    """Convertit les scalaires pandas/numpy en valeurs JSON simples."""
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
