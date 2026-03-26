"""
Configuration du logging structuré (JSON).

Fournit des logs JSON pour faciliter l'analyse en production.
Compatible avec ELK, Datadog, CloudWatch, etc.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Formatter qui produit des logs en JSON."""

    def __init__(
        self,
        service_name: str = "ai-lead-scoring",
        include_timestamp: bool = True,
    ):
        super().__init__()
        self.service_name = service_name
        self.include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "service": self.service_name,
        }

        if self.include_timestamp:
            log_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Ajouter le contexte si présent
        if hasattr(record, "context") and record.context:
            log_data["context"] = record.context

        # Ajouter les infos d'exception si présentes
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Ajouter les infos de localisation en mode debug
        if record.levelno <= logging.DEBUG:
            log_data["location"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        return json.dumps(log_data, ensure_ascii=False)


class ContextLogger(logging.LoggerAdapter):
    """Logger qui permet d'ajouter du contexte aux logs."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        # Extraire le contexte des kwargs
        context = kwargs.pop("context", None) or self.extra.get("context", {})

        # Fusionner avec le contexte existant
        if context:
            extra = kwargs.get("extra", {})
            extra["context"] = context
            kwargs["extra"] = extra

        return msg, kwargs


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    service_name: str = "ai-lead-scoring",
) -> logging.Logger:
    """
    Configure le logging pour l'application.

    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR).
        json_format: Si True, utilise le format JSON.
        service_name: Nom du service pour les logs.

    Returns:
        Logger configuré.
    """
    logger = logging.getLogger("lead_scoring")
    logger.setLevel(getattr(logging, level.upper()))

    # Éviter les doublons
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    if json_format:
        handler.setFormatter(JSONFormatter(service_name=service_name))
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    logger.addHandler(handler)

    return logger


def get_logger(
    name: Optional[str] = None, context: Optional[Dict[str, Any]] = None
) -> ContextLogger:
    """
    Obtient un logger avec contexte optionnel.

    Args:
        name: Nom du logger (suffixe après 'lead_scoring').
        context: Contexte par défaut à inclure dans tous les logs.

    Returns:
        ContextLogger prêt à l'emploi.
    """
    logger_name = f"lead_scoring.{name}" if name else "lead_scoring"
    base_logger = logging.getLogger(logger_name)

    return ContextLogger(base_logger, {"context": context or {}})


# Logger pré-configuré pour import direct
def log_prediction(
    lead_id: Optional[str],
    score: float,
    priority: str,
    latency_ms: float,
) -> None:
    """Log une prédiction avec contexte structuré."""
    logger = get_logger("prediction")
    logger.info(
        "Lead scored",
        context={
            "lead_id": lead_id,
            "score": round(score, 4),
            "priority": priority,
            "latency_ms": round(latency_ms, 2),
        },
    )


def log_batch_prediction(
    total_leads: int,
    high_priority_count: int,
    latency_ms: float,
) -> None:
    """Log un batch de prédictions avec contexte structuré."""
    logger = get_logger("prediction")
    logger.info(
        "Batch scored",
        context={
            "total_leads": total_leads,
            "high_priority_count": high_priority_count,
            "high_priority_ratio": round(high_priority_count / total_leads, 4)
            if total_leads > 0
            else 0,
            "latency_ms": round(latency_ms, 2),
        },
    )


def log_model_loaded(model_path: str, load_time_ms: float) -> None:
    """Log le chargement du modèle."""
    logger = get_logger("model")
    logger.info(
        "Model loaded",
        context={
            "model_path": model_path,
            "load_time_ms": round(load_time_ms, 2),
        },
    )


def log_validation_error(error_type: str, details: str) -> None:
    """Log une erreur de validation."""
    logger = get_logger("validation")
    logger.warning(
        "Validation error",
        context={
            "error_type": error_type,
            "details": details,
        },
    )
