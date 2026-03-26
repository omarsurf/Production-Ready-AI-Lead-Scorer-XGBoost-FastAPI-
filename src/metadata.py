"""
Gestion des métadonnées du modèle.

Stocke et charge les informations sur le modèle :
- Date d'entraînement
- Métriques de performance
- Schéma des features attendues
- Hyperparamètres
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import (
    MODELS_DIR,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    REQUIRED_COLUMNS,
    TARGET,
)


METADATA_PATH = MODELS_DIR / "model_metadata.json"
SCHEMA_PATH = MODELS_DIR / "input_schema.json"


@dataclass
class ModelMetadata:
    """Métadonnées complètes du modèle."""

    # Identification
    model_name: str = "tuned_xgb_pipeline"
    model_version: str = "1.0.0"

    # Temporel
    training_date: str = ""
    training_duration_seconds: float = 0.0

    # Dataset
    training_samples: int = 0
    test_samples: int = 0
    target_column: str = TARGET
    positive_class_ratio: float = 0.0

    # Performance
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Hyperparamètres
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Features
    feature_count: int = len(REQUIRED_COLUMNS)
    numeric_features: List[str] = field(default_factory=lambda: NUMERIC_FEATURES.copy())
    ordinal_features: List[str] = field(default_factory=lambda: ORDINAL_FEATURES.copy())
    nominal_features: List[str] = field(default_factory=lambda: NOMINAL_FEATURES.copy())

    # Artifact
    model_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Crée une instance depuis un dictionnaire."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def save_metadata(metadata: ModelMetadata, path: Optional[Path] = None) -> Path:
    """
    Sauvegarde les métadonnées du modèle.

    Args:
        metadata: Métadonnées à sauvegarder.
        path: Chemin de destination (défaut: METADATA_PATH).

    Returns:
        Chemin du fichier créé.
    """
    path = path or METADATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False, allow_nan=False)

    return path


def load_metadata(path: Optional[Path] = None) -> Optional[ModelMetadata]:
    """
    Charge les métadonnées du modèle.

    Args:
        path: Chemin du fichier (défaut: METADATA_PATH).

    Returns:
        ModelMetadata ou None si fichier absent.
    """
    path = path or METADATA_PATH

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ModelMetadata.from_dict(data)


def generate_input_schema() -> Dict[str, Any]:
    """
    Génère le schéma JSON des features attendues.

    Returns:
        Schéma JSON Schema compatible.
    """
    from src.config import ALLOWED_CATEGORICAL_VALUES

    properties = {}

    # Features numériques
    for feat in NUMERIC_FEATURES:
        properties[feat] = {
            "type": "number",
            "description": f"Feature numérique: {feat}",
        }

    # Features ordinales
    for feat in ORDINAL_FEATURES:
        allowed = ALLOWED_CATEGORICAL_VALUES.get(feat, [])
        properties[feat] = {
            "type": "string",
            "enum": allowed,
            "description": f"Feature ordinale: {feat}",
        }

    # Features nominales
    for feat in NOMINAL_FEATURES:
        allowed = ALLOWED_CATEGORICAL_VALUES.get(feat, [])
        properties[feat] = {
            "type": "string",
            "enum": allowed if allowed else None,
            "description": f"Feature nominale: {feat}",
        }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "LeadInput",
        "description": "Schéma des features pour le scoring de leads",
        "type": "object",
        "properties": properties,
        "required": REQUIRED_COLUMNS,
        "additionalProperties": True,
    }


def save_input_schema(path: Optional[Path] = None) -> Path:
    """
    Sauvegarde le schéma JSON des features.

    Args:
        path: Chemin de destination (défaut: SCHEMA_PATH).

    Returns:
        Chemin du fichier créé.
    """
    path = path or SCHEMA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    schema = generate_input_schema()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    return path


def create_training_metadata(
    training_samples: int,
    test_samples: int,
    positive_ratio: float,
    metrics: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    model_path: str,
    training_duration: float = 0.0,
    model_version: str = "1.0.0",
) -> ModelMetadata:
    """
    Crée les métadonnées après un entraînement.

    Args:
        training_samples: Nombre d'échantillons d'entraînement.
        test_samples: Nombre d'échantillons de test.
        positive_ratio: Ratio de la classe positive.
        metrics: Métriques de performance (roc_auc, precision, etc.).
        hyperparameters: Hyperparamètres du modèle.
        model_path: Chemin vers l'artifact.
        training_duration: Durée de l'entraînement en secondes.
        model_version: Version du modèle.

    Returns:
        ModelMetadata prêt à être sauvegardé.
    """
    return ModelMetadata(
        model_version=model_version,
        training_date=datetime.now().isoformat(),
        training_duration_seconds=round(training_duration, 2),
        training_samples=training_samples,
        test_samples=test_samples,
        positive_class_ratio=round(positive_ratio, 4),
        metrics=metrics,
        hyperparameters=hyperparameters,
        model_path=str(model_path),
    )
