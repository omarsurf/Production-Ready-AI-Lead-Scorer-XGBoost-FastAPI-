"""
Module d'inférence pour le scoring de leads.

Fonctions principales:
- load_model(): charge le pipeline
- score_leads(): score un DataFrame
- score_csv(): CLI pour batch scoring
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.config import (
    CLASSIFICATION_THRESHOLD,
    EXCLUDED_FEATURES,
    MODEL_PATH,
    OUTPUTS_DIR,
    PRIORITY_THRESHOLDS,
    REQUIRED_COLUMNS,
)
from src.schema import validate_input


_model_cache: Dict[Path, Any] = {}


def _normalize_model_path(model_path: Optional[Path] = None) -> Path:
    """Normalise le chemin du modèle pour une utilisation sûre dans le cache."""
    return Path(model_path or MODEL_PATH).expanduser().resolve()


def load_model(model_path: Optional[Path] = None):
    """
    Charge le pipeline de scoring.

    Args:
        model_path: Chemin vers le modèle (défaut: MODEL_PATH).

    Returns:
        Pipeline sklearn chargé.
    """
    path = _normalize_model_path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {path}")

    if path in _model_cache:
        return _model_cache[path]

    _model_cache[path] = joblib.load(path)
    return _model_cache[path]


def score_leads(
    df: pd.DataFrame,
    model=None,
    include_priority: bool = True,
) -> pd.DataFrame:
    """
    Score un DataFrame de leads et retourne les résultats avec ranking.

    Args:
        df: DataFrame contenant les features des leads.
        model: Pipeline sklearn (optionnel, chargé si None).
        include_priority: Ajouter la colonne priority (high/medium/low).

    Returns:
        DataFrame avec colonnes ajoutées:
        - score: probabilité de conversion [0, 1]
        - predicted_label: 0 ou 1
        - priority_rank: rang (1 = meilleur lead)
        - priority: high/medium/low (si include_priority=True)
    """
    # Validation du schéma
    validate_input(df)

    # Chargement du modèle si nécessaire
    if model is None:
        model = load_model()

    # Préparer les features (exclure duration si présent)
    X = df.drop(
        columns=[c for c in EXCLUDED_FEATURES if c in df.columns],
        errors="ignore",
    )
    X = X[REQUIRED_COLUMNS]

    # Scoring
    result = df.copy()
    result["score"] = model.predict_proba(X)[:, 1]
    result["predicted_label"] = (result["score"] >= CLASSIFICATION_THRESHOLD).astype(
        int
    )
    result["priority_rank"] = (
        result["score"].rank(ascending=False, method="first").astype(int)
    )

    # Ajout de la priorité textuelle
    if include_priority:
        result["priority"] = result["score"].apply(_get_priority)

    # Tri par score décroissant
    return result.sort_values("priority_rank")


def _get_priority(score: float) -> str:
    """Convertit un score en niveau de priorité."""
    if score >= PRIORITY_THRESHOLDS["high"]:
        return "high"
    elif score >= PRIORITY_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def score_csv(input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Score un fichier CSV de leads.

    Args:
        input_path: Chemin vers le CSV d'entrée.
        output_path: Chemin vers le CSV de sortie (optionnel).

    Returns:
        DataFrame scoré.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {input_path}")

    # Détecter le séparateur
    with open(input_path) as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","

    df = pd.read_csv(input_path, sep=sep)
    scored = score_leads(df)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_path, index=False)
        print(f"✓ {len(scored)} leads scorés → {output_path}")

    return scored


def main():
    """Point d'entrée CLI pour le batch scoring."""
    parser = argparse.ArgumentParser(
        description="Score des leads depuis un fichier CSV"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Chemin vers le CSV de leads à scorer",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(OUTPUTS_DIR / "scored_leads.csv"),
        help="Chemin vers le CSV de sortie",
    )
    args = parser.parse_args()

    score_csv(args.input, args.output)


if __name__ == "__main__":
    main()
