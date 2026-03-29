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
    MODEL_CACHE_SIZE,
    OUTPUTS_DIR,
    PRIORITY_THRESHOLDS,
    REQUIRED_COLUMNS,
)
from src.schema import validate_input


_model_cache: Dict[Path, Any] = {}


def _normalize_model_path(model_path: Optional[Path] = None) -> Path:
    """Normalise le chemin du modèle pour une utilisation sûre dans le cache."""
    if model_path is not None:
        return Path(model_path).expanduser().resolve()

    # Try registry first, fallback to legacy MODEL_PATH
    try:
        from src.registry import get_production_model_path

        return get_production_model_path()
    except (ImportError, FileNotFoundError):
        return MODEL_PATH.expanduser().resolve()


def resolve_model_path(model_path: Optional[Path] = None) -> Path:
    """Expose la résolution du modèle actif pour les workflows batch/reporting."""
    return _normalize_model_path(model_path)


def clear_model_cache(model_path: Optional[Path] = None) -> None:
    """Vide tout le cache ou une seule entrée si un chemin est fourni."""
    if model_path is None:
        _model_cache.clear()
        return

    _model_cache.pop(_normalize_model_path(model_path), None)


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

    if len(_model_cache) >= MODEL_CACHE_SIZE:
        _model_cache.pop(next(iter(_model_cache)))

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
    parser.add_argument(
        "--check-drift",
        action="store_true",
        help="Check for data drift before scoring (batch only)",
    )
    parser.add_argument(
        "--drift-report",
        help="Path to save drift report JSON (requires --check-drift)",
    )
    args = parser.parse_args()

    # Check drift if requested
    if args.check_drift:
        try:
            from src.drift import detect_drift, resolve_reference_path

            # Load input data
            input_path = Path(args.input)
            with open(input_path) as f:
                first_line = f.readline()
            sep = ";" if ";" in first_line else ","
            df = pd.read_csv(input_path, sep=sep)

            resolved_reference_path, model_version = resolve_reference_path()
            try:
                report = detect_drift(
                    X_current=df,
                    reference_path=resolved_reference_path,
                    model_version=model_version,
                )

                print(f"Drift status: {report.overall_status.upper()}")
                if report.major_drift_features:
                    print(f"⚠ Major drift: {', '.join(report.major_drift_features)}")

                if args.drift_report:
                    import json

                    report_path = Path(args.drift_report)
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(report_path, "w") as f:
                        json.dump(report.to_dict(), f, indent=2)
                    print(f"✓ Drift report: {report_path}")

            except ValueError as e:
                print(f"⚠ Drift check skipped: {e}")

        except FileNotFoundError as e:
            print(f"⚠ Drift check skipped: {e}")
        except Exception as e:
            print(f"⚠ Drift check failed: {e}")

    score_csv(args.input, args.output)


if __name__ == "__main__":
    main()
