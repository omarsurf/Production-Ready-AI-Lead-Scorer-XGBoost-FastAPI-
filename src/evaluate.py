"""
Module d'évaluation business pour le scoring de leads.

Métriques calculées :
- ROC-AUC, Precision, Recall, F1
- Precision@K pour différents K
- Uplift vs baseline (taux de conversion moyen)
- Cumulative Gain (% conversions captées en contactant X% des leads)

Usage CLI :
    python -m src.evaluate --input data/test.csv --output outputs/evaluation_report.json
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    CLASSIFICATION_THRESHOLD,
    EXCLUDED_FEATURES,
    MODEL_PATH,
    OUTPUTS_DIR,
    REQUIRED_COLUMNS,
    TARGET,
    TOP_K_FRACTIONS,
)
from src.inference import load_model
from src.schema import validate_input


@dataclass
class EvaluationReport:
    """Rapport d'évaluation complet."""

    # Métriques classiques
    roc_auc: Optional[float]
    accuracy: float
    precision: float
    recall: float
    f1: float

    # Métriques business
    baseline_conversion_rate: float
    precision_at_k: Dict[str, float]
    uplift_at_k: Dict[str, float]
    cumulative_gain: Dict[str, float]

    # Metadata
    total_samples: int
    total_conversions: int
    model_path: str
    warnings: List[str]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Tuple[Dict[str, Optional[float]], List[str]]:
    """
    Calcule les métriques de classification avec gestion explicite du cas mono-classe.

    Args:
        y_true: Labels réels binaires.
        y_scores: Scores de probabilité prédits.

    Returns:
        Tuple (metrics, warnings).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    y_pred = (y_scores >= CLASSIFICATION_THRESHOLD).astype(int)

    warnings: List[str] = []
    roc_auc: Optional[float]

    if len(np.unique(y_true)) < 2:
        roc_auc = None
        warnings.append(
            "ROC-AUC non defini: une seule classe est presente dans y_true."
        )
    else:
        roc_auc = round(float(roc_auc_score(y_true, y_scores)), 4)

    metrics: Dict[str, Optional[float]] = {
        "roc_auc": roc_auc,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }

    return metrics, warnings


def evaluate_model(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k_fractions: Tuple[float, ...] = TOP_K_FRACTIONS,
) -> EvaluationReport:
    """
    Évalue un modèle avec métriques classiques et business.

    Args:
        y_true: Labels réels (0/1).
        y_scores: Scores de probabilité prédits.
        k_fractions: Fractions pour Precision@K (ex: 0.1 = top 10%).

    Returns:
        EvaluationReport avec toutes les métriques.
    """
    if len(y_true) == 0:
        raise ValueError("Le dataset d'évaluation est vide")
    if len(y_true) != len(y_scores):
        raise ValueError("y_true et y_scores doivent avoir la même longueur")

    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    metrics, metric_warnings = compute_classification_metrics(y_true, y_scores)

    # Baseline
    baseline_rate = float(y_true.mean())
    total_conversions = int(y_true.sum())

    # Métriques business par K
    precision_at_k = {}
    uplift_at_k = {}
    cumulative_gain = {}
    sorted_indices = np.argsort(y_scores)[::-1]

    for k_frac in k_fractions:
        k = max(1, int(len(y_true) * k_frac))
        if k == 0:
            continue

        # Trier par score décroissant
        top_k_indices = sorted_indices[:k]
        top_k_labels = y_true[top_k_indices]

        # Precision@K : % de conversions dans le top K
        p_at_k = float(top_k_labels.mean())
        precision_at_k[f"top_{int(k_frac * 100)}%"] = round(p_at_k, 4)

        # Uplift : combien de fois mieux que le baseline
        uplift = p_at_k / baseline_rate if baseline_rate > 0 else 0
        uplift_at_k[f"top_{int(k_frac * 100)}%"] = round(uplift, 2)

        # Cumulative Gain : % des conversions totales captées
        conversions_in_top_k = top_k_labels.sum()
        gain = conversions_in_top_k / total_conversions if total_conversions > 0 else 0
        cumulative_gain[f"top_{int(k_frac * 100)}%"] = round(gain, 4)

    return EvaluationReport(
        roc_auc=metrics["roc_auc"],
        accuracy=float(metrics["accuracy"]),
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1=float(metrics["f1"]),
        baseline_conversion_rate=round(baseline_rate, 4),
        precision_at_k=precision_at_k,
        uplift_at_k=uplift_at_k,
        cumulative_gain=cumulative_gain,
        total_samples=len(y_true),
        total_conversions=total_conversions,
        model_path=str(MODEL_PATH),
        warnings=metric_warnings,
    )


def evaluate_csv(
    input_path: str,
    output_path: Optional[str] = None,
    target_col: str = TARGET,
) -> EvaluationReport:
    """
    Évalue le modèle sur un fichier CSV avec labels.

    Args:
        input_path: Chemin vers le CSV avec features + target.
        output_path: Chemin vers le rapport JSON (optionnel).
        target_col: Nom de la colonne target.

    Returns:
        EvaluationReport.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {input_path}")

    # Détecter le séparateur
    with open(input_path) as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","

    df = pd.read_csv(input_path, sep=sep)

    if target_col not in df.columns:
        raise ValueError(f"Colonne target '{target_col}' non trouvée dans {input_path}")

    y_true = normalize_target(df[target_col], target_col=target_col)
    features = df.drop(columns=[target_col], errors="ignore")
    validate_input(features, warn_on_excluded_features=False)

    # Scorer les leads sans réordonner les lignes pour garder l'alignement avec y_true
    model = load_model()
    X = features.drop(
        columns=[column for column in EXCLUDED_FEATURES if column in features.columns],
        errors="ignore",
    )
    X = X[REQUIRED_COLUMNS]
    y_scores = model.predict_proba(X)[:, 1]

    # Évaluer
    report = evaluate_model(y_true, y_scores)

    # Sauvegarder si demandé
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(asdict(report), f, indent=2, allow_nan=False)
        print(f"✓ Rapport sauvegardé → {output_path}")

    return report


def normalize_target(target: pd.Series, target_col: str = TARGET) -> np.ndarray:
    """
    Normalise une target hétérogène vers un vecteur binaire 0/1.

    Args:
        target: Série pandas contenant la target brute.
        target_col: Nom logique de la colonne.

    Returns:
        Numpy array d'entiers 0/1.
    """
    if target.isna().any():
        raise ValueError(
            f"La colonne target '{target_col}' contient des valeurs nulles"
        )

    if pd.api.types.is_bool_dtype(target):
        return target.astype(int).to_numpy()

    if pd.api.types.is_numeric_dtype(target):
        numeric_target = pd.to_numeric(target)
        unique_values = set(numeric_target.unique().tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"Valeurs inattendues dans '{target_col}': {sorted(unique_values)}. "
                "Valeurs attendues: 0/1"
            )
        return numeric_target.astype(int).to_numpy()

    normalized = (
        target.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0,
                "true": 1,
                "false": 0,
            }
        )
    )
    if normalized.isna().any():
        invalid_values = sorted(
            target[normalized.isna()].astype(str).str.strip().unique().tolist()
        )
        raise ValueError(
            f"Valeurs inattendues dans '{target_col}': {invalid_values}. "
            "Valeurs attendues: yes/no, true/false ou 0/1"
        )

    return normalized.astype(int).to_numpy()


def print_report(report: EvaluationReport) -> None:
    """Affiche le rapport de manière lisible."""
    print("\n" + "=" * 60)
    print("RAPPORT D'ÉVALUATION - AI LEAD SCORING")
    print("=" * 60)

    print(
        f"\n📊 Dataset: {report.total_samples} leads, {report.total_conversions} conversions"
    )
    print(f"   Taux de conversion baseline: {report.baseline_conversion_rate:.2%}")

    print("\n📈 Métriques classiques:")
    if report.roc_auc is None:
        print("   ROC-AUC:   n/a")
    else:
        print(f"   ROC-AUC:   {report.roc_auc:.4f}")
    print(f"   Accuracy:  {report.accuracy:.4f}")
    print(f"   Precision: {report.precision:.4f}")
    print(f"   Recall:    {report.recall:.4f}")
    print(f"   F1-Score:  {report.f1:.4f}")

    if report.warnings:
        print("\n⚠️ Warnings:")
        for warning in report.warnings:
            print(f"   - {warning}")

    print("\n🎯 Precision@K (% de conversions dans le top K):")
    for k, p in report.precision_at_k.items():
        print(f"   {k}: {p:.2%}")

    print("\n🚀 Uplift vs baseline:")
    for k, u in report.uplift_at_k.items():
        print(f"   {k}: {u:.1f}x mieux que le random")

    print("\n📦 Cumulative Gain (% des conversions captées):")
    for k, g in report.cumulative_gain.items():
        print(f"   {k}: {g:.1%} des conversions")

    print("\n" + "=" * 60)


def main():
    """Point d'entrée CLI pour l'évaluation."""
    parser = argparse.ArgumentParser(
        description="Évaluer le modèle de scoring sur un dataset avec labels"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Chemin vers le CSV avec features + target",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(OUTPUTS_DIR / "evaluation_report.json"),
        help="Chemin vers le rapport JSON de sortie",
    )
    parser.add_argument(
        "--target",
        "-t",
        default=TARGET,
        help="Nom de la colonne target (défaut: 'y')",
    )
    args = parser.parse_args()

    report = evaluate_csv(args.input, args.output, args.target)
    print_report(report)


if __name__ == "__main__":
    main()
