"""
Validation de schéma pour les données entrantes.

Fail fast sur colonnes manquantes.
Warning sur features de leakage détectées.
"""

import warnings
from typing import Dict, List

import pandas as pd

from src.config import (
    ALLOWED_CATEGORICAL_VALUES,
    EXCLUDED_FEATURES,
    REQUIRED_COLUMNS,
)


class SchemaValidationError(ValueError):
    """Erreur levée quand le schéma d'entrée est invalide."""

    pass


def validate_input(
    df: pd.DataFrame,
    warn_on_excluded_features: bool = True,
) -> None:
    """
    Vérifie que toutes les colonnes requises sont présentes.

    Args:
        df: DataFrame de leads à valider.
        warn_on_excluded_features: Émettre un warning si une feature exclue
            comme 'duration' est détectée.

    Raises:
        SchemaValidationError: Si des colonnes requises sont manquantes.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"Colonnes manquantes: {sorted(missing)}. "
            f"Colonnes requises: {REQUIRED_COLUMNS}"
        )

    invalid_categories = get_invalid_categorical_values(df)
    if invalid_categories:
        details = []
        for column, values in invalid_categories.items():
            allowed_values = ALLOWED_CATEGORICAL_VALUES[column]
            details.append(
                f"Valeurs invalides pour '{column}': {values}. "
                f"Valeurs autorisées: {allowed_values}"
            )
        raise SchemaValidationError(" ; ".join(details))

    # Warning si feature de leakage détectée
    if warn_on_excluded_features:
        for feature in EXCLUDED_FEATURES:
            if feature in df.columns:
                warnings.warn(
                    f"'{feature}' détecté dans les données. "
                    f"Cette feature sera ignorée (leakage post-contact).",
                    UserWarning,
                )


def get_missing_columns(df: pd.DataFrame) -> List[str]:
    """
    Retourne la liste des colonnes manquantes.

    Args:
        df: DataFrame à vérifier.

    Returns:
        Liste des colonnes manquantes (vide si tout est ok).
    """
    return sorted(set(REQUIRED_COLUMNS) - set(df.columns))


def get_invalid_categorical_values(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Retourne les valeurs catégorielles inconnues par colonne.

    Args:
        df: DataFrame à vérifier.

    Returns:
        Mapping colonne -> valeurs invalides.
    """
    invalid: Dict[str, List[str]] = {}

    for column, allowed_values in ALLOWED_CATEGORICAL_VALUES.items():
        if column not in df.columns:
            continue

        allowed = set(allowed_values)
        values = {
            str(value)
            for value in df[column].dropna().unique().tolist()
            if value not in allowed
        }
        if values:
            invalid[column] = sorted(values)

    return invalid


def has_leakage_features(df: pd.DataFrame) -> bool:
    """
    Vérifie si des features de leakage sont présentes.

    Args:
        df: DataFrame à vérifier.

    Returns:
        True si au moins une feature de leakage est présente.
    """
    return any(feature in df.columns for feature in EXCLUDED_FEATURES)
