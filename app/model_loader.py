"""
Chargement et gestion du modèle pour l'API.

Le modèle est chargé une seule fois au démarrage.
"""

from pathlib import Path
from typing import Optional

from src.inference import load_model as load_shared_model


_loaded_model_path: Optional[Path] = None


def _resolve_api_model_path() -> Path:
    """Résout le chemin réellement servi par l'API via le registry."""
    from src.registry import get_production_model_path

    return get_production_model_path().expanduser().resolve()


def load_model_at_startup() -> bool:
    """
    Charge le modèle au démarrage de l'API.

    Returns:
        True si le chargement a réussi, False sinon.
    """
    global _loaded_model_path

    if _loaded_model_path is not None:
        return True

    try:
        resolved_path = _resolve_api_model_path()
        load_shared_model(resolved_path)
        _loaded_model_path = resolved_path
        print(f"✓ Modèle chargé: {resolved_path}")
        return True
    except Exception as e:
        print(f"✗ Erreur chargement modèle: {e}")
        _loaded_model_path = None
        return False


def get_model():
    """
    Retourne le modèle chargé.

    Returns:
        Pipeline sklearn ou None si non chargé.
    """
    if _loaded_model_path is None:
        return None
    return load_shared_model(_loaded_model_path)


def is_model_loaded() -> bool:
    """Vérifie si le modèle est chargé."""
    return _loaded_model_path is not None


def get_model_path() -> Optional[str]:
    """Retourne le chemin du modèle si chargé."""
    if _loaded_model_path is not None:
        return str(_loaded_model_path)
    return None
