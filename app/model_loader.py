"""
Chargement et gestion du modèle pour l'API.

Le modèle est chargé une seule fois au démarrage.
"""

from typing import Optional

from src.config import MODEL_PATH
from src.inference import load_model as load_shared_model


_model_loaded = False


def load_model_at_startup() -> bool:
    """
    Charge le modèle au démarrage de l'API.

    Returns:
        True si le chargement a réussi, False sinon.
    """
    global _model_loaded

    if _model_loaded:
        return True

    try:
        if not MODEL_PATH.exists():
            print(f"⚠️ Modèle non trouvé: {MODEL_PATH}")
            return False

        load_shared_model(MODEL_PATH)
        _model_loaded = True
        print(f"✓ Modèle chargé: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"✗ Erreur chargement modèle: {e}")
        _model_loaded = False
        return False


def get_model():
    """
    Retourne le modèle chargé.

    Returns:
        Pipeline sklearn ou None si non chargé.
    """
    if not _model_loaded:
        return None
    return load_shared_model(MODEL_PATH)


def is_model_loaded() -> bool:
    """Vérifie si le modèle est chargé."""
    return _model_loaded


def get_model_path() -> Optional[str]:
    """Retourne le chemin du modèle si chargé."""
    if _model_loaded:
        return str(MODEL_PATH)
    return None
