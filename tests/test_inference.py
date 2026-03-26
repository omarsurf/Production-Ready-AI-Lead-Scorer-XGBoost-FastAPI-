"""
Tests pour le module d'inférence.

Tests:
- Chargement du modèle
- Scoring d'un lead unique
- Scoring batch
- Gestion de 'duration' (leakage)
- Colonnes manquantes
"""

import warnings
import shutil

import pandas as pd
import pytest

from src.config import MODEL_PATH
from src.inference import load_model, score_leads
from src.schema import SchemaValidationError, validate_input


# Lead exemple valide
SAMPLE_LEAD = {
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}


class TestModelLoading:
    """Tests de chargement du modèle."""

    def test_model_file_exists(self):
        """Vérifie que le fichier modèle existe."""
        assert MODEL_PATH.exists(), f"Modèle non trouvé: {MODEL_PATH}"

    def test_model_loads_successfully(self):
        """Vérifie que le modèle se charge sans erreur."""
        model = load_model()
        assert model is not None

    def test_model_has_predict_proba(self):
        """Vérifie que le modèle a la méthode predict_proba."""
        model = load_model()
        assert hasattr(model, "predict_proba")

    def test_same_model_path_returns_same_instance(self):
        """Le cache doit réutiliser l'instance déjà chargée pour le même chemin."""
        default_model = load_model()
        explicit_model = load_model(MODEL_PATH)

        assert default_model is explicit_model

    def test_different_model_paths_return_distinct_instances(self, tmp_path):
        """Deux chemins distincts doivent produire deux entrées de cache distinctes."""
        copied_model_path = tmp_path / MODEL_PATH.name
        shutil.copy2(MODEL_PATH, copied_model_path)

        default_model = load_model(MODEL_PATH)
        copied_model = load_model(copied_model_path)

        assert default_model is not copied_model
        assert load_model(copied_model_path) is copied_model


class TestScoring:
    """Tests de scoring."""

    def test_single_lead_scoring(self):
        """Score un lead unique."""
        df = pd.DataFrame([SAMPLE_LEAD])
        result = score_leads(df)

        assert "score" in result.columns
        assert "predicted_label" in result.columns
        assert "priority_rank" in result.columns
        assert "priority" in result.columns

    def test_score_in_valid_range(self):
        """Le score doit être entre 0 et 1."""
        df = pd.DataFrame([SAMPLE_LEAD])
        result = score_leads(df)

        score = result["score"].iloc[0]
        assert 0 <= score <= 1

    def test_predicted_label_is_binary(self):
        """Le label doit être 0 ou 1."""
        df = pd.DataFrame([SAMPLE_LEAD])
        result = score_leads(df)

        label = result["predicted_label"].iloc[0]
        assert label in [0, 1]

    def test_priority_is_valid(self):
        """La priorité doit être high, medium ou low."""
        df = pd.DataFrame([SAMPLE_LEAD])
        result = score_leads(df)

        priority = result["priority"].iloc[0]
        assert priority in ["high", "medium", "low"]

    def test_batch_scoring(self):
        """Score plusieurs leads."""
        leads = [SAMPLE_LEAD.copy() for _ in range(10)]
        # Varier les données
        for i, lead in enumerate(leads):
            lead["age"] = 25 + i * 5
            lead["balance"] = 500 + i * 200

        df = pd.DataFrame(leads)
        result = score_leads(df)

        assert len(result) == 10
        assert result["priority_rank"].min() == 1
        assert result["priority_rank"].max() == 10

    def test_result_sorted_by_rank(self):
        """Les résultats sont triés par priority_rank."""
        leads = [SAMPLE_LEAD.copy() for _ in range(5)]
        df = pd.DataFrame(leads)
        result = score_leads(df)

        ranks = result["priority_rank"].tolist()
        assert ranks == sorted(ranks)


class TestLeakagePrevention:
    """Tests de prévention du leakage."""

    def test_duration_ignored(self):
        """La colonne duration est ignorée sans erreur."""
        lead_with_duration = {**SAMPLE_LEAD, "duration": 999}
        df = pd.DataFrame([lead_with_duration])

        # Doit fonctionner sans erreur
        result = score_leads(df)
        assert "score" in result.columns

    def test_duration_warning(self):
        """Un warning est émis si duration est présent."""
        lead_with_duration = {**SAMPLE_LEAD, "duration": 999}
        df = pd.DataFrame([lead_with_duration])

        with pytest.warns(UserWarning, match="duration"):
            validate_input(df)

    def test_duration_warning_can_be_disabled(self):
        """Le warning duration peut être désactivé pour des workflows spécifiques."""
        lead_with_duration = {**SAMPLE_LEAD, "duration": 999}
        df = pd.DataFrame([lead_with_duration])

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            validate_input(df, warn_on_excluded_features=False)

        assert not any(
            "duration" in str(warning.message).lower() for warning in captured
        )


class TestSchemaValidation:
    """Tests de validation du schéma."""

    def test_missing_column_raises_error(self):
        """Erreur si une colonne requise manque."""
        incomplete_lead = SAMPLE_LEAD.copy()
        del incomplete_lead["age"]

        df = pd.DataFrame([incomplete_lead])

        with pytest.raises(SchemaValidationError, match="age"):
            validate_input(df)

    def test_all_required_columns_present(self):
        """Pas d'erreur si toutes les colonnes sont présentes."""
        df = pd.DataFrame([SAMPLE_LEAD])
        validate_input(df)  # Ne doit pas lever d'exception

    def test_extra_columns_allowed(self):
        """Les colonnes supplémentaires sont autorisées."""
        lead_with_extra = {**SAMPLE_LEAD, "extra_col": "value"}
        df = pd.DataFrame([lead_with_extra])
        validate_input(df)  # Ne doit pas lever d'exception

    def test_invalid_categorical_value_raises_error(self):
        """Erreur si une valeur catégorielle sort du vocabulaire appris."""
        invalid_lead = SAMPLE_LEAD.copy()
        invalid_lead["education"] = "doctorate"

        df = pd.DataFrame([invalid_lead])

        with pytest.raises(SchemaValidationError, match="education"):
            validate_input(df)


class TestOriginalDataPreserved:
    """Tests de préservation des données originales."""

    def test_original_columns_preserved(self):
        """Les colonnes originales sont préservées."""
        df = pd.DataFrame([SAMPLE_LEAD])
        result = score_leads(df)

        for col in SAMPLE_LEAD.keys():
            assert col in result.columns

    def test_original_values_unchanged(self):
        """Les valeurs originales ne sont pas modifiées."""
        df = pd.DataFrame([SAMPLE_LEAD])
        result = score_leads(df)

        for col, value in SAMPLE_LEAD.items():
            assert result[col].iloc[0] == value
