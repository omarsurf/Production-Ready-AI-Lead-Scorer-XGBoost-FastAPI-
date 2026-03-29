"""
Schémas Pydantic pour la validation des requêtes/réponses API.
"""

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from src.config import ALLOWED_CATEGORICAL_VALUES


class LeadInput(BaseModel):
    """Données d'un lead pour scoring."""

    lead_id: Optional[str] = Field(
        default=None,
        description="Identifiant optionnel du lead, renvoyé dans les réponses batch",
    )
    age: int = Field(..., ge=18, le=100, description="Âge du client")
    job: str = Field(..., description="Type d'emploi")
    marital: str = Field(..., description="Statut marital")
    education: str = Field(..., description="Niveau d'éducation")
    default: str = Field(..., description="Crédit en défaut (yes/no)")
    balance: int = Field(..., description="Solde moyen annuel en euros")
    housing: str = Field(..., description="Prêt immobilier (yes/no)")
    loan: str = Field(..., description="Prêt personnel (yes/no)")
    contact: str = Field(..., description="Type de contact")
    day: int = Field(..., ge=1, le=31, description="Jour du dernier contact")
    month: str = Field(..., description="Mois du dernier contact")
    campaign: int = Field(..., ge=1, description="Nombre de contacts cette campagne")
    pdays: int = Field(..., description="Jours depuis dernier contact précédent")
    previous: int = Field(..., ge=0, description="Contacts campagnes précédentes")
    poutcome: str = Field(..., description="Résultat campagne précédente")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "lead_id": "lead-001",
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
        },
    )

    @field_validator(*ALLOWED_CATEGORICAL_VALUES.keys())
    @classmethod
    def validate_known_categories(cls, value: str, info: ValidationInfo) -> str:
        """Refuse les catégories hors vocabulaire appris par le modèle."""
        allowed_values = ALLOWED_CATEGORICAL_VALUES[info.field_name]
        if value not in allowed_values:
            raise ValueError(
                f"'{value}' n'est pas une valeur valide pour '{info.field_name}'. "
                f"Valeurs autorisées: {allowed_values}"
            )
        return value

    @field_validator("lead_id", mode="before")
    @classmethod
    def normalize_lead_id(cls, value: Any) -> Any:
        """Normalise lead_id en string pour simplifier la traçabilité downstream."""
        if value is None:
            return None
        return str(value)


class LeadScoreResponse(BaseModel):
    """Réponse de scoring pour un lead."""

    score: float = Field(
        ..., ge=0, le=1, description="Probabilité de conversion [0, 1]"
    )
    predicted_label: int = Field(..., ge=0, le=1, description="Prédiction binaire 0/1")
    priority: Literal["high", "medium", "low"] = Field(
        ..., description="Niveau de priorité"
    )


class LeadBatchResult(LeadScoreResponse):
    """Résultat de scoring batch avec traçabilité d'entrée."""

    input_index: int = Field(
        ..., ge=0, description="Position d'origine du lead dans la requête"
    )
    priority_rank: int = Field(..., ge=1, description="Rang de priorité (1 = meilleur)")
    lead_id: Optional[str] = Field(
        default=None,
        description="Identifiant métier renvoyé si fourni dans la requête",
    )

    @field_validator("lead_id", mode="before")
    @classmethod
    def normalize_lead_id(cls, value: Any) -> Any:
        """Garantit une sérialisation homogène du lead_id côté réponse."""
        if value is None:
            return None
        return str(value)


class LeadBatchInput(BaseModel):
    """Liste de leads pour scoring batch."""

    leads: List[LeadInput] = Field(..., min_length=1, max_length=1000)


class LeadBatchScoreResponse(BaseModel):
    """Réponse de scoring batch avec ranking."""

    results: List[LeadBatchResult]
    total: int = Field(..., description="Nombre total de leads scorés")
    high_priority_count: int = Field(..., description="Nombre de leads priorité haute")


class HealthResponse(BaseModel):
    """Réponse du health check."""

    status: str
    model_loaded: bool
    model_path: Optional[str] = None
