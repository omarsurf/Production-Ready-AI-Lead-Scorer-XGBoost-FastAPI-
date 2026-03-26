"""
Configuration centralisée du projet AI Lead Scoring.

Toutes les constantes, chemins et listes de features sont définis ici.
"""

from pathlib import Path

# Chemins du projet
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Artifact principal
MODEL_PATH = MODELS_DIR / "tuned_xgb_pipeline.joblib"
TRAINING_DATA_PATH = DATA_DIR / "raw" / "bank+marketing" / "bank" / "bank-full.csv"

# Paramètres globaux
RANDOM_STATE = 42
TEST_SIZE = 0.2
CLASSIFICATION_THRESHOLD = 0.5

# Features numériques
NUMERIC_FEATURES = [
    "age",
    "balance",
    "campaign",
    "pdays",
    "previous",
    "day",
]

# Features catégorielles ordinales (avec ordre explicite)
ORDINAL_FEATURES = ["education"]
EDUCATION_ORDER = ["unknown", "primary", "secondary", "tertiary"]
JOB_CATEGORIES = [
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]
MARITAL_CATEGORIES = ["divorced", "married", "single"]
YES_NO_CATEGORIES = ["no", "yes"]
CONTACT_CATEGORIES = ["cellular", "telephone", "unknown"]
MONTH_CATEGORIES = [
    "apr",
    "aug",
    "dec",
    "feb",
    "jan",
    "jul",
    "jun",
    "mar",
    "may",
    "nov",
    "oct",
    "sep",
]
POUTCOME_CATEGORIES = ["failure", "other", "success", "unknown"]

# Features catégorielles nominales
NOMINAL_FEATURES = [
    "job",
    "marital",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

# Features interdites en production (leakage post-contact)
EXCLUDED_FEATURES = ["duration"]

# Target
TARGET = "y"

# Colonnes requises pour l'inférence
REQUIRED_COLUMNS = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES

# Valeurs catégorielles autorisées
ALLOWED_CATEGORICAL_VALUES = {
    "education": EDUCATION_ORDER,
    "job": JOB_CATEGORIES,
    "marital": MARITAL_CATEGORIES,
    "default": YES_NO_CATEGORIES,
    "housing": YES_NO_CATEGORIES,
    "loan": YES_NO_CATEGORIES,
    "contact": CONTACT_CATEGORIES,
    "month": MONTH_CATEGORIES,
    "poutcome": POUTCOME_CATEGORIES,
}

# Paramètres du modèle final
FINAL_XGB_PARAMS = {
    "subsample": 0.9,
    "reg_lambda": 1,
    "reg_alpha": 0,
    "n_estimators": 200,
    "min_child_weight": 1,
    "max_depth": 3,
    "learning_rate": 0.1,
    "gamma": 0.1,
    "colsample_bytree": 0.7,
}
XGB_TUNING_SPACE = {
    "classifier__n_estimators": [100, 150, 200, 300],
    "classifier__max_depth": [3, 4, 5, 6, 8],
    "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
    "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "classifier__min_child_weight": [1, 3, 5, 7],
    "classifier__gamma": [0, 0.1, 0.3, 0.5],
    "classifier__reg_alpha": [0, 0.01, 0.1, 1],
    "classifier__reg_lambda": [1, 1.5, 2, 3],
}
TOP_K_FRACTIONS = (0.1, 0.2, 0.3, 0.5)

# Seuils de priorité pour le scoring
PRIORITY_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.4,
}
