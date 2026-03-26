"""
Entraînement reproductible du modèle de lead scoring.

Le script permet soit :
- d'entraîner directement le pipeline final avec les meilleurs hyperparamètres connus
- de relancer un RandomizedSearchCV pour retrouver/mettre à jour ces hyperparamètres
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    EDUCATION_ORDER,
    EXCLUDED_FEATURES,
    FINAL_XGB_PARAMS,
    MODEL_PATH,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    OUTPUTS_DIR,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    TOP_K_FRACTIONS,
    TRAINING_DATA_PATH,
    XGB_TUNING_SPACE,
)
from src.metadata import (
    create_training_metadata,
    save_input_schema,
    save_metadata,
)
from src.evaluate import compute_classification_metrics


def load_training_frame(input_path: Path) -> pd.DataFrame:
    """Charge le dataset d'entraînement avec détection simple du séparateur."""
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset non trouvé: {input_path}")

    with input_path.open() as file_handle:
        first_line = file_handle.readline()
    sep = ";" if ";" in first_line else ","
    return pd.read_csv(input_path, sep=sep)


def build_preprocessor() -> ColumnTransformer:
    """Construit le preprocessing sklearn partagé par le training et l'inférence."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(categories=[EDUCATION_ORDER])),
        ]
    )

    nominal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("ord", ordinal_transformer, ORDINAL_FEATURES),
            ("nom", nominal_transformer, NOMINAL_FEATURES),
        ]
    )


def build_pipeline(
    scale_pos_weight: float,
    classifier_params: Optional[Dict[str, Any]] = None,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Construit le pipeline XGBoost complet."""
    params = {
        "random_state": random_state,
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight,
    }
    if classifier_params:
        params.update(classifier_params)

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", XGBClassifier(**params)),
        ]
    )


def prepare_training_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
):
    """Prépare le split stratifié train/test à partir du dataset brut."""
    target_series = df[TARGET].map({"no": 0, "yes": 1})
    if target_series.isna().any():
        raise ValueError(
            f"Valeurs inattendues dans la target '{TARGET}': "
            f"{sorted(df[TARGET].dropna().unique().tolist())}"
        )

    drop_columns = [TARGET, *EXCLUDED_FEATURES]
    features = df.drop(columns=drop_columns, errors="ignore")

    return train_test_split(
        features,
        target_series.astype(int),
        test_size=test_size,
        random_state=random_state,
        stratify=target_series,
    )


def precision_at_fraction(scored_frame: pd.DataFrame, fraction: float) -> float:
    """Calcule la précision sur le top-k% des leads triés par score."""
    top_k = max(1, int(len(scored_frame) * fraction))
    return float(scored_frame.head(top_k)["y_true"].mean())


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Produit les métriques de classification et de ranking business."""
    y_proba = model.predict_proba(X_test)[:, 1]
    classification_metrics, metric_warnings = compute_classification_metrics(
        y_test.to_numpy(),
        y_proba,
    )
    metrics: Dict[str, Any] = {
        "accuracy": classification_metrics["accuracy"],
        "precision": classification_metrics["precision"],
        "recall": classification_metrics["recall"],
        "f1_score": classification_metrics["f1"],
        "roc_auc": classification_metrics["roc_auc"],
    }

    scored_leads = X_test.copy()
    scored_leads["y_true"] = y_test.values
    scored_leads["score"] = y_proba
    scored_leads = scored_leads.sort_values("score", ascending=False).reset_index(
        drop=True
    )

    baseline_conversion_rate = float(scored_leads["y_true"].mean())
    ranking_metrics: Dict[str, Any] = {
        "baseline_conversion_rate": round(baseline_conversion_rate, 4),
    }

    for fraction in TOP_K_FRACTIONS:
        percent = int(fraction * 100)
        precision_value = precision_at_fraction(scored_leads, fraction)
        ranking_metrics[f"precision_at_{percent}"] = round(precision_value, 4)
        ranking_metrics[f"uplift_at_{percent}"] = round(
            precision_value / baseline_conversion_rate
            if baseline_conversion_rate
            else 0.0,
            4,
        )

    return {
        "classification_metrics": metrics,
        "ranking_metrics": ranking_metrics,
        "warnings": metric_warnings,
    }


def train_model(
    input_path: Path,
    model_output: Path,
    metrics_output: Optional[Path] = None,
    tune: bool = False,
    n_iter: int = 20,
    cv: int = 3,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save_model_metadata: bool = True,
) -> Dict[str, Any]:
    """Entraîne le modèle final et persiste l'artefact."""
    import time

    start_time = time.time()

    df = load_training_frame(input_path)
    X_train, X_test, y_train, y_test = prepare_training_split(
        df=df,
        test_size=test_size,
        random_state=random_state,
    )

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    if tune:
        search = RandomizedSearchCV(
            estimator=build_pipeline(
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
            ),
            param_distributions=XGB_TUNING_SPACE,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            verbose=1,
            random_state=random_state,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        model_params = search.best_params_
        best_cv_score = round(float(search.best_score_), 4)
    else:
        model = build_pipeline(
            scale_pos_weight=scale_pos_weight,
            classifier_params=FINAL_XGB_PARAMS,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        model_params = {
            f"classifier__{key}": value for key, value in FINAL_XGB_PARAMS.items()
        }
        best_cv_score = None

    evaluation = evaluate_model(model, X_test, y_test)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output)

    result: Dict[str, Any] = {
        "input_path": str(input_path),
        "model_output": str(model_output),
        "tuned": tune,
        "random_state": random_state,
        "test_size": test_size,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "scale_pos_weight": round(scale_pos_weight, 4),
        "best_cv_roc_auc": best_cv_score,
        "model_params": model_params,
        **evaluation,
    }

    if metrics_output is not None:
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        metrics_output.write_text(
            json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        )

    # Sauvegarder les métadonnées du modèle
    if save_model_metadata:
        training_duration = time.time() - start_time
        positive_ratio = float(y_train.mean())

        metadata = create_training_metadata(
            training_samples=len(X_train),
            test_samples=len(X_test),
            positive_ratio=positive_ratio,
            metrics=evaluation["classification_metrics"],
            hyperparameters=model_params,
            model_path=str(model_output),
            training_duration=training_duration,
        )
        metadata_path = save_metadata(metadata)
        schema_path = save_input_schema()
        print(f"✓ Métadonnées sauvegardées: {metadata_path}")
        print(f"✓ Schéma sauvegardé: {schema_path}")

    return result


def main() -> None:
    """Point d'entrée CLI pour l'entraînement reproductible."""
    parser = argparse.ArgumentParser(
        description="Entraîne le modèle de lead scoring et sauvegarde le pipeline."
    )
    parser.add_argument(
        "--input",
        default=str(TRAINING_DATA_PATH),
        help="Chemin du dataset source",
    )
    parser.add_argument(
        "--model-output",
        default=str(MODEL_PATH),
        help="Chemin de sortie du pipeline joblib",
    )
    parser.add_argument(
        "--metrics-output",
        default=str(OUTPUTS_DIR / "training_metrics.json"),
        help="Chemin optionnel du rapport JSON de métriques",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Relance un RandomizedSearchCV avant d'entraîner le modèle final",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Nombre d'itérations de recherche aléatoire si --tune est activé",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Nombre de folds de cross-validation si --tune est activé",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Proportion du jeu de test",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Seed global pour le split et le modèle",
    )

    args = parser.parse_args()
    metrics_output = Path(args.metrics_output) if args.metrics_output else None

    result = train_model(
        input_path=Path(args.input),
        model_output=Path(args.model_output),
        metrics_output=metrics_output,
        tune=args.tune,
        n_iter=args.n_iter,
        cv=args.cv,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(
        "Modele entraine: "
        f"ROC-AUC={result['classification_metrics']['roc_auc'] if result['classification_metrics']['roc_auc'] is not None else 'n/a'} "
        f"P@10={result['ranking_metrics']['precision_at_10']} "
        f"sortie={result['model_output']}"
    )


if __name__ == "__main__":
    main()
