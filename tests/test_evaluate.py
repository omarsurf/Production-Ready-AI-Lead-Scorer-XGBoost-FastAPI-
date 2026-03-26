"""
Tests pour le module d'évaluation business.
"""

from dataclasses import asdict
import json
import warnings

import numpy as np
import pandas as pd

from src.config import REQUIRED_COLUMNS
from src.evaluate import evaluate_csv, evaluate_model, normalize_target, print_report
from src.inference import load_model


def test_normalize_target_accepts_yes_no_strings():
    """La target yes/no doit être convertie proprement en 0/1."""
    target = pd.Series(["yes", "no", "yes", "no"])

    normalized = normalize_target(target, target_col="y")

    assert normalized.tolist() == [1, 0, 1, 0]


def test_normalize_target_rejects_unknown_values():
    """Les labels hors vocabulaire doivent être refusés explicitement."""
    target = pd.Series(["yes", "maybe", "no"])

    try:
        normalize_target(target, target_col="y")
    except ValueError as exc:
        assert "maybe" in str(exc)
    else:
        raise AssertionError("normalize_target aurait dû lever une ValueError")


def test_evaluate_model_computes_business_metrics():
    """Precision@K, uplift et gain cumulatif doivent être cohérents."""
    y_true = np.array([1, 0, 1, 0, 1])
    y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])

    report = evaluate_model(y_true, y_scores, k_fractions=(0.2, 0.4, 1.0))

    assert report.total_samples == 5
    assert report.total_conversions == 3
    assert report.baseline_conversion_rate == 0.6
    assert report.precision_at_k == {
        "top_20%": 1.0,
        "top_40%": 0.5,
        "top_100%": 0.6,
    }
    assert report.uplift_at_k == {
        "top_20%": 1.67,
        "top_40%": 0.83,
        "top_100%": 1.0,
    }
    assert report.cumulative_gain == {
        "top_20%": 0.3333,
        "top_40%": 0.3333,
        "top_100%": 1.0,
    }
    assert report.warnings == []


def test_evaluate_model_handles_single_class_target():
    """Le ROC-AUC doit devenir null-friendly sur un dataset mono-classe."""
    y_true = np.array([0, 0, 0])
    y_scores = np.array([0.3, 0.2, 0.1])

    report = evaluate_model(y_true, y_scores, k_fractions=(1.0,))

    assert report.roc_auc is None
    assert report.warnings == [
        "ROC-AUC non defini: une seule classe est presente dans y_true."
    ]
    assert report.accuracy == 1.0
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.f1 == 0.0


def test_print_report_displays_na_for_undefined_roc_auc(capsys):
    """L'affichage CLI doit rester lisible quand ROC-AUC est indisponible."""
    report = evaluate_model(
        np.array([0, 0, 0]),
        np.array([0.3, 0.2, 0.1]),
        k_fractions=(1.0,),
    )

    print_report(report)
    captured = capsys.readouterr()

    assert "ROC-AUC:   n/a" in captured.out
    assert "ROC-AUC non defini" in captured.out


def test_single_class_report_serializes_to_strict_json(tmp_path):
    """Le rapport doit être sérialisable en JSON strict avec roc_auc=null."""
    report = evaluate_model(
        np.array([0, 0, 0]),
        np.array([0.3, 0.2, 0.1]),
        k_fractions=(1.0,),
    )
    output_path = tmp_path / "single_class_report.json"

    output_path.write_text(json.dumps(asdict(report), allow_nan=False))
    payload = json.loads(output_path.read_text())

    assert payload["roc_auc"] is None
    assert payload["warnings"]
    assert "NaN" not in output_path.read_text()


def test_evaluate_csv_matches_manual_model_scoring(tmp_path):
    """Le rapport CSV doit être aligné avec un scoring manuel non réordonné."""
    input_path = "data/raw/bank+marketing/bank/bank.csv"
    output_path = tmp_path / "evaluation_report.json"

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        report = evaluate_csv(input_path, output_path=str(output_path))

    df = pd.read_csv(input_path, sep=";")
    y_true = normalize_target(df["y"], target_col="y")
    X = df.drop(columns=["y", "duration"])
    X = X[REQUIRED_COLUMNS]
    y_scores = load_model().predict_proba(X)[:, 1]
    manual_report = evaluate_model(y_true, y_scores)

    assert report.precision_at_k == manual_report.precision_at_k
    assert report.uplift_at_k == manual_report.uplift_at_k
    assert report.cumulative_gain == manual_report.cumulative_gain
    assert report.roc_auc == manual_report.roc_auc

    payload = json.loads(output_path.read_text())
    assert payload["total_samples"] == 4521
    assert "top_10%" in payload["precision_at_k"]
    assert payload["warnings"] == []
    assert not any("duration" in str(warning.message).lower() for warning in captured)


def test_evaluate_csv_writes_null_roc_auc_for_single_class_dataset(tmp_path):
    """L'évaluation CSV ne doit jamais écrire NaN quand une seule classe est présente."""
    source_path = "data/raw/bank+marketing/bank/bank.csv"
    df = pd.read_csv(source_path, sep=";")
    single_class_df = df[df["y"] == "no"].head(20)
    input_path = tmp_path / "single_class_bank.csv"
    output_path = tmp_path / "single_class_report.json"
    single_class_df.to_csv(input_path, sep=";", index=False)

    report = evaluate_csv(str(input_path), output_path=str(output_path))
    payload = json.loads(output_path.read_text())

    assert report.roc_auc is None
    assert payload["roc_auc"] is None
    assert payload["warnings"] == [
        "ROC-AUC non defini: une seule classe est presente dans y_true."
    ]
    assert "NaN" not in output_path.read_text()
