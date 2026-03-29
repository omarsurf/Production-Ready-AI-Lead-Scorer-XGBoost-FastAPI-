"""
Drift detection using Population Stability Index (PSI).

This module provides PSI-based drift detection for both numeric and categorical
features, comparing training distributions against inference data.

PSI Interpretation:
- PSI < 0.1: No significant drift
- 0.1 <= PSI < 0.25: Minor drift (monitor)
- PSI >= 0.25: Major drift (action required)
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
)

# PSI thresholds
PSI_BINS = 10
PSI_THRESHOLD_MINOR = 0.1
PSI_THRESHOLD_MAJOR = 0.25
MIN_DRIFT_SAMPLE_SIZE = 100

# All categorical features (both nominal and ordinal)
CATEGORICAL_FEATURES = NOMINAL_FEATURES + ORDINAL_FEATURES
LEGACY_REFERENCE_DISTRIBUTIONS_PATH = MODELS_DIR / "reference_distributions.json"


@dataclass
class FeatureDrift:
    """Drift result for a single feature."""

    feature_name: str
    feature_type: str  # "numeric" or "categorical"
    psi: float
    drift_level: str  # "none", "minor", "major"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftReport:
    """Full drift report for all features."""

    timestamp: str
    model_version: Optional[str]
    sample_size: int
    features: List[FeatureDrift] = field(default_factory=list)
    overall_status: str = "healthy"  # "healthy", "warning", "critical"
    major_drift_features: List[str] = field(default_factory=list)
    minor_drift_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "sample_size": self.sample_size,
            "overall_status": self.overall_status,
            "major_drift_features": self.major_drift_features,
            "minor_drift_features": self.minor_drift_features,
            "features": [f.to_dict() for f in self.features],
        }


def _get_drift_level(psi: float) -> str:
    """Classify PSI score into drift level."""
    if psi >= PSI_THRESHOLD_MAJOR:
        return "major"
    elif psi >= PSI_THRESHOLD_MINOR:
        return "minor"
    return "none"


def compute_numeric_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = PSI_BINS,
    epsilon: float = 1e-6,
) -> Tuple[float, List[float], List[float], List[float]]:
    """
    Compute PSI for numeric features using binning.

    Args:
        expected: Training data values (reference distribution).
        actual: Inference data values (current distribution).
        n_bins: Number of bins for histogram.
        epsilon: Small value to avoid log(0).

    Returns:
        Tuple of (psi_score, bin_edges, expected_percentages, actual_percentages)
    """
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0, [], [], []

    # Create bins based on expected distribution (training data)
    _, bin_edges = np.histogram(expected, bins=n_bins)

    # Count samples in each bin
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to percentages
    expected_pct = expected_counts / len(expected) + epsilon
    actual_pct = actual_counts / len(actual) + epsilon

    # Compute PSI
    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

    return psi, bin_edges.tolist(), expected_pct.tolist(), actual_pct.tolist()


def compute_categorical_psi(
    expected: pd.Series,
    actual: pd.Series,
    known_categories: Optional[List[str]] = None,
    epsilon: float = 1e-6,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Compute PSI for categorical features using category proportions.

    Args:
        expected: Training data values (reference distribution).
        actual: Inference data values (current distribution).
        known_categories: List of known categories from training.
        epsilon: Small value to avoid log(0).

    Returns:
        Tuple of (psi_score, expected_proportions, actual_proportions)
    """
    # Handle missing values as a category
    expected = expected.fillna("__MISSING__")
    actual = actual.fillna("__MISSING__")

    # Get all categories (from training + any new ones in actual)
    if known_categories is None:
        all_categories = set(expected.unique()) | set(actual.unique())
    else:
        all_categories = set(known_categories) | {"__MISSING__"}
        # Include new categories from actual (they'll have 0 in expected)
        all_categories |= set(actual.unique())

    # Compute proportions
    expected_counts = expected.value_counts()
    actual_counts = actual.value_counts()

    expected_pct = {}
    actual_pct = {}
    psi = 0.0

    for cat in all_categories:
        exp_p = expected_counts.get(cat, 0) / len(expected) + epsilon
        act_p = actual_counts.get(cat, 0) / len(actual) + epsilon

        expected_pct[str(cat)] = float(exp_p)
        actual_pct[str(cat)] = float(act_p)

        psi += (act_p - exp_p) * np.log(act_p / exp_p)

    return float(psi), expected_pct, actual_pct


def save_reference_distributions(
    X_train: pd.DataFrame,
    output_path: Path,
    numeric_features: List[str] = NUMERIC_FEATURES,
    categorical_features: List[str] = CATEGORICAL_FEATURES,
    n_bins: int = PSI_BINS,
) -> Path:
    """
    Save training distributions as reference for future drift detection.

    Args:
        X_train: Training features DataFrame.
        output_path: Path to save distributions JSON.
        numeric_features: List of numeric feature names.
        categorical_features: List of categorical feature names.
        n_bins: Number of bins for numeric features.

    Returns:
        Path to saved distributions file.
    """
    distributions: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(X_train),
        "numeric_features": {},
        "categorical_features": {},
    }

    # Numeric features: store bin edges and percentages
    for feature in numeric_features:
        if feature in X_train.columns:
            values = X_train[feature].values
            _, bin_edges, expected_pct, _ = compute_numeric_psi(
                values, values, n_bins=n_bins
            )
            distributions["numeric_features"][feature] = {
                "bin_edges": bin_edges,
                "percentages": expected_pct,
                "n_missing": int(pd.isna(X_train[feature]).sum()),
            }

    # Categorical features: store category proportions
    for feature in categorical_features:
        if feature in X_train.columns:
            series = X_train[feature].fillna("__MISSING__")
            proportions = (series.value_counts() / len(series)).to_dict()
            distributions["categorical_features"][feature] = {
                "proportions": {str(k): float(v) for k, v in proportions.items()},
                "categories": list(series.unique()),
            }

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(distributions, f, indent=2)
        f.write("\n")

    return output_path


def load_reference_distributions(path: Path) -> Dict[str, Any]:
    """Load reference distributions from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Reference distributions not found: {path}")

    with open(path) as f:
        return json.load(f)


def resolve_reference_path(
    reference_path: Optional[Path] = None,
) -> Tuple[Path, Optional[str]]:
    """
    Resolve the drift reference path with registry-aware fallback order.

    Priority:
    1. Explicit reference_path provided by operator
    2. Production model reference distributions from registry
    3. Legacy models/reference_distributions.json
    4. FileNotFoundError
    """
    if reference_path is not None:
        return Path(reference_path).expanduser().resolve(), None

    production_model = None
    try:
        from src.registry import get_production_model

        production_model = get_production_model()
    except Exception:
        production_model = None

    if production_model and production_model.reference_distributions_path:
        from src.registry import resolve_registry_entry_path

        resolved_path = resolve_registry_entry_path(
            production_model.reference_distributions_path
        )
        if resolved_path.exists():
            return resolved_path, production_model.version

        if LEGACY_REFERENCE_DISTRIBUTIONS_PATH.exists():
            return LEGACY_REFERENCE_DISTRIBUTIONS_PATH.expanduser().resolve(), None

        raise FileNotFoundError(
            "Reference distributions declared for production model "
            f"v{production_model.version} but file not found: {resolved_path}"
        )

    if LEGACY_REFERENCE_DISTRIBUTIONS_PATH.exists():
        return LEGACY_REFERENCE_DISTRIBUTIONS_PATH.expanduser().resolve(), None

    raise FileNotFoundError(
        "No reference distributions found in the production registry entry and "
        f"no legacy reference distributions at {LEGACY_REFERENCE_DISTRIBUTIONS_PATH}"
    )


def detect_drift(
    X_current: pd.DataFrame,
    reference_path: Path,
    model_version: Optional[str] = None,
    min_sample_size: int = MIN_DRIFT_SAMPLE_SIZE,
) -> DriftReport:
    """
    Compare current data against reference distributions.

    Args:
        X_current: Current inference data.
        reference_path: Path to reference distributions JSON.
        model_version: Version of model for the report.
        min_sample_size: Minimum samples required for meaningful PSI.

    Returns:
        DriftReport with per-feature PSI scores and overall status.

    Raises:
        ValueError: If sample size is too small.
    """
    if len(X_current) < min_sample_size:
        raise ValueError(
            f"Sample size ({len(X_current)}) is below minimum ({min_sample_size}). "
            "PSI is not meaningful on small samples."
        )

    reference = load_reference_distributions(reference_path)

    features: List[FeatureDrift] = []
    major_drift: List[str] = []
    minor_drift: List[str] = []

    # Check numeric features
    for feature, ref_data in reference.get("numeric_features", {}).items():
        if feature not in X_current.columns:
            continue

        bin_edges = np.array(ref_data["bin_edges"])
        expected_pct = np.array(ref_data["percentages"])

        # Get actual distribution
        actual = X_current[feature].values
        actual = actual[~np.isnan(actual)]

        if len(actual) == 0:
            psi = 0.0
        else:
            actual_counts, _ = np.histogram(actual, bins=bin_edges)
            actual_pct = actual_counts / len(actual) + 1e-6
            psi = float(
                np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
            )

        drift_level = _get_drift_level(psi)
        features.append(
            FeatureDrift(
                feature_name=feature,
                feature_type="numeric",
                psi=round(psi, 4),
                drift_level=drift_level,
            )
        )

        if drift_level == "major":
            major_drift.append(feature)
        elif drift_level == "minor":
            minor_drift.append(feature)

    # Check categorical features
    for feature, ref_data in reference.get("categorical_features", {}).items():
        if feature not in X_current.columns:
            continue

        expected_props = ref_data["proportions"]

        # Compute PSI
        actual = X_current[feature].fillna("__MISSING__")
        actual_counts = actual.value_counts()

        psi = 0.0
        epsilon = 1e-6
        all_cats = set(expected_props.keys()) | set(actual.unique())

        for cat in all_cats:
            exp_p = expected_props.get(str(cat), 0.0) + epsilon
            act_p = actual_counts.get(cat, 0) / len(actual) + epsilon
            psi += (act_p - exp_p) * np.log(act_p / exp_p)

        drift_level = _get_drift_level(psi)
        features.append(
            FeatureDrift(
                feature_name=feature,
                feature_type="categorical",
                psi=round(float(psi), 4),
                drift_level=drift_level,
            )
        )

        if drift_level == "major":
            major_drift.append(feature)
        elif drift_level == "minor":
            minor_drift.append(feature)

    # Determine overall status
    if major_drift:
        overall_status = "critical"
    elif minor_drift:
        overall_status = "warning"
    else:
        overall_status = "healthy"

    return DriftReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_version=model_version,
        sample_size=len(X_current),
        features=features,
        overall_status=overall_status,
        major_drift_features=major_drift,
        minor_drift_features=minor_drift,
    )


def main() -> None:
    """CLI for drift detection."""
    parser = argparse.ArgumentParser(description="Detect data drift using PSI")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--reference",
        "-r",
        help="Path to reference distributions JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to output drift report JSON (optional)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=MIN_DRIFT_SAMPLE_SIZE,
        help=f"Minimum sample size (default: {MIN_DRIFT_SAMPLE_SIZE})",
    )

    args = parser.parse_args()

    # Load data
    input_path = Path(args.input)
    with open(input_path) as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","
    df = pd.read_csv(input_path, sep=sep)

    resolved_reference_path, model_version = resolve_reference_path(
        Path(args.reference) if args.reference else None
    )

    # Detect drift
    report = detect_drift(
        X_current=df,
        reference_path=resolved_reference_path,
        model_version=model_version,
        min_sample_size=args.min_samples,
    )

    # Output results
    print(f"\n{'=' * 50}")
    print(f"DRIFT REPORT - {report.timestamp[:19]}")
    print(f"{'=' * 50}")
    print(f"Sample size: {report.sample_size}")
    print(f"Overall status: {report.overall_status.upper()}")
    print()

    if report.major_drift_features:
        print(f"MAJOR DRIFT ({len(report.major_drift_features)} features):")
        for f in report.major_drift_features:
            print(f"  - {f}")
        print()

    if report.minor_drift_features:
        print(f"Minor drift ({len(report.minor_drift_features)} features):")
        for f in report.minor_drift_features:
            print(f"  - {f}")
        print()

    # Save report if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"✓ Report saved to: {output_path}")


if __name__ == "__main__":
    main()
