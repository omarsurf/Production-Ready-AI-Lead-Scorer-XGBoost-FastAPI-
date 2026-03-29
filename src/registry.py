"""
File-based model registry for versioning, promotion, and rollback.

This module provides a simple, file-based alternative to MLflow for
tracking model versions in a production-ready manner.
"""

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import MODEL_PATH, MODELS_DIR, REGISTRY_PATH

SCHEMA_VERSION = "1.0"


@dataclass
class ModelEntry:
    """A single model version in the registry."""

    version: str
    created_at: str
    artifact_path: str
    metadata_path: str
    reference_distributions_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "archived"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelEntry":
        return cls(
            version=data["version"],
            created_at=data["created_at"],
            artifact_path=data["artifact_path"],
            metadata_path=data["metadata_path"],
            reference_distributions_path=data.get("reference_distributions_path"),
            metrics=data.get("metrics") or {},
            status=data.get("status") or "archived",
        )


@dataclass
class Registry:
    """The full model registry."""

    schema_version: str = SCHEMA_VERSION
    production_version: Optional[str] = None
    models: List[ModelEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "production_version": self.production_version,
            "models": [m.to_dict() for m in self.models],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Registry":
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            production_version=data.get("production_version"),
            models=[ModelEntry.from_dict(m) for m in data.get("models", [])],
        )

    def get_model(self, version: str) -> Optional[ModelEntry]:
        """Get a specific model version."""
        for model in self.models:
            if model.version == version:
                return model
        return None

    def get_latest_version(self) -> Optional[str]:
        """Get the latest registered version."""
        if not self.models:
            return None
        return self.models[-1].version


def _registry_storage_root(registry_path: Path) -> Path:
    """Base directory used to persist relative artifact paths."""
    return Path(registry_path).expanduser().resolve().parent.parent


def _serialize_registry_path(
    path: Optional[Path], registry_path: Path
) -> Optional[str]:
    """Persist artifact paths relative to the project root when possible."""
    if path is None:
        return None

    resolved_path = Path(path).expanduser().resolve()
    storage_root = _registry_storage_root(registry_path)
    try:
        return str(resolved_path.relative_to(storage_root))
    except ValueError:
        return str(resolved_path)


def resolve_registry_entry_path(
    stored_path: Optional[str],
    registry_path: Path = REGISTRY_PATH,
) -> Optional[Path]:
    """Resolve a stored registry path, supporting both relative and legacy absolute entries."""
    if stored_path is None:
        return None

    path = Path(stored_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    return (_registry_storage_root(registry_path) / path).resolve()


def _sync_statuses(registry: Registry) -> None:
    """Keep persisted per-model status aligned with production_version."""
    for model in registry.models:
        model.status = "archived"

    if registry.production_version is None:
        return

    production_model = registry.get_model(registry.production_version)
    if production_model is not None:
        production_model.status = "production"


def load_registry(path: Path = REGISTRY_PATH) -> Registry:
    """Load registry from disk, or return empty registry if not found."""
    if not path.exists():
        return Registry()

    with open(path) as f:
        data = json.load(f)
    registry = Registry.from_dict(data)
    _sync_statuses(registry)
    return registry


def save_registry_atomic(registry: Registry, path: Path = REGISTRY_PATH) -> None:
    """
    Save registry to disk atomically.

    Uses temp file + os.replace() to ensure the registry is never corrupted
    even if the process is interrupted mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _sync_statuses(registry)

    # Write to temp file in same directory (ensures same filesystem for atomic rename)
    fd, temp_path = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(registry.to_dict(), f, indent=2, allow_nan=False)
            f.write("\n")
        # Atomic rename
        os.replace(temp_path, path)
    except Exception:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def parse_version(version: str) -> tuple:
    """Parse semantic version string to tuple of ints."""
    parts = version.split(".")
    return tuple(int(p) for p in parts)


def format_version(major: int, minor: int, patch: int) -> str:
    """Format version tuple to string."""
    return f"{major}.{minor}.{patch}"


def get_next_version(registry: Registry, bump: str = "patch") -> str:
    """
    Compute next semantic version.

    Args:
        registry: Current registry.
        bump: Type of version bump ("major", "minor", "patch").

    Returns:
        Next version string.

    Examples:
        1.0.0 + patch -> 1.0.1
        1.0.1 + minor -> 1.1.0
        1.1.0 + major -> 2.0.0
    """
    latest = registry.get_latest_version()

    if latest is None:
        return "1.0.0"

    major, minor, patch = parse_version(latest)

    if bump == "major":
        return format_version(major + 1, 0, 0)
    elif bump == "minor":
        return format_version(major, minor + 1, 0)
    else:  # patch
        return format_version(major, minor, patch + 1)


def get_version_dir(version: str) -> Path:
    """Get the directory path for a model version."""
    return MODELS_DIR / f"v{version}"


def register_model(
    model_path: Path,
    metadata_path: Path,
    reference_distributions_path: Optional[Path] = None,
    metrics: Optional[Dict[str, Any]] = None,
    set_as_production: bool = True,
    version_bump: str = "patch",
    registry_path: Path = REGISTRY_PATH,
) -> ModelEntry:
    """
    Register a new model version.

    This function:
    1. Computes the next version number
    2. Creates the version directory
    3. Copies artifacts to the version directory
    4. Updates the registry

    Args:
        model_path: Path to the model artifact (.joblib).
        metadata_path: Path to the metadata JSON.
        reference_distributions_path: Optional path to drift reference distributions.
        set_as_production: Whether to set this version as production.
        version_bump: Type of version bump ("major", "minor", "patch").
        registry_path: Path to registry file.

    Returns:
        The new ModelEntry.
    """
    registry = load_registry(registry_path)
    version = get_next_version(registry, version_bump)
    version_dir = get_version_dir(version)

    # Create version directory
    version_dir.mkdir(parents=True, exist_ok=True)

    # Copy artifacts to version directory
    new_model_path = version_dir / "model.joblib"
    new_metadata_path = version_dir / "metadata.json"
    shutil.copy2(model_path, new_model_path)
    shutil.copy2(metadata_path, new_metadata_path)

    new_ref_dist_path = None
    if reference_distributions_path and reference_distributions_path.exists():
        new_ref_dist_path = version_dir / "reference_distributions.json"
        shutil.copy2(reference_distributions_path, new_ref_dist_path)

    # Create entry
    entry = ModelEntry(
        version=version,
        created_at=datetime.now(timezone.utc).isoformat(),
        artifact_path=_serialize_registry_path(new_model_path, registry_path) or "",
        metadata_path=_serialize_registry_path(new_metadata_path, registry_path) or "",
        reference_distributions_path=_serialize_registry_path(
            new_ref_dist_path, registry_path
        )
        if new_ref_dist_path
        else None,
        metrics=metrics or {},
        status="production" if set_as_production else "archived",
    )

    # Update registry
    registry.models.append(entry)
    if set_as_production:
        registry.production_version = version

    save_registry_atomic(registry, registry_path)

    return entry


def get_production_model(registry_path: Path = REGISTRY_PATH) -> Optional[ModelEntry]:
    """Get the current production model entry."""
    registry = load_registry(registry_path)

    if registry.production_version is None:
        return None

    return registry.get_model(registry.production_version)


def get_production_model_path(registry_path: Path = REGISTRY_PATH) -> Path:
    """
    Get path to production model, with fallback to legacy MODEL_PATH.

    This maintains backward compatibility with the old single-model setup.
    """
    entry = get_production_model(registry_path)

    if entry is not None:
        resolved_path = resolve_registry_entry_path(entry.artifact_path, registry_path)
        if resolved_path is not None:
            return resolved_path

    # Fallback to legacy path
    if MODEL_PATH.exists():
        return MODEL_PATH

    raise FileNotFoundError(
        f"No production model found in registry and no legacy model at {MODEL_PATH}"
    )


def promote_to_production(version: str, registry_path: Path = REGISTRY_PATH) -> None:
    """
    Promote a specific version to production.

    This only changes the production_version pointer, no files are copied.

    Args:
        version: Version to promote.
        registry_path: Path to registry file.

    Raises:
        ValueError: If version doesn't exist.
    """
    registry = load_registry(registry_path)

    if registry.get_model(version) is None:
        available = [m.version for m in registry.models]
        raise ValueError(
            f"Version '{version}' not found in registry. "
            f"Available versions: {available}"
        )

    registry.production_version = version
    save_registry_atomic(registry, registry_path)


def rollback_production(to_version: str, registry_path: Path = REGISTRY_PATH) -> None:
    """
    Rollback production to a previous version.

    This is an alias for promote_to_production for semantic clarity.
    """
    promote_to_production(to_version, registry_path)


def list_versions(registry_path: Path = REGISTRY_PATH) -> List[Dict[str, Any]]:
    """
    List all registered model versions with their status.

    Returns list of dicts with version info and production status.
    """
    registry = load_registry(registry_path)

    result = []
    for model in registry.models:
        info = model.to_dict()
        info["is_production"] = model.status == "production"
        result.append(info)

    return result


# CLI interface
def main() -> None:
    """CLI for model registry operations."""
    import argparse

    parser = argparse.ArgumentParser(description="Model registry management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    subparsers.add_parser("list", help="List all model versions")

    # Promote command
    promote_parser = subparsers.add_parser(
        "promote", help="Promote version to production"
    )
    promote_parser.add_argument("--version", required=True, help="Version to promote")

    # Rollback command
    rollback_parser = subparsers.add_parser(
        "rollback", help="Rollback to previous version"
    )
    rollback_parser.add_argument(
        "--version", required=True, help="Version to rollback to"
    )

    args = parser.parse_args()

    if args.command == "list":
        versions = list_versions()
        if not versions:
            print("No models registered yet.")
            return

        print(
            f"{'Version':<10} {'Created':<25} {'Status':<12} {'ROC-AUC':<10} {'P@10':<10}"
        )
        print("-" * 74)
        for v in versions:
            created = v["created_at"][:19].replace("T", " ")
            metrics = v.get("metrics", {})
            roc_auc = metrics.get("roc_auc")
            precision_at_10 = metrics.get("precision_at_10")
            roc_auc_display = (
                f"{roc_auc:.4f}" if isinstance(roc_auc, (int, float)) else "n/a"
            )
            precision_display = (
                f"{precision_at_10:.4f}"
                if isinstance(precision_at_10, (int, float))
                else "n/a"
            )
            print(
                f"{v['version']:<10} {created:<25} {v['status']:<12} "
                f"{roc_auc_display:<10} {precision_display:<10}"
            )

    elif args.command == "promote":
        promote_to_production(args.version)
        print(f"✓ Promoted version {args.version} to production")

    elif args.command == "rollback":
        rollback_production(args.version)
        print(f"✓ Rolled back production to version {args.version}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
