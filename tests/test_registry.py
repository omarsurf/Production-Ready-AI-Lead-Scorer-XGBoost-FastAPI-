"""Tests for src/registry.py - Model versioning and registry."""

import json

import pytest

from src.registry import (
    ModelEntry,
    Registry,
    format_version,
    get_next_version,
    get_production_model,
    get_production_model_path,
    list_versions,
    load_registry,
    parse_version,
    promote_to_production,
    register_model,
    resolve_registry_entry_path,
    rollback_production,
    save_registry_atomic,
)


class TestVersioning:
    """Tests for version parsing and incrementing."""

    def test_parse_version(self):
        """Should parse semantic version string to tuple."""
        assert parse_version("1.0.0") == (1, 0, 0)
        assert parse_version("2.3.4") == (2, 3, 4)
        assert parse_version("10.20.30") == (10, 20, 30)

    def test_format_version(self):
        """Should format tuple to version string."""
        assert format_version(1, 0, 0) == "1.0.0"
        assert format_version(2, 3, 4) == "2.3.4"

    def test_get_next_version_empty_registry(self):
        """Empty registry should start at 1.0.0."""
        registry = Registry()
        assert get_next_version(registry) == "1.0.0"

    def test_get_next_version_patch(self):
        """Patch bump: 1.0.0 -> 1.0.1"""
        registry = Registry(
            models=[
                ModelEntry(
                    version="1.0.0", created_at="", artifact_path="", metadata_path=""
                )
            ]
        )
        assert get_next_version(registry, "patch") == "1.0.1"

    def test_get_next_version_minor(self):
        """Minor bump: 1.0.5 -> 1.1.0"""
        registry = Registry(
            models=[
                ModelEntry(
                    version="1.0.5", created_at="", artifact_path="", metadata_path=""
                )
            ]
        )
        assert get_next_version(registry, "minor") == "1.1.0"

    def test_get_next_version_major(self):
        """Major bump: 1.2.3 -> 2.0.0"""
        registry = Registry(
            models=[
                ModelEntry(
                    version="1.2.3", created_at="", artifact_path="", metadata_path=""
                )
            ]
        )
        assert get_next_version(registry, "major") == "2.0.0"


class TestRegistryPersistence:
    """Tests for registry save/load."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Registry should survive save/load roundtrip."""
        registry_path = tmp_path / "registry.json"

        registry = Registry(
            production_version="1.0.0",
            models=[
                ModelEntry(
                    version="1.0.0",
                    created_at="2024-01-01T00:00:00Z",
                    artifact_path="models/v1.0.0/model.joblib",
                    metadata_path="models/v1.0.0/metadata.json",
                )
            ],
        )

        save_registry_atomic(registry, registry_path)
        loaded = load_registry(registry_path)

        assert loaded.production_version == "1.0.0"
        assert len(loaded.models) == 1
        assert loaded.models[0].version == "1.0.0"
        assert loaded.models[0].status == "production"

    def test_load_nonexistent_returns_empty(self, tmp_path):
        """Loading non-existent registry should return empty registry."""
        registry_path = tmp_path / "nonexistent.json"

        registry = load_registry(registry_path)

        assert registry.production_version is None
        assert len(registry.models) == 0

    def test_atomic_write_creates_valid_json(self, tmp_path):
        """Atomic write should create valid JSON file."""
        registry_path = tmp_path / "registry.json"
        registry = Registry(production_version="1.0.0", models=[])

        save_registry_atomic(registry, registry_path)

        # Verify it's valid JSON
        with open(registry_path) as f:
            data = json.load(f)

        assert data["production_version"] == "1.0.0"


class TestModelRegistration:
    """Tests for registering new models."""

    def test_register_model_creates_version_dir(self, tmp_path):
        """Registration should create version directory with artifacts."""
        # Setup
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"fake model")
        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text('{"version": "test"}')
        registry_path = tmp_path / "models" / "registry.json"

        # Patch MODELS_DIR for test
        import src.registry as reg

        original_models_dir = reg.MODELS_DIR
        reg.MODELS_DIR = tmp_path / "models"

        try:
            entry = register_model(
                model_path=model_path,
                metadata_path=metadata_path,
                registry_path=registry_path,
            )

            assert entry.version == "1.0.0"
            assert entry.artifact_path == "models/v1.0.0/model.joblib"
            assert entry.metadata_path == "models/v1.0.0/metadata.json"
            assert resolve_registry_entry_path(
                entry.artifact_path, registry_path
            ).exists()
            assert resolve_registry_entry_path(
                entry.metadata_path, registry_path
            ).exists()

            # Check registry was updated
            registry = load_registry(registry_path)
            assert registry.production_version == "1.0.0"
            assert len(registry.models) == 1
        finally:
            reg.MODELS_DIR = original_models_dir

    def test_register_model_increments_version(self, tmp_path):
        """Second registration should increment version."""
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"fake model")
        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text('{"version": "test"}')
        registry_path = tmp_path / "models" / "registry.json"

        import src.registry as reg

        original_models_dir = reg.MODELS_DIR
        reg.MODELS_DIR = tmp_path / "models"

        try:
            entry1 = register_model(
                model_path=model_path,
                metadata_path=metadata_path,
                registry_path=registry_path,
            )
            entry2 = register_model(
                model_path=model_path,
                metadata_path=metadata_path,
                registry_path=registry_path,
            )

            assert entry1.version == "1.0.0"
            assert entry2.version == "1.0.1"
        finally:
            reg.MODELS_DIR = original_models_dir

    def test_register_model_persists_metrics_and_production_status(self, tmp_path):
        """Registered entries should persist metrics and expose production status."""
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"fake model")
        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text('{"version": "test"}')
        registry_path = tmp_path / "models" / "registry.json"
        metrics = {"roc_auc": 0.8123, "precision_at_10": 0.45}

        import src.registry as reg

        original_models_dir = reg.MODELS_DIR
        reg.MODELS_DIR = tmp_path / "models"

        try:
            entry = register_model(
                model_path=model_path,
                metadata_path=metadata_path,
                metrics=metrics,
                registry_path=registry_path,
            )
            loaded = load_registry(registry_path)
            registered_model = loaded.get_model(entry.version)

            assert registered_model is not None
            assert registered_model.metrics == metrics
            assert registered_model.status == "production"
        finally:
            reg.MODELS_DIR = original_models_dir


class TestPromotion:
    """Tests for promote and rollback."""

    def test_promote_to_production(self, tmp_path):
        """Should change production_version without copying files."""
        registry_path = tmp_path / "registry.json"
        registry = Registry(
            production_version="1.0.0",
            models=[
                ModelEntry(
                    version="1.0.0", created_at="", artifact_path="a", metadata_path="m"
                ),
                ModelEntry(
                    version="1.0.1", created_at="", artifact_path="b", metadata_path="n"
                ),
            ],
        )
        save_registry_atomic(registry, registry_path)

        promote_to_production("1.0.1", registry_path)

        loaded = load_registry(registry_path)
        assert loaded.production_version == "1.0.1"
        assert loaded.get_model("1.0.0").status == "archived"
        assert loaded.get_model("1.0.1").status == "production"

    def test_promote_nonexistent_version_raises(self, tmp_path):
        """Should raise ValueError for non-existent version."""
        registry_path = tmp_path / "registry.json"
        registry = Registry(
            production_version="1.0.0",
            models=[
                ModelEntry(
                    version="1.0.0", created_at="", artifact_path="a", metadata_path="m"
                ),
            ],
        )
        save_registry_atomic(registry, registry_path)

        with pytest.raises(ValueError, match="not found"):
            promote_to_production("9.9.9", registry_path)

    def test_rollback_production(self, tmp_path):
        """Rollback should change production_version."""
        registry_path = tmp_path / "registry.json"
        registry = Registry(
            production_version="1.0.1",
            models=[
                ModelEntry(
                    version="1.0.0", created_at="", artifact_path="a", metadata_path="m"
                ),
                ModelEntry(
                    version="1.0.1", created_at="", artifact_path="b", metadata_path="n"
                ),
            ],
        )
        save_registry_atomic(registry, registry_path)

        rollback_production("1.0.0", registry_path)

        loaded = load_registry(registry_path)
        assert loaded.production_version == "1.0.0"


class TestProductionModel:
    """Tests for getting production model."""

    def test_get_production_model(self, tmp_path):
        """Should return production model entry."""
        registry_path = tmp_path / "registry.json"
        registry = Registry(
            production_version="1.0.1",
            models=[
                ModelEntry(
                    version="1.0.0", created_at="", artifact_path="a", metadata_path="m"
                ),
                ModelEntry(
                    version="1.0.1", created_at="", artifact_path="b", metadata_path="n"
                ),
            ],
        )
        save_registry_atomic(registry, registry_path)

        entry = get_production_model(registry_path)

        assert entry is not None
        assert entry.version == "1.0.1"

    def test_get_production_model_none_when_empty(self, tmp_path):
        """Should return None when no production version set."""
        registry_path = tmp_path / "registry.json"
        registry = Registry()
        save_registry_atomic(registry, registry_path)

        entry = get_production_model(registry_path)

        assert entry is None


class TestListVersions:
    """Tests for listing versions."""

    def test_list_versions_includes_production_flag(self, tmp_path):
        """Should include is_production flag for each version."""
        registry_path = tmp_path / "registry.json"
        registry = Registry(
            production_version="1.0.1",
            models=[
                ModelEntry(
                    version="1.0.0",
                    created_at="2024-01-01",
                    artifact_path="a",
                    metadata_path="m",
                ),
                ModelEntry(
                    version="1.0.1",
                    created_at="2024-01-02",
                    artifact_path="b",
                    metadata_path="n",
                ),
            ],
        )
        save_registry_atomic(registry, registry_path)

        versions = list_versions(registry_path)

        assert len(versions) == 2
        assert versions[0]["version"] == "1.0.0"
        assert versions[0]["is_production"] is False
        assert versions[0]["status"] == "archived"
        assert versions[1]["version"] == "1.0.1"
        assert versions[1]["is_production"] is True
        assert versions[1]["status"] == "production"
        assert versions[1]["metrics"] == {}

    def test_list_versions_empty_registry(self, tmp_path):
        """Should return empty list for empty registry."""
        registry_path = tmp_path / "nonexistent.json"

        versions = list_versions(registry_path)

        assert versions == []


class TestBackwardCompatibility:
    """Tests for backward compatibility with legacy MODEL_PATH."""

    def test_load_registry_accepts_old_entries_without_metrics_or_status(
        self, tmp_path
    ):
        """Older registry files should remain readable without migration."""
        registry_path = tmp_path / "registry.json"
        registry_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "production_version": "1.0.0",
                    "models": [
                        {
                            "version": "1.0.0",
                            "created_at": "2024-01-01T00:00:00Z",
                            "artifact_path": "models/v1.0.0/model.joblib",
                            "metadata_path": "models/v1.0.0/metadata.json",
                        }
                    ],
                }
            )
        )

        loaded = load_registry(registry_path)

        assert loaded.models[0].metrics == {}
        assert loaded.models[0].status == "production"

    def test_get_production_model_path_fallback(self, tmp_path):
        """Should fallback to MODEL_PATH when no registry exists."""
        # Create a legacy model file
        legacy_model = tmp_path / "legacy_model.joblib"
        legacy_model.write_bytes(b"fake model")

        # Patch MODEL_PATH
        import src.registry as reg

        original_model_path = reg.MODEL_PATH
        reg.MODEL_PATH = legacy_model

        try:
            # No registry exists
            registry_path = tmp_path / "nonexistent_registry.json"
            path = get_production_model_path(registry_path)

            assert path == legacy_model
        finally:
            reg.MODEL_PATH = original_model_path

    def test_get_production_model_path_uses_registry(self, tmp_path):
        """Should use registry when it exists."""
        registry_path = tmp_path / "registry.json"
        model_path = tmp_path / "v1.0.0" / "model.joblib"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"fake model")

        registry = Registry(
            production_version="1.0.0",
            models=[
                ModelEntry(
                    version="1.0.0",
                    created_at="",
                    artifact_path=str(model_path),
                    metadata_path="",
                ),
            ],
        )
        save_registry_atomic(registry, registry_path)

        path = get_production_model_path(registry_path)

        assert path == model_path

    def test_resolve_registry_entry_path_supports_relative_entries(self, tmp_path):
        """Relative paths in the registry should resolve from the project root."""
        registry_path = tmp_path / "models" / "registry.json"
        stored_path = "models/v1.2.3/model.joblib"

        resolved = resolve_registry_entry_path(stored_path, registry_path)

        assert resolved == (tmp_path / stored_path).resolve()
