.PHONY: install install-runtime test serve score train tune tune-business list-models promote rollback drift-check evaluate docker-build docker-run docker-stop clean help

# Variables
PYTHON = python
PIP = pip
UVICORN = uvicorn
PYTEST = pytest

# Fichiers par défaut
INPUT_FILE = data/raw/bank+marketing/bank/bank-full.csv
OUTPUT_FILE = outputs/scored_leads.csv

help:
	@echo "AI Lead Scoring - Commandes disponibles"
	@echo ""
	@echo "  make install    Installer les dépendances"
	@echo "  make install-runtime  Installer uniquement les dépendances runtime"
	@echo "  make test       Lancer les tests"
	@echo "  make serve      Démarrer l'API FastAPI"
	@echo "  make score      Scorer le fichier exemple"
	@echo "  make train      Réentraîner le pipeline final"
	@echo "  make tune       Relancer le tuning XGBoost"
	@echo "  make evaluate   Évaluer le modèle (métriques business)"
	@echo "  make docker-build  Construire l'image Docker"
	@echo "  make docker-run    Lancer l'API via Docker"
	@echo "  make docker-stop   Arrêter le container Docker"
	@echo "  make clean      Nettoyer les fichiers générés"
	@echo ""

install:
	$(PIP) install -r requirements.txt

install-runtime:
	$(PIP) install -r requirements-runtime.txt

test:
	$(PYTEST) tests/ -v

serve:
	$(UVICORN) app.main:app --reload --host 0.0.0.0 --port 8000

score:
	$(PYTHON) -m src.inference --input $(INPUT_FILE) --output $(OUTPUT_FILE)

train:
	$(PYTHON) -m src.training

tune:
	$(PYTHON) -m src.training --tune

tune-business:
	@echo "Alias of make tune: multi-metric tuning (ROC-AUC + Precision@10), refit on business metric..."
	$(PYTHON) -m src.training --tune

# Model registry commands
list-models:
	$(PYTHON) -m src.registry list

promote:
	$(PYTHON) -m src.registry promote --version $(VERSION)

rollback:
	$(PYTHON) -m src.registry rollback --version $(VERSION)

# Drift detection
drift-check:
	$(PYTHON) -m src.drift --input $(INPUT_FILE) --output outputs/drift_report.json

evaluate:
	$(PYTHON) -m src.evaluate --input $(INPUT_FILE) --output outputs/evaluation_report.json

docker-build:
	docker compose build api

docker-run:
	docker compose up -d api
	@echo "✓ API disponible sur http://localhost:8000"
	@echo "  Health check: curl http://localhost:8000/health"

docker-stop:
	docker compose down

clean:
	rm -f outputs/*.csv
	rm -f outputs/*.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
