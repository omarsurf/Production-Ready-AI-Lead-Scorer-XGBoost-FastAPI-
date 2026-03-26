# AI Lead Scoring System
![CI](https://github.com/omarsurf/Production-Ready-AI-Lead-Scorer-XGBoost-FastAPI-/actions/workflows/ci.yml/badge.svg) ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)

I built a production-ready AI lead scoring system that optimizes sales outreach. Instead of aiming for pure statistical accuracy, it strictly avoids data leakage (pre-contact scoring), ranks incoming leads by conversion probability, and exposes both a batch CLI and a real-time FastAPI inference service to focus the sales effort on the most promising prospects.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Challenges Solved](#challenges-solved)
- [Built By Me](#built-by-me)
- [Production Mindset](#production-mindset)
- [Tech Stack](#tech-stack)
- [Lessons Learned](#lessons-learned)
- [Quick Start](#quick-start)
- [API & CLI Reference](#api--cli-reference)
- [Project Structure](#project-structure)

---

## Problem Statement

### The Business Challenge
Sales teams lose hundreds of hours annually calling low-qualifying leads. The naive approach - "predict who will convert and call everyone flagged" - fails because:

| Challenge | Why It's Hard |
| :--- | :--- |
| **Effort asymmetry** | Contacting non-converting leads wastes valuable sales time |
| **Data leakage trap** | Models trained on call `duration` look great on paper but fail in production because duration is a *post-contact* metric |
| **Accuracy is misleading** | Binary classification ignores the business need to *prioritize* a queue |

### The Solution
This pipeline reframes lead scoring as a **priority ranking problem**:

Instead of a simple "yes/no", the model assigns a **conversion probability** to every lead *before* contact. Leads are then sorted and banded into `HIGH`, `MEDIUM`, and `LOW` priority segments. This allows the business to measure success using **Precision@K**, **Uplift over baseline**, and **Cumulative Gain**—metrics that directly map to sales ROI.

---

## Architecture

### Artifact Flow
| Stage | Input | Output |
| :--- | :--- | :--- |
| **Prepare & Train** | Raw Bank Marketing CSV (45k rows) | `tuned_xgb_pipeline.joblib` (Includes sklearn preprocessor + model) |
| **Batch Predict** | New Leads CSV | `scored_leads.csv` (Appended with `score`, `predicted_label`, `priority_rank`) |
| **Live API** | JSON Payload (Single/Batch) | JSON Response with probability and ranking |

---

## Key Features

| Feature | Description | Why It Matters |
| :--- | :--- | :--- |
| **Pre-Contact Scoring** | Explicit exclusion of `duration` feature | Prevents data leakage; predictions remain entirely valid for untouched leads |
| **Unified Artifact** | Scikit-learn preprocessing integrated with XGBoost | Prevents training/serving skew by exporting a single `.joblib` pipeline |
| **Batch CLI** | Appends scores and priority ranks to CSVs | Perfect for nightly lead queue generation and CRM injestion |
| **FastAPI Service** | Real-time endpoints (`/predict`, `/predict/batch`) | Allows live webhooks to score inbound leads the moment they hit the form |
| **Schema Discipline** | Pydantic v2 validation via API | Rejects invalid payloads with HTTP 422 before they can crash the model |
| **Dockerized** | Multi-stage Python 3.12-slim build | Runtime image CPU-only allégée, prête à déployer |

---

## Challenges Solved

Real engineering problems encountered and resolved during development:

### 1. The Data Leakage Trap
**Problem:** The original UCI dataset includes a `duration` column (length of the phone call). Training on this feature yields artificially high accuracy metrics, but the model is useless in production since `duration` is only known *after* the sales team completes the call.

**Solution:** Enforced strict schema constraints to exclude `duration` completely, focusing entirely on pre-contact attributes (demographics, financial history, previous campaigns).

### 2. Business vs. Statistical Optimization
**Problem:** A standard classification model outputs binary labels (0 or 1). For a sales team of 5 people with 5,000 leads, "1" is not granular enough to dictate who to call first.

**Solution:** Optimized the pipeline to output `predict_proba`. Developed batch logic that sorts the output queue descending by `score` and assigns a definitive `priority_rank`, directly enabling "Precision@Top10%" business goals.

### 3. API Silent Failures & Unknown Categories
**Problem:** Passing unseen categorical data (e.g., a new `education` level) to a deployed model often results in silent failures, NaN predictions, or 500 server crashes.

**Solution:** 
1. Used `handle_unknown="ignore"` in the OneHotEncoder to ensure model stability.
2. Backed the FastAPI endpoints with Pydantic schemas validating all closed-vocabulary enumerations, failing fast with a 422 Unprocessable Entity error instead of a generic backend crash.

---

## Built By Me

This project demonstrates end-to-end ML engineering capabilities tailored for business operations:

| Capability | Evidence |
| :--- | :--- |
| **ML Pipeline Design** | Reproducible `src.training` pipeline outputting a cohesive unified artifact |
| **Business Alignment** | Prioritizing Cumulative Gain and Precision@K over theoretical accuracy |
| **Production Reliability** | Schema constraints and explicit REST APIs over notebook state |
| **Deployment readiness** | Multi-stage Dockerfile, Makefile shortcuts, and Uvicorn ASGI server |
| **Code Modularity** | Clean separation of `src/` (modeling), `app/` (API), and `tests/` |

---

## Production Mindset

This project was built with a real-world production lifecycle in mind:
- **No notebook sprawl** - Notebooks are strictly for EDA. All logic lives in modular Python packages.
- **Pinned Requirements** - `requirements-runtime.txt` et `requirements-dev.txt` séparent le runtime du setup local/test pour limiter le drift et garder l'image plus légère.
- **Schema Validation** - Pydantic models strictly validate every inbound API request.
- **Test Coverage** - Pytest suite covers both ML inference logic and HTTP API contracts.
- **Health Probes** - Standard `/health` endpoint for Kubernetes/Docker orchestrators.

---

## Tech Stack

| Category | Technology | Why This Choice |
| :--- | :--- | :--- |
| **ML Framework** | Scikit-learn + XGBoost | Unmatched robustness for tabular data and complex tree building |
| **API Serving** | FastAPI + Uvicorn | High performance, native async, auto-generates OpenAPI docs |
| **Validation** | Pydantic | Type-safe schema validation |
| **Data Processing** | Pandas, Numpy | Standardized data manipulation |
| **Testing** | Pytest | Fast and reliable unit/integration testing |
| **Containerization** | Docker | Consistent CPU-only runtime via a lean multi-stage build |

---

## Lessons Learned

1. **Beware the "Too Good To Be True" Feature:** 
Identify post-facto variables quickly during EDA. The business wants to predict the future, not summarize the past. Excluding `duration` was painful for the ROC curve but absolutely essential for reality.

2. **Categorical Encoders need Production Guards:**
Always set `handle_unknown='ignore'` when using OneHotEncoder. New categorical values *will* appear in production data, and the model must degrade gracefully rather than crash.

3. **Batch and Live serving require different UX:**
Batch scoring needs to preserve the input rows and append new columns (`score`, `rank`). Live scoring needs to be minimal, lighting fast, and return simple JSON objects.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/omarsurf/Production-Ready-AI-Lead-Scorer-XGBoost-FastAPI-.git
cd Production-Ready-AI-Lead-Scorer-XGBoost-FastAPI-

# Install local/dev dependencies via Make
make install
```

```bash
# Optional: install only production/runtime dependencies
make install-runtime
```

### Run Full Pipeline

```bash
# Train the model and generate the .joblib artifact
make train

# Run batch scoring on sample leads
make score
```

### Docker Deployment

```bash
# Build the CPU-only runtime image
make docker-build

# Run the FastAPI service detached
make docker-run

# Verify service health
curl http://localhost:8000/health
```

---

## API & CLI Reference

### Management Commands (CLI)

| Command | Description |
| :--- | :--- |
| `make train` | Reproducibly train the model pipeline and artifact |
| `make tune` | Run hyperparameter tuning for XGBoost |
| `make score` | Run the CLI batch scoring script on a test file |
| `make serve` | Spin up the FastAPI server locally |
| `make install-runtime` | Install only runtime dependencies |
| `make test` | Run the pytest suite |

### FastAPI Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status |
| `POST` | `/predict` | Single lead prediction |
| `POST` | `/predict/batch`| Batch prediction returning `input_index` AND `priority_rank` |

### Dependency Files

| File | Scope |
| :--- | :--- |
| `requirements-runtime.txt` | Production/runtime dependencies used by Docker |
| `requirements-dev.txt` | Runtime + test dependencies for local development |
| `requirements.txt` | Thin alias to the dev environment |

### Docker Notes

- The Docker image installs only `requirements-runtime.txt`.
- Healthchecks rely on Python's standard library, so test-only packages are not shipped in production.
- The optional `batch-scorer` profile expects the mounted sample file `/app/data/raw/bank+marketing/bank/bank.csv`.

---

## Project Structure

```text
AI_LEAD_SCORE/
├── app/
│   ├── main.py              # FastAPI application & endpoints
│   ├── schemas.py           # Pydantic input/output schemas
│   └── model_loader.py      # Artifact loading logic
├── data/                    # UCI Bank dataset
├── models/
│   └── tuned_xgb_pipeline.joblib # Production artifact
├── notebooks/               # EDA and exploration only
├── outputs/                 # Scored batch CSVs
├── src/
│   ├── config.py            # Feature definitions & constants
│   ├── evaluate.py          # Business evaluation metrics and reporting
│   ├── schema.py            # Data validation helpers
│   ├── inference.py         # Batch scoring logic
│   ├── metadata.py          # Model metadata + schema persistence
│   └── training.py          # ML pipeline construction
├── tests/
│   ├── test_api.py          # API contract tests
│   ├── test_evaluate.py     # Evaluation/report tests
│   ├── test_inference.py    # Batch logic tests
│   └── test_training.py     # Training/report serialization tests
├── Dockerfile               # Multi-stage production image
├── docker-compose.yml       # Local deployment stack
├── Makefile                 # Developer shortcuts
├── requirements-runtime.txt # Runtime dependencies only
├── requirements-dev.txt     # Runtime + test dependencies
└── requirements.txt         # Dev alias
```
