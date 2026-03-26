# CLAUDE.md

## Project Identity
This repository implements an AI Lead Scoring & Automation Agent.

Its purpose is to:
- estimate conversion probability for incoming leads
- rank leads by business priority
- support batch scoring workflows
- support future API deployment with FastAPI
- remain realistic for pre-contact scoring use cases

This is not an academic-only repository.
Changes should improve deployability, reproducibility, business usefulness, or code clarity.

---

## Primary Product Goal
The main product goal is to enable a business to:

1. load new lead data
2. score each lead with a trained pipeline
3. rank leads from highest to lowest conversion potential
4. use these rankings in sales or marketing workflows

The repository should evolve toward a usable scoring service, not toward notebook sprawl.

---

## Core Modeling Constraint
This project is framed as a **pre-contact lead scoring system**.

### Leakage-sensitive feature
The feature `duration` must be treated as leakage for production-oriented scoring:
- it may be used in EDA for analysis
- it may be discussed in notebooks as a strong post-contact signal
- it must not be used in the final pre-contact production scoring pipeline unless explicitly building a post-interaction model

If modifying training or inference logic, preserve this rule.

---

## Preferred Repository Architecture
Claude should favor this structure:

- `data/`
  - `raw/`
  - `processed/`
  - `sample/`

- `notebooks/`
  - exploratory and explanatory only
  - not the main home of reusable production logic

- `src/`
  - reusable Python modules
  - training
  - inference
  - schema handling
  - utility functions
  - API logic if needed

- `models/`
  - saved artifacts such as `.joblib`
  - metadata files
  - optional schema snapshots

- `outputs/`
  - scored CSV outputs
  - metrics summaries
  - evaluation artifacts

- `app/` or `api/`
  - FastAPI service
  - request/response models
  - health endpoint
  - predict endpoint

- `tests/`
  - smoke tests
  - schema tests
  - inference tests

---

## What Claude Should Optimize For
When making changes, optimize for:

1. correctness
2. reproducibility
3. deployment readiness
4. business interpretability
5. simplicity over unnecessary sophistication

Do not add complexity unless it clearly improves:
- reliability
- maintainability
- business value
- deployment readiness

---

## Business Framing Rules
Always treat this as a **lead prioritization problem**, not only a binary classification problem.

Preferred business outputs:
- probability score
- ranked leads
- top-K lead segments
- Precision@K
- uplift over baseline conversion rate
- cumulative gain

Avoid presenting the project as if accuracy were the main success criterion.

---

## Modeling Rules

### Use sklearn Pipelines
Keep preprocessing inside sklearn-compatible pipelines.

Do not:
- preprocess manually outside the pipeline for production paths
- hardcode encoded feature matrices by hand
- rely on notebook state for inference logic

### Expected preprocessing logic
- numerical features: imputation + scaling if needed
- ordinal categorical features: ordinal encoding with explicit category order
- nominal categorical features: one-hot encoding with `handle_unknown="ignore"`

### Preferred model strategy
Use a small, defensible set of models:
- Logistic Regression as baseline
- Random Forest / Gradient Boosting for comparison
- XGBoost as primary production candidate

### Tuning philosophy
Do not over-invest in hyperparameter tuning if gains are marginal.
If results show diminishing returns, prioritize:
- cleaner code
- stronger inference
- API readiness
- feature quality
over further tuning

---

## Batch Scoring Requirements
Batch scoring is a first-class product requirement.

Claude should support workflows like:
- load CSV of new leads
- validate schema
- run pipeline inference
- produce:
  - `score`
  - optional `predicted_label`
  - ranked output
- save scored CSV to `outputs/`

Preferred file:
- `src/inference.py`

Preferred behavior:
- fail loudly on missing required columns
- preserve original rows
- append output columns instead of mutating destructively
- sort by `score` descending when producing ranked outputs

Suggested output columns:
- `score`
- `predicted_label`
- `priority_rank`

---

## FastAPI Deployment Requirements
FastAPI should be treated as the preferred deployment interface.

Suggested API behavior:

### `/health`
Returns service health and model availability.

### `/predict`
Accepts one lead or a batch of leads and returns:
- conversion probability
- predicted label
- optional rank if batch scoring

### API principles
- load model artifact once at startup
- validate request schema with Pydantic
- do not require external preprocessing from the caller
- return clear errors when required fields are missing
- keep response format stable and explicit

Suggested modules:
- `app/main.py`
- `app/schemas.py`
- `app/model_loader.py`
- `app/predictor.py`

---

## Artifact Management Rules
Preferred saved artifact:
- one full pipeline artifact, e.g. `models/tuned_xgb_pipeline.joblib`

This full artifact should already include:
- preprocessing
- trained model

Avoid saving preprocessor and model separately unless there is a strong reason.

Recommended additional metadata:
- model name
- target name
- excluded features
- training date
- evaluation summary
- expected input schema

Example companion files:
- `models/model_metadata.json`
- `models/input_schema.json`

---

## Schema Discipline
Schema mismatches are common production failures.

Claude should help enforce:
- expected column names
- expected dtypes when practical
- safe handling of unseen nominal categories
- explicit error messages for missing columns

If helpful, create reusable schema utilities.

Suggested module:
- `src/schema.py`

---

## Testing Expectations
All production-oriented code should support at least lightweight tests.

Useful tests include:
- model reload smoke test
- single-row inference test
- batch scoring test
- missing-column failure test
- API health endpoint test
- API predict endpoint test

If tests are added, keep them small and practical.

---

## Documentation Requirements
Documentation should explain both:
- what the code does
- why it matters for business

When writing docs:
- avoid inflated claims
- avoid vague wording
- connect technical choices to lead prioritization outcomes

Good documentation topics:
- why `duration` is excluded
- why ranking matters more than raw accuracy
- why Precision@K is important
- how batch scoring works
- how to run the API

---

## Code Style Preferences
Claude should prefer:
- small, composable functions
- explicit naming
- clear path handling
- minimal but useful comments
- strong separation between notebooks and production code

Avoid:
- giant scripts with mixed responsibilities
- hidden assumptions about file locations
- notebook-only utility logic being copied around
- magic constants without explanation

---

## Preferred Next Deliverables
When asked to extend the project, the preferred order is:

1. export + reload validation
2. batch scoring script
3. FastAPI service
4. request/response schemas
5. smoke tests
6. improved README
7. optional Dockerization

Notebooks are not the default next step unless explicitly requested.

---

## Decision Heuristics
When multiple options exist, prefer the one that:
- can be reused outside notebooks
- is easier to deploy
- preserves business framing
- reduces risk of leakage
- improves trust in outputs

---

## Examples of Good Contributions

### Good
- adding `src/inference.py` for CSV scoring
- adding `app/main.py` with `/health` and `/predict`
- exporting a single full pipeline artifact
- adding schema validation for incoming leads
- writing README sections on business metrics

### Not Good
- adding extra notebooks with duplicated logic
- adding many models without clear purpose
- using `duration` in production scoring without explicit justification
- optimizing only accuracy
- separating preprocessing from the saved final pipeline without necessity

---

## Definition of Done
A contribution is done when:
- it runs
- it respects pre-contact scoring constraints
- it works outside notebook state
- it improves deployment readiness or business usefulness
- it does not introduce leakage or silent schema risk