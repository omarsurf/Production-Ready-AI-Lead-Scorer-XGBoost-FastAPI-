# SKILL.md

## Skill Identity

You are working on a production-oriented machine learning project:
an AI Lead Scoring & Automation Agent.

This is not an academic exercise.

Your role is to:
- build a reliable lead scoring system
- make it usable outside notebooks
- ensure business relevance
- prepare it for deployment (batch + API)

---

## Core Mindset

Always think in this order:

1. Can this run outside a notebook?
2. Does this respect pre-contact constraints?
3. Does this improve business usefulness?
4. Is this reproducible?
5. Is this simple and maintainable?

If the answer is "no" to any of these, revise the solution.

---

## Project Framing

This is a lead prioritization system, not just a classifier.

The model must:
- output probabilities
- enable ranking
- support top-K decision making

Key concept:
The goal is NOT perfect classification  
The goal is better allocation of sales effort

---

## Critical Constraint: Data Leakage

Forbidden in production scoring:
- duration must NOT be used

Reason:
- it is only known after interaction
- it creates unrealistic performance

Allowed:
- use in EDA
- use in analysis notebooks (clearly labeled)

Required:
- exclude from final pipeline used in inference/API

---

## Pipeline Discipline

Always use a single pipeline that includes:
- preprocessing
- encoding
- model

Do NOT:
- preprocess manually before inference
- rely on notebook-transformed data
- separate preprocessing from model unless absolutely necessary

Correct pattern:

model = Pipeline([
    ("preprocessor", ...),
    ("classifier", ...)
])

---

## Feature Handling Rules

Numerical:
- impute (median preferred)
- scale if needed

Ordinal categorical:
- explicit order required
- use OrdinalEncoder with defined categories

Example:
education = ["unknown", "primary", "secondary", "tertiary"]

Nominal categorical:
- use OneHotEncoder
- must use handle_unknown="ignore"

---

## Modeling Strategy

Baseline first:
- Logistic Regression

Then comparison:
- Random Forest
- Gradient Boosting

Production candidate:
- XGBoost

---

## Tuning Philosophy

Do NOT over-optimize.

If tuning yields small gains:
- stop tuning
- move to deployment

Prefer:
- better features
- cleaner inference
- better documentation

over:
- marginal ROC-AUC gains

---

## Evaluation Philosophy

Avoid focusing on:
- accuracy alone

Focus on:
- ROC-AUC
- precision
- recall
- F1
- Precision@K
- uplift vs baseline
- cumulative gain

---

## Business Metrics First-Class

Always consider:

Precision@K:
What percentage of top leads convert?

Uplift:
How much better than baseline?

Gain curve:
How fast do we capture conversions?

---

## Inference Design Rules

Input:
- tabular data (DataFrame or JSON)
- must match training schema

Output:
- score (probability)
- predicted label (optional)
- ranking (for batch)

Required behavior:
- do not modify original columns destructively
- append results instead

Example output:
score
predicted_label
priority_rank

---

## Batch Scoring Expectations

Must support:

input: CSV of leads  
output: CSV with scores + ranking  

Requirements:
- sort by score descending
- preserve row integrity
- fail on missing required columns

---

## API Design Expectations

FastAPI should provide:

/health:
- returns status
- confirms model is loaded

/predict:
- accepts one or multiple leads
- returns:
  - probability
  - label

Rules:
- load model once at startup
- no external preprocessing required
- validate input schema
- clear error messages

---

## Artifact Management

Preferred artifact:
models/tuned_xgb_pipeline.joblib

This must include:
- preprocessing
- model

Optional:
- metadata JSON
- schema JSON

---

## Schema Discipline

Always assume input data can be wrong.

Must handle:
- missing columns → error
- unknown categories → safe (handled by encoder)
- unexpected format → explicit failure

---

## Code Organization Rules

src/:
- reusable logic only
- no notebook dependencies
- functions and classes only

notebooks/:
- exploration + storytelling
- no critical logic exclusively here

app/:
- API only
- no training logic

---

## Error Handling Philosophy

Fail fast and clearly.

Bad:
- silent failures
- implicit assumptions

Good:
- explicit error messages
- clear requirements

---

## Testing Expectations

Minimum:
- model reload works
- prediction works on 1 row
- prediction works on batch
- missing column triggers error

---

## Documentation Requirements

Every important component must answer:

- What does it do?
- Why does it matter for business?

Avoid:
- vague explanations
- purely technical descriptions

---

## What to Avoid

- notebook-only logic
- duplicated code
- over-tuning
- unnecessary models
- using leakage features silently
- mixing training and inference logic
- magic numbers without explanation

---

## What Good Work Looks Like

A strong contribution:

- improves ranking quality OR
- improves inference reliability OR
- improves deployment readiness OR
- improves clarity of business value

---

## Decision Heuristic

If unsure, choose the option that:

- works outside notebooks
- is easier to deploy
- preserves business logic
- reduces risk of errors
- keeps the system understandable

---

## Definition of Done

A task is complete when:

- code runs independently
- respects pre-contact constraints
- works with saved pipeline
- produces usable outputs
- improves business usefulness
- is understandable by another developer