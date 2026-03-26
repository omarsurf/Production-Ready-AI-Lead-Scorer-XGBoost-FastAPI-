# claude_tasks.md

## Completed Tasks ✓
1. ✓ Build a reusable batch scoring script (`src/inference.py`)
2. ✓ Export and reload the final pipeline artifact
3. ✓ Create a FastAPI app with `/health` and `/predict` (`app/main.py`)
4. ✓ Add schema validation for incoming lead data (`src/schema.py`, `app/schemas.py`)
5. ✓ Improve README with business-oriented explanations
6. ✓ Add training script (`src/training.py`)
7. ✓ Add API tests (`tests/test_api.py`)
8. ✓ Pin dependencies (`requirements.txt`)
9. ✓ Add evaluation module with business metrics (`src/evaluate.py`)
10. ✓ Dockerfile + docker-compose for deployment
11. ✓ Add model metadata JSON (`src/metadata.py`)
12. ✓ Add structured logging (`src/logging_config.py`)
13. ✓ CI/CD pipeline (`.github/workflows/ci.yml`)

## Next Tasks
1. Add /metrics endpoint (Prometheus format)
2. Add rate limiting
3. Add model versioning / A/B testing support

## Important Constraints
- Do not use `duration` in the production scoring pipeline
- Keep preprocessing inside the saved pipeline
- Prefer ranking-oriented outputs over raw classification outputs
- Avoid adding complexity unless it improves deployability or reliability
