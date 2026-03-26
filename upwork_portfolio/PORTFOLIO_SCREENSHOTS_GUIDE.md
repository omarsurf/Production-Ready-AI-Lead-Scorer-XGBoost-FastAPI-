# Portfolio Screenshots Guide for Upwork

## Generated Visuals (Ready to Use)

All these files are in `upwork_portfolio/screenshots/`:

### 1. Feature Importance (`feature_importance.png`)
**Purpose:** Proves NO DATA LEAKAGE - shows `duration` is excluded from the model.

**Use Case:** Add to portfolio with caption:
> "XGBoost Feature Importance - Pre-contact features only. The 'duration' feature (call time) was explicitly excluded to prevent data leakage, ensuring the model works for real-world lead prioritization BEFORE contact."

---

### 2. Cumulative Gain Curve (`cumulative_gain_curve.png`)
**Purpose:** THE business value chart - shows ROI of using the model.

**Use Case:** Add to portfolio with caption:
> "By targeting just the top 10% of leads ranked by our model, sales teams capture 45% of all conversions - 4.5x better than random calling."

---

### 3. Precision@K Comparison (`precision_at_k_comparison.png`)
**Purpose:** Concrete conversion rate improvements per segment.

**Use Case:** Add to portfolio with caption:
> "Top 10% leads show 53% conversion rate vs 11.7% baseline - enabling focused, high-ROI sales efforts."

---

### 4. Scored Leads Preview (`scored_leads_preview.png`)
**Purpose:** Shows the DELIVERABLE - what the client's sales team actually receives.

**Use Case:** Add to portfolio with caption:
> "Daily batch scoring output: Each lead gets a conversion probability score (0-100%) and priority ranking (HIGH/MEDIUM/LOW) for CRM integration."

---

## Manual Screenshots Needed

### 5. Cover Dashboard (`cover_dashboard.html`)
**How to capture:**
```bash
# Open the dashboard in your browser
open upwork_portfolio/cover_dashboard.html
```
Then take a full-page screenshot (1200x750px optimal).

**Purpose:** Your Upwork thumbnail/cover image showing a polished, production-ready interface.

---

### 6. FastAPI Swagger UI (API Documentation)
**How to capture:**
```bash
# Start the API server
cd /Users/omarpiro/ML_DL_Projects/AI_LEAD_SCORE
make serve
# OR
uvicorn app.main:app --reload
```

Then open: http://localhost:8000/docs

**What to capture:**
1. Full Swagger UI overview showing all 3 endpoints
2. Expanded `/predict/batch` endpoint showing the request schema

**Purpose:** Proves you can DEPLOY models, not just train them in notebooks.

---

## Recommended Upwork Portfolio Layout

### Image 1: Cover (Thumbnail)
Use `cover_dashboard.html` screenshot - it's designed for this purpose.

### Image 2: Business Value
Use `cumulative_gain_curve.png` - speaks directly to ROI.

### Image 3: API Deployment
Use FastAPI Swagger screenshot - proves production capability.

### Image 4: Model Rigor
Use `feature_importance.png` - shows ML best practices (no leakage).

### Image 5: Deliverable
Use `scored_leads_preview.png` - shows concrete output.

---

## Screenshot Tips for Upwork

1. **Resolution:** Use 2x Retina display if available, export at 1920x1080 minimum
2. **Browser:** Use Chrome/Safari in incognito mode (no extensions/bookmarks visible)
3. **Dark Mode:** All visuals are designed with dark mode - looks professional
4. **Consistency:** Keep the same browser window size for all screenshots

---

## Quick Regeneration

If you need to regenerate the automated visuals:
```bash
python upwork_portfolio/generate_portfolio_visuals.py
```

---

## File Checklist

- [ ] `screenshots/feature_importance.png` (auto-generated)
- [ ] `screenshots/cumulative_gain_curve.png` (auto-generated)
- [ ] `screenshots/precision_at_k_comparison.png` (auto-generated)
- [ ] `screenshots/scored_leads_preview.png` (auto-generated)
- [ ] `screenshots/cover_dashboard.png` (manual - from HTML)
- [ ] `screenshots/fastapi_swagger.png` (manual - from localhost)

Good luck with your Upwork portfolio!
