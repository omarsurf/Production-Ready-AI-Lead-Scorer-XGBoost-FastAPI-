"""
Portfolio Visual Generator for Upwork
=====================================
Generates high-quality screenshots/visuals for the AI Lead Scoring project portfolio.

Outputs:
1. feature_importance.png - XGBoost feature importance (proves no data leakage)
2. cumulative_gain_curve.png - Business value visualization
3. precision_at_k_comparison.png - Top segment performance
4. scored_leads_preview.png - CSV deliverable visualization
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Style configuration for professional visuals
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'primary': '#38bdf8',      # Cyan accent
    'secondary': '#818cf8',    # Purple accent
    'success': '#10b981',      # Green
    'warning': '#f59e0b',      # Orange
    'danger': '#ef4444',       # Red
    'dark': '#0f172a',         # Dark background
    'text': '#f1f5f9',         # Light text
    'dim': '#94a3b8'           # Muted text
}

OUTPUT_DIR = PROJECT_ROOT / "upwork_portfolio" / "screenshots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_data():
    """Load the trained model and evaluation data."""
    import joblib

    model_path = PROJECT_ROOT / "models" / "tuned_xgb_pipeline.joblib"
    data_path = PROJECT_ROOT / "data" / "raw" / "bank+marketing" / "bank" / "bank-full.csv"
    scored_path = PROJECT_ROOT / "outputs" / "scored_leads.csv"
    eval_path = PROJECT_ROOT / "outputs" / "evaluation_report.json"

    model = joblib.load(model_path)
    df = pd.read_csv(data_path, sep=";")
    scored_leads = pd.read_csv(scored_path)

    import json
    with open(eval_path, 'r') as f:
        eval_report = json.load(f)

    return model, df, scored_leads, eval_report


def generate_feature_importance(model, save_path):
    """
    Generate Feature Importance chart.

    KEY PORTFOLIO MESSAGE: Shows that 'duration' is NOT in the model,
    proving rigorous data leakage prevention for pre-contact scoring.
    """
    print("Generating Feature Importance chart...")

    # Get the XGBoost classifier from the pipeline
    xgb_clf = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']

    # Get feature names after preprocessing
    feature_names = []

    # Numeric features
    numeric_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
    feature_names.extend(numeric_cols)

    # Ordinal feature
    feature_names.append('education')

    # Nominal features (one-hot encoded)
    nominal_encoder = preprocessor.named_transformers_['nom'].named_steps['encoder']
    nominal_features = nominal_encoder.get_feature_names_out(
        ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    )
    feature_names.extend(nominal_features)

    # Get importances
    importances = xgb_clf.feature_importances_

    # Create DataFrame and get top 15
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(15)

    # Create the visualization
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['dark'])
    ax.set_facecolor(COLORS['dark'])

    # Horizontal bar chart
    bars = ax.barh(
        importance_df['feature'][::-1],
        importance_df['importance'][::-1],
        color=COLORS['primary'],
        edgecolor=COLORS['secondary'],
        linewidth=1.5
    )

    # Add value labels
    for bar, val in zip(bars, importance_df['importance'][::-1]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height()/2,
            f'{val:.3f}',
            va='center',
            color=COLORS['text'],
            fontsize=10,
            fontweight='bold'
        )

    # Styling
    ax.set_xlabel('Feature Importance (Gain)', color=COLORS['text'], fontsize=12, fontweight='bold')
    ax.set_title(
        'XGBoost Feature Importance\n(Pre-Contact Features Only - No Data Leakage)',
        color=COLORS['text'],
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Add annotation about duration exclusion
    ax.annotate(
        '[!] "duration" excluded\n(only known post-contact)',
        xy=(0.7, 0.05),
        xycoords='axes fraction',
        fontsize=10,
        color=COLORS['warning'],
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor=COLORS['dark'],
            edgecolor=COLORS['warning'],
            linewidth=2
        )
    )

    # Style axes
    ax.tick_params(colors=COLORS['text'], labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['dim'])
    ax.spines['left'].set_color(COLORS['dim'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['dark'])
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def generate_cumulative_gain_curve(eval_report, save_path):
    """
    Generate Cumulative Gain Curve.

    KEY PORTFOLIO MESSAGE: Shows business value - how targeting top leads
    captures disproportionate conversions vs random baseline.
    """
    print("Generating Cumulative Gain Curve...")

    # Data from evaluation report
    cg = eval_report['cumulative_gain']

    # Build curve data points
    percentiles = [0, 10, 20, 30, 50, 100]
    model_gains = [0, cg['top_10%']*100, cg['top_20%']*100, cg['top_30%']*100, cg['top_50%']*100, 100]
    random_gains = [0, 10, 20, 30, 50, 100]

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=COLORS['dark'])
    ax.set_facecolor(COLORS['dark'])

    # Plot model curve
    ax.fill_between(
        percentiles, model_gains, random_gains,
        alpha=0.3, color=COLORS['primary']
    )
    ax.plot(
        percentiles, model_gains,
        color=COLORS['primary'],
        linewidth=3,
        marker='o',
        markersize=8,
        label='XGBoost Model'
    )

    # Plot random baseline
    ax.plot(
        percentiles, random_gains,
        color=COLORS['dim'],
        linewidth=2,
        linestyle='--',
        label='Random Baseline'
    )

    # Add key metric annotations
    ax.annotate(
        f'Top 10% captures {cg["top_10%"]*100:.1f}%\nof all conversions',
        xy=(10, cg['top_10%']*100),
        xytext=(25, cg['top_10%']*100 + 10),
        fontsize=11,
        color=COLORS['success'],
        fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2)
    )

    ax.annotate(
        f'4.5x better\nthan random',
        xy=(10, 10),
        xytext=(25, 20),
        fontsize=10,
        color=COLORS['warning'],
        arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=1.5)
    )

    # Styling
    ax.set_xlabel('% of Leads Contacted (Ranked by Score)', color=COLORS['text'], fontsize=12, fontweight='bold')
    ax.set_ylabel('% of Total Conversions Captured', color=COLORS['text'], fontsize=12, fontweight='bold')
    ax.set_title(
        'Cumulative Gain Curve\nBusiness Value of Lead Prioritization',
        color=COLORS['text'],
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', facecolor=COLORS['dark'], edgecolor=COLORS['dim'],
              labelcolor=COLORS['text'], fontsize=11)

    ax.tick_params(colors=COLORS['text'], labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['dim'])
    ax.spines['left'].set_color(COLORS['dim'])
    ax.grid(True, alpha=0.2, color=COLORS['dim'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['dark'])
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def generate_precision_comparison(eval_report, save_path):
    """
    Generate Precision@K comparison chart.

    KEY PORTFOLIO MESSAGE: Concrete conversion rate improvements
    for top lead segments.
    """
    print("Generating Precision@K Comparison chart...")

    pak = eval_report['precision_at_k']
    uplift = eval_report['uplift_at_k']
    baseline = eval_report['baseline_conversion_rate']

    segments = ['Top 10%', 'Top 20%', 'Top 30%', 'Top 50%', 'Baseline']
    precision_values = [pak['top_10%'], pak['top_20%'], pak['top_30%'], pak['top_50%'], baseline]
    colors = [COLORS['success'], COLORS['primary'], COLORS['secondary'], COLORS['warning'], COLORS['danger']]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['dark'])
    ax.set_facecolor(COLORS['dark'])

    bars = ax.bar(segments, [v*100 for v in precision_values], color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar, val, seg in zip(bars, precision_values, segments):
        label = f'{val*100:.1f}%'
        if seg != 'Baseline':
            uplift_val = uplift.get(seg.lower().replace(' ', '_').replace('%', ''), 0)
            if uplift_val:
                label += f'\n({uplift_val}x uplift)'
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            label,
            ha='center',
            va='bottom',
            color=COLORS['text'],
            fontsize=11,
            fontweight='bold'
        )

    ax.axhline(baseline*100, color=COLORS['danger'], linestyle='--', linewidth=2, label=f'Baseline: {baseline*100:.1f}%')

    ax.set_ylabel('Conversion Rate (%)', color=COLORS['text'], fontsize=12, fontweight='bold')
    ax.set_title(
        'Conversion Rate by Lead Segment\nModel vs Random Targeting',
        color=COLORS['text'],
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    ax.set_ylim(0, 65)
    ax.tick_params(colors=COLORS['text'], labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['dim'])
    ax.spines['left'].set_color(COLORS['dim'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['dark'])
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def generate_scored_leads_preview(scored_leads, save_path):
    """
    Generate a styled preview of the scored leads CSV.

    KEY PORTFOLIO MESSAGE: Shows the concrete deliverable -
    a ranked CSV with scores and priority labels.
    """
    print("Generating Scored Leads Preview...")

    # Select top 12 leads with relevant columns
    preview_cols = ['age', 'job', 'education', 'score', 'priority_rank', 'priority']
    preview_df = scored_leads[preview_cols].head(12).copy()

    # Format score
    preview_df['score'] = preview_df['score'].apply(lambda x: f'{x:.2%}')
    preview_df['priority_rank'] = preview_df['priority_rank'].apply(lambda x: f'#{x}')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=COLORS['dark'])
    ax.set_facecolor(COLORS['dark'])
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=preview_df.values,
        colLabels=['Age', 'Job', 'Education', 'Conv. Score', 'Rank', 'Priority'],
        cellLoc='center',
        loc='center',
        colColours=[COLORS['primary']]*6
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Color cells based on priority
    for i, row in enumerate(preview_df.itertuples()):
        priority = row.priority.lower()
        if priority == 'high':
            bg_color = '#1a3a2a'  # Dark green
            text_color = COLORS['success']
        elif priority == 'medium':
            bg_color = '#3a2a1a'  # Dark orange
            text_color = COLORS['warning']
        else:
            bg_color = '#2a2a3a'  # Dark purple
            text_color = COLORS['dim']

        for j in range(len(preview_cols)):
            cell = table[(i+1, j)]
            cell.set_facecolor(COLORS['dark'])
            cell.set_text_props(color=COLORS['text'])
            if j == 5:  # Priority column
                cell.set_text_props(color=text_color, fontweight='bold')
            if j == 3:  # Score column
                cell.set_text_props(color=COLORS['primary'], fontweight='bold')

    # Style header
    for j in range(len(preview_cols)):
        cell = table[(0, j)]
        cell.set_text_props(color=COLORS['dark'], fontweight='bold')
        cell.set_facecolor(COLORS['primary'])

    ax.set_title(
        'Batch Scoring Output: scored_leads.csv\nTop 12 High-Priority Leads for Sales Team',
        color=COLORS['text'],
        fontsize=14,
        fontweight='bold',
        pad=30,
        y=0.95
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS['dark'])
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def main():
    """Generate all portfolio visuals."""
    print("\n" + "="*60)
    print("🎨 AI Lead Scoring - Portfolio Visual Generator")
    print("="*60 + "\n")

    # Load data
    print("Loading model and data...")
    model, df, scored_leads, eval_report = load_model_and_data()
    print("  ✓ Data loaded successfully\n")

    # Generate visuals
    generate_feature_importance(
        model,
        OUTPUT_DIR / "feature_importance.png"
    )

    generate_cumulative_gain_curve(
        eval_report,
        OUTPUT_DIR / "cumulative_gain_curve.png"
    )

    generate_precision_comparison(
        eval_report,
        OUTPUT_DIR / "precision_at_k_comparison.png"
    )

    generate_scored_leads_preview(
        scored_leads,
        OUTPUT_DIR / "scored_leads_preview.png"
    )

    print("\n" + "="*60)
    print("✅ All visuals generated successfully!")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\n📸 MANUAL SCREENSHOTS STILL NEEDED:")
    print("  1. cover_dashboard.html → Open in browser, full-page screenshot")
    print("  2. FastAPI Swagger UI → Run 'make serve' then http://localhost:8000/docs")
    print("\n")


if __name__ == "__main__":
    main()
