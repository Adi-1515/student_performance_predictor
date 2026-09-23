"""
utils.py
--------
Shared utility functions used across the project.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# â”€â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR     = os.path.join(os.path.dirname(__file__), "..")
VIZ_DIR      = os.path.join(BASE_DIR, "visualizations")
DATA_PROC    = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR    = os.path.join(BASE_DIR, "models")


def ensure_dirs():
    """Create all required project directories if they don't exist."""
    for d in [VIZ_DIR, DATA_PROC, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)


def save_figure(fig, filename: str):
    """Save a matplotlib figure to the visualizations directory."""
    ensure_dirs()
    path = os.path.join(VIZ_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ðŸ’¾  Saved â†’ {path}")
    return path


def print_divider(title: str = "", width: int = 60):
    if title:
        pad = (width - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * pad)
    else:
        print("=" * width)


def performance_label(score: float) -> str:
    """Map a numeric score to a performance category string."""
    if score >= 80:
        return "Excellent"
    elif score >= 65:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Low"


def risk_label(score: float) -> str:
    """Map a numeric score to a risk level string."""
    if score >= 80:
        return "Low Risk"
    elif score >= 65:
        return "Medium Risk"
    elif score >= 50:
        return "High Risk"
    else:
        return "Very High Risk"


def generate_ai_insights(df: pd.DataFrame) -> list[str]:
    """
    Automatically generate human-readable analytical insights
    from the cleaned dataset.  No external API required.

    Returns a list of insight strings.
    """
    insights = []

    # â”€â”€ Attendance insight â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    high_att  = df[df["attendance_percentage"] >= 85]["final_score"].mean()
    low_att   = df[df["attendance_percentage"] < 60]["final_score"].mean()
    if pd.notna(high_att) and pd.notna(low_att):
        insights.append(
            f"Students with â‰¥85% attendance have an average final score of "
            f"{high_att:.1f}, compared to {low_att:.1f} for those with <60% attendance "
            f"in this dataset."
        )

    # â”€â”€ Study hours insight â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    high_study = df[df["study_hours"] >= 6]["final_score"].mean()
    low_study  = df[df["study_hours"] <  3]["final_score"].mean()
    if pd.notna(high_study) and pd.notna(low_study):
        insights.append(
            f"Students studying 6+ hours/day average a score of {high_study:.1f}, "
            f"versus {low_study:.1f} for those studying fewer than 3 hours/day."
        )

    # â”€â”€ Stress insight â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if "stress_level" in df.columns:
        stress_scores = df.groupby("stress_level")["final_score"].mean()
        if "High" in stress_scores and "Low" in stress_scores:
            insights.append(
                f"Students with High stress have an average final score of "
                f"{stress_scores['High']:.1f} vs {stress_scores['Low']:.1f} "
                f"for Low-stress students."
            )

    # â”€â”€ Pass rate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if "pass_fail" in df.columns:
        pass_rate = (df["pass_fail"] == "Pass").mean() * 100
        insights.append(f"Overall pass rate in the dataset: {pass_rate:.1f}%.")

    # â”€â”€ At-risk students â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    at_risk = (df["final_score"] < 50).sum()
    insights.append(
        f"{at_risk} students ({at_risk/len(df)*100:.1f}%) are predicted to score "
        f"below 50 â€” these students may benefit from early academic intervention."
    )

    # â”€â”€ Correlation insight â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    corr_cols = ["study_hours","attendance_percentage","previous_cgpa",
                 "assignment_score","final_score"]
    available = [c for c in corr_cols if c in df.columns]
    if len(available) > 1:
        corr = df[available].corr()["final_score"].drop("final_score")
        top_feat  = corr.idxmax()
        top_corr  = corr.max()
        insights.append(
            f"In this dataset, '{top_feat}' shows the strongest positive "
            f"association with final_score (r = {top_corr:.3f}). "
            f"Correlation indicates association, not causation."
        )

    # â”€â”€ Extracurricular insight â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if "extracurricular_activity" in df.columns:
        yes_score = df[df["extracurricular_activity"]=="Yes"]["final_score"].mean()
        no_score  = df[df["extracurricular_activity"]=="No"]["final_score"].mean()
        if pd.notna(yes_score) and pd.notna(no_score):
            diff = yes_score - no_score
            direction = "higher" if diff > 0 else "lower"
            insights.append(
                f"Students who participate in extracurricular activities score "
                f"on average {abs(diff):.1f} points {direction} than those who don't "
                f"(in this synthetic dataset)."
            )

    return insights

