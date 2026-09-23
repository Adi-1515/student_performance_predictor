"""
app.py — Student Performance Prediction Web Application
--------------------------------------------------------
A Streamlit-based interface that:
  • Accepts student feature inputs via sidebar widgets
  • Loads trained regression + classification models
  • Predicts final score, performance category, and risk level
  • Displays AI-driven insights from the dataset
  • Includes an EDA overview tab

Run:
    streamlit run app/app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Add src/ to path so preprocessing + predict can be imported
APP_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(APP_DIR, "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

from predict import predict_student, load_models, PERF_LABELS
from utils import generate_ai_insights, performance_label, risk_label

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(APP_DIR, ".."))
DATA_PATH  = os.path.join(BASE_DIR, "data", "processed", "student_performance_cleaned.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
VIZ_DIR    = os.path.join(BASE_DIR, "visualizations")

# ─── Styling helpers ─────────────────────────────────────────────────────────
def colored_badge(text, color):
    return f'<span style="background:{color};color:white;padding:4px 12px;border-radius:12px;font-weight:bold;">{text}</span>'

RISK_COLORS = {
    "Low Risk":       "#22c55e",
    "Medium Risk":    "#f97316",
    "High Risk":      "#ef4444",
    "Very High Risk": "#9b1c1c",
}
PERF_COLORS = {
    "Excellent": "#3b82f6",
    "Good":      "#22c55e",
    "Average":   "#f97316",
    "Low":       "#ef4444",
}

# ─── Load dataset ─────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("## 🎓 Student Performance Prediction & Analytics")
st.markdown(
    "_IBM SkillsBuild Data Analytics with AI Internship 2026 — Final Project_  \n"
    "**Synthetic dataset · For educational purposes only**"
)
st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_predict, tab_eda, tab_insights, tab_about = st.tabs([
    "🔮 Predict", "📊 EDA Overview", "💡 AI Insights", "ℹ️ About"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### Enter Student Information")
    st.markdown("Fill in the details on the left sidebar and click **Predict**.")

    # ── Sidebar inputs ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎓 Student Features")
        st.markdown("---")
        st.markdown("**📚 Academic**")

        study_hours            = st.slider("Study Hours / Day",             0.0, 12.0, 5.0, 0.5)
        attendance_percentage  = st.slider("Attendance (%)",                 40.0, 100.0, 78.0, 1.0)
        previous_semester_marks= st.slider("Previous Semester Marks",        20.0, 100.0, 65.0, 1.0)
        assignment_score       = st.slider("Assignment Score",                20.0, 100.0, 72.0, 1.0)
        internal_exam_score    = st.slider("Internal Exam Score",             15.0, 100.0, 68.0, 1.0)
        previous_cgpa          = st.slider("Previous CGPA",                   4.0, 10.0,  7.2, 0.1)
        semester               = st.selectbox("Semester", list(range(1, 9)), index=3)

        st.markdown("---")
        st.markdown("**🌙 Lifestyle**")
        sleep_hours            = st.slider("Sleep Hours / Day",              3.0, 10.0, 6.5, 0.5)
        internet_usage_hours   = st.slider("Internet Usage Hours / Day",     0.0, 12.0, 4.0, 0.5)
        physical_activity_hours= st.slider("Physical Activity Hours / Day",  0.0, 8.0,  2.5, 0.5)
        stress_level           = st.selectbox("Stress Level", ["Low", "Medium", "High"], index=1)
        extracurricular        = st.selectbox("Extracurricular Activity", ["Yes", "No"], index=0)

        st.markdown("---")
        st.markdown("**👤 Demographic**")
        age    = st.number_input("Age", min_value=15, max_value=35, value=20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        course = st.selectbox(
            "Course",
            ["Computer Science", "Electronics", "Mechanical", "Civil", "Business"],
        )

        st.markdown("---")
        predict_btn = st.button("🔮 Predict Performance", use_container_width=True, type="primary")

    # ── Prediction ─────────────────────────────────────────────────
    if predict_btn:
        student_input = {
            "study_hours":              study_hours,
            "sleep_hours":              sleep_hours,
            "internet_usage_hours":     internet_usage_hours,
            "physical_activity_hours":  physical_activity_hours,
            "attendance_percentage":    attendance_percentage,
            "assignment_score":         assignment_score,
            "internal_exam_score":      internal_exam_score,
            "previous_semester_marks":  previous_semester_marks,
            "previous_cgpa":            previous_cgpa,
            "semester":                 semester,
            "age":                      age,
            "stress_level":             stress_level,
            "extracurricular_activity": extracurricular,
            "gender":                   gender,
            "course":                   course,
        }

        try:
            result = predict_student(student_input)

            # ── Metric cards ───────────────────────────────────────
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Predicted Final Score",
                          f"{result['predicted_score']:.1f} / 100")
            with col2:
                st.markdown("**🏅 Performance Category**")
                pcolor = PERF_COLORS.get(result["performance_category"], "#6b7280")
                st.markdown(
                    colored_badge(result["performance_category"], pcolor),
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown("**⚠️ Risk Level**")
                rcolor = RISK_COLORS.get(result["risk_level"], "#6b7280")
                st.markdown(
                    colored_badge(result["risk_level"], rcolor),
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.info(f"💬 **Explanation:** {result['explanation']}")

            # ── Score gauge ────────────────────────────────────────
            score = result["predicted_score"]
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.barh(["Score"], [score], color=pcolor, height=0.35)
            ax.barh(["Score"], [100 - score], left=score,
                    color="#e5e7eb", height=0.35)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Score")
            ax.set_title(f"Predicted Score: {score:.1f}/100", fontweight="bold")
            ax.axvline(50, color="#f97316", linestyle="--", linewidth=1, label="Pass threshold (50)")
            ax.legend(fontsize=7)
            ax.set_yticks([])
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except FileNotFoundError as e:
            st.error(
                f"⚠️ Model files not found. Please train the models first.\n\n"
                f"Run in your terminal:\n"
                f"```\npython src/train_regression.py\npython src/train_classification.py\n```\n\n"
                f"Error: {e}"
            )
    else:
        st.info("👈 Fill in student details in the sidebar and click **Predict Performance**.")

        # Show sample prediction cards as placeholders
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Predicted Final Score", "—")
        with col2:
            st.markdown("**🏅 Performance Category**")
            st.markdown(colored_badge("—", "#6b7280"), unsafe_allow_html=True)
        with col3:
            st.markdown("**⚠️ Risk Level**")
            st.markdown(colored_badge("—", "#6b7280"), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — EDA Overview
# ════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("### 📊 Exploratory Data Analysis Overview")

    df = load_dataset()
    if df is None:
        st.warning("Dataset not found. Please run the data pipeline first.")
    else:
        # ── KPIs ───────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Students",     len(df))
        k2.metric("Avg Final Score",    f"{df['final_score'].mean():.1f}")
        k3.metric("Avg Attendance (%)", f"{df['attendance_percentage'].mean():.1f}")
        k4.metric("Avg Study Hours",    f"{df['study_hours'].mean():.1f}")
        pass_pct = (df["pass_fail"] == "Pass").mean() * 100 if "pass_fail" in df.columns else 0
        k5.metric("Pass Rate (%)",      f"{pass_pct:.1f}")

        st.markdown("---")
        st.markdown("#### Pre-generated Visualizations")
        st.markdown("_Run `python src/eda.py` to generate all charts._")

        # Show any saved PNG files
        if os.path.exists(VIZ_DIR):
            images = sorted([f for f in os.listdir(VIZ_DIR) if f.endswith(".png")])
            if images:
                for i in range(0, len(images), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        if i + j < len(images):
                            img_path = os.path.join(VIZ_DIR, images[i + j])
                            col.image(img_path, use_container_width=True,
                                      caption=images[i + j].replace("_"," ").replace(".png",""))
            else:
                st.info("No visualization images found. Run `python src/eda.py` first.")

        # ── Quick live charts ───────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Live Dataset Statistics")

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df["final_score"].dropna(), bins=30, color="#3b82f6", edgecolor="white", alpha=0.85)
            ax.set_title("Final Score Distribution", fontweight="bold")
            ax.set_xlabel("Final Score")
            ax.set_ylabel("Frequency")
            ax.axvline(df["final_score"].mean(), color="crimson", linestyle="--",
                       label=f"Mean: {df['final_score'].mean():.1f}")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            if "performance_category" in df.columns:
                cat_counts = df["performance_category"].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                colors_bar = ["#ef4444","#f97316","#22c55e","#3b82f6"]
                ax.bar(cat_counts.index, cat_counts.values,
                       color=colors_bar[:len(cat_counts)], edgecolor="white")
                ax.set_title("Performance Category Distribution", fontweight="bold")
                ax.set_xlabel("Category")
                ax.set_ylabel("Count")
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("#### Correlation with Final Score")
        num_cols = ["study_hours","attendance_percentage","previous_cgpa",
                    "assignment_score","sleep_hours","internet_usage_hours",
                    "stress_level_encoded","internal_exam_score","previous_semester_marks"]
        available = [c for c in num_cols if c in df.columns]
        corr = df[available + ["final_score"]].corr()["final_score"].drop("final_score").sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors_corr = ["#ef4444" if v < 0 else "#3b82f6" for v in corr.values]
        ax.barh(corr.index, corr.values, color=colors_corr, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Feature Correlation with Final Score", fontweight="bold")
        ax.set_xlabel("Pearson r")
        st.pyplot(fig)
        plt.close(fig)

# ════════════════════════════════════════════════════════════════════
# TAB 3 — AI INSIGHTS
# ════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown("### 💡 AI-Driven Insights")
    st.markdown(
        "_Automatically generated observations from the dataset. "
        "These reflect patterns in the synthetic data and should not "
        "be interpreted as universal truths._"
    )

    df = load_dataset()
    if df is None:
        st.warning("Dataset not found.")
    else:
        insights = generate_ai_insights(df)
        for i, insight in enumerate(insights, 1):
            st.markdown(f"**{i}.** {insight}")

        st.markdown("---")
        st.markdown("#### Top Students by Predicted Score")
        top_cols = ["student_id","study_hours","attendance_percentage",
                    "previous_cgpa","final_score","performance_category"]
        available_top = [c for c in top_cols if c in df.columns]
        top_students = df.nlargest(10, "final_score")[available_top]
        st.dataframe(top_students.reset_index(drop=True), use_container_width=True)

        st.markdown("#### Students Needing Support (final_score < 50)")
        at_risk_cols = ["student_id","study_hours","attendance_percentage",
                        "stress_level","final_score"] if "student_id" in df.columns else \
                       ["study_hours","attendance_percentage","stress_level","final_score"]
        at_risk_avail = [c for c in at_risk_cols if c in df.columns]
        at_risk = df[df["final_score"] < 50][at_risk_avail]
        st.dataframe(at_risk.head(15).reset_index(drop=True), use_container_width=True)
        st.caption(
            f"⚠️ {len(at_risk)} students have a predicted final score below 50. "
            "This is a model-based indicator only. Human judgment is essential."
        )

# ════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### ℹ️ About This Project")
    st.markdown("""
| Field | Details |
|---|---|
| **Project Title** | Student Performance Prediction & Analytics Using Machine Learning |
| **Internship** | IBM SkillsBuild Data Analytics with AI — AICTE 2026 |
| **Dataset** | Synthetic (1,200 student records, generated for educational purposes) |
| **ML Models** | Linear Regression, Decision Tree, Random Forest (Regression + Classification) |
| **Unsupervised** | K-Means Clustering (4 segments) |
| **Framework** | Python · scikit-learn · Streamlit · Pandas · Matplotlib · Seaborn |

---

**⚠️ Ethical Disclaimer**

This application uses a **synthetic dataset** generated solely for demonstrating
machine learning concepts. Predictions are model-based indicators and should
**never** be used to make real academic decisions about real students without
proper human oversight and consent.

**Model limitations:**
- Trained on synthetic data; real-world accuracy may differ significantly.
- No personally identifiable information is used.
- Predictions reflect patterns in the training data, not causal relationships.
- Educators and counsellors should be involved in any academic intervention.

---

**Tech Stack**

```
Python 3.x | pandas | numpy | scikit-learn | matplotlib
seaborn | joblib | streamlit | plotly
```
    """)
    st.markdown("---")
    st.caption("IBM SkillsBuild Data Analytics with AI Internship 2026 — Final Project")
