"""
predict.py
----------
Standalone prediction utility.

Load trained models and predict for a new student.
Can be called from the command line or imported by app.py.

Usage (CLI):
    python src/predict.py

Usage (import):
    from predict import predict_student
    result = predict_student({...})
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import NUMERIC_FEATURES, ENCODED_FEATURES

BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Category decode maps
PERF_LABELS = {0: "Low", 1: "Average", 2: "Good", 3: "Excellent"}


def load_models():
    """Load regression and classification pipelines."""
    reg_path  = os.path.join(MODEL_DIR, "regression_model.pkl")
    clf_path  = os.path.join(MODEL_DIR, "classification_model.pkl")

    if not os.path.exists(reg_path):
        raise FileNotFoundError(
            f"Regression model not found at {reg_path}. "
            "Please run: python src/train_regression.py"
        )
    if not os.path.exists(clf_path):
        raise FileNotFoundError(
            f"Classification model not found at {clf_path}. "
            "Please run: python src/train_classification.py"
        )

    reg_pipeline = joblib.load(reg_path)
    clf_pipeline = joblib.load(clf_path)
    return reg_pipeline, clf_pipeline


def build_input_row(student_dict: dict) -> pd.DataFrame:
    """
    Convert a dictionary of student features into a single-row DataFrame
    that matches the feature schema expected by the model pipelines.

    Required keys in student_dict (all others default to 0):
        study_hours, sleep_hours, internet_usage_hours,
        physical_activity_hours, attendance_percentage,
        assignment_score, internal_exam_score,
        previous_semester_marks, previous_cgpa,
        semester, age,
        stress_level, extracurricular_activity, gender,
        course
    """
    # Stress encoding
    stress_map = {"Low": 0, "Medium": 1, "High": 2}
    gender_map = {"Male": 0, "Female": 1, "Other": 2}

    row = {col: 0 for col in NUMERIC_FEATURES + ENCODED_FEATURES}

    # Numeric
    for col in NUMERIC_FEATURES:
        row[col] = student_dict.get(col, 0)

    # Encoded categoricals
    row["stress_level_encoded"] = stress_map.get(student_dict.get("stress_level", "Medium"), 1)
    row["extracurricular_encoded"] = 1 if student_dict.get("extracurricular_activity","No") == "Yes" else 0
    row["gender_encoded"] = gender_map.get(student_dict.get("gender","Male"), 0)

    # One-hot course columns: drop_first=True dropped "Business" (alphabetically first).
    # Remaining dummies: Civil, Computer Science, Electronics, Mechanical
    all_courses = ["Civil", "Computer Science", "Electronics", "Mechanical"]
    for crs in all_courses:
        col_name = f"course_{crs}"
        row[col_name] = 1 if student_dict.get("course", "Computer Science") == crs else 0

    return pd.DataFrame([row])


def get_risk_level(predicted_score: float) -> str:
    """Derive a risk label from the predicted score."""
    if predicted_score >= 80:
        return "Low Risk"
    elif predicted_score >= 65:
        return "Medium Risk"
    elif predicted_score >= 50:
        return "High Risk"
    else:
        return "Very High Risk"


def predict_student(student_dict: dict) -> dict:
    """
    Given a dict of student features, return predictions.

    Returns:
        {
          "predicted_score":        float,
          "performance_category":   str,
          "risk_level":             str,
          "explanation":            str,
        }
    """
    reg_pipeline, clf_pipeline = load_models()
    input_df = build_input_row(student_dict)

    predicted_score = float(np.clip(reg_pipeline.predict(input_df)[0], 0, 100))
    perf_encoded    = int(clf_pipeline.predict(input_df)[0])
    performance_cat = PERF_LABELS.get(perf_encoded, "Unknown")
    risk_level      = get_risk_level(predicted_score)

    # Auto-generate a short plain-language explanation
    study_h   = student_dict.get("study_hours", 0)
    attend    = student_dict.get("attendance_percentage", 0)
    stress    = student_dict.get("stress_level", "Medium")

    explanation_parts = []
    if study_h >= 6:
        explanation_parts.append("strong study hours")
    elif study_h <= 2:
        explanation_parts.append("low study hours (consider increasing)")

    if attend >= 80:
        explanation_parts.append("good attendance")
    elif attend < 60:
        explanation_parts.append("low attendance (attendance is a key factor)")

    if stress == "High":
        explanation_parts.append("high stress (may be impacting performance)")

    explanation = (
        f"Predicted score: {predicted_score:.1f}/100 â†’ {performance_cat} ({risk_level}). "
        + (f"Key observations: {', '.join(explanation_parts)}." if explanation_parts else "")
    )

    return {
        "predicted_score":      round(predicted_score, 2),
        "performance_category": performance_cat,
        "risk_level":           risk_level,
        "explanation":          explanation,
    }


# â”€â”€â”€ CLI demo â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    sample_student = {
        "study_hours":             6.5,
        "sleep_hours":             7.0,
        "internet_usage_hours":    3.0,
        "physical_activity_hours": 2.5,
        "attendance_percentage":   85.0,
        "assignment_score":        78.0,
        "internal_exam_score":     72.0,
        "previous_semester_marks": 70.0,
        "previous_cgpa":           7.8,
        "semester":                4,
        "age":                     20,
        "stress_level":            "Medium",
        "extracurricular_activity":"Yes",
        "gender":                  "Female",
        "course":                  "Computer Science",
    }

    print("\nâ”€â”€â”€ Sample Student Prediction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    for k, v in sample_student.items():
        print(f"  {k:<28} : {v}")

    result = predict_student(sample_student)
    print("\nâ”€â”€â”€ Prediction Results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    print(f"  Predicted Score      : {result['predicted_score']}")
    print(f"  Performance Category : {result['performance_category']}")
    print(f"  Risk Level           : {result['risk_level']}")
    print(f"  Explanation          : {result['explanation']}")

