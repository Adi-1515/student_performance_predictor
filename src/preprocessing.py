"""
preprocessing.py
----------------
Shared preprocessing utilities used by both regression and classification
training scripts.  Separating preprocessing here avoids code duplication
and makes it easy to load the same pipeline when serving predictions.

Key responsibilities:
  â€¢ Define feature columns used for modelling
  â€¢ Build a scikit-learn ColumnTransformer pipeline
  â€¢ Expose a helper that returns X, y splits ready for training
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# â”€â”€â”€ Feature definitions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Numeric features fed to models
NUMERIC_FEATURES = [
    "study_hours",
    "sleep_hours",
    "internet_usage_hours",
    "physical_activity_hours",
    "attendance_percentage",
    "assignment_score",
    "internal_exam_score",
    "previous_semester_marks",
    "previous_cgpa",
    "semester",
    "age",
]

# Already-encoded binary/ordinal columns
ENCODED_FEATURES = [
    "stress_level_encoded",
    "extracurricular_encoded",
    "gender_encoded",
]

# One-hot course columns produced by data_cleaning.py
# (all columns whose names start with "course_")
def get_course_dummies(df):
    return [c for c in df.columns if c.startswith("course_")]


# Combined feature list builder
def get_feature_columns(df):
    return NUMERIC_FEATURES + ENCODED_FEATURES + get_course_dummies(df)


# â”€â”€â”€ Preprocessing pipeline builder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_preprocessor(df):
    """
    Returns a fitted ColumnTransformer that:
      - Imputes (median) + scales (StandardScaler) numeric features
      - Passes encoded/dummy features through unchanged

    Parameters
    ----------
    df : pd.DataFrame  â€” cleaned dataset (used only to discover column names)

    Returns
    -------
    preprocessor : ColumnTransformer  (unfitted â€” fit during model.fit())
    feature_names : list[str]         (ordered list of all feature columns)
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    passthrough_cols = ENCODED_FEATURES + get_course_dummies(df)

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", "passthrough",       passthrough_cols),
    ])

    feature_names = NUMERIC_FEATURES + passthrough_cols
    return preprocessor, feature_names


# â”€â”€â”€ Data loader helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_data(path=None):
    """
    Load the cleaned CSV.  Falls back to the default processed path.
    """
    if path is None:
        base = os.path.join(os.path.dirname(__file__), "..")
        path = os.path.join(base, "data", "processed",
                            "student_performance_cleaned.csv")
    df = pd.read_csv(path)
    return df


def get_regression_data(df):
    """
    Returns (X, y) for regression: predict final_score.
    """
    features = get_feature_columns(df)
    X = df[features].copy()
    y = df["final_score"].copy()
    return X, y


def get_classification_data(df, target="performance_category"):
    """
    Returns (X, y) for classification.
    target options:
        'performance_category'  â€” multiclass (Low/Average/Good/Excellent)
        'pass_fail'             â€” binary (Pass/Fail)
    """
    features = get_feature_columns(df)
    X = df[features].copy()
    if target == "performance_category":
        y = df["performance_encoded"].copy()   # 0â€“3 ordinal
    elif target == "pass_fail":
        y = df["pass_fail_encoded"].copy()      # 0/1
    else:
        raise ValueError(f"Unknown target: {target}")
    return X, y

