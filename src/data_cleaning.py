"""
data_cleaning.py
----------------
Complete data-cleaning pipeline for Student Performance Prediction.

Steps performed:
  1.  Load raw CSV
  2.  Inspect shape, dtypes, missing values, duplicates
  3.  Remove duplicate rows
  4.  Handle missing numerical values (median imputation)
  5.  Handle missing categorical values (mode imputation)
  6.  Detect and fix invalid values (age out of range, study_hours > 12)
  7.  Cap outliers in numerical columns using IQR method
  8.  Encode categorical columns (Label + One-Hot where appropriate)
  9.  Print before/after statistics
  10. Save cleaned CSV

Run:
    python src/data_cleaning.py
Output:
    data/processed/student_performance_cleaned.csv
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# â”€â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
RAW_PATH   = os.path.join(BASE_DIR, "data", "raw",       "student_performance_raw.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "student_performance_cleaned.csv")
os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)

# â”€â”€â”€ 1. Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=" * 60)
print("STUDENT PERFORMANCE â€” DATA CLEANING PIPELINE")
print("=" * 60)

df = pd.read_csv(RAW_PATH)
print(f"\n[1] Raw dataset loaded")
print(f"    Shape : {df.shape[0]} rows Ã— {df.shape[1]} columns")

# â”€â”€â”€ 2. Initial inspection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n[2] Data types:\n{df.dtypes}")
print(f"\n[2] Missing values (before cleaning):")
missing_before = df.isnull().sum()
print(missing_before[missing_before > 0])
print(f"\n[2] Duplicate rows: {df.duplicated().sum()}")

raw_shape = df.shape

# â”€â”€â”€ 3. Remove duplicates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
dupes_removed = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"\n[3] Removed {dupes_removed} duplicate rows â†’ {df.shape[0]} rows remaining")

# â”€â”€â”€ 4. Fix invalid values â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Age must be between 15 and 35 for a college student
invalid_age = ((df["age"] < 15) | (df["age"] > 35)).sum()
df.loc[(df["age"] < 15) | (df["age"] > 35), "age"] = np.nan
print(f"\n[4] Replaced {invalid_age} invalid age values with NaN")

# study_hours physically cannot exceed 20 h/day; cap at 16 (realistic max)
invalid_study = (df["study_hours"] > 16).sum()
df.loc[df["study_hours"] > 16, "study_hours"] = np.nan
print(f"[4] Replaced {invalid_study} implausible study_hours (>16) with NaN")

# â”€â”€â”€ 5. Handle missing numerical values â€” median imputation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
numerical_cols = [
    "age", "study_hours", "sleep_hours", "internet_usage_hours",
    "physical_activity_hours", "attendance_percentage",
    "assignment_score", "internal_exam_score",
    "previous_semester_marks", "previous_cgpa",
]

print(f"\n[5] Imputing missing numerical values with column medians â€¦")
for col in numerical_cols:
    n_missing = df[col].isnull().sum()
    if n_missing > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"    {col:<28} â€” {n_missing} values â†’ median {median_val:.2f}")

# â”€â”€â”€ 6. Handle missing categorical values â€” mode imputation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
categorical_cols = ["gender", "course", "stress_level", "extracurricular_activity"]

print(f"\n[6] Imputing missing categorical values with column mode â€¦")
for col in categorical_cols:
    n_missing = df[col].isnull().sum()
    if n_missing > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"    {col:<28} â€” {n_missing} values â†’ mode '{mode_val}'")

# â”€â”€â”€ 7. Outlier capping using IQR (Winsorisation) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
outlier_cols = [
    "study_hours", "sleep_hours", "internet_usage_hours",
    "physical_activity_hours", "attendance_percentage",
    "assignment_score", "internal_exam_score",
    "previous_semester_marks", "previous_cgpa",
]

print(f"\n[7] Capping outliers (IQR method) â€¦")
for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower, upper)
    if n_outliers:
        print(f"    {col:<28} â€” capped {n_outliers} outliers [{lower:.1f}, {upper:.1f}]")

# â”€â”€â”€ 8. Encode categorical columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n[8] Encoding categorical columns â€¦")

# Ordinal / label encoding
stress_map = {"Low": 0, "Medium": 1, "High": 2}
df["stress_level_encoded"] = df["stress_level"].map(stress_map)
print("    stress_level â†’ stress_level_encoded  (0=Low, 1=Medium, 2=High)")

gender_map = {"Male": 0, "Female": 1, "Other": 2}
df["gender_encoded"] = df["gender"].map(gender_map)
print("    gender â†’ gender_encoded  (0=Male, 1=Female, 2=Other)")

df["extracurricular_encoded"] = (df["extracurricular_activity"] == "Yes").astype(int)
print("    extracurricular_activity â†’ extracurricular_encoded  (1=Yes, 0=No)")

df["pass_fail_encoded"] = (df["pass_fail"] == "Pass").astype(int)
print("    pass_fail â†’ pass_fail_encoded  (1=Pass, 0=Fail)")

# Ordinal for performance category
perf_map = {"Low": 0, "Average": 1, "Good": 2, "Excellent": 3}
df["performance_encoded"] = df["performance_category"].map(perf_map)
print("    performance_category â†’ performance_encoded  (0=Low â€¦ 3=Excellent)")

# One-hot encoding for course (drop_first to avoid multicollinearity)
course_dummies = pd.get_dummies(df["course"], prefix="course", drop_first=True)
df = pd.concat([df, course_dummies], axis=1)
print(f"    course â†’ one-hot encoded ({list(course_dummies.columns)})")

# â”€â”€â”€ 9. Final verification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n[9] Final dataset stats:")
print(f"    Shape after cleaning : {df.shape[0]} rows Ã— {df.shape[1]} columns")
print(f"    Remaining NaN values : {df.isnull().sum().sum()}")
print(f"    Duplicate rows       : {df.duplicated().sum()}")

print(f"\n    Numerical summary (key columns):")
summary_cols = ["study_hours", "attendance_percentage", "final_score",
                "previous_cgpa", "sleep_hours"]
print(df[summary_cols].describe().round(2).to_string())

print(f"\n    Performance category distribution:")
print(df["performance_category"].value_counts())
print(f"\n    Pass / Fail distribution:")
print(df["pass_fail"].value_counts())

# â”€â”€â”€ 10. Save cleaned CSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df.to_csv(CLEAN_PATH, index=False)
print(f"\n[10] âœ…  Cleaned dataset saved â†’ {CLEAN_PATH}")
print("=" * 60)

