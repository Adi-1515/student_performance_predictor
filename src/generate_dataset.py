"""
generate_dataset.py
-------------------
Synthetic dataset generator for Student Performance Prediction project.
This script creates a realistic (but fictional) student dataset for
educational and demonstration purposes.

NOTE: All data is SYNTHETIC (artificially generated). It does NOT
represent real students and should NOT be used for actual academic decisions.

Run:
    python src/generate_dataset.py
Output:
    data/raw/student_performance_raw.csv
"""

import numpy as np
import pandas as pd
import os

# â”€â”€â”€ Reproducibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEED = 42
np.random.seed(SEED)
N = 1200  # number of synthetic student records

# â”€â”€â”€ Helper distributions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def random_choice(options, size, weights=None):
    return np.random.choice(options, size=size, p=weights)


# â”€â”€â”€ Base features (independent) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
age = np.random.randint(17, 25, N)
gender = random_choice(["Male", "Female", "Other"], N, [0.52, 0.45, 0.03])
course = random_choice(
    ["Computer Science", "Electronics", "Mechanical", "Civil", "Business"],
    N,
    [0.30, 0.20, 0.20, 0.15, 0.15],
)
semester = np.random.randint(1, 9, N)

# Academic lifestyle base (with some noise)
study_hours = clip(np.random.normal(5, 2.5, N), 0, 12)
sleep_hours = clip(np.random.normal(6.5, 1.2, N), 3, 10)
internet_usage_hours = clip(np.random.normal(4, 2, N), 0, 12)
extracurricular = random_choice(["Yes", "No"], N, [0.45, 0.55])
stress_level = random_choice(["Low", "Medium", "High"], N, [0.25, 0.50, 0.25])
physical_activity_hours = clip(np.random.normal(3, 1.5, N), 0, 8)

# Previous academic background
previous_cgpa = clip(np.random.normal(7.2, 1.1, N), 4.0, 10.0)
attendance_percentage = clip(np.random.normal(78, 12, N), 40, 100)
assignment_score = clip(np.random.normal(72, 12, N), 20, 100)
internal_exam_score = clip(np.random.normal(68, 14, N), 15, 100)
previous_semester_marks = clip(np.random.normal(65, 15, N), 25, 100)

# â”€â”€â”€ Build final_score with realistic relationships â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Each factor contributes with noise; no single factor is perfectly predictive

stress_penalty = {"Low": 0, "Medium": -3, "High": -7}
stress_arr = np.array([stress_penalty[s] for s in stress_level])

extra_bonus = np.array([2 if e == "Yes" else 0 for e in extracurricular])

final_score = (
    0.25 * previous_semester_marks    # prior performance
    + 0.20 * attendance_percentage     # attendance matters
    + 0.18 * study_hours * 4.5        # study effort (scaled)
    + 0.15 * assignment_score          # continuous assessment
    + 0.10 * internal_exam_score       # mid-term
    + 0.08 * previous_cgpa * 5        # overall GPA context
    + 0.04 * sleep_hours * 2          # rest (small positive)
    - 0.03 * internet_usage_hours * 2 # distraction (small negative)
    + stress_arr                       # stress penalty
    + extra_bonus                      # extracurricular bonus
    + np.random.normal(0, 5, N)       # irreducible noise
)

# Normalise to a realistic exam scale (0â€“100)
final_score = final_score - final_score.min()
final_score = (final_score / final_score.max()) * 100
final_score = clip(np.round(final_score, 2), 0, 100)

# â”€â”€â”€ Performance category (classification target) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def assign_category(score):
    if score >= 80:
        return "Excellent"
    elif score >= 65:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Low"

performance_category = np.array([assign_category(s) for s in final_score])
pass_fail = np.where(final_score >= 50, "Pass", "Fail")

# â”€â”€â”€ Assemble DataFrame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
student_ids = [f"STU{str(i+1001).zfill(4)}" for i in range(N)]

df = pd.DataFrame({
    "student_id": student_ids,
    "age": age,
    "gender": gender,
    "course": course,
    "semester": semester,
    "study_hours": np.round(study_hours, 2),
    "sleep_hours": np.round(sleep_hours, 2),
    "internet_usage_hours": np.round(internet_usage_hours, 2),
    "extracurricular_activity": extracurricular,
    "stress_level": stress_level,
    "physical_activity_hours": np.round(physical_activity_hours, 2),
    "attendance_percentage": np.round(attendance_percentage, 2),
    "assignment_score": np.round(assignment_score, 2),
    "internal_exam_score": np.round(internal_exam_score, 2),
    "previous_semester_marks": np.round(previous_semester_marks, 2),
    "previous_cgpa": np.round(previous_cgpa, 2),
    "final_score": final_score,
    "performance_category": performance_category,
    "pass_fail": pass_fail,
})

# â”€â”€â”€ Inject realistic data quality issues â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Missing values (~3â€“5% per selected column)
missing_cols = [
    "study_hours", "sleep_hours", "attendance_percentage",
    "assignment_score", "stress_level", "physical_activity_hours",
    "previous_cgpa", "internet_usage_hours",
]
for col in missing_cols:
    missing_idx = np.random.choice(N, size=int(N * 0.04), replace=False)
    df.loc[missing_idx, col] = np.nan

# 2. Duplicate rows (~1%)
dup_idx = np.random.choice(N, size=12, replace=False)
df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

# 3. Outliers in study_hours (a handful of extreme values)
outlier_idx = np.random.choice(len(df), size=5, replace=False)
df.loc[outlier_idx, "study_hours"] = np.random.uniform(13, 18, 5)

# 4. A few invalid age values
# Convert age to float first so NaN and out-of-range floats are accepted (pandas 3.x)
df["age"] = df["age"].astype(float)
invalid_idx = np.random.choice(len(df), size=4, replace=False)
df.loc[invalid_idx, "age"] = [float(v) for v in np.random.choice([-1, 0, 99, 150], size=4)]

# â”€â”€â”€ Save â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "student_performance_raw.csv")
df.to_csv(output_path, index=False)

print(f"âœ…  Raw dataset generated: {output_path}")
print(f"    Rows (incl. duplicates): {len(df)}")
print(f"    Columns               : {len(df.columns)}")
print(f"\nColumn list:")
for col in df.columns:
    print(f"  â€¢ {col}")
print(f"\nPerformance category distribution:")
print(df["performance_category"].value_counts())
print(f"\nPass/Fail distribution:")
print(df["pass_fail"].value_counts())
print(f"\nMissing values injected (sample):")
print(df[missing_cols].isnull().sum())

