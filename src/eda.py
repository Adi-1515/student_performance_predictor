"""
eda.py
------
Exploratory Data Analysis for Student Performance Prediction.

Analyses performed:
  â€¢ Univariate: distributions of key numeric columns
  â€¢ Bivariate:  scatter plots of features vs final_score
  â€¢ Multivariate: correlation heatmap, pair plot (subset)
  â€¢ Categorical comparisons: boxplots by gender, course, stress level
  â€¢ Performance category distribution

Run:
    python src/eda.py
Output:
    visualizations/ (PNG files for every chart)
    Prints summary statistics to console
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# â”€â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "student_performance_cleaned.csv")
VIZ_DIR   = os.path.join(BASE_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

# â”€â”€â”€ Style â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = "Blues_d"
ACCENT  = "#3b82f6"

def save(fig, name):
    path = os.path.join(VIZ_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ðŸ’¾  Saved â†’ {path}")

# â”€â”€â”€ Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df = pd.read_csv(DATA_PATH)
print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)
print(f"Dataset: {df.shape[0]} rows Ã— {df.shape[1]} columns\n")

# â”€â”€â”€ Statistical Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[STATISTICAL SUMMARY â€” Key Numeric Columns]")
key_cols = [
    "study_hours", "attendance_percentage", "sleep_hours",
    "internet_usage_hours", "previous_cgpa", "assignment_score",
    "internal_exam_score", "previous_semester_marks", "final_score",
]
summary = df[key_cols].describe().T
summary["median"] = df[key_cols].median()
summary["mode"]   = df[key_cols].mode().iloc[0]
print(summary[["count","mean","median","mode","std","min","25%","75%","max"]].round(2).to_string())

# â”€â”€â”€ UNIVARIATE ANALYSIS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[UNIVARIATE ANALYSIS]")

uni_cols = [
    ("study_hours",            "Study Hours per Day"),
    ("attendance_percentage",  "Attendance (%)"),
    ("final_score",            "Final Score"),
    ("sleep_hours",            "Sleep Hours per Day"),
    ("previous_cgpa",          "Previous CGPA"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, (col, label) in enumerate(uni_cols):
    ax = axes[i]
    ax.hist(df[col].dropna(), bins=30, color=ACCENT, edgecolor="white", alpha=0.85)
    ax.set_title(label, fontweight="bold")
    ax.set_xlabel(label)
    ax.set_ylabel("Frequency")
    mean_val = df[col].mean()
    ax.axvline(mean_val, color="crimson", linestyle="--", label=f"Mean: {mean_val:.1f}")
    ax.legend(fontsize=9)
axes[-1].set_visible(False)
fig.suptitle("Univariate Distributions â€” Key Features", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "01_univariate_distributions.png")

# â”€â”€â”€ Performance Category Distribution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cat_counts = df["performance_category"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ["#ef4444","#f97316","#22c55e","#3b82f6"]
axes[0].bar(cat_counts.index, cat_counts.values,
            color=colors[:len(cat_counts)], edgecolor="white")
axes[0].set_title("Performance Category Distribution", fontweight="bold")
axes[0].set_xlabel("Performance Category")
axes[0].set_ylabel("Number of Students")
for bar, val in zip(axes[0].patches, cat_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha="center", va="bottom", fontweight="bold")

pf_counts = df["pass_fail"].value_counts()
axes[1].pie(pf_counts.values, labels=pf_counts.index,
            colors=["#22c55e","#ef4444"], autopct="%1.1f%%",
            startangle=90, wedgeprops=dict(edgecolor="white"))
axes[1].set_title("Pass / Fail Distribution", fontweight="bold")
plt.tight_layout()
save(fig, "02_performance_distribution.png")

# â”€â”€â”€ BIVARIATE ANALYSIS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[BIVARIATE ANALYSIS]")

bivariate_pairs = [
    ("study_hours",            "Study Hours vs Final Score"),
    ("attendance_percentage",  "Attendance vs Final Score"),
    ("previous_semester_marks","Previous Marks vs Final Score"),
    ("assignment_score",       "Assignment Score vs Final Score"),
    ("sleep_hours",            "Sleep Hours vs Final Score"),
    ("internet_usage_hours",   "Internet Usage vs Final Score"),
    ("stress_level_encoded",   "Stress Level (encoded) vs Final Score"),
    ("previous_cgpa",          "Previous CGPA vs Final Score"),
]

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()
for i, (feat, title) in enumerate(bivariate_pairs):
    ax = axes[i]
    ax.scatter(df[feat], df["final_score"],
               alpha=0.35, s=18, color=ACCENT, edgecolors="none")
    # correlation line (polyfit)
    valid = df[[feat,"final_score"]].dropna()
    m, b = np.polyfit(valid[feat], valid["final_score"], 1)
    x_line = np.linspace(valid[feat].min(), valid[feat].max(), 100)
    ax.plot(x_line, m*x_line + b, color="crimson", linewidth=1.5)
    corr = valid[feat].corr(valid["final_score"])
    ax.set_title(f"{title}\n(r = {corr:.3f})", fontsize=9, fontweight="bold")
    ax.set_xlabel(feat, fontsize=8)
    ax.set_ylabel("Final Score", fontsize=8)
axes[-1].set_visible(False)
fig.suptitle("Bivariate Analysis â€” Feature vs Final Score", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "03_bivariate_scatter.png")

# â”€â”€â”€ Boxplots â€” Categorical vs Final Score â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

sns.boxplot(data=df, x="stress_level", y="final_score",
            order=["Low","Medium","High"],
            palette=["#22c55e","#f97316","#ef4444"], ax=axes[0])
axes[0].set_title("Stress Level vs Final Score", fontweight="bold")

sns.boxplot(data=df, x="extracurricular_activity", y="final_score",
            palette=["#3b82f6","#a855f7"], ax=axes[1])
axes[1].set_title("Extracurricular Activity vs Final Score", fontweight="bold")

sns.boxplot(data=df, x="gender", y="final_score",
            palette="pastel", ax=axes[2])
axes[2].set_title("Gender vs Final Score", fontweight="bold")

for ax in axes:
    ax.set_xlabel("")
    ax.set_ylabel("Final Score")
plt.tight_layout()
save(fig, "04_categorical_boxplots.png")

# â”€â”€â”€ MULTIVARIATE â€” Correlation Heatmap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[MULTIVARIATE ANALYSIS]")

numeric_df = df[key_cols + ["stress_level_encoded","extracurricular_encoded","semester"]]
corr_matrix = numeric_df.corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0,
    linewidths=0.5, square=True, ax=ax,
    annot_kws={"size": 8},
)
ax.set_title("Correlation Matrix â€” Student Features", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
save(fig, "05_correlation_heatmap.png")

# Print top correlations with final_score
corr_with_target = corr_matrix["final_score"].drop("final_score").sort_values(ascending=False)
print("\nTop correlations with final_score:")
print(corr_with_target.round(3).to_string())

# â”€â”€â”€ Course & Semester Analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

course_avg = df.groupby("course")["final_score"].mean().sort_values(ascending=False)
axes[0].barh(course_avg.index, course_avg.values, color=ACCENT, edgecolor="white")
axes[0].set_title("Average Final Score by Course", fontweight="bold")
axes[0].set_xlabel("Average Final Score")
for bar in axes[0].patches:
    axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{bar.get_width():.1f}", va="center", fontsize=9)

sem_avg = df.groupby("semester")["final_score"].mean().sort_index()
axes[1].plot(sem_avg.index, sem_avg.values, marker="o", color=ACCENT, linewidth=2.5, markersize=8)
axes[1].set_title("Average Final Score by Semester", fontweight="bold")
axes[1].set_xlabel("Semester")
axes[1].set_ylabel("Average Final Score")
axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
save(fig, "06_course_semester_analysis.png")

# â”€â”€â”€ Study Hours Bucketed Analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df["study_bucket"] = pd.cut(
    df["study_hours"],
    bins=[0, 2, 4, 6, 8, 20],
    labels=["0â€“2 h", "2â€“4 h", "4â€“6 h", "6â€“8 h", "8+ h"]
)
bucket_avg = df.groupby("study_bucket", observed=True)["final_score"].mean()
bucket_cnt = df.groupby("study_bucket", observed=True)["final_score"].count()

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(bucket_avg.index.astype(str), bucket_avg.values,
              color=ACCENT, edgecolor="white")
ax.set_title("Average Final Score by Daily Study Hours", fontweight="bold", fontsize=12)
ax.set_xlabel("Daily Study Hours Bucket")
ax.set_ylabel("Average Final Score")
for bar, cnt, val in zip(bars, bucket_cnt, bucket_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}\n(n={cnt})", ha="center", fontsize=9)
plt.tight_layout()
save(fig, "07_study_hours_buckets.png")

# â”€â”€â”€ Attendance Bucket Analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df["attendance_bucket"] = pd.cut(
    df["attendance_percentage"],
    bins=[0, 50, 65, 75, 85, 100],
    labels=["<50%", "50â€“65%", "65â€“75%", "75â€“85%", "85â€“100%"]
)
att_avg = df.groupby("attendance_bucket", observed=True)["final_score"].mean()
att_cnt = df.groupby("attendance_bucket", observed=True)["final_score"].count()

fig, ax = plt.subplots(figsize=(9, 5))
colors_att = ["#ef4444","#f97316","#eab308","#22c55e","#3b82f6"]
bars = ax.bar(att_avg.index.astype(str), att_avg.values,
              color=colors_att[:len(att_avg)], edgecolor="white")
ax.set_title("Average Final Score by Attendance Range", fontweight="bold", fontsize=12)
ax.set_xlabel("Attendance Range")
ax.set_ylabel("Average Final Score")
for bar, cnt, val in zip(bars, att_cnt, att_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}\n(n={cnt})", ha="center", fontsize=9)
plt.tight_layout()
save(fig, "08_attendance_buckets.png")

print(f"\nâœ…  EDA complete â€” {len(os.listdir(VIZ_DIR))} visualizations saved in: {VIZ_DIR}")
print("=" * 60)

