"""
train_classification.py
-----------------------
Train and evaluate three classification models to predict
student performance category (Low / Average / Good / Excellent).

Models trained:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier

Evaluation metrics:
  â€¢ Accuracy
  â€¢ Precision (weighted)
  â€¢ Recall    (weighted)
  â€¢ F1-Score  (weighted)
  â€¢ Confusion Matrix

Outputs:
  â€¢ models/classification_model.pkl
  â€¢ visualizations/13_classification_*

Run:
    python src/train_classification.py
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_data, get_classification_data, build_preprocessor

warnings.filterwarnings("ignore")

# â”€â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")
VIZ_DIR   = os.path.join(BASE_DIR, "visualizations")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

SEED = 42
CLASS_NAMES = ["Low", "Average", "Good", "Excellent"]
sns.set_theme(style="whitegrid", font_scale=1.05)

# â”€â”€â”€ Load data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=" * 60)
print("CLASSIFICATION â€” PREDICT PERFORMANCE CATEGORY")
print("=" * 60)

df = load_data()
X, y = get_classification_data(df, target="performance_category")
preprocessor, feature_names = build_preprocessor(df)

print(f"Dataset : {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes : {dict(zip(range(4), CLASS_NAMES))}")
print(f"Class distribution:")
for i, cls in enumerate(CLASS_NAMES):
    cnt = (y == i).sum()
    print(f"  {i} ({cls:>10}) : {cnt} ({cnt/len(y)*100:.1f}%)")

# â”€â”€â”€ Train / Test split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# â”€â”€â”€ Model definitions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=SEED, C=1.0, solver="lbfgs",
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=SEED, max_depth=8, min_samples_leaf=5
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=SEED, max_depth=10, min_samples_leaf=3
    ),
}

# â”€â”€â”€ Train, evaluate, collect metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
results = {}
conf_matrices = {}

for name, estimator in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",        estimator),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    cv_acc = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy").mean()

    results[name] = {
        "Accuracy":  round(acc, 4),
        "Precision": round(prec, 4),
        "Recall":    round(rec, 4),
        "F1 Score":  round(f1, 4),
        "CV Acc":    round(cv_acc, 4),
        "pipeline":  pipeline,
        "y_pred":    y_pred,
    }
    conf_matrices[name] = cm

    print(f"\n  {name}")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    CV Acc (5-fold): {cv_acc:.4f}")
    print(f"\n  Classification Report ({name}):")
    print(classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        digits=3,
        zero_division=0,
    ))

# â”€â”€â”€ Results table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=" * 60)
print("MODEL COMPARISON TABLE")
print("=" * 60)
results_df = pd.DataFrame({
    k: {m: results[k][m] for m in ["Accuracy","Precision","Recall","F1 Score","CV Acc"]}
    for k in results
}).T
print(results_df.to_string())

# â”€â”€â”€ Best model selection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
best_name = results_df["F1 Score"].idxmax()
best_pipeline = results[best_name]["pipeline"]
print(f"\nâœ…  Best model: {best_name}  (highest F1: {results_df.loc[best_name,'F1 Score']})")

# â”€â”€â”€ Save best model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model_path = os.path.join(MODEL_DIR, "classification_model.pkl")
joblib.dump(best_pipeline, model_path)
print(f"    Saved â†’ {model_path}")

all_clf_path = os.path.join(MODEL_DIR, "classification_all_models.pkl")
joblib.dump({k: results[k]["pipeline"] for k in results}, all_clf_path)

# â”€â”€â”€ VISUALISATIONS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def save(fig, name):
    path = os.path.join(VIZ_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ðŸ’¾  {path}")

# 1. Confusion matrices â€” all 3 models
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, cm) in zip(axes, conf_matrices.items()):
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5,
    )
    ax.set_title(f"Confusion Matrix\n{name}", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
save(fig, "13_classification_confusion_matrices.png")

# 2. Model comparison â€” Accuracy & F1
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors_bar = ["#3b82f6", "#22c55e", "#f97316"]
for i, metric in enumerate(["Accuracy", "F1 Score"]):
    vals  = results_df[metric].values
    names = results_df.index.tolist()
    bars  = axes[i].bar(names, vals, color=colors_bar, edgecolor="white")
    axes[i].set_title(f"Model Comparison â€” {metric}", fontweight="bold")
    axes[i].set_ylabel(metric)
    axes[i].set_ylim(0, 1.10)
    for bar, v in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
save(fig, "14_classification_model_comparison.png")

# 3. Feature importance â€” Random Forest Classifier
rf_name = "Random Forest"
if rf_name in results:
    rf_pipeline = results[rf_name]["pipeline"]
    rf_model    = rf_pipeline.named_steps["model"]
    importances = rf_model.feature_importances_

    from preprocessing import ENCODED_FEATURES, NUMERIC_FEATURES
    passthrough_cols = [c for c in df.columns if c.startswith("course_")]
    all_feature_names = NUMERIC_FEATURES + ENCODED_FEATURES + passthrough_cols

    imp_df = pd.DataFrame({
        "Feature":    all_feature_names[:len(importances)],
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1],
            color="#a855f7", edgecolor="white")
    ax.set_title("Random Forest â€” Feature Importances (Classification)", fontweight="bold")
    ax.set_xlabel("Mean Decrease in Impurity (Gini)")
    ax.axvline(imp_df["Importance"].mean(), color="crimson",
               linestyle="--", label="Mean importance")
    ax.legend()
    save(fig, "15_classification_feature_importance.png")

    print(f"\nTop-5 features by RF Classifier importance:")
    print(imp_df.head(5)[["Feature","Importance"]].to_string(index=False))

# 4. Risk analysis â€” derive risk level and plot
df["risk_level"] = pd.cut(
    df["final_score"],
    bins=[0, 50, 65, 80, 100],
    labels=["High Risk", "Medium Risk", "Low Risk", "No Risk"],
)
risk_counts = df["risk_level"].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
colors_risk = {"High Risk":"#ef4444", "Medium Risk":"#f97316",
               "Low Risk":"#eab308", "No Risk":"#22c55e"}
bars = ax.bar(
    risk_counts.index,
    risk_counts.values,
    color=[colors_risk.get(x, "#3b82f6") for x in risk_counts.index],
    edgecolor="white"
)
ax.set_title("Student Risk Level Distribution", fontweight="bold", fontsize=12)
ax.set_xlabel("Risk Level")
ax.set_ylabel("Number of Students")
for bar, v in zip(bars, risk_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(v), ha="center", fontweight="bold")
plt.tight_layout()
save(fig, "16_risk_level_distribution.png")

print(f"\nRisk Level Distribution:")
print(risk_counts.to_string())

print(f"\nâœ…  Classification training complete.")
print("=" * 60)

