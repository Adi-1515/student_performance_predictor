"""
train_regression.py
-------------------
Train and evaluate three regression models to predict student final_score.

Models trained:
  1. Linear Regression
  2. Decision Tree Regressor
  3. Random Forest Regressor

Evaluation metrics:
  â€¢ MAE   (Mean Absolute Error)
  â€¢ MSE   (Mean Squared Error)
  â€¢ RMSE  (Root Mean Squared Error)
  â€¢ RÂ²    (Coefficient of Determination)

Outputs:
  â€¢ models/regression_model.pkl     (best model pipeline)
  â€¢ models/scaler.pkl               (StandardScaler â€” also embedded in pipeline)
  â€¢ visualizations/09_regression_*  (actual vs predicted, residuals, comparison)

Run:
    python src/train_regression.py
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
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add src to path so preprocessing.py can be imported
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_data, get_regression_data, build_preprocessor

warnings.filterwarnings("ignore")

# â”€â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")
VIZ_DIR   = os.path.join(BASE_DIR, "visualizations")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

SEED = 42
sns.set_theme(style="whitegrid", font_scale=1.05)

# â”€â”€â”€ Load data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=" * 60)
print("REGRESSION â€” PREDICT FINAL SCORE")
print("=" * 60)

df = load_data()
X, y = get_regression_data(df)
preprocessor, feature_names = build_preprocessor(df)

print(f"Dataset : {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target  : final_score  |  range [{y.min():.1f}, {y.max():.1f}]")

# â”€â”€â”€ Train / Test split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED
)
print(f"\nTrain size : {len(X_train)}  |  Test size : {len(X_test)}")

# â”€â”€â”€ Model definitions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree":     DecisionTreeRegressor(random_state=SEED, max_depth=8),
    "Random Forest":     RandomForestRegressor(
                             n_estimators=200, random_state=SEED,
                             max_depth=10, min_samples_leaf=5,
                         ),
}

# â”€â”€â”€ Train, evaluate, collect metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
results = {}
predictions = {}

for name, estimator in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",        estimator),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    # 5-fold cross-validation RÂ²
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")

    results[name] = {
        "MAE":    round(mae, 4),
        "MSE":    round(mse, 4),
        "RMSE":   round(rmse, 4),
        "RÂ²":     round(r2, 4),
        "CV RÂ²":  round(cv_scores.mean(), 4),
        "pipeline": pipeline,
    }
    predictions[name] = y_pred
    print(f"\n  {name}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    RÂ²   : {r2:.4f}")
    print(f"    CV RÂ² (5-fold): {cv_scores.mean():.4f} Â± {cv_scores.std():.4f}")

# â”€â”€â”€ Results table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 60)
print("MODEL COMPARISON TABLE")
print("=" * 60)
results_df = pd.DataFrame({
    k: {m: results[k][m] for m in ["MAE","MSE","RMSE","RÂ²","CV RÂ²"]}
    for k in results
}).T
print(results_df.to_string())

# â”€â”€â”€ Best model selection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
best_name = results_df["RÂ²"].idxmax()
best_pipeline = results[best_name]["pipeline"]
print(f"\nâœ…  Best model: {best_name}  (highest RÂ²: {results_df.loc[best_name,'RÂ²']})")

# â”€â”€â”€ Save best model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model_path = os.path.join(MODEL_DIR, "regression_model.pkl")
joblib.dump(best_pipeline, model_path)
print(f"    Saved â†’ {model_path}")

# Save all pipelines (for Streamlit / predict.py)
all_models_path = os.path.join(MODEL_DIR, "regression_all_models.pkl")
joblib.dump({k: results[k]["pipeline"] for k in results}, all_models_path)

# â”€â”€â”€ VISUALISATIONS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def save(fig, name):
    path = os.path.join(VIZ_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ðŸ’¾  {path}")

# 1. Actual vs Predicted â€” best model
y_pred_best = predictions[best_name]
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred_best, alpha=0.45, s=20, color="#3b82f6", edgecolors="none")
lo, hi = min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())
ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
ax.set_title(f"Actual vs Predicted â€” {best_name}\n(RÂ² = {results_df.loc[best_name,'RÂ²']})",
             fontweight="bold")
ax.set_xlabel("Actual Final Score")
ax.set_ylabel("Predicted Final Score")
ax.legend()
save(fig, "09_regression_actual_vs_predicted.png")

# 2. Residual plot â€” best model
residuals = y_test.values - y_pred_best
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_pred_best, residuals, alpha=0.4, s=18, color="#f97316", edgecolors="none")
ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
ax.set_title(f"Residual Plot â€” {best_name}", fontweight="bold")
ax.set_xlabel("Predicted Final Score")
ax.set_ylabel("Residual (Actual âˆ’ Predicted)")
save(fig, "10_regression_residuals.png")

# 3. Model comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
metrics = ["RMSE", "RÂ²"]
colors = ["#ef4444", "#22c55e"]
for i, metric in enumerate(metrics):
    vals = results_df[metric].values
    names = results_df.index.tolist()
    bars = axes[i].bar(names, vals, color=colors[i], edgecolor="white")
    axes[i].set_title(f"Model Comparison â€” {metric}", fontweight="bold")
    axes[i].set_ylabel(metric)
    for bar, v in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    axes[i].set_ylim(0, vals.max() * 1.15)
plt.tight_layout()
save(fig, "11_regression_model_comparison.png")

# 4. Feature importance (Random Forest only)
rf_name = "Random Forest"
if rf_name in results:
    rf_pipeline = results[rf_name]["pipeline"]
    rf_model    = rf_pipeline.named_steps["model"]
    importances = rf_model.feature_importances_

    # Reconstruct feature names after preprocessing (numeric scaled + passthrough)
    prep = rf_pipeline.named_steps["preprocessor"]
    passthrough_cols = [c for c in df.columns if c.startswith("course_")]
    from preprocessing import ENCODED_FEATURES, NUMERIC_FEATURES
    all_feature_names = NUMERIC_FEATURES + ENCODED_FEATURES + passthrough_cols

    imp_df = pd.DataFrame({
        "Feature":    all_feature_names[:len(importances)],
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1],
            color="#3b82f6", edgecolor="white")
    ax.set_title("Random Forest â€” Feature Importances (Regression)", fontweight="bold")
    ax.set_xlabel("Mean Decrease in Impurity (Gini)")
    ax.axvline(imp_df["Importance"].mean(), color="crimson",
               linestyle="--", label="Mean importance")
    ax.legend()
    save(fig, "12_regression_feature_importance.png")

    print(f"\nTop-5 features by Random Forest importance:")
    print(imp_df.head(5)[["Feature","Importance"]].to_string(index=False))

print(f"\nâœ…  Regression training complete.")
print("=" * 60)

