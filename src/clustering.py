"""
clustering.py
-------------
Unsupervised Learning â€” K-Means Student Segmentation.

Groups students into clusters based on academic and lifestyle features,
then describes each cluster analytically.

NOTE: Cluster labels are analytical descriptions derived from the data;
they do NOT guarantee real-world groupings or carry diagnostic meaning.

Evaluation:
  â€¢ Elbow Method    â€” choose optimal k
  â€¢ Silhouette Score â€” measure cluster quality

Outputs:
  â€¢ models/kmeans_model.pkl
  â€¢ data/processed/student_performance_clustered.csv
  â€¢ visualizations/17_clustering_*

Run:
    python src/clustering.py
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

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_data, NUMERIC_FEATURES

warnings.filterwarnings("ignore")

SEED    = 42
BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")
VIZ_DIR   = os.path.join(BASE_DIR, "visualizations")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.05)

# â”€â”€â”€ Load & prepare â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=" * 60)
print("UNSUPERVISED LEARNING â€” K-MEANS STUDENT SEGMENTATION")
print("=" * 60)

df = load_data()

# Use a focused subset of features for clustering
CLUSTER_FEATURES = [
    "study_hours", "attendance_percentage",
    "previous_semester_marks", "final_score",
    "sleep_hours", "internet_usage_hours",
    "stress_level_encoded", "previous_cgpa",
]

X_clust = df[CLUSTER_FEATURES].dropna()
scaler_clust = StandardScaler()
X_scaled = scaler_clust.fit_transform(X_clust)
print(f"Clustering on {len(X_clust)} students Ã— {len(CLUSTER_FEATURES)} features")

# â”€â”€â”€ Elbow Method â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[Elbow Method] Computing WCSS for k = 2 â€¦ 10 â€¦")
wcss = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(k_range), wcss, marker="o", color="#3b82f6", linewidth=2.5, markersize=8)
ax.set_title("Elbow Method â€” Optimal Number of Clusters", fontweight="bold")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
ax.axvline(4, color="crimson", linestyle="--", label="Chosen k=4")
ax.legend()
elbow_path = os.path.join(VIZ_DIR, "17_clustering_elbow.png")
fig.savefig(elbow_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ðŸ’¾  {elbow_path}")

# â”€â”€â”€ Silhouette Scores â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[Silhouette Scores] for k = 2 â€¦ 8 â€¦")
sil_scores = []
sil_k_range = range(2, 9)
for k in sil_k_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_scaled)
    score  = silhouette_score(X_scaled, labels)
    sil_scores.append(score)
    print(f"  k={k}  silhouette={score:.4f}")

best_k = list(sil_k_range)[np.argmax(sil_scores)]
print(f"\n  Best k by silhouette: {best_k}  (score: {max(sil_scores):.4f})")

# Use k=4 for interpretability (common academic groupings)
CHOSEN_K = 4
print(f"\n  Using k={CHOSEN_K} for analytical interpretability")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(sil_k_range), sil_scores, marker="s", color="#22c55e", linewidth=2.5, markersize=8)
ax.set_title("Silhouette Score vs k", fontweight="bold")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Silhouette Score")
ax.axvline(CHOSEN_K, color="crimson", linestyle="--", label=f"Chosen k={CHOSEN_K}")
ax.legend()
sil_path = os.path.join(VIZ_DIR, "18_clustering_silhouette.png")
fig.savefig(sil_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ðŸ’¾  {sil_path}")

# â”€â”€â”€ Final KMeans (k=4) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
km_final = KMeans(n_clusters=CHOSEN_K, random_state=SEED, n_init=20)
cluster_labels = km_final.fit_predict(X_scaled)

X_clust = X_clust.copy()
X_clust["cluster"] = cluster_labels

# â”€â”€â”€ Cluster profiling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cluster_means = X_clust.groupby("cluster")[CLUSTER_FEATURES].mean().round(2)
print(f"\nCluster Centroids (original scale):")
print(cluster_means.to_string())

# Assign analytical labels based on final_score & study_hours
# (descriptive, not prescriptive)
score_rank = cluster_means["final_score"].rank()
study_rank = cluster_means["study_hours"].rank()

label_map = {}
for c in range(CHOSEN_K):
    sr = score_rank[c]
    if sr == CHOSEN_K:
        label_map[c] = "High Achievers"
    elif sr == CHOSEN_K - 1:
        label_map[c] = "Above-Average Students"
    elif sr == 2:
        label_map[c] = "Students Needing Support"
    else:
        label_map[c] = "At-Risk Students"

X_clust["cluster_label"] = X_clust["cluster"].map(label_map)
print(f"\nCluster labels assigned:")
for c, lbl in label_map.items():
    cnt = (X_clust["cluster"] == c).sum()
    avg_score = cluster_means.loc[c, "final_score"]
    print(f"  Cluster {c} â†’ {lbl:>30}  (n={cnt}, avg final_score={avg_score:.1f})")

# â”€â”€â”€ PCA 2D visualisation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_

colors_clust = ["#3b82f6","#22c55e","#f97316","#a855f7"]
fig, ax = plt.subplots(figsize=(10, 7))
for c, lbl in label_map.items():
    mask = X_clust["cluster"] == c
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        s=30, alpha=0.6, color=colors_clust[c],
        edgecolors="none", label=lbl,
    )
ax.set_title(
    f"Student Clusters (PCA 2D) â€” k={CHOSEN_K}\n"
    f"Explained variance: PC1={explained[0]*100:.1f}%, PC2={explained[1]*100:.1f}%",
    fontweight="bold",
)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend(title="Cluster", loc="best")
pca_path = os.path.join(VIZ_DIR, "19_clustering_pca.png")
fig.savefig(pca_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ðŸ’¾  {pca_path}")

# â”€â”€â”€ Cluster boxplot â€” final_score â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, ax = plt.subplots(figsize=(10, 6))
order = sorted(label_map.values(), key=lambda l: X_clust.groupby("cluster_label")["final_score"].mean()[l], reverse=True)
sns.boxplot(
    data=X_clust, x="cluster_label", y="final_score",
    order=order,
    palette=colors_clust, ax=ax
)
ax.set_title("Final Score Distribution by Student Cluster", fontweight="bold")
ax.set_xlabel("Cluster")
ax.set_ylabel("Final Score")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
box_path = os.path.join(VIZ_DIR, "20_clustering_boxplot.png")
fig.savefig(box_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ðŸ’¾  {box_path}")

# â”€â”€â”€ Save model & annotated dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
joblib.dump({"kmeans": km_final, "scaler": scaler_clust, "label_map": label_map,
             "features": CLUSTER_FEATURES}, model_path)
print(f"\n  ðŸ’¾  KMeans model saved â†’ {model_path}")

# Merge cluster info back to main df
df_out = df.copy()
df_out.loc[X_clust.index, "cluster"]       = cluster_labels
df_out.loc[X_clust.index, "cluster_label"] = X_clust["cluster_label"].values
out_path = os.path.join(BASE_DIR, "data", "processed", "student_performance_clustered.csv")
df_out.to_csv(out_path, index=False)
print(f"  ðŸ’¾  Clustered dataset saved â†’ {out_path}")

print(f"\nâœ…  Clustering complete.")
print("=" * 60)

