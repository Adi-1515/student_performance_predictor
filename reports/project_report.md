# Student Performance Prediction & Analytics Using Machine Learning
## Academic Project Report

**Submitted by:** [Your Name]
**Roll No.:** [Your Roll Number]
**College:** [Your College Name]
**Program:** B.Tech Computer Science Engineering
**Internship:** IBM SkillsBuild Data Analytics with AI — AICTE 2026
**Submission Date:** [Date]

---

## Table of Contents

1. Introduction
2. Problem Statement
3. Objectives
4. Existing System
5. Proposed System
6. Dataset Description
7. Data Preprocessing
8. Exploratory Data Analysis
9. Machine Learning Methodology
10. Regression Models
11. Classification Models
12. Model Evaluation
13. Unsupervised Learning — Clustering
14. Power BI Dashboard
15. Results and Findings
16. AI-Driven Insights
17. Limitations
18. Future Scope
19. Conclusion
20. Ethics Statement
21. References

---

## Chapter 1 — Introduction

Education is one of the most critical pillars of human development.
Academic institutions generate large volumes of student performance data every semester —
attendance records, assignment scores, internal exam results, and final grades.
This data, if analysed effectively, can reveal patterns that help educators and
academic counsellors identify students who may need support before it is too late.

Machine Learning (ML) provides powerful tools to mine patterns from educational data.
This project applies data analytics and supervised machine learning techniques to a
synthetic student dataset in order to:

- Understand the factors most strongly associated with academic performance in the dataset.
- Build predictive models for student final scores and performance categories.
- Create a practical tool (Streamlit application) for academic outcome prediction.
- Develop a Power BI dashboard for institutional overview.

The project was developed as the final submission for the
**IBM SkillsBuild Data Analytics with AI Internship 2026**, conducted in
collaboration with AICTE.

---

## Chapter 2 — Problem Statement

Academic institutions often detect underperforming students only after final examinations,
at which point it is too late for meaningful intervention.

**The core problem:**
How can we use mid-semester student data — including study habits, attendance,
assignment scores, and lifestyle factors — to predict academic outcomes early
and identify students who may need academic support?

**Secondary challenges:**
- Student data is noisy (missing values, outliers, inconsistencies).
- No single feature completely determines academic performance.
- Multiple interacting factors (academic and non-academic) influence outcomes.
- Predictions must be transparent and explainable to be useful in practice.

---

## Chapter 3 — Objectives

1. Generate and preprocess a realistic synthetic student dataset.
2. Perform thorough exploratory data analysis (EDA).
3. Identify key features associated with final academic score.
4. Build and compare regression models to predict final score.
5. Build and compare classification models to predict performance category.
6. Perform K-Means clustering for student segmentation.
7. Evaluate all models using appropriate metrics.
8. Build a Streamlit prediction interface.
9. Design a Power BI dashboard for stakeholder reporting.
10. Generate automated AI-driven insights from the data.
11. Document the project for academic and GitHub portfolio purposes.

---

## Chapter 4 — Existing System

Existing approaches to student performance analysis typically include:

### 4.1 Manual Grade Monitoring
Academic staff review student marks at semester end.
**Limitation:** Reactive — intervention comes after failure.

### 4.2 Basic Statistical Reports
Institutions compute averages and pass rates.
**Limitation:** No predictive capability; no feature-level analysis.

### 4.3 Prior Research
Several academic papers have applied ML to educational datasets:
- Cortez & Silva (2008) used Decision Trees on a Portuguese student dataset.
- Kotsiantis et al. (2010) applied Naive Bayes and SVM for grade prediction.
- Hussain et al. (2019) used Random Forest on student data.

**Limitation of prior work:** Most use small, institution-specific datasets that
do not generalise. Few include lifestyle factors.

---

## Chapter 5 — Proposed System

The proposed system is a complete data analytics and ML pipeline that:

1. **Ingests** student academic and lifestyle data.
2. **Cleans** the data (missing values, duplicates, outliers).
3. **Analyses** the data via EDA and statistical methods.
4. **Trains** multiple regression and classification models.
5. **Evaluates** and compares models using standard metrics.
6. **Segments** students via unsupervised K-Means clustering.
7. **Predicts** outcomes for new students via a Streamlit interface.
8. **Visualises** results in a Power BI dashboard.
9. **Generates** plain-language AI insights automatically.

**Advantages over existing systems:**
- Proactive: mid-semester prediction rather than post-exam reporting.
- Multi-model: compares multiple algorithms and selects the best.
- Explainable: feature importance charts clarify model behaviour.
- Accessible: no-code Streamlit interface for non-technical users.

---

## Chapter 6 — Dataset Description

### 6.1 Dataset Type
**Synthetic** — artificially generated using Python (NumPy/Pandas) with realistic
statistical distributions and inter-variable correlations.

> All data is fictional. It does NOT represent real students.
> It was created solely for demonstrating data analytics concepts.

### 6.2 Dataset Size
| Property | Value |
|---|---|
| Raw records (incl. duplicates) | ~1,212 |
| After deduplication | ~1,200 |
| Features | 19 |
| Missing values (injected) | ~4% per selected column |
| Duplicate rows (injected) | 12 |
| Outlier rows (injected) | 5–9 |

### 6.3 Feature Descriptions

| Column | Type | Description |
|---|---|---|
| student_id | String | Unique student identifier (e.g. STU1001) |
| age | Integer | Student age (expected: 17–24) |
| gender | Categorical | Male / Female / Other |
| course | Categorical | Computer Science / Electronics / Mechanical / Civil / Business |
| semester | Integer | Current semester (1–8) |
| study_hours | Float | Average daily study hours |
| sleep_hours | Float | Average daily sleep hours |
| internet_usage_hours | Float | Average daily internet usage hours |
| extracurricular_activity | Categorical | Yes / No |
| stress_level | Categorical | Low / Medium / High |
| physical_activity_hours | Float | Average daily physical activity hours |
| attendance_percentage | Float | Attendance percentage (0–100) |
| assignment_score | Float | Average assignment score (0–100) |
| internal_exam_score | Float | Internal/mid-term exam score (0–100) |
| previous_semester_marks | Float | Marks from the previous semester |
| previous_cgpa | Float | Cumulative GPA from previous semesters |
| **final_score** | **Float** | **Target: regression** |
| **performance_category** | **Categorical** | **Target: classification (Low/Average/Good/Excellent)** |
| **pass_fail** | **Categorical** | **Binary target (Pass/Fail)** |

### 6.4 Performance Category Thresholds
| Category | Score Range |
|---|---|
| Excellent | ≥ 80 |
| Good | 65 – 79 |
| Average | 50 – 64 |
| Low | < 50 |

### 6.5 Synthetic Data Design Rationale
The final_score was computed as a weighted combination of:
- previous_semester_marks (25%)
- attendance_percentage (20%)
- study_hours × 4.5 (18%)
- assignment_score (15%)
- internal_exam_score (10%)
- previous_cgpa × 5 (8%)
- sleep_hours, internet_usage, stress, extracurricular (remaining)
- Gaussian noise σ=5 added to prevent perfect prediction.

---

## Chapter 7 — Data Preprocessing

### 7.1 Loading and Inspection
The raw CSV was loaded using Pandas. Shape, data types, missing values,
and duplicate counts were printed for initial inspection.

### 7.2 Duplicate Removal
`DataFrame.drop_duplicates()` was used to remove 12 injected duplicate rows.

### 7.3 Invalid Value Detection
- Age values outside [15, 35] were replaced with NaN and subsequently imputed.
- study_hours > 16 were treated as implausible and replaced with NaN.

### 7.4 Missing Value Handling

| Strategy | Columns |
|---|---|
| Median imputation | All numerical features |
| Mode imputation | All categorical features |

Median was chosen over mean for numerical imputation because it is
more robust to outliers.

### 7.5 Outlier Handling
IQR-based Winsorisation was applied to numerical columns:
- Lower fence = Q1 − 1.5 × IQR
- Upper fence = Q3 + 1.5 × IQR
- Values outside fences were clipped (not removed).

### 7.6 Encoding

| Column | Encoding Method |
|---|---|
| stress_level | Ordinal: Low=0, Medium=1, High=2 |
| gender | Label: Male=0, Female=1, Other=2 |
| extracurricular_activity | Binary: Yes=1, No=0 |
| pass_fail | Binary: Pass=1, Fail=0 |
| performance_category | Ordinal: Low=0, Average=1, Good=2, Excellent=3 |
| course | One-hot encoding (drop_first=True) |

### 7.7 StandardScaler
Numerical features were standardised (zero mean, unit variance) inside the
scikit-learn Pipeline during model training. This is important for
Logistic Regression, which is sensitive to feature scale.

---

## Chapter 8 — Exploratory Data Analysis

EDA was performed using Matplotlib and Seaborn.

### 8.1 Univariate Analysis
Key distributions examined:
- **study_hours**: Right-skewed; most students study 3–7 hours/day.
- **attendance_percentage**: Approximately normal; mean ~78%.
- **final_score**: Roughly normal after transformation; range 0–100.
- **sleep_hours**: Approximately normal; mean ~6.5 hours.
- **previous_cgpa**: Slightly left-skewed; most students have CGPA 6–9.

### 8.2 Bivariate Analysis
Scatter plots with correlation coefficients (r):
- **study_hours vs final_score**: Positive association.
- **attendance_percentage vs final_score**: Positive association.
- **previous_semester_marks vs final_score**: Strong positive association.
- **assignment_score vs final_score**: Moderate positive association.
- **internet_usage_hours vs final_score**: Slight negative association.
- **stress_level_encoded vs final_score**: Negative association.

> Note: These are associations within the synthetic dataset, not causal claims.

### 8.3 Multivariate Analysis
A correlation heatmap was generated for all numeric features.
Features most positively correlated with final_score (in this dataset):
- previous_semester_marks
- attendance_percentage
- study_hours
- assignment_score
- previous_cgpa

### 8.4 Categorical Analysis
Boxplots showed:
- High-stress students tend to have lower average scores.
- Students with extracurricular activities show slightly higher average scores.
- Performance differences across gender groups are small.

### 8.5 Statistical Summary

| Metric | study_hours | attendance | final_score |
|---|---|---|---|
| Mean | ~4.92 | ~77.5 | ~47.7 |
| Median | ~4.92 | ~77.5 | ~47.7 |
| Std Dev | ~1.96 | ~12.4 | ~15.2 |
| Min | ~0.30 | ~40.0 | ~0.0 |
| Max | ~11.4 | ~100.0 | ~100.0 |

> Run `python src/eda.py` to populate actual values.

---

## Chapter 9 — Machine Learning Methodology

### 9.1 Framework
All ML models were implemented using **scikit-learn**.

### 9.2 Pipeline Architecture
Each model was wrapped in a scikit-learn `Pipeline` consisting of:
1. `ColumnTransformer` (StandardScaler for numeric; passthrough for encoded/dummies)
2. The estimator (LinearRegression, RandomForest, etc.)

Using a pipeline ensures that:
- Preprocessing is applied consistently to training and test data.
- No data leakage occurs (scaler is fit only on training data).

### 9.3 Train/Test Split
- 80% training, 20% testing
- `random_state=42` for reproducibility
- Stratified split used for classification to preserve class balance

### 9.4 Cross-Validation
5-fold cross-validation was used to obtain more reliable performance estimates.

---

## Chapter 10 — Regression Models

**Objective:** Predict `final_score` (continuous numeric target).

### 10.1 Linear Regression
A baseline model that assumes a linear relationship between features and target.
- Simple, interpretable, fast to train.
- Sensitive to outliers and assumes linearity.

### 10.2 Decision Tree Regressor
A non-linear model that recursively splits data based on feature thresholds.
- Interpretable (visualisable tree).
- Can overfit if not constrained; `max_depth=8` used.

### 10.3 Random Forest Regressor
An ensemble of 200 decision trees with feature bagging.
- More robust than a single tree.
- Handles non-linearity and feature interactions.
- Provides feature importance scores.
- Parameters: `n_estimators=200, max_depth=10, min_samples_leaf=5`.

### 10.4 Results

| Model | MAE | MSE | RMSE | R² | CV R² |
|---|---|---|---|---|---|
| **Linear Regression** | **8.04** | 103.07 | **10.15** | **0.5398** | **0.5635** |
| Decision Tree | 11.50 | 206.75 | 14.38 | 0.0769 | 0.1109 |
| Random Forest | 8.99 | 123.78 | 11.13 | 0.4473 | 0.4652 |

**Model selected as best: Linear Regression (R² = 0.5398)**
Saved as `models/regression_model.pkl`.

---

## Chapter 11 — Classification Models

**Objective:** Predict `performance_category` (Low / Average / Good / Excellent).

### 11.1 Logistic Regression
A linear probabilistic classifier for multi-class problems.
- Effective for linearly separable classes.
- Requires feature scaling (handled by pipeline).
- Parameters: `C=1.0, max_iter=1000, multi_class='auto'`.

### 11.2 Decision Tree Classifier
A tree-based classifier using Gini impurity.
- Interpretable splits.
- Parameters: `max_depth=8, min_samples_leaf=5`.

### 11.3 Random Forest Classifier
Ensemble of 200 decision trees.
- More robust than single tree; handles noisy data well.
- Parameters: `n_estimators=200, max_depth=10, min_samples_leaf=3`.

### 11.4 Results

| Model | Accuracy | Precision | Recall | F1 Score | CV Acc |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.7250** | **0.7046** | **0.7250** | **0.7135** | **0.6483** |
| Decision Tree | 0.5458 | 0.5283 | 0.5458 | 0.5364 | 0.5617 |
| Random Forest | 0.6333 | 0.5743 | 0.6333 | 0.5904 | 0.6225 |

Metrics use **weighted average** to account for class imbalance.

---

## Chapter 12 — Model Evaluation

### 12.1 Regression Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | Mean(|actual - pred|) | Average prediction error in score units |
| MSE | Mean((actual - pred)²) | Penalises large errors more than MAE |
| RMSE | √MSE | Same units as target; intuitive |
| R² | 1 - SS_res/SS_tot | Proportion of variance explained (1.0 = perfect) |

### 12.2 Classification Metrics

| Metric | Formula | When to Use |
|---|---|---|
| Accuracy | Correct / Total | Only reliable when classes are balanced |
| Precision | TP / (TP + FP) | Important when false positives are costly |
| Recall | TP / (TP + FN) | Important when false negatives are costly |
| F1 Score | 2×P×R / (P+R) | Balance between precision and recall |
| Confusion Matrix | Grid of TP/FP/TN/FN | Detailed per-class breakdown |

### 12.3 Feature Importance
Random Forest feature importances (Mean Decrease in Impurity) show which
features the model found most informative.

> Note: Model feature importance is a model-internal measure.
> It reflects the model's learned structure, not proven causal relationships.

---

## Chapter 13 — Unsupervised Learning: K-Means Clustering

### 13.1 Purpose
Group students into segments based on academic and lifestyle features,
without using the target label (final_score used as a feature, not a label).

### 13.2 Features Used
study_hours, attendance_percentage, previous_semester_marks, final_score,
sleep_hours, internet_usage_hours, stress_level_encoded, previous_cgpa.

### 13.3 Optimal k Selection
- **Elbow Method**: WCSS plotted for k=2 to 10.
- **Silhouette Score**: Computed for k=2 to 8.
- **Chosen k=4** for four-segment interpretability.

### 13.4 Cluster Descriptions
(Actual cluster means — replace with values from execution)

| Cluster | Analytical Label | Avg Score | Avg Study Hours |
|---|---|---|---|
| 0 | Students Needing Support | 39.6 | 5.1 |
| 1 | Above-Average Students | 53.8 | 5.0 |
| 2 | At-Risk Students | 33.0 | 4.6 |
| 3 | High Achievers | 60.8 | 5.3 |

> ⚠️ Cluster labels are analytical descriptions, not definitive student classifications.
> The same student may fall into different clusters if features change.

---

## Chapter 14 — Power BI Dashboard

A 4-page Power BI dashboard was designed using the cleaned CSV as the data source.

| Page | Purpose |
|---|---|
| Executive Overview | KPI cards, performance distribution, course & semester averages |
| Academic Analysis | Scatter plots of academic features vs final score |
| Lifestyle Analysis | Stress, sleep, internet, extracurricular vs performance |
| Risk Analysis | Risk category distribution, at-risk student table |

Detailed construction instructions are in `powerbi/powerbi_dashboard_guide.md`.

---

## Chapter 15 — Results and Findings

### 15.1 Key Dataset Observations

1. Attendance percentage and previous semester marks show the strongest positive
   associations with final score in this dataset.
2. Study hours show a meaningful positive association with final score.
3. High-stress students tend to score lower on average (within this dataset).
4. Students participating in extracurricular activities show a slightly higher average score.
5. Internet usage hours show a slight negative association with final score.

> All findings are observations about the synthetic dataset.
> They should not be interpreted as universal causal claims.

### 15.2 Model Performance
(Fill in actual results after running the code)

- **Best Regression Model:** Linear Regression with R² = **0.5398**
- **Best Classification Model:** Logistic Regression with F1 = **0.7135**

### 15.3 Risk Analysis
- Students with final_score < 50 are classified as at-risk.
- Number of at-risk students: **667** (55.6% of dataset)

---

## Chapter 16 — AI-Driven Insights

The `utils.py` module's `generate_ai_insights()` function automatically produces
plain-language observations including:

1. Attendance comparison: high vs low attendance groups.
2. Study hours comparison: high vs low study groups.
3. Stress level comparison: Low vs High stress groups.
4. Overall pass rate.
5. Count of at-risk students.
6. Top feature correlation with final_score.
7. Extracurricular activity score comparison.

These insights are displayed in the Streamlit application's "AI Insights" tab.

**No external API** is required. All insights are derived from simple dataset
statistics and are regenerated each time the app runs.

---

## Chapter 17 — Limitations

1. **Synthetic dataset**: All data was artificially generated. Model performance
   on real student data may differ significantly.

2. **No causality**: Correlations observed do not prove causation. Study hours
   being associated with better scores does not mean increasing study hours alone
   will improve scores.

3. **Static model**: Models are trained once. They do not update as new student
   data arrives.

4. **Feature completeness**: Many real-world factors affecting performance
   (family background, economic status, learning disabilities, teacher quality)
   are not included.

5. **Class imbalance**: If some performance categories have very few examples,
   classification metrics may be inflated.

6. **Generalisation**: Models trained on data from one institution may not
   generalise well to others.

---

## Chapter 18 — Future Scope

1. Collect real anonymised student data with institutional ethics approval.
2. Add time-series modelling (e.g. LSTM for grade trend prediction).
3. Implement SHAP values for improved model explainability.
4. Connect to a live database (SQLite / PostgreSQL) for real-time updates.
5. Build a REST API for LMS (Learning Management System) integration.
6. Add automated email alerts for at-risk students.
7. Extend to multi-institution comparative analysis.
8. Explore ensemble stacking to improve model accuracy.

---

## Chapter 19 — Conclusion

This project demonstrates a complete data analytics and machine learning pipeline
applied to student performance data.

Starting from a synthetic raw dataset with intentional quality issues,
the project covered:
- Data cleaning and preprocessing
- Exploratory data analysis
- Feature engineering
- Regression modelling (three algorithms)
- Classification modelling (three algorithms)
- Unsupervised clustering (K-Means)
- Automated insight generation
- An interactive Streamlit prediction interface
- A Power BI executive dashboard
- Complete project documentation

The project successfully demonstrates the core skills covered in the
IBM SkillsBuild Data Analytics with AI internship, including data analytics,
supervised learning, unsupervised learning, model evaluation, visualisation,
and responsible AI considerations.

---

## Chapter 20 — Ethics Statement

This project takes student data ethics seriously.

1. **Synthetic data only**: No real student information was used.
2. **No PII**: No personally identifiable information is stored or processed.
3. **Model as support tool**: Predictions are designed to support educators,
   not replace human judgment.
4. **Transparency**: Model assumptions and limitations are documented.
5. **Bias awareness**: Synthetic data may embed unintended biases from its
   generation process. Real-world deployment would require bias auditing.
6. **Consent**: Any real-world deployment must include informed student consent
   as required by applicable data protection regulations (e.g., GDPR, FERPA,
   or applicable Indian regulations).

---

## References

1. Cortez, P. & Silva, A. (2008). *Using data mining to predict secondary school
   student performance*. Proceedings of 5th FUture BUsiness TEChnology Conference.

2. Kotsiantis, S. et al. (2010). *Educational data mining: A case study for
   predicting dropout-prone students*. IFIP Advances in Information and Communication Technology.

3. Hussain, S. et al. (2019). *Using machine learning to predict student difficulties
   from learning management system data*. Knowledge and Information Systems.

4. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*.
   Journal of Machine Learning Research, 12, pp. 2825–2830.

5. McKinney, W. (2010). *Data Structures for Statistical Computing in Python*.
   Proceedings of the 9th Python in Science Conference.

6. IBM SkillsBuild Data Analytics with AI Curriculum. (2026). AICTE.

---

*This report was prepared as the final project submission for the
IBM SkillsBuild Data Analytics with AI Internship 2026, AICTE.*
