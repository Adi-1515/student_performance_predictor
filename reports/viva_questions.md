# Viva Questions & Answers
## Student Performance Prediction & Analytics — IBM SkillsBuild Internship

---

## SECTION A — Data Analytics & EDA

**Q1. What is Exploratory Data Analysis (EDA)?**

EDA is the process of examining a dataset to understand its structure, identify patterns,
detect anomalies, and form hypotheses before applying formal modelling.
It uses statistical summaries (mean, median, standard deviation) and visualisations
(histograms, scatter plots, heatmaps) to gain insight.
In this project, EDA revealed which features are most associated with final_score.

---

**Q2. Why is data cleaning important?**

Real-world (and even synthetic) datasets contain:
- Missing values that would cause model errors.
- Duplicate rows that would bias model training.
- Outliers that can skew statistical summaries and model coefficients.
- Invalid values (e.g., negative age) that make no domain sense.

Without cleaning, model performance is unreliable and results are misleading.

---

**Q3. What is data preprocessing?**

Data preprocessing converts raw data into a form suitable for machine learning.
Steps include:
- Handling missing values (imputation)
- Encoding categorical variables (label/one-hot encoding)
- Feature scaling (StandardScaler)
- Feature selection or engineering

In this project, preprocessing was implemented as a scikit-learn Pipeline.

---

**Q4. What is the difference between correlation and causation?**

Correlation measures the statistical relationship between two variables.
Causation means one variable directly causes a change in another.
High correlation does NOT imply causation.
For example: study_hours may be correlated with final_score in this dataset,
but we cannot conclude that simply increasing study hours guarantees a higher score.

---

**Q5. What is a correlation heatmap and why is it useful?**

A correlation heatmap visualises the Pearson correlation coefficient between all
pairs of numeric features as a colour matrix.
Values range from −1 (perfect negative correlation) to +1 (perfect positive correlation).
It helps quickly identify which features are strongly related to each other
and to the target variable, and which features may be redundant (multicollinearity).

---

**Q6. What is a boxplot and what does it show?**

A boxplot (box-and-whisker plot) shows the distribution of a numeric variable:
- Box: Q1 (25th percentile) to Q3 (75th percentile)
- Median line inside the box
- Whiskers extend to Q1 − 1.5×IQR and Q3 + 1.5×IQR
- Dots beyond whiskers are outliers.
In this project, boxplots compared final_score across stress levels.

---

**Q7. What is IQR and how is it used for outlier detection?**

IQR (Interquartile Range) = Q3 − Q1.
Outliers are defined as values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR.
This project uses Winsorisation: values beyond these fences are clipped to the fence
value rather than removed, preserving dataset size.

---

**Q8. What is imputation?**

Imputation means filling in missing values with estimated values.
- **Median imputation**: Replace NaN with the column median. Robust to outliers.
- **Mode imputation**: Replace NaN with the most frequent category value.
This project uses both, depending on column type.

---

## SECTION B — Machine Learning Fundamentals

**Q9. What is supervised learning?**

Supervised learning trains a model on labelled data (input-output pairs).
The model learns to map inputs to outputs.
Examples:
- Regression: predict a continuous value (final_score)
- Classification: predict a category (Low / Average / Good / Excellent)

---

**Q10. What is the difference between regression and classification?**

| | Regression | Classification |
|---|---|---|
| Target | Continuous (e.g., 73.5) | Discrete category (e.g., "Good") |
| Example metric | RMSE, R² | Accuracy, F1 |
| Example model | Linear Regression | Logistic Regression |

This project uses both.

---

**Q11. What is Linear Regression?**

Linear Regression fits a straight-line (hyperplane in multiple dimensions) to the data:
`y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ + ε`

It minimises the Sum of Squared Errors (SSE).
Assumptions: linearity, no multicollinearity, homoscedasticity.
Used as a **baseline** in this project.

---

**Q12. What is a Decision Tree?**

A Decision Tree splits data recursively based on feature thresholds.
At each node, the split that minimises impurity (Gini for classification,
MSE for regression) is chosen.
- Interpretable (human-readable tree structure).
- Can overfit if not regularised (max_depth controls this).

---

**Q13. What is Random Forest and why is it better than a single Decision Tree?**

Random Forest is an **ensemble** of many Decision Trees trained on:
- Bootstrap samples (random rows)
- Random feature subsets at each split

Final prediction: average (regression) or majority vote (classification).
Advantages over single tree:
- Reduces variance (less overfitting)
- More robust to noise
- Provides feature importance scores

---

**Q14. What is Logistic Regression?**

Despite its name, Logistic Regression is a **classification** model.
It applies the sigmoid function to produce probabilities:
`P(y=1) = 1 / (1 + e^(-z))`  where `z = w₀ + w₁x₁ + ...`

For multiclass problems, it uses softmax (one-vs-rest or multinomial).
Requires feature scaling — handled by the pipeline's StandardScaler.

---

**Q15. What is overfitting and underfitting?**

| | Overfitting | Underfitting |
|---|---|---|
| Definition | Model memorises training data; fails on new data | Model is too simple; poor on both train and test |
| Train accuracy | Very high | Low |
| Test accuracy | Much lower than train | Low |
| Fix | Regularisation, max_depth, more data | More complex model, more features |

Decision Trees without depth limits overfit easily.
Random Forest mitigates overfitting via averaging.

---

**Q16. What is cross-validation?**

Cross-validation evaluates a model more reliably by training and testing on
different data splits.
In 5-fold CV:
1. Dataset is split into 5 equal folds.
2. Train on 4 folds, test on 1 fold — repeat 5 times.
3. Report mean ± std of the metric.
This provides a better estimate of generalisation than a single train/test split.

---

**Q17. What is a train-test split and why is it needed?**

We split data into training (80%) and testing (20%) sets.
- **Training set**: used to fit the model.
- **Testing set**: used to evaluate the model on unseen data.
Without this split, we cannot know whether the model generalises or merely memorises
the training data.

---

**Q18. What is data leakage and why is it dangerous?**

Data leakage occurs when information from the test set (or future data) is used
during training, making the model appear better than it really is.
Example: fitting the StandardScaler on the full dataset before splitting would
leak test set statistics into training.
This project avoids leakage by fitting the pipeline only on training data.

---

## SECTION C — Metrics

**Q19. What is MAE (Mean Absolute Error)?**

`MAE = (1/n) × Σ|yᵢ − ŷᵢ|`
The average absolute difference between actual and predicted values.
Easy to interpret: "On average, predictions are off by X score points."
Less sensitive to large errors than RMSE.

---

**Q20. What is RMSE (Root Mean Squared Error)?**

`RMSE = √[(1/n) × Σ(yᵢ − ŷᵢ)²]`
Like MAE but penalises large errors more heavily (due to squaring).
In the same units as the target variable.
Preferred when large errors are particularly undesirable.

---

**Q21. What is R² (R-squared)?**

`R² = 1 − SS_res / SS_tot`
Proportion of variance in the target variable explained by the model.
- R² = 1.0 → perfect fit
- R² = 0.0 → model performs no better than predicting the mean
- R² < 0 → model is worse than predicting the mean
Higher R² indicates better fit, but R² alone does not guarantee a good model.

---

**Q22. What is accuracy and when is it not enough?**

`Accuracy = Correct predictions / Total predictions`
Simple, intuitive — but misleading when classes are imbalanced.
Example: If 90% of students pass, a model that always predicts "Pass" achieves
90% accuracy without learning anything useful.
F1 Score, Precision, and Recall are better in such cases.

---

**Q23. What is Precision?**

`Precision = TP / (TP + FP)`
Of all students the model predicted as "Excellent", how many actually were?
Important when false positives are costly.

---

**Q24. What is Recall?**

`Recall = TP / (TP + FN)`
Of all students who were actually "Low" performers, how many did the model correctly identify?
Important when false negatives are costly (missing at-risk students is bad).

---

**Q25. What is F1 Score?**

`F1 = 2 × (Precision × Recall) / (Precision + Recall)`
Harmonic mean of Precision and Recall.
Useful when you need a balance between both.
This project reports weighted F1 across all classes.

---

**Q26. What is a Confusion Matrix?**

A confusion matrix shows the counts of:
- True Positives (TP): correctly predicted positive class
- True Negatives (TN): correctly predicted negative class
- False Positives (FP): incorrectly predicted as positive
- False Negatives (FN): incorrectly predicted as negative

For multiclass, it is an N×N matrix (N = number of classes).
Visualised as a heatmap using Seaborn in this project.

---

## SECTION D — Unsupervised Learning

**Q27. What is K-Means Clustering?**

K-Means partitions data into k clusters by:
1. Randomly initialising k centroids.
2. Assigning each point to the nearest centroid.
3. Recomputing centroids as cluster means.
4. Repeating until assignments stabilise.

In this project, k=4 was used to segment students into performance groups.

---

**Q28. How do you choose the optimal k in K-Means?**

Two methods were used:
1. **Elbow Method**: Plot WCSS (Within-Cluster Sum of Squares) vs k.
   The "elbow" point (where WCSS decrease flattens) suggests the optimal k.
2. **Silhouette Score**: Measures how similar each point is to its own cluster
   vs other clusters. Higher is better (max 1.0).

k=4 was chosen for interpretability (four analytical student segments).

---

**Q29. What is PCA (Principal Component Analysis)?**

PCA reduces high-dimensional data to fewer dimensions while preserving
maximum variance.
In this project, PCA was used to create a 2D visualisation of the 8-dimensional
clustering space, making it possible to see cluster separations in a scatter plot.

---

## SECTION E — Project-Specific

**Q30. Why did you select this project?**

Student performance prediction directly applies data analytics concepts
(cleaning, EDA, feature engineering), supervised ML (regression + classification),
unsupervised ML (clustering), and practical tools (Streamlit, Power BI).
It is socially relevant — educators can use such tools to support students early.
It aligns perfectly with the IBM SkillsBuild Data Analytics with AI curriculum.

---

**Q31. What dataset did you use?**

A synthetic dataset of ~1,200 student records with 19 features,
generated using Python (NumPy/Pandas) with realistic statistical distributions.
All data is fictional — no real students are involved.
Missing values, duplicates, and outliers were intentionally injected to
demonstrate the full data cleaning workflow.

---

**Q32. What were the most important features in your models?**

Based on Random Forest feature importance (fill in after running code):
- `previous_semester_marks` — strongest predictor
- `attendance_percentage` — second strongest
- `study_hours` — important lifestyle factor
- `assignment_score` — continuous assessment signal
- `previous_cgpa` — cumulative academic history

> Note: These importances reflect the synthetic data's design and the model's
> internal structure, not proven real-world causal factors.

---

**Q33. Which model performed best and why?**

For regression: likely **Random Forest Regressor** (higher R², lower RMSE)
due to its ability to capture non-linear feature interactions and robustness to outliers.

For classification: likely **Random Forest Classifier** (higher F1)
for similar reasons.

However, final selection was based on actual test metrics, not assumption.

---

**Q34. What challenges did you face?**

1. Designing a realistic synthetic dataset without making relationships
   too perfect or too noisy.
2. Avoiding data leakage when building the preprocessing pipeline.
3. Handling class imbalance in the performance category classification.
4. Choosing appropriate metrics for multiclass classification.
5. Making the Streamlit application work correctly with the saved pipeline.

---

**Q35. What are the ethical implications of this project?**

- Uses synthetic data — no real student privacy is at risk.
- Model predictions should support educators, not replace human judgment.
- Over-reliance on model predictions could unfairly stigmatise students.
- Bias in training data would propagate into predictions.
- Any real deployment requires institutional ethics approval and student consent.

---

**Q36. How can this project be improved?**

1. Use real (anonymised, consented) student data.
2. Add SHAP values for model explainability.
3. Connect to a live database for real-time data ingestion.
4. Build a REST API for LMS integration.
5. Add time-series analysis (grade trends over semesters).
6. Implement retraining when model performance degrades.
7. Include more features: teacher quality, socioeconomic background.

---

**Q37. What is feature engineering?**

Creating new features from existing ones to improve model performance.
In this project:
- `study_bucket`: bucketed study_hours (0–2, 2–4, etc.)
- `attendance_bucket`: bucketed attendance ranges
- `stress_level_encoded`: ordinal encoding of stress
- `extracurricular_encoded`: binary encoding
- `performance_encoded`: ordinal encoding of category

---

**Q38. What is StandardScaler and why is it needed?**

StandardScaler transforms each feature to have zero mean and unit variance:
`z = (x − μ) / σ`

Required for:
- Logistic Regression (gradient descent converges faster with scaled features)
- Distance-based algorithms (K-Means)

Not strictly needed for tree-based models (Decision Tree, Random Forest)
but included in the pipeline for consistency.

---

**Q39. What is joblib and why do we use it?**

Joblib is a Python library for saving and loading Python objects, especially
large NumPy arrays and scikit-learn models, efficiently.
`joblib.dump(model, 'model.pkl')` saves a trained pipeline.
`joblib.load('model.pkl')` loads it back for prediction.
It is faster than Python's built-in `pickle` for large array objects.

---

**Q40. Why use a Pipeline in scikit-learn?**

A Pipeline chains preprocessing and model into a single object:
1. Prevents data leakage (transforms are fit on training data only).
2. Simplifies code (one `fit()` and `predict()` call).
3. Makes deployment easier (save one pipeline, not separate scaler + model).
4. Enables cross-validation of the full pipeline.
