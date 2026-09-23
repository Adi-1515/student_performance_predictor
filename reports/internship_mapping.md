# IBM SkillsBuild Internship Curriculum Mapping
## Student Performance Prediction & Analytics Project

---

## Curriculum Alignment Table

| IBM SkillsBuild Topic | Project Implementation | File(s) |
|---|---|---|
| **Data Collection & Loading** | Load CSV using Pandas; inspect shape, dtypes | `src/data_cleaning.py` |
| **Data Cleaning** | Remove duplicates; fix invalid values; impute NaN | `src/data_cleaning.py` |
| **Handling Missing Values** | Median imputation (numeric), mode (categorical) | `src/data_cleaning.py` |
| **Outlier Detection & Treatment** | IQR-based Winsorisation | `src/data_cleaning.py` |
| **Data Wrangling** | Feature encoding, type conversion, column creation | `src/data_cleaning.py`, `preprocessing.py` |
| **Exploratory Data Analysis** | Univariate, bivariate, multivariate analysis | `src/eda.py` |
| **Statistical Analysis** | Mean, median, mode, std, quartiles, correlation | `src/eda.py`, `utils.py` |
| **Data Visualisation** | 8+ Matplotlib/Seaborn charts saved as PNG | `src/eda.py`, `src/train_*.py` |
| **Feature Engineering** | Bucketed study/attendance columns, encoded features | `src/data_cleaning.py`, `preprocessing.py` |
| **Supervised Learning** | Regression + Classification with scikit-learn | `src/train_regression.py`, `train_classification.py` |
| **Regression** | Linear Regression, Decision Tree, Random Forest | `src/train_regression.py` |
| **Classification** | Logistic Regression, Decision Tree, Random Forest | `src/train_classification.py` |
| **Decision Trees** | DecisionTreeRegressor + DecisionTreeClassifier | Both training scripts |
| **Random Forest** | RandomForestRegressor + RandomForestClassifier | Both training scripts |
| **Model Evaluation (Regression)** | MAE, MSE, RMSE, R², 5-fold CV | `src/train_regression.py` |
| **Model Evaluation (Classification)** | Accuracy, Precision, Recall, F1, Confusion Matrix | `src/train_classification.py` |
| **Model Comparison** | Comparison tables for 3 models each | Both training scripts |
| **Feature Importance** | Random Forest feature importance bar charts | Both training scripts |
| **Unsupervised Learning** | K-Means clustering; Elbow + Silhouette evaluation | `src/clustering.py` |
| **Dimensionality Reduction** | PCA 2D for cluster visualisation | `src/clustering.py` |
| **Predictive Analytics** | End-to-end pipeline: clean → train → predict | All src files |
| **Model Persistence** | Save/load models with Joblib (`.pkl`) | All training scripts, `predict.py` |
| **AI-Driven Insights** | Auto-generated plain-language dataset observations | `src/utils.py`, `app/app.py` |
| **Business Decision Support** | Risk level categorisation; at-risk student identification | `src/predict.py`, `app/app.py` |
| **Power BI / BI Dashboard** | 4-page Power BI dashboard with DAX measures | `powerbi/powerbi_dashboard_guide.md` |
| **Web Application Deployment** | Streamlit interactive prediction interface | `app/app.py` |
| **Responsible AI / Ethics** | Ethics statement; disclaimer on synthetic data | `reports/project_report.md` |
| **Project Documentation** | README, academic report, viva Q&A, PPT content | `reports/`, `presentation/` |

---

## Topic Coverage Summary

| Category | Topics Covered | Status |
|---|---|---|
| Data Analytics | Collection, Cleaning, Wrangling, EDA, Stats | ✅ Complete |
| Visualisation | Matplotlib, Seaborn, Power BI | ✅ Complete |
| Supervised ML | Regression (3 models), Classification (3 models) | ✅ Complete |
| Unsupervised ML | K-Means Clustering | ✅ Complete |
| Model Evaluation | Regression + Classification metrics | ✅ Complete |
| Deployment | Streamlit web app | ✅ Complete |
| BI Dashboard | Power BI (4-page guide + DAX formulas) | ✅ Complete |
| AI Insights | Automated insight generation | ✅ Complete |
| Documentation | Report, README, Viva, PPT | ✅ Complete |
| Ethics | Data privacy, responsible use, limitations | ✅ Complete |

---

## Ethics Statement

Because this project involves student-related data (even synthetic), responsible
handling is important to model for future real-world applications.

### 1. Student Privacy
- No real student data was used in this project.
- The dataset is entirely synthetic, generated programmatically.
- No personally identifiable information (PII) is stored or processed.

### 2. Responsible Use of Predictions
- Model predictions are analytical indicators, NOT definitive judgments.
- No prediction should be used to label, stigmatise, or disadvantage a student.
- Educators and academic counsellors must be involved in any academic intervention.
- The system is designed to **support** human decision-making, not replace it.

### 3. Model Bias
- The synthetic dataset was designed to be balanced across demographic groups.
- However, any dataset may contain unintentional biases.
- Real-world deployment would require formal bias auditing.
- Bias in training data will propagate to model predictions.

### 4. Limitations of Predictive Models
- Models are based on historical patterns in training data.
- They cannot account for all factors affecting student performance.
- A low predicted score does not mean a student cannot succeed.
- Predictions represent probabilities, not certainties.

### 5. Data Protection
- Any real-world deployment must comply with applicable data protection regulations.
- In India: relevant regulations include the Digital Personal Data Protection Act 2023.
- Globally: GDPR (EU), FERPA (USA), and institutional ethics board approval apply.
- Student and parental consent must be obtained before collecting or processing real data.

### 6. Transparency
- The model architecture, training data, and limitations are fully documented.
- Feature importance charts help explain which factors influence predictions.
- Plain-language explanations are provided for every prediction in the Streamlit app.

---

## Final Deliverable Checklist

| Deliverable | Status |
|---|---|
| Raw dataset (`student_performance_raw.csv`) | ✅ Generated by `src/generate_dataset.py` |
| Cleaned dataset (`student_performance_cleaned.csv`) | ✅ Generated by `src/data_cleaning.py` |
| Data cleaning script | ✅ `src/data_cleaning.py` |
| EDA script + visualizations | ✅ `src/eda.py` → `visualizations/` |
| Regression training script | ✅ `src/train_regression.py` |
| Classification training script | ✅ `src/train_classification.py` |
| Clustering script | ✅ `src/clustering.py` |
| Predict utility | ✅ `src/predict.py` |
| Trained regression model | ✅ `models/regression_model.pkl` (after running) |
| Trained classification model | ✅ `models/classification_model.pkl` (after running) |
| Evaluation metrics (regression) | ✅ Printed + plotted by training script |
| Evaluation metrics (classification) | ✅ Printed + plotted by training script |
| Feature importance visualizations | ✅ `visualizations/12_regression_feature_importance.png` etc. |
| Risk analysis | ✅ `visualizations/16_risk_level_distribution.png` |
| AI-driven insights | ✅ `src/utils.py` + Streamlit app |
| Streamlit application | ✅ `app/app.py` |
| Power BI dashboard guide | ✅ `powerbi/powerbi_dashboard_guide.md` |
| requirements.txt | ✅ `requirements.txt` |
| README.md | ✅ `README.md` |
| Project report | ✅ `reports/project_report.md` |
| PPT content | ✅ `presentation/presentation_content.md` |
| Viva questions & answers | ✅ `reports/viva_questions.md` |
| Internship curriculum mapping | ✅ This file |
| Ethics statement | ✅ This file + project report |
| GitHub-ready folder structure | ✅ Complete |
