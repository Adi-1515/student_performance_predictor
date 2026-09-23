# 🎓 Student Performance Prediction & Analytics Using Machine Learning

> **IBM SkillsBuild Data Analytics with AI Internship 2026 — AICTE Final Project**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Project Overview

A complete, end-to-end **Data Analytics + Machine Learning** project that analyses
synthetic student academic and lifestyle data to:

- Identify key factors associated with student performance
- Predict a student's **final exam score** (regression)
- Classify students into **performance categories** (Low / Average / Good / Excellent)
- Segment students into groups using **K-Means clustering**
- Generate **AI-driven insights** automatically from the data
- Provide an interactive **Streamlit prediction interface**
- Support academic dashboarding via **Power BI**

> ⚠️ **All data is synthetic** — generated for educational and demonstration purposes only.
> This project does NOT use real student data.

---

## 🎯 Problem Statement

Educational institutions need data-driven tools to identify students who may be
at academic risk early enough to provide timely support.  Traditional methods rely
on end-of-term results, which are too late for intervention.

This project builds a machine learning system that uses mid-semester features
(study habits, attendance, assignment scores, lifestyle indicators) to predict
likely academic outcomes and flag students who may need additional support.

---

## 🏆 Objectives

1. Perform comprehensive data cleaning and preprocessing
2. Conduct exploratory data analysis with statistical and visual techniques
3. Build and compare regression models to predict final score
4. Build and compare classification models to predict performance category
5. Perform K-Means clustering for student segmentation
6. Generate automated analytical insights
7. Deploy an interactive Streamlit prediction interface
8. Build a Power BI executive dashboard

---

## ✨ Features

| Feature | Description |
|---|---|
| Data Generation | Realistic synthetic dataset (1,200 records, 19 columns) |
| Data Cleaning | Missing values, duplicates, outliers, type conversion |
| EDA | 8+ visualizations covering uni/bi/multivariate analysis |
| Regression | Linear Regression, Decision Tree, Random Forest |
| Classification | Logistic Regression, Decision Tree, Random Forest |
| Clustering | K-Means with Elbow + Silhouette evaluation |
| Prediction UI | Streamlit web application with live predictions |
| Risk Analysis | Automatic student risk categorisation |
| AI Insights | Auto-generated plain-language dataset observations |
| Power BI | Full dashboard guide with DAX formulas |
| Documentation | Academic report, PPT content, viva Q&A |

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Data Analysis | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | scikit-learn |
| Model Saving | Joblib |
| Web Interface | Streamlit |
| BI Dashboard | Microsoft Power BI |
| Version Control | Git / GitHub |

---

## 📊 Dataset

| Property | Value |
|---|---|
| Type | Synthetic (generated for education) |
| Records | ~1,200 students |
| Columns | 19 (academic + lifestyle + demographic) |
| Missing Values | ~4% injected per selected column |
| Duplicates | ~12 rows injected |
| Outliers | A few injected extreme values |
| Generator | `src/generate_dataset.py` |

**Key columns:**
`student_id`, `study_hours`, `attendance_percentage`, `previous_semester_marks`,
`assignment_score`, `internal_exam_score`, `previous_cgpa`, `sleep_hours`,
`internet_usage_hours`, `extracurricular_activity`, `stress_level`,
`physical_activity_hours`, `age`, `gender`, `course`, `semester`,
`final_score` *(regression target)*, `performance_category` *(classification target)*,
`pass_fail` *(binary target)*

---

## 🔬 Methodology

```
Raw Data
   ↓
Data Cleaning  →  cleaned CSV
   ↓
EDA + Statistical Analysis  →  visualizations/
   ↓
Feature Engineering  →  encoded columns
   ↓
┌──────────────────────────────────┐
│  Regression (predict final_score) │
│  Classification (predict category)│
│  Clustering (K-Means segments)    │
└──────────────────────────────────┘
   ↓
Model Evaluation + Comparison
   ↓
Best Model → models/*.pkl
   ↓
Streamlit App  +  Power BI Dashboard
```

---

## 📁 Project Structure

```
student-performance-prediction/
│
├── data/
│   ├── raw/                      ← Raw synthetic dataset
│   └── processed/                ← Cleaned & clustered datasets
│
├── notebooks/                    ← Jupyter Notebooks (EDA, models)
│
├── src/
│   ├── generate_dataset.py       ← Create synthetic raw dataset
│   ├── data_cleaning.py          ← Full cleaning pipeline
│   ├── eda.py                    ← EDA + visualizations
│   ├── preprocessing.py          ← Shared feature/pipeline utilities
│   ├── train_regression.py       ← Train & evaluate regression models
│   ├── train_classification.py   ← Train & evaluate classification models
│   ├── clustering.py             ← K-Means segmentation
│   ├── predict.py                ← Standalone prediction utility
│   └── utils.py                  ← Shared helpers + AI insights
│
├── models/                       ← Saved .pkl model files
├── app/
│   └── app.py                    ← Streamlit web application
│
├── visualizations/               ← Generated PNG charts
├── powerbi/
│   └── powerbi_dashboard_guide.md
│
├── reports/
│   ├── project_report.md         ← Full academic report
│   ├── project_summary.md
│   ├── viva_questions.md
│   └── internship_mapping.md
│
├── presentation/
│   └── presentation_content.md   ← PPT slide content
│
├── requirements.txt
├── README.md
└── gitignore.txt                 ← Rename to .gitignore
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10 or newer
- pip (comes with Python)
- Optional: Microsoft Power BI Desktop

### 2. Clone / Download
```bash
git clone https://github.com/YOUR_USERNAME/student-performance-prediction.git
cd student-performance-prediction
```

### 3. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Generate Dataset
```bash
python src/generate_dataset.py
```

### 6. Clean Data
```bash
python src/data_cleaning.py
```

### 7. Run EDA
```bash
python src/eda.py
```

### 8. Train Regression Models
```bash
python src/train_regression.py
```

### 9. Train Classification Models
```bash
python src/train_classification.py
```

### 10. Run Clustering
```bash
python src/clustering.py
```

### 11. Launch Streamlit App
```bash
streamlit run app/app.py
```
Open: http://localhost:8501

---

## 📈 Results

> Results below are placeholders — run the code to obtain actual values.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** | **8.04** | **10.15** | **0.5398** |
| Decision Tree Regressor | 11.50 | 14.38 | 0.0769 |
| Random Forest Regressor | 8.99 | 11.13 | 0.4473 |

| Model | Accuracy | F1 Score |
|---|---|---|
| **Logistic Regression** | **0.7250** | **0.7135** |
| Decision Tree Classifier | 0.5458 | 0.5364 |
| Random Forest Classifier | 0.6333 | 0.5904 |

---

## ⚠️ Limitations

1. **Synthetic data**: All data is artificially generated; results may not generalise to real students.
2. **No causal inference**: Correlations observed do not imply causation.
3. **Static model**: Trained once; does not update dynamically as new data arrives.
4. **Simplified features**: Real academic performance depends on many more factors.
5. **Ethical use**: Model predictions should support, not replace, educator judgment.

---

## 🔭 Future Scope

- Collect real anonymised student data (with institutional ethics approval)
- Add deep learning models (LSTM for time-series grade trends)
- Integrate a live database (SQLite / PostgreSQL)
- Add REST API for LMS integration
- Implement explainability (SHAP values)
- Add automated email alerts for at-risk students

---

## 🎓 Internship Relevance

This project directly demonstrates the following IBM SkillsBuild
Data Analytics with AI curriculum topics:
Data Cleaning, Data Wrangling, EDA, Predictive Analytics,
Supervised Learning (Regression + Classification), Decision Trees,
Random Forest, Unsupervised Learning (K-Means), Data Visualisation,
AI Insights, and Business Decision Support.

---

## 👤 Author

**[Your Name]**
B.Tech Computer Science | [Your College Name]
IBM SkillsBuild Data Analytics with AI Internship 2026 (AICTE)

---

## 📄 License

This project is released under the MIT License for educational purposes.
The synthetic dataset is freely usable for learning and demonstration.
