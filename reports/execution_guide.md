# STEP-BY-STEP EXECUTION GUIDE
## Student Performance Prediction & Analytics
### For Beginners — Windows (PowerShell)

---

## Prerequisites Check

Before starting, verify:
- [ ] You have Python 3.10 or newer installed
- [ ] You have internet access (for pip install)
- [ ] You have Microsoft Power BI Desktop installed (optional)

---

## STEP 1 — Verify Python Installation

Open **PowerShell** (search "PowerShell" in Start menu) and run:
```powershell
python3 --version
```
Expected output: `Python 3.11.x` (or 3.10/3.12)

If Python is not installed:
1. Go to https://python.org/downloads
2. Download Python 3.11 or newer
3. During installation, check **"Add Python to PATH"**
4. Restart PowerShell after installation

---

## STEP 2 — Navigate to the Project Folder

```powershell
# Open PowerShell and navigate to where you saved the project:
cd "E:\StudentPredectionIBM\student-performance-prediction"

# Confirm you are in the right folder:
Get-ChildItem
# You should see: src/, data/, app/, reports/, etc.
```

---

## STEP 3 — Create a Virtual Environment (Recommended)

```powershell
# Create virtual environment named 'venv'
python3 -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# If you get an error about execution policy, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try activating again
.\venv\Scripts\Activate.ps1

# Your prompt should now show (venv) at the beginning
```

---

## STEP 4 — Install Dependencies

```powershell
# Make sure venv is active (you should see (venv) in your prompt)
# Install all required packages:
pip install -r requirements.txt

# This installs: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, streamlit, plotly
# Wait for all packages to finish downloading (may take 2-5 minutes)
```

**Verify installation:**
```powershell
python3 -c "import pandas, numpy, sklearn, matplotlib, seaborn, joblib, streamlit; print('All packages installed successfully!')"
```
Expected output: `All packages installed successfully!`

**Common error:** `ModuleNotFoundError: No module named 'xxx'`
Fix: `pip install xxx`

---

## STEP 5 — Generate the Dataset

```powershell
python3 src/generate_dataset.py
```

**Expected output:**
```
Raw dataset generated: data/raw/student_performance_raw.csv
Rows (incl. duplicates): 1212
Columns               : 19
...
```

**What this does:** Creates a synthetic dataset of ~1,212 student records with
realistic academic and lifestyle features, including intentional missing values,
duplicates, and outliers for the cleaning demonstration.

**Output file:** `data/raw/student_performance_raw.csv`

---

## STEP 6 — Clean the Data

```powershell
python3 src/data_cleaning.py
```

**Expected output:**
```
[1] Raw dataset loaded  →  1212 rows x 19 columns
[3] Removed 12 duplicate rows → 1200 rows remaining
[4] Replaced X invalid age values with NaN
[5] Imputing missing numerical values with column medians ...
...
[10] Cleaned dataset saved → data/processed/student_performance_cleaned.csv
```

**What this does:**
- Removes 12 duplicate rows
- Fixes invalid values
- Imputes missing values (median for numbers, mode for categories)
- Caps outliers using IQR method
- Encodes categorical columns
- Saves cleaned data

**Output file:** `data/processed/student_performance_cleaned.csv`

---

## STEP 7 — Run Exploratory Data Analysis

```powershell
python3 src/eda.py
```

**Expected output:**
```
[STATISTICAL SUMMARY - Key Numeric Columns]
...
[UNIVARIATE ANALYSIS]
[SAVED] → visualizations/01_univariate_distributions.png
...
EDA complete - 8 visualizations saved in: visualizations/
```

**What this does:** Creates 8 analysis charts saved as PNG files:
1. Univariate distributions
2. Performance category distribution
3. Bivariate scatter plots (features vs final score)
4. Categorical boxplots
5. Correlation heatmap
6. Course and semester analysis
7. Study hours buckets
8. Attendance buckets

**View charts:** Open the `visualizations/` folder to see the PNG files.

---

## STEP 8 — Train Regression Models

```powershell
python3 src/train_regression.py
```

**Expected output:**
```
REGRESSION - PREDICT FINAL SCORE
Dataset : 1200 samples, 15 features
Train size : 960  |  Test size : 240

  Linear Regression
    MAE  : X.XXXX
    RMSE : X.XXXX
    R2   : X.XXXX

  Decision Tree
    ...

  Random Forest
    ...

MODEL COMPARISON TABLE
...
Best model: XXX (highest R2: X.XXXX)
Saved → models/regression_model.pkl
```

**What this does:**
- Trains 3 regression models to predict final_score
- Evaluates each model (MAE, RMSE, R²)
- Selects and saves the best model

**Output files:**
- `models/regression_model.pkl` (best model)
- `models/regression_all_models.pkl` (all 3 models)
- `visualizations/09_regression_actual_vs_predicted.png`
- `visualizations/10_regression_residuals.png`
- `visualizations/11_regression_model_comparison.png`
- `visualizations/12_regression_feature_importance.png`

---

## STEP 9 — Train Classification Models

```powershell
python3 src/train_classification.py
```

**Expected output:**
```
CLASSIFICATION - PREDICT PERFORMANCE CATEGORY
...
MODEL COMPARISON TABLE
...
Best model: XXX (highest F1: X.XXXX)
Saved → models/classification_model.pkl
```

**Output files:**
- `models/classification_model.pkl` (best model)
- `visualizations/13_classification_confusion_matrices.png`
- `visualizations/14_classification_model_comparison.png`
- `visualizations/15_classification_feature_importance.png`
- `visualizations/16_risk_level_distribution.png`

---

## STEP 10 — Run K-Means Clustering

```powershell
python3 src/clustering.py
```

**Expected output:**
```
K-MEANS STUDENT SEGMENTATION
...
Cluster labels assigned:
  Cluster 0 → High Achievers          (n=XXX, avg final_score=XX.X)
  Cluster 1 → Above-Average Students  (n=XXX, avg final_score=XX.X)
  ...
KMeans model saved → models/kmeans_model.pkl
```

**Output files:**
- `models/kmeans_model.pkl`
- `data/processed/student_performance_clustered.csv`
- `visualizations/17_clustering_elbow.png`
- `visualizations/18_clustering_silhouette.png`
- `visualizations/19_clustering_pca.png`
- `visualizations/20_clustering_boxplot.png`

---

## STEP 11 — Test the Prediction Utility

```powershell
python3 src/predict.py
```

**Expected output:**
```
--- Sample Student Prediction ---
  study_hours              : 6.5
  ...

--- Prediction Results ---
  Predicted Score      : XX.X
  Performance Category : Good/Average/etc.
  Risk Level           : Medium Risk
  Explanation          : Predicted score: XX.X/100 ...
```

---

## STEP 12 — Launch the Streamlit Application

```powershell
streamlit run app/app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Then:** Open your browser and go to: **http://localhost:8501**

You will see:
- **Predict tab**: Enter student details → click Predict → see score + category + risk
- **EDA Overview tab**: Dataset statistics and pre-generated charts
- **AI Insights tab**: Automated observations from the dataset
- **About tab**: Project info and ethics disclaimer

**To stop Streamlit:** Press `Ctrl + C` in PowerShell.

**Common error:** `ModuleNotFoundError: No module named 'streamlit'`
Fix: `pip install streamlit`

---

## STEP 13 — Import Cleaned Data into Power BI

1. Download and install **Power BI Desktop** (free):
   https://powerbi.microsoft.com/desktop

2. Open Power BI Desktop.

3. Click **Home → Get Data → Text/CSV**

4. Navigate to:
   `data/processed/student_performance_cleaned.csv`

5. Click **Load**.

6. Follow the instructions in:
   `powerbi/powerbi_dashboard_guide.md`

---

## STEP 14 — Build the Power BI Dashboard

Follow the step-by-step guide in `powerbi/powerbi_dashboard_guide.md`:
- Create 4 DAX measures
- Build Page 1: Executive Overview (KPI cards + charts)
- Build Page 2: Academic Analysis (scatter plots)
- Build Page 3: Lifestyle Analysis (boxplots)
- Build Page 4: Risk Analysis (donut chart + table)

Save the file as: `powerbi/student_performance_dashboard.pbix`

---

## STEP 15 — Prepare the Final Report

1. Open `reports/project_report.md` in any text editor or VS Code.
2. Replace all `[RUN_CODE]` placeholders with actual values from your executed scripts.
3. Fill in your name, roll number, college name, and date.
4. Export to PDF using VS Code's "Markdown PDF" extension (or copy into Word).

---

## STEP 16 — Prepare the Presentation

1. Open `presentation/presentation_content.md`.
2. Create a new PowerPoint presentation.
3. Copy the slide content — one slide per section.
4. Insert actual chart images from `visualizations/`:
   - Slide 6: `05_correlation_heatmap.png`
   - Slide 8: `09_regression_actual_vs_predicted.png`
   - Slide 9: `13_classification_confusion_matrices.png`
   - Slide 11: `12_regression_feature_importance.png`
5. Replace `[RUN_CODE]` with actual metric values.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError` | `pip install <module_name>` |
| `FileNotFoundError: ...cleaned.csv` | Run `data_cleaning.py` first |
| `FileNotFoundError: ...model.pkl` | Run training scripts first |
| Streamlit not found | `pip install streamlit` |
| PowerShell execution policy error | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Encoding error (emoji) | The scripts use ASCII — no issue expected |
| Port 8501 in use | `streamlit run app/app.py --server.port 8502` |

---

## Full Run Order (Quick Reference)

```powershell
# 1. Activate venv (if created)
.\venv\Scripts\Activate.ps1

# 2. Generate raw data
python3 src/generate_dataset.py

# 3. Clean data
python3 src/data_cleaning.py

# 4. EDA and visualizations
python3 src/eda.py

# 5. Train regression models
python3 src/train_regression.py

# 6. Train classification models
python3 src/train_classification.py

# 7. K-Means clustering
python3 src/clustering.py

# 8. Test predictions
python3 src/predict.py

# 9. Launch web app
streamlit run app/app.py
```

Total estimated time: 5–10 minutes (excluding Power BI).
