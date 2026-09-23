# Power BI Dashboard Guide
## Student Performance Prediction & Analytics

> This guide provides step-by-step instructions to build a 4-page Power BI dashboard
> using the cleaned dataset (`student_performance_cleaned.csv`).

---

## STEP 1 — Import Data

1. Open **Power BI Desktop**.
2. Click **Home → Get Data → Text/CSV**.
3. Navigate to `data/processed/student_performance_cleaned.csv`.
4. Click **Load** (or **Transform Data** to preview first).
5. The table will appear in the **Fields** pane as `student_performance_cleaned`.

---

## STEP 2 — Create Calculated Columns

In **Data View**, right-click the table → **New Column**:

### Column 1: Risk Level
```DAX
Risk Level =
IF(
    'student_performance_cleaned'[final_score] >= 80, "Low Risk",
    IF(
        'student_performance_cleaned'[final_score] >= 65, "Medium Risk",
        IF(
            'student_performance_cleaned'[final_score] >= 50, "High Risk",
            "Very High Risk"
        )
    )
)
```

### Column 2: Study Category
```DAX
Study Category =
SWITCH(
    TRUE(),
    'student_performance_cleaned'[study_hours] < 2, "< 2 hrs",
    'student_performance_cleaned'[study_hours] < 4, "2–4 hrs",
    'student_performance_cleaned'[study_hours] < 6, "4–6 hrs",
    'student_performance_cleaned'[study_hours] < 8, "6–8 hrs",
    "8+ hrs"
)
```

### Column 3: Attendance Category
```DAX
Attendance Category =
SWITCH(
    TRUE(),
    'student_performance_cleaned'[attendance_percentage] < 50, "< 50%",
    'student_performance_cleaned'[attendance_percentage] < 65, "50–65%",
    'student_performance_cleaned'[attendance_percentage] < 75, "65–75%",
    'student_performance_cleaned'[attendance_percentage] < 85, "75–85%",
    "85–100%"
)
```

---

## STEP 3 — Create DAX Measures

In **Report View**, right-click the table → **New Measure**:

```DAX
Total Students = COUNTROWS('student_performance_cleaned')
```

```DAX
Avg Final Score = AVERAGE('student_performance_cleaned'[final_score])
```

```DAX
Avg Attendance = AVERAGE('student_performance_cleaned'[attendance_percentage])
```

```DAX
Avg Study Hours = AVERAGE('student_performance_cleaned'[study_hours])
```

```DAX
Pass Count =
CALCULATE(
    COUNTROWS('student_performance_cleaned'),
    'student_performance_cleaned'[pass_fail] = "Pass"
)
```

```DAX
Pass Percentage =
DIVIDE(
    [Pass Count],
    [Total Students],
    0
) * 100
```

```DAX
At Risk Students =
CALCULATE(
    COUNTROWS('student_performance_cleaned'),
    'student_performance_cleaned'[final_score] < 50
)
```

```DAX
Fail Count = [Total Students] - [Pass Count]
```

```DAX
High Risk Students =
CALCULATE(
    COUNTROWS('student_performance_cleaned'),
    'student_performance_cleaned'[Risk Level] = "Very High Risk"
        || 'student_performance_cleaned'[Risk Level] = "High Risk"
)
```

```DAX
Avg CGPA = AVERAGE('student_performance_cleaned'[previous_cgpa])
```

---

## STEP 4 — Page 1: Executive Overview

**Page name:** Executive Overview

### KPI Cards (top row)
Add 6 **Card** visuals:
| Card | Measure / Field |
|---|---|
| Total Students | `Total Students` |
| Average Final Score | `Avg Final Score` |
| Average Attendance | `Avg Attendance` |
| Pass Percentage | `Pass Percentage` |
| At-Risk Students | `At Risk Students` |
| Average CGPA | `Avg CGPA` |

### Chart 1 — Performance Distribution (Bar Chart)
- X-axis: `performance_category`
- Y-axis: Count of rows  (use `COUNTROWS`)
- Sort by: Performance category order (Low → Excellent)
- Title: "Student Performance Distribution"

### Chart 2 — Average Score by Course (Horizontal Bar)
- Y-axis: `course`
- X-axis: `Avg Final Score`
- Title: "Avg Final Score by Course"

### Chart 3 — Average Score by Semester (Line Chart)
- X-axis: `semester`
- Y-axis: `Avg Final Score`
- Title: "Avg Final Score by Semester"

### Chart 4 — Pass / Fail Pie Chart
- Values: `Pass Count`, `Fail Count`
- Legend: Pass / Fail
- Title: "Pass / Fail Distribution"

### Slicer — Course
- Field: `course`
- Style: Dropdown

### Slicer — Gender
- Field: `gender`
- Style: Tile

---

## STEP 5 — Page 2: Academic Analysis

**Page name:** Academic Analysis

### Scatter Plot 1 — Study Hours vs Final Score
- X-axis: `study_hours`
- Y-axis: `final_score`
- Details: `student_id` (hover tooltip)
- Title: "Study Hours vs Final Score"
- Add trend line: Analytics pane → Trend Line

### Scatter Plot 2 — Attendance vs Final Score
- X-axis: `attendance_percentage`
- Y-axis: `final_score`
- Title: "Attendance % vs Final Score"
- Add trend line

### Scatter Plot 3 — Previous Marks vs Final Score
- X-axis: `previous_semester_marks`
- Y-axis: `final_score`
- Title: "Previous Marks vs Final Score"

### Bar Chart — Avg Final Score by Study Category
- X-axis: `Study Category`
- Y-axis: `Avg Final Score`
- Sort: Custom order (< 2 hrs → 8+ hrs)

### Bar Chart — Avg Final Score by Attendance Category
- X-axis: `Attendance Category`
- Y-axis: `Avg Final Score`

### Slicer — Semester
- Field: `semester`
- Style: Between (range slider)

---

## STEP 6 — Page 3: Lifestyle Analysis

**Page name:** Lifestyle Analysis

### Box Plot (using Violin approximation with bar + error bars)
- Group: `stress_level`
- Value: `final_score`
- Title: "Stress Level vs Final Score"

> Power BI does not have a native box plot — use a **Clustered Bar Chart**
> with `stress_level` on X-axis and `Avg Final Score` on Y-axis as an alternative.
> Or download the **Box and Whisker chart** from AppSource.

### Column Chart — Avg Score by Extracurricular Activity
```DAX
Avg Score Extracurricular Yes =
CALCULATE(
    AVERAGE('student_performance_cleaned'[final_score]),
    'student_performance_cleaned'[extracurricular_activity] = "Yes"
)

Avg Score Extracurricular No =
CALCULATE(
    AVERAGE('student_performance_cleaned'[final_score]),
    'student_performance_cleaned'[extracurricular_activity] = "No"
)
```
- Use a **Clustered Column Chart** with these two measures.

### Scatter Plot — Sleep Hours vs Final Score
- X-axis: `sleep_hours`
- Y-axis: `final_score`
- Title: "Sleep Hours vs Final Score"
- Add trend line

### Scatter Plot — Internet Usage vs Final Score
- X-axis: `internet_usage_hours`
- Y-axis: `final_score`
- Title: "Internet Usage vs Final Score"

### Slicers
- Stress Level
- Extracurricular Activity

---

## STEP 7 — Page 4: Risk Analysis

**Page name:** Risk & Prediction Analysis

### KPI Cards
| Card | Value |
|---|---|
| High Risk Students | `High Risk Students` |
| At-Risk Students | `At Risk Students` |
| Pass Percentage | `Pass Percentage` |

### Donut Chart — Risk Level Distribution
- Values: Count of rows
- Legend: `Risk Level`
- Title: "Student Risk Level Distribution"
- Colors:
  - Low Risk → Green (#22c55e)
  - Medium Risk → Orange (#f97316)
  - High Risk → Red (#ef4444)
  - Very High Risk → Dark Red (#9b1c1c)

### Bar Chart — Avg Final Score by Risk Level
- X-axis: `Risk Level`
- Y-axis: `Avg Final Score`
- Sort: Very High Risk → Low Risk

### Table — At-Risk Students
Add a **Table** visual filtered to `final_score < 50`:
- Columns: `student_id`, `course`, `semester`, `study_hours`,
  `attendance_percentage`, `final_score`, `Risk Level`
- Filter: `final_score < 50`

### Conditional Formatting
In the table, apply conditional formatting to `final_score`:
- Background color scale: Red (0) → Yellow (50) → Green (100)

### Slicer — Risk Level
- Field: `Risk Level`

### Slicer — Course
- Field: `course`

---

## STEP 8 — Dashboard Formatting Tips

1. **Theme**: Use "Executive" or "City Park" built-in theme, or set:
   - Background: #f7f8fa (light grey)
   - Text: #1f2328 (dark)
   - Accent: #3b82f6 (blue)

2. **Page navigation**: Add Button → Page Navigator to move between pages.

3. **Header text box**: On each page, add a text box at the top:
   - Page 1: "Student Performance — Executive Overview"
   - Font: Segoe UI, 18pt, bold

4. **Logo**: Insert → Image → add your college or IBM SkillsBuild logo.

5. **Tooltips**: Enable tooltips on all charts to show `student_id` + `final_score`.

6. **Cross-filtering**: Keep default cross-filtering ON — clicking a bar in one
   chart automatically filters other charts on the same page.

7. **Mobile layout**: View → Mobile Layout — arrange key KPI cards for mobile.

---

## STEP 9 — Publish (Optional)

1. Save the `.pbix` file as `powerbi/student_performance_dashboard.pbix`.
2. If you have a Power BI Pro account: **Home → Publish → My Workspace**.
3. Share the link with your internship mentor.

---

## Summary of DAX Formulas

| Measure | Formula |
|---|---|
| Total Students | `COUNTROWS(table)` |
| Avg Final Score | `AVERAGE(final_score)` |
| Avg Attendance | `AVERAGE(attendance_percentage)` |
| Pass Percentage | `DIVIDE(CALCULATE(COUNTROWS,pass_fail="Pass"), COUNTROWS, 0) * 100` |
| At Risk Students | `CALCULATE(COUNTROWS, final_score < 50)` |
| High Risk Students | `CALCULATE(COUNTROWS, Risk Level IN {"High Risk","Very High Risk"})` |
| Avg CGPA | `AVERAGE(previous_cgpa)` |

---

*Power BI Desktop is free to download from: https://powerbi.microsoft.com/desktop*
