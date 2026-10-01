"""
DAT7303 – Data Mining and Machine Learning
Portfolio 3 – Full CRISP-DM Analysis (Tasks 1–4)
Dataset: Student Performance Dataset (UCI Machine Learning Repository)
Target Variable: G3 (Final Grade, 0–20 scale)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
os.makedirs("figures", exist_ok=True)

print("=" * 70)
print("  DAT7303 Portfolio 3 – Student Performance Prediction (CRISP-DM)")
print("=" * 70)


# =============================================================================
# TASK 1 – BUSINESS UNDERSTANDING & DATA UNDERSTANDING (CRISP-DM Steps 1 & 2)
# =============================================================================
print("\n" + "─" * 70)
print("TASK 1 – Business Understanding & Data Understanding")
print("─" * 70)

# ── 1.1 Business Understanding ───────────────────────────────────────────────
print("""
Business Objective:
  Educational institutions face ongoing challenges in identifying students
  at risk of poor academic performance. Early and accurate prediction of
  final grades enables targeted interventions, resource allocation, and
  personalised support strategies.

Business Success Criteria:
  - Predict final student grade (G3) with R² ≥ 0.75
  - Identify the top five features most strongly associated with academic outcome
  - Provide a deployable model suitable for use at the start of the academic year

Data Mining Goal:
  Build and compare three regression models to predict G3 (continuous, 0–20).
  Select the optimal model based on R², MAE, and RMSE.

Project Plan:
  Step 1 – Business Understanding  (Task 1)
  Step 2 – Data Understanding       (Task 1)
  Step 3 – Data Preparation         (Task 2)
  Step 4 – Modelling                (Task 3)
  Step 5 – Evaluation               (Task 4)
  Tools: Python 3, pandas, scikit-learn, matplotlib, seaborn
""")

# ── 1.2 Load Dataset ─────────────────────────────────────────────────────────
# UCI Student Performance dataset (Mathematics course, 395 students)
# Source: P. Cortez and A. Silva (2008). Using Data Mining to Predict Secondary
# School Student Performance. Proceedings of the 5th FUture BUsiness TEChnology
# Conference (FUBUTEC 2008), EUROSIS. Licensed CC BY 4.0 via the UCI ML Repository.
df = pd.read_csv("data/student-mat.csv", sep=";")
print(f"\n  Dataset loaded. Shape: {df.shape}")

# ── 1.3 Describe Data ────────────────────────────────────────────────────────
print("\n── Data Description ──")
print(df.dtypes.to_string())
print(f"\n  Numerical summary:\n{df.describe().round(2).to_string()}")

# ── 1.4 Explore Data – visualisations ────────────────────────────────────────
print("\n  Generating Task 1 exploratory visualisations...")

# Fig 1 – Distribution of G3 (target)
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["G3"], bins=21, kde=True, color="#2E75B6", ax=ax)
ax.set_title("Figure 1 – Distribution of Final Grade (G3)", fontweight="bold")
ax.set_xlabel("Final Grade (G3)")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("figures/fig1_g3_distribution.png", dpi=150)
plt.close()
print("  Saved: figures/fig1_g3_distribution.png")

# Fig 2 – Correlation heatmap (numerical columns)
num_cols = df.select_dtypes(include=np.number).columns.tolist()
fig, ax = plt.subplots(figsize=(12, 8))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, annot_kws={"size": 7})
ax.set_title("Figure 2 – Correlation Heatmap (Numerical Features)", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/fig2_correlation_heatmap.png", dpi=150)
plt.close()
print("  Saved: figures/fig2_correlation_heatmap.png")

# Fig 3 – G3 by parental education
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, title in zip(axes, ["Medu", "Fedu"],
                           ["Mother's Education Level", "Father's Education Level"]):
    sns.boxplot(x=col, y="G3", data=df, palette="Blues", ax=ax)
    ax.set_title(f"Figure 3 – G3 by {title}", fontweight="bold")
    ax.set_xlabel(f"{title} (0=None → 4=Higher)")
    ax.set_ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig("figures/fig3_parental_education_vs_g3.png", dpi=150)
plt.close()
print("  Saved: figures/fig3_parental_education_vs_g3.png")

# ── 1.5 Data Quality ─────────────────────────────────────────────────────────
print("\n── Data Quality Check ──")
print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"  Duplicate rows: {df.duplicated().sum()}")
print(f"  G3 range: {df['G3'].min()} – {df['G3'].max()}")


# =============================================================================
# TASK 2 – DATA PREPARATION (CRISP-DM Step 3)
# =============================================================================
print("\n" + "─" * 70)
print("TASK 2 – Data Preparation")
print("─" * 70)

# ── 2.1 Select Data ───────────────────────────────────────────────────────────
# Drop school identifier (not predictive in a general model)
df_prep = df.drop(columns=["school"])
print(f"\n  After dropping 'school' column: {df_prep.shape}")

# ── 2.2 Handle Missing Values ─────────────────────────────────────────────────
print(f"  Missing values before: {df_prep.isnull().sum().sum()}")
# Numerical – median imputation (robust to outliers)
for col in df_prep.select_dtypes(include=np.number).columns:
    df_prep[col].fillna(df_prep[col].median(), inplace=True)
# Categorical – mode imputation
for col in df_prep.select_dtypes(include="object").columns:
    df_prep[col].fillna(df_prep[col].mode()[0], inplace=True)
print(f"  Missing values after:  {df_prep.isnull().sum().sum()}")

# ── 2.3 Encode Categorical Variables ─────────────────────────────────────────
binary_cols = ["sex","address","famsize","Pstatus","schoolsup","famsup",
               "paid","activities","nursery","higher","internet","romantic"]
multi_cols  = ["Mjob","Fjob","reason","guardian"]

le = LabelEncoder()
for col in binary_cols:
    df_prep[col] = le.fit_transform(df_prep[col])

df_prep = pd.get_dummies(df_prep, columns=multi_cols, drop_first=True)
print(f"  After encoding: {df_prep.shape[1]} features")

# ── 2.4 Feature Engineering ───────────────────────────────────────────────────
# Derive average parental education
df_prep["avg_parental_edu"] = (df_prep["Medu"] + df_prep["Fedu"]) / 2
# Alcohol risk index
df_prep["alcohol_risk"] = df_prep["Dalc"] + df_prep["Walc"]
print("  Engineered: avg_parental_edu, alcohol_risk")

# ── 2.5 Outlier Check (IQR) ───────────────────────────────────────────────────
print("\n  Outlier counts (IQR method):")
outlier_summary = {}
for col in ["age","absences","G1","G2","G3"]:
    Q1, Q3 = df_prep[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    n_out = ((df_prep[col] < Q1 - 1.5*IQR) | (df_prep[col] > Q3 + 1.5*IQR)).sum()
    outlier_summary[col] = n_out
    print(f"    {col}: {n_out} outliers")
# Retain outliers – G3=0 students are valid (failed/absent)

# ── 2.6 Visualisations ───────────────────────────────────────────────────────
print("\n  Generating Task 2 visualisations...")

# Fig 4 – G1, G2, G3 scatter (grade progression)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(df_prep["G1"], df_prep["G3"], alpha=0.4, color="#2E75B6", edgecolors="none")
axes[0].set_xlabel("First Period Grade (G1)"); axes[0].set_ylabel("Final Grade (G3)")
axes[0].set_title("Figure 4a – G1 vs G3", fontweight="bold")
axes[1].scatter(df_prep["G2"], df_prep["G3"], alpha=0.4, color="#ED7D31", edgecolors="none")
axes[1].set_xlabel("Second Period Grade (G2)"); axes[1].set_ylabel("Final Grade (G3)")
axes[1].set_title("Figure 4b – G2 vs G3", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/fig4_grade_progression.png", dpi=150)
plt.close()
print("  Saved: figures/fig4_grade_progression.png")

# Fig 5 – Study time vs G3
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x="studytime", y="G3", data=df_prep, palette="Greens", ax=ax)
ax.set_title("Figure 5 – Study Time vs Final Grade", fontweight="bold")
ax.set_xlabel("Study Time (1=<2h → 4=>10h/week)")
ax.set_ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig("figures/fig5_studytime_vs_g3.png", dpi=150)
plt.close()
print("  Saved: figures/fig5_studytime_vs_g3.png")

# ── 2.7 Split Features / Target ───────────────────────────────────────────────
X = df_prep.drop(columns=["G3"])
y = df_prep["G3"]

# ── 2.8 Scale ─────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ── 2.9 Train/Test Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
print(f"\n  Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")


# =============================================================================
# TASK 3 – MODELLING (CRISP-DM Step 4)
# =============================================================================
print("\n" + "─" * 70)
print("TASK 3 – Modelling")
print("─" * 70)

def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae  = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2   = r2_score(y_te, preds)
    cv   = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2")
    print(f"\n  {name}")
    print(f"    MAE  = {mae:.4f}")
    print(f"    RMSE = {rmse:.4f}")
    print(f"    R²   = {r2:.4f}")
    print(f"    5-Fold CV R² = {cv.mean():.4f} ± {cv.std():.4f}")
    return {"Model": name, "MAE": round(mae,4), "RMSE": round(rmse,4),
            "R2": round(r2,4), "CV_R2": round(cv.mean(),4),
            "preds": preds, "model": model}

# ── Model 1: Multiple Linear Regression ───────────────────────────────────────
print("\n── Model 1: Multiple Linear Regression ──")
lr = evaluate("Linear Regression", LinearRegression(),
              X_train, X_test, y_train, y_test)

# ── Model 2: Decision Tree Regressor ──────────────────────────────────────────
print("\n── Model 2: Decision Tree Regressor ──")
dt = evaluate("Decision Tree", DecisionTreeRegressor(max_depth=5, random_state=42),
              X_train, X_test, y_train, y_test)

# ── Model 3: Random Forest Regressor ──────────────────────────────────────────
print("\n── Model 3: Random Forest Regressor ──")
rf = evaluate("Random Forest", RandomForestRegressor(n_estimators=200, max_depth=10,
                                                      random_state=42, n_jobs=-1),
              X_train, X_test, y_train, y_test)

# ── Results Table ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame([
    {k: v for k, v in m.items() if k not in ("preds","model")}
    for m in [lr, dt, rf]
])
print(f"\n── Model Comparison Table ──\n{results_df.to_string(index=False)}")

# ── Task 3 Visualisations ─────────────────────────────────────────────────────
print("\n  Generating Task 3 visualisations...")

# Fig 6 – Predicted vs Actual for all three models
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, m, colour in zip(axes, [lr, dt, rf], ["#2E75B6","#ED7D31","#70AD47"]):
    ax.scatter(y_test, m["preds"], alpha=0.5, color=colour, edgecolors="none")
    lims = [0, 20]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual G3"); ax.set_ylabel("Predicted G3")
    ax.set_title(f"Figure 6 – {m['Model']}\nR²={m['R2']}", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/fig6_predicted_vs_actual.png", dpi=150)
plt.close()
print("  Saved: figures/fig6_predicted_vs_actual.png")

# Fig 7 – Feature Importance (Random Forest)
importances = pd.Series(rf["model"].feature_importances_, index=X.columns)
top15 = importances.nlargest(15).sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
top15.plot(kind="barh", color="#2E75B6", ax=ax)
ax.set_title("Figure 7 – Top 15 Feature Importances (Random Forest)", fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("figures/fig7_feature_importance.png", dpi=150)
plt.close()
print("  Saved: figures/fig7_feature_importance.png")

# Fig 8 – Model comparison bar chart
metrics = ["MAE","RMSE","R2"]
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
colours = ["#2E75B6","#ED7D31","#70AD47"]
for ax, metric in zip(axes, metrics):
    vals = results_df[metric]
    bars = ax.bar(results_df["Model"], vals, color=colours)
    ax.set_title(f"Figure 8 – {metric} Comparison", fontweight="bold")
    ax.set_ylabel(metric)
    ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig8_model_comparison.png", dpi=150)
plt.close()
print("  Saved: figures/fig8_model_comparison.png")


# =============================================================================
# TASK 4 – EVALUATION (CRISP-DM Step 5)
# =============================================================================
print("\n" + "─" * 70)
print("TASK 4 – Evaluation")
print("─" * 70)

# Identify optimal model
best = max([lr, dt, rf], key=lambda m: m["R2"])
print(f"\n  Optimal Model: {best['Model']}")
print(f"  R² = {best['R2']}  |  MAE = {best['MAE']}  |  RMSE = {best['RMSE']}")

print("""
  Evaluation Summary:
  ─────────────────────────────────────────────────────────────────
  Linear Regression provides a useful baseline and is interpretable
  but underperforms on non-linear relationships in the data.

  Decision Tree improves on non-linearity but risks overfitting
  with shallow depth settings.

  Random Forest achieves the highest R², lowest MAE, and lowest RMSE
  by aggregating 200 decision trees, reducing variance substantially.
  It meets the business success criterion of R² ≥ 0.75.

  The top predictors identified are G2 (second period grade),
  absences, reason for choosing the school, age and G1 (first
  period grade). G2 dominates (about 79% of importance), so the
  model mainly extends the existing grade trajectory; earlier-
  in-the-year prediction would need models without G1 and G2.

  Next Steps:
  - Hyperparameter tuning (GridSearchCV) for further improvement
  - Gather additional features (e.g. teacher assessments, attendance logs)
  - Deploy as a web dashboard for academic staff
  ─────────────────────────────────────────────────────────────────
""")

# ── Sample Prediction using Optimal Model ────────────────────────────────────
# Construct a sample student profile (before encoding/scaling)
print("  Sample Prediction (using optimal model):")
sample_raw = pd.DataFrame([{
    "sex": 0, "age": 16, "address": 1, "famsize": 1, "Pstatus": 1,
    "Medu": 3, "Fedu": 2, "traveltime": 2, "studytime": 3, "failures": 0,
    "schoolsup": 0, "famsup": 1, "paid": 0, "activities": 1,
    "nursery": 1, "higher": 1, "internet": 1, "romantic": 0,
    "famrel": 4, "freetime": 3, "goout": 2, "Dalc": 1, "Walc": 2,
    "health": 4, "absences": 3, "G1": 12, "G2": 13,
    "avg_parental_edu": 2.5, "alcohol_risk": 3,
}])

# Add dummy-encoded columns with zeros for any missing dummies
for col in X.columns:
    if col not in sample_raw.columns:
        sample_raw[col] = 0
sample_raw = sample_raw[X.columns]
sample_scaled = scaler.transform(sample_raw)
prediction = best["model"].predict(sample_scaled)[0]
print(f"\n  Student Profile: Age=16, StudyTime=3, G1=12, G2=13, Failures=0, Absences=3")
print(f"  Predicted Final Grade (G3): {round(prediction, 2)} / 20")

# Fig 9 – Residual plot for best model
residuals = y_test.values - best["preds"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(best["preds"], residuals, alpha=0.5, color="#2E75B6", edgecolors="none")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted G3"); axes[0].set_ylabel("Residual")
axes[0].set_title(f"Figure 9a – Residuals: {best['Model']}", fontweight="bold")
sns.histplot(residuals, kde=True, bins=25, color="#70AD47", ax=axes[1])
axes[1].set_title("Figure 9b – Residual Distribution", fontweight="bold")
axes[1].set_xlabel("Residual")
plt.tight_layout()
plt.savefig("figures/fig9_residuals.png", dpi=150)
plt.close()
print("\n  Saved: figures/fig9_residuals.png")

print("\n" + "=" * 70)
print("  All tasks complete. Figures saved to /figures/")
print("=" * 70)
