import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LOAD REFINED DATA

df = pd.read_csv('Final_Cleaned_Refined_Unemployment_India.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

print("      EVALUATING MODEL PERFORMANCE: SIMPLE VS BOOSTING\n  ")

# MODEL 1: BASELINE MODEL (SIMPLE INDEPENDENT TREES - NO MEMORY)

print("[MODEL 1] Running Baseline Random Forest Regressor...")

# Features available natively in the standard dataset rows
base_features = ['Region', 'Area', 'Estimated Labour Participation Rate (%)', 'Year', 'Month_Name', 'Is_Lockdown_Period']
X_base = df[base_features].copy()
y_base = df['Estimated Unemployment Rate (%)']

# One-Hot Encode categorical text variables so the machine can read them
X_base = pd.get_dummies(X_base, columns=['Region', 'Area', 'Month_Name'], drop_first=True)

# 80/20 Train-Test Exam Split
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_base, y_base, test_size=0.2, random_state=42)

# Training independent parallel trees
base_model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
base_model.fit(X_train_b, y_train_b)

# Predictions & Final Exam Scoring for Model 1
y_pred_b = base_model.predict(X_test_b)
mae_b = mean_absolute_error(y_test_b, y_pred_b)
r2_b = r2_score(y_test_b, y_pred_b)


# MODEL 2: UPGRADED GRADIENT BOOSTING MODEL (WITH SEQUENTIAL MEMORY)
print("[MODEL 2] Engineering Lag Features & Running Sequential Gradient Boosting...")

# Group by State & Sector, then shift rows to teach the machine what happened last month
df_sorted = df.sort_values(by=['Region', 'Area', 'Date']).copy()
df_sorted['Prev_Month_Unemployment'] = df_sorted.groupby(['Region', 'Area'])['Estimated Unemployment Rate (%)'].shift(1)

# Drop rows belonging to the very first month because they don't have a past to look back on
df_lagged = df_sorted.dropna(subset=['Prev_Month_Unemployment']).copy()

# Add the new "Memory Column" into our list of clues
upgraded_features = ['Region', 'Area', 'Estimated Labour Participation Rate (%)', 'Year', 'Month_Name', 'Is_Lockdown_Period', 'Prev_Month_Unemployment']
X_up = df_lagged[upgraded_features].copy()
y_up = df_lagged['Estimated Unemployment Rate (%)']

# One-Hot Encode categorical strings
X_up = pd.get_dummies(X_up, columns=['Region', 'Area', 'Month_Name'], drop_first=True)

# 80/20 Train-Test Exam Split (Aligned with the new lagged dimensions)
X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_up, y_up, test_size=0.2, random_state=42)

# Training sequential error-correcting trees (Gradient Boosting)
boosting_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
boosting_model.fit(X_train_u, y_train_u)

# Predictions & Final Exam Scoring for Model 2
y_pred_u = boosting_model.predict(X_test_u)
mae_u = mean_absolute_error(y_test_u, y_pred_u)
r2_u = r2_score(y_test_u, y_pred_u)

# HEAD-TO-HEAD SUMMARY REPORT

print("\n             FINAL PERFORMANCE COMPARISON             ")
print(f"MODEL 1: Baseline Random Forest (No Memory Context)")
print(f" -> Mean Absolute Error (MAE) : {mae_b:.2f}%")
print(f" -> R-squared (R² Score)      : {r2_b * 100:.2f}%")

print(f"MODEL 2: Gradient Boosting Regressor (With 1-Month Lag Memory)")
print(f" -> Mean Absolute Error (MAE) : {mae_u:.2f}%  (Lower is Better)")
print(f" -> R-squared (R² Score)      : {r2_u * 100:.2f}% (Higher is Better)")
