# -*- coding: utf-8 -*-
"""
Created on Fri Oct  31 19:56:20 2025

@author: Piotr Slomka, Jianhang Zhou
"""

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


# Assume df is available

# Clip LVEF above 75% and below 10%
# Soft thresholding with 0.5
df["LVEF"] = np.where(df["LVEF"] > 75.0, 75.0 + ((df["LVEF"] - 75.0) * 0.5), df["LVEF"])
df["LVEF"] = np.where(df["LVEF"] < 10.0, 10.0 - ((10.0 - df["LVEF"]) * 0.5), df["LVEF"])

# List sites
if 'Site' in df.columns:
    ls_sites = df['Site'].unique()

# Create X and y
X = df.drop(columns=["LVEF"])
if 'Site' in X.columns:
    X = X.drop(columns=['Site'])

for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])

y = df["LVEF"]

# Hyperparameter grid
param_grid = {
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.02],
    "n_estimators": [1500, 1000, 500],
    "subsample": [0.9],
    "colsample_bytree": [0.9]
}

# Store results
results = []

# Leave-one-site-out validation
for site in ls_sites:

    # Define train/test split
    is_test = df["Site"] == site

    X_train, X_val = X[~is_test], X[is_test]
    y_train, y_val = y[~is_test], y[is_test]

    model = XGBRegressor(objective="reg:squarederror", random_state=42)
    grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_val)

    # Non-linear calibration using actual value as reference
    calib_model = LinearRegression()
    calib_model.fit(X_val, y_val - y_pred)
    y_pred_residual = calib_model.predict(X_val)
    y_pred_calib = y_pred + y_pred_residual
    
    # Store for fold-wise analysis
    site_results = pd.DataFrame({
        "site": site,
        "y_val": y_val.values,
        "y_pred": y_pred,
        "y_pred_calib": y_pred_calib
    })
    results.append(site_results)


# Concatenate all site results
results_df = pd.concat(results, ignore_index=True)
y_pred_calibrated_all = results_df['y_pred_calib'].values
y_pred_calibrated_all = np.where(y_pred_calibrated_all > 75.0, 75.0 + ((y_pred_calibrated_all - 75.0) * 0.5), y_pred_calibrated_all)
y_pred_calibrated_all = np.where(y_pred_calibrated_all < 10.0, 10.0 - ((10.0 - y_pred_calibrated_all) * 0.5), y_pred_calibrated_all)
results_df['y_pred_calib_clip'] = y_pred_calibrated_all