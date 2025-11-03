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
import shap


# Assume df is available

# Clip LVEF above 75% and below 10%
df["LVEF"] = np.where(df["LVEF"] > 75.0, 75.0 + ((df["LVEF"] - 75.0) * 0.5), df["LVEF"])
df["LVEF"] = np.where(df["LVEF"] < 10.0, 10.0 - ((10.0 - df["LVEF"]) * 0.5), df["LVEF"])


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

grid_all = GridSearchCV(
    XGBRegressor(objective="reg:squarederror", random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_all.fit(X, y)
best_params_all = grid_all.best_params_

final_model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_params_all)
final_model.fit(X, y)


explainer = shap.Explainer(final_model, X)
shap_values = explainer(X)

# Plot global importance
shap.plots.beeswarm(shap_values, order=shap_values.abs.mean(0), max_display=50)

# Plot individual importance
# ptid = patient index
shap.plots.waterfall(shap_values[ptid], max_display=11)