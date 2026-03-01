from scripts.s02_feature_dataset import get_train_test_val_split
from src.XGBoost import getXGBoost
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    explained_variance_score
)

def make_shap_friendly(df, cat_cols):
    df2 = df.copy()

    for col in cat_cols:
        # Ensure true categorical dtype first
        df2[col] = df2[col].astype("category")

        # Convert categories to integer codes for SHAP
        df2[col] = df2[col].cat.codes

        # SHAP/XGBoost can dislike -1 for missing categories; make them NaN instead
        df2[col] = df2[col].replace(-1, np.nan)

    # Force any lingering object columns to numeric if possible
    for col in df2.columns:
        if df2[col].dtype == "object":
            df2[col] = pd.to_numeric(df2[col], errors="coerce")

    return df2

if __name__ == "__main__":

    X_train, X_test, X_val, y_train, y_test, y_val = get_train_test_val_split()

    cat_cols = ["city_1", "city_2", "state_1", "state_2", "carrier_low", "metro_1", "metro_2"]

    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    model = getXGBoost(X_train, X_val, y_train, y_val)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")
    print(f"Median AE: {medae:.4f}")
    print(f"Explained Variance: {evs:.4f}")

    # Save predictions for fare-gap analysis
    results = X_test.copy()
    results["actual_fare"] = y_test.values
    results["predicted_fare"] = y_pred
    results["fare_gap"] = results["actual_fare"] - results["predicted_fare"]

    os.makedirs("artifacts/shap", exist_ok=True)
    # Sample rows
X_shap_raw = X_test.sample(min(300, len(X_test)), random_state=42).copy()

# SHAP-friendly numeric copies
X_train_shap = make_shap_friendly(X_train, cat_cols)
X_shap = make_shap_friendly(X_shap_raw, cat_cols)

# Optional sanity check
print(X_shap.dtypes)
print(X_shap.isnull().sum().sort_values(ascending=False).head(10))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

plt.figure()
shap.summary_plot(shap_values, X_shap, show=False)
plt.tight_layout()
plt.savefig("artifacts/shap/shap_summary.png", dpi=200, bbox_inches="tight")
plt.close()

print("Saved SHAP summary plot.")