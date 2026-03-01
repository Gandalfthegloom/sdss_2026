import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Get up one folder to see scripts
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.s02_feature_dataset import get_train_test_val_split

# Import all our models
from src.XGBoost import getXGBoost
from src.LightGBM import getLightGBM
from src.CatBoost import getCatBoost
from src.RandomForest import getRandomForest
from src.RidgeRegression import getRidge

STRING_COLS = [
    # "city_1",
    # "city_2",
    # "state_1",
    # "state_2",
    "carrier_low",
    # "metro_1",
    # "metro_2",
]

NUMERIC_COLS = [
    "Year",
    "quarter",
    "nsmiles",
    "passengers",
    "fare_real", # We make sure the thing we want to predict is actually in the thing for now. This will later be separated into the y data during split
    "large_ms",
    # "fare_lg_real",
    "lf_ms",
    # "fare_low_real",
    "TotalFaredPax_city1",
    "TotalPerLFMkts_city1",
    "TotalPerPrem_city1",
    "TotalFaredPax_city2",
    "TotalPerLFMkts_city2",
    "TotalPerPrem_city2",
    "median_income_1",
    "median_income_2",
]

def evaluate_model(model_name, model_func, X_train, X_valid, X_test, y_train, y_valid, y_test):
    print(f"\n{'='*50}")
    print(f" Training {model_name}...")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # Train the model
    model = model_func(X_train, X_valid, y_train, y_valid)
    
    train_time = time.time() - start_time
    
    # Predict on unseen test data
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "Model": model_name,
        "RMSE ($)": round(rmse, 2),
        "MAE ($)": round(mae, 2),
        "R-Squared": round(r2, 4),
        "Train Time (s)": round(train_time, 2)
    }

if __name__ == "__main__":
    # 1. Load Data
    print("Loading data...")
    X_train, X_test, X_val, y_train, y_test, y_val = get_train_test_val_split(string_cols=STRING_COLS, numeric_cols=NUMERIC_COLS)
    
    # CRITICAL CHECK: Ensure target variable is NOT in features. NO LEAKSSSS
    for df in [X_train, X_test, X_val]:
        if "fare_real" in df.columns:
            df.drop(columns=["fare_real"], inplace=True)
            print("Dropped 'fare_real' from features to prevent target leakage!")
    
    # Convert all string/object columns to 'category' globally
    # This prevents XGBoost and LightGBM from crashing on the Pandas "string" dtype
    cat_cols = X_train.select_dtypes(exclude=['number', 'bool']).columns
    for df in [X_train, X_test, X_val]:
        for col in cat_cols:
            df[col] = df[col].astype('category')
    print(f"Converted {len(cat_cols)} text columns to 'category' dtype.")

    # 2. Define the competitors
    models_to_test = {
        "Ridge Baseline": getRidge,
        "Random Forest": getRandomForest,
        "XGBoost": getXGBoost,
        "LightGBM": getLightGBM,
        "CatBoost": getCatBoost
    }

    # 3. Run the evaluation loop
    results = []
    for name, func in models_to_test.items():
        metrics = evaluate_model(name, func, X_train, X_val, X_test, y_train, y_val, y_test)
        results.append(metrics)

    # 4. Display Results
    print("\n\n" + " MODEL BAKE-OFF RESULTS ".center(60))
    print("-" * 60)
    
    results_df = pd.DataFrame(results)
    # Sort by RMSE (lowest is best)
    results_df = results_df.sort_values(by="RMSE ($)", ascending=True).reset_index(drop=True)
    
    # Print a nice Markdown table to the console
    print(results_df.to_string(index=False))