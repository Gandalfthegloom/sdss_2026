#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score


# # 1) Build modeling table (NO LOG): route_id, target, lags

# In[2]:


def build_model_table_no_log(
    df: pd.DataFrame,
    target_col: str = "fare_real",
    horizon: int = 1,                 # quarters ahead
    lags: tuple = (1, 4),
    lag_feature_cols: list | None = None,
):
    d = df.copy()

    # Required columns check
    required_cols = ["Year", "quarter", "citymarketid_1", "citymarketid_2", "nsmiles", target_col]
    missing = [c for c in required_cols if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # route_id (directionless)
    a = d["citymarketid_1"].astype(int)
    b = d["citymarketid_2"].astype(int)
    d["route_from"] = np.minimum(a, b)
    d["route_to"]   = np.maximum(a, b)
    d["route_id"]   = d["route_from"].astype(str) + "_" + d["route_to"].astype(str)

    # time index
    d["qtr_index"] = d["Year"].astype(int) * 4 + (d["quarter"].astype(int) - 1)
    d = d.sort_values(["route_id", "qtr_index"]).reset_index(drop=True)

    # Target in levels (NO LOG)
    d["y"] = pd.to_numeric(d[target_col], errors="coerce")
    d["y_target"] = d.groupby("route_id")["y"].shift(-horizon)

    # Lags of target
    for L in lags:
        d[f"y_lag{L}"] = d.groupby("route_id")["y"].shift(L)

    # Lag other features (forecast-safe)
    if lag_feature_cols is None:
        lag_feature_cols = []
    for c in lag_feature_cols:
        if c not in d.columns:
            continue
        for L in lags:
            d[f"{c}_lag{L}"] = d.groupby("route_id")[c].shift(L)

    # Feature list
    feature_cols = ["route_id", "quarter", "nsmiles"] + [f"y_lag{L}" for L in lags]
    for c in lag_feature_cols:
        for L in lags:
            colname = f"{c}_lag{L}"
            if colname in d.columns:
                feature_cols.append(colname)

    # Drop rows missing target or required lag history
    required_for_model = ["y_target"] + [f"y_lag{L}" for L in lags]
    model_df = d.dropna(subset=required_for_model).copy()

    # Make categorical
    model_df["route_id"] = model_df["route_id"].astype("category")

    return model_df, feature_cols


# # 2) Time-series cross validation with LightGBM + R^2

# In[3]:


def lgbm_timeseries_cv(
    model_df: pd.DataFrame,
    feature_cols: list,
    *,
    target_col: str = "y_target",
    time_col: str = "qtr_index",
    n_splits: int = 5,
    weight_col: str | None = "passengers",   # set None to disable
    params: dict | None = None,
    num_boost_round: int = 5000,
    early_stopping_rounds: int = 200,
    verbose_eval: int = 200,
    seed: int = 42,
):
    if params is None:
        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "seed": seed,
        }

    dfv = model_df.sort_values(time_col).reset_index(drop=True).copy()
    dfv["route_id"] = dfv["route_id"].astype("category")

    X = dfv[feature_cols]
    y = dfv[target_col].to_numpy()

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_rows = []
    models = []

    all_y = []
    all_pred = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Optional weights
        w_tr = None
        w_va = None
        if weight_col is not None and weight_col in dfv.columns:
            w_tr = dfv.iloc[tr_idx][weight_col].fillna(0).clip(lower=0)
            w_va = dfv.iloc[va_idx][weight_col].fillna(0).clip(lower=0)

        dtrain = lgb.Dataset(
            X_tr, label=y_tr, weight=w_tr,
            categorical_feature=["route_id"], free_raw_data=False
        )
        dvalid = lgb.Dataset(
            X_va, label=y_va, weight=w_va,
            categorical_feature=["route_id"], reference=dtrain, free_raw_data=False
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(verbose_eval)
            ],
        )

        pred = model.predict(X_va, num_iteration=model.best_iteration)

        mae = float(mean_absolute_error(y_va, pred))
        rmse = float(np.sqrt(np.mean((y_va - pred) ** 2)))
        r2 = float(r2_score(y_va, pred))

        fold_rows.append({
            "fold": fold,
            "train_rows": int(len(tr_idx)),
            "valid_rows": int(len(va_idx)),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "best_iteration": int(model.best_iteration),
            "valid_time_min": int(dfv.loc[va_idx, time_col].min()),
            "valid_time_max": int(dfv.loc[va_idx, time_col].max()),
        })
        models.append(model)

        all_y.append(y_va)
        all_pred.append(pred)

    results_df = pd.DataFrame(fold_rows)

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)

    summary = {
        "mae_mean": float(results_df["mae"].mean()),
        "mae_std": float(results_df["mae"].std(ddof=1)),
        "rmse_mean": float(results_df["rmse"].mean()),
        "rmse_std": float(results_df["rmse"].std(ddof=1)),
        "r2_mean": float(results_df["r2"].mean()),
        "r2_std": float(results_df["r2"].std(ddof=1)),
        "overall_cv_r2": float(r2_score(all_y, all_pred)),
    }

    return results_df, summary, models


# # 3) RUN: build table -> CV -> print results

# In[5]:


df = pd.read_excel("airline_ticket_dataset.xlsx")
cpi = pd.read_excel("CPI US.xlsx", sheet_name="Monthly")

def add_cpi(airline_ticket: pd.DataFrame, cpi:pd.DataFrame):
    airline_ticket["fare_per_miles"] = airline_ticket["fare"]/df["nsmiles"]

    cpi["Year"] = cpi["observation_date"].dt.year
    cpi["month"] = cpi["observation_date"].dt.month
    cpi["quarter"] = (cpi["month"] - 1) // 3 + 1

    cpi_q = (
        cpi.groupby(["Year", "quarter"], as_index=False)
           .agg(cpi_q=("CPIAUCSL", "mean"),
                months_in_q=("CPIAUCSL", "count"))
           .sort_values(["Year", "quarter"])
    )

    cpi_q["cpi_adj"] = cpi_q["cpi_q"]/284.905667 * 100
    cpi_q.drop([14, 15, 16], axis=0, inplace=True)

    airline_ticket = airline_ticket.merge(cpi_q[["Year", "quarter", "cpi_adj"]], on=["Year", "quarter"], how="right")
    nom_price = ["fare", "fare_lg", "fare_low"]

    for x in nom_price:
        airline_ticket[x + "_real"] = airline_ticket[x] * (100 / airline_ticket["cpi_adj"])

    return airline_ticket

df = add_cpi(df, cpi)

import warnings
warnings.filterwarnings("ignore")


# In[6]:


# Choose which columns you want lagged (forecast-safe)
lag_feature_cols = [
    "passengers",
    "large_ms", "lf_ms",
    "fare_lg_real", "fare_low_real",
    "TotalPerLFMkts_city1", "TotalPerPrem_city1",
    "TotalPerLFMkts_city2", "TotalPerPrem_city2",
]

model_df, feature_cols = build_model_table_no_log(
    df,
    target_col="fare_real",
    horizon=1,              # next quarter
    lags=(1, 4),
    lag_feature_cols=lag_feature_cols
)

cv_results, cv_summary, cv_models = lgbm_timeseries_cv(
    model_df=model_df,
    feature_cols=feature_cols,
    target_col="y_target",
    time_col="qtr_index",
    n_splits=5,
    weight_col="passengers",   # set None if you don't want weighting
    # params=...,              # optionally pass your own params dict
)

print("FEATURES USED:")
print(feature_cols)

print("\nCV RESULTS (per fold):")
print(cv_results)

print("\nCV SUMMARY:")
print(cv_summary)


# In[ ]:




