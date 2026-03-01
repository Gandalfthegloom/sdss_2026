from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_CSV_PATH = "Data/Interim/adjusted_airline_tickets.csv"
OUTPUT_CSV_PATH = "Data/Processed/model_ready_airline_fares.csv"
TARGET_COL = "fare_real"
RANDOM_STATE = 42
TEST_SIZE = 0.3


KEEP_COLS = [
    "Year",
    "quarter",
    "city_1",
    "city_2",
    "state_1",
    "state_2",
    "nsmiles",
    "passengers",
    "fare_real",
    "large_ms",
    "fare_lg_real",
    "carrier_low",
    "lf_ms",
    "fare_low_real",
    "TotalFaredPax_city1",
    "TotalPerLFMkts_city1",
    "TotalPerPrem_city1",
    "TotalFaredPax_city2",
    "TotalPerLFMkts_city2",
    "TotalPerPrem_city2",
    "median_income_1",
    "median_income_2",
    "metro_1",
    "metro_2",
]




def build_filtered_dataset(
        string_cols,
        numeric_cols,
        target_col,
        csv_path: str | Path = RAW_CSV_PATH
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Load, filter, and clean the airline fare dataset.

    Returns a model-ready dataframe that still contains the target column.
    """
    df = pd.read_csv(csv_path)

    df = df.loc[:, ~df.columns.duplicated()]

    keep_cols = numeric_cols + string_cols
    if target_col not in keep_cols:
        keep_cols.append(target_col)

    missing_cols = [col for col in keep_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    model_df = df[keep_cols].copy()

    # Create aggregate feature requested earlier
    model_df["TotalFaredTotal"] = (
        pd.to_numeric(model_df["TotalFaredPax_city1"], errors="coerce").fillna(0)
        + pd.to_numeric(model_df["TotalFaredPax_city2"], errors="coerce").fillna(0)
    )

    # Standardize text columns
    for col in string_cols:
        model_df[col] = model_df[col].astype("string").str.strip()
        model_df[col] = model_df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Convert numerics
    for col in (numeric_cols + ["TotalFaredTotal"]):
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    # Basic row filtering
    model_df = model_df.dropna(subset=[target_col])
    model_df = model_df.drop_duplicates()
    model_df = model_df.loc[
        (model_df["fare_real"] > 0)
        & (model_df["passengers"] > 0)
        & (model_df["nsmiles"] > 0)
    ].copy()

    # Impute numeric columns with median
    for col in numeric_cols + ["TotalFaredTotal"]:
        model_df[col] = model_df[col].fillna(model_df[col].median())

    # Impute categorical columns with mode
    for col in string_cols:
        mode = model_df[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        model_df[col] = model_df[col].fillna(fill_value)

    # Optional rename for readability
    model_df = model_df.rename(columns={"quarter": "Quarter"})

    valid_mask = (model_df["Year"] == 2024) & (model_df["Quarter"] >= 3)
    test_mask = (model_df["Year"] == 2025)
    train_mask = ~(valid_mask | test_mask)

    model_df_train = model_df.loc[train_mask].reset_index(drop=True)
    model_df_valid = model_df.loc[valid_mask].reset_index(drop=True)
    model_df_test = model_df.loc[test_mask].reset_index(drop=True)

    return model_df_train, model_df_valid, model_df_test


def get_train_test_val_split(
    string_cols: list,
    numeric_cols: list,
    csv_path: str | Path = RAW_CSV_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    target_col: str = TARGET_COL,
):
    """
    Return X_train, X_test, x_val, y_train, y_test, y_val from the filtered dataset.
    """
    model_df_train, model_df_valid, model_df_test = build_filtered_dataset(string_cols=string_cols, numeric_cols=numeric_cols, target_col=target_col, csv_path=csv_path)

    X_train = model_df_train.drop(columns=[target_col])
    y_train = model_df_train[target_col]

    X_val = model_df_valid.drop(columns=[target_col])
    y_val = model_df_valid[target_col]

    X_test = model_df_test.drop(columns=[target_col])
    y_test = model_df_test[target_col]

    # X = model_df.drop(columns=[target_col])
    # y = model_df[target_col]
    #
    # X_train, X_temp, y_train, y_temp = train_test_split(
    #     X,
    #     y,
    #     test_size=test_size,
    #     random_state=random_state,
    # )
    #
    # X_val, X_test, y_val, y_test = train_test_split(
    #     X_temp,
    #     y_temp,
    #     test_size=0.5,
    #     random_state=random_state,
    # )

    return X_train, X_test, X_val, y_train, y_test, y_val


if __name__ == "__main__":
    filtered_df = build_filtered_dataset(RAW_CSV_PATH)
    filtered_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved filtered dataset to {OUTPUT_CSV_PATH} with shape {filtered_df.shape}")
