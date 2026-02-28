import pandas as pd
from pathlib import Path
from getMedianIncome import getMedianIncome


def joinMedianIncome(origin, state_column_origin):
    """
    Merge median dataset into origin dataset
    :param origin: Original/Target Dataset
    :param state_column_origin: Column Name for State in Original Dataset
    :return: New Dataset
    """
    "state_1"
    median_path = Path("Data/Median_Income.csv")
    if not median_path.exists():
        getMedianIncome()

    number = state_column_origin[-1]
    median = pd.read_csv("Data/Median_Income.csv")

    median = median.rename(columns={"median_household_income": f"median_income_{number}"})

    return origin.merge(
        median,
        left_on=[state_column_origin, "Year"],
        right_on=["STATE", "year"],
        how="left"
    ).drop(columns=["STATE", "year"])


def extractCityStateMetropolitan(df):
    """
    Make a new column contains metro indicator, city, and state
    :param df: Original dataset
    :return: new processed dataset
    """
    df["Temp_State"] = df["city1"].apply(lambda x: x.split(","))
    df["metro_1"] = df["city1"].apply(lambda x: "(Metropolitan Area)" in x)
    df["city_1"] = df["Temp_State"].apply(lambda x: x[0])
    df["state_1"] = df["Temp_State"].apply(lambda x: x[1].split(" ")[1])

    df["Temp_State"] = df["city2"].apply(lambda x: x.split(","))
    df["metro_2"] = df["city2"].apply(lambda x: "(Metropolitan Area)" in x)
    df["city_2"] = df["Temp_State"].apply(lambda x: x[0])
    df["state_2"] = df["Temp_State"].apply(lambda x: x[1].split(" ")[1])

    return df.drop(columns=["city1", "city2"])
