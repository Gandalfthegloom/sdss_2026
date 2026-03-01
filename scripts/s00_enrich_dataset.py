import json
from pathlib import Path

import pandas as pd
from Preprocessing import extractCityStateMetropolitan, joinMedianIncome, getCityLookUp
import geopandas as gpd
from shapely.geometry import Point

def process_airline_data():
    # Read Airline Ticket Dataset
    df = pd.read_excel("Data/Raw/airline_ticket_dataset.xlsx")
    df["fare_per_miles"] = df["fare"] / df["nsmiles"]

    # Read and Process CPI Dataset
    cpi = pd.read_excel("Data/Raw/CPI US.xlsx", sheet_name="Monthly")

    cpi["Year"] = cpi["observation_date"].dt.year
    cpi["month"] = cpi["observation_date"].dt.month
    cpi["quarter"] = (cpi["month"] - 1) // 3 + 1

    cpi_q = (
        cpi.groupby(["Year", "quarter"], as_index=False)
        .agg(
            cpi_q=("CPIAUCSL", "mean"),
            months_in_q=("CPIAUCSL", "count")
        )
        .sort_values(["Year", "quarter"])
    )

    # Adjust CPI so that Q1 2022 becomes the base year
    cpi_q["cpi_adj"] = (cpi_q["cpi_q"] / 284.905667) * 100
    cpi_q.drop([14, 15, 16], axis=0, inplace=True)

    # Merge Datasets
    df2 = df.merge(
        cpi_q[["Year", "quarter", "cpi_adj"]], 
        on=["Year", "quarter"], 
        how="right"
    )

    # Calculate Real Prices
    nom_price = ["fare", "fare_lg", "fare_low"]
    for col in nom_price:
        df2[f"{col}_real"] = df2[col] * (100 / df2["cpi_adj"])
        
    df2 = extractCityStateMetropolitan(df2)
    df2 = joinMedianIncome(df2, "state_1")
    df2 = joinMedianIncome(df2, "state_2")

    # Adding Coordinate
    df = df2.copy()
    df["city_1_clean"] = df["city_1"].apply(lambda x: x.split("/")[0])
    df["city_2_clean"] = df["city_2"].apply(lambda x: x.split("/")[0])

    # Build LookUp Table
    city_path = Path("scripts/cityLookUp.json")
    if not city_path.exists():
        city_lookup = getCityLookUp(df)
        with open("scripts/cityLookUp.json", "w", encoding="utf-8") as f:
            json.dump(city_lookup, f, ensure_ascii=False, indent=2)

    with open("scripts/cityLookUp.json", "r", encoding="utf-8") as f:
        city_lookup  = json.load(f)


    df["coord_1"] = df["city_1_clean"].map(city_lookup)
    df["coord_2"] = df["city_2_clean"].map(city_lookup)

    df = df.dropna()
    # clean data
    df["lat_1"] = df["coord_1"].apply(lambda x: x[0])
    df["lon_1"] = df["coord_1"].apply(lambda x: x[1])
    df["lat_2"] = df["coord_2"].apply(lambda x: x[0])
    df["lon_2"] = df["coord_2"].apply(lambda x: x[1])


    # If you have a folder with the .shp + .dbf + .shx + .prj
    poly = gpd.read_file("Data/Raw/cb_2018_us_state_500k.zip")

    # For Folium, use lat/lon CRS
    poly = poly.to_crs(epsg=4326)

    # 0) Make sure both layers share CRS
    poly = poly.to_crs("EPSG:4326")

    df2 = df.copy()
    df2["row_id"] = df2.index  # stable key to merge back

    # 1) Origin points
    orig = gpd.GeoDataFrame(
        df2[["row_id"]],
        geometry=gpd.points_from_xy(df2["lon_1"], df2["lat_1"]),
        crs="EPSG:4326"
    )

    orig_join = (
        gpd.sjoin(orig, poly[["STUSPS", "geometry"]], predicate="within", how="left")
        .drop(columns=["index_right"])
        .rename(columns={"STUSPS": "orig_STUSPS"})
    )

    # 2) Destination points
    dest = gpd.GeoDataFrame(
        df2[["row_id"]],
        geometry=gpd.points_from_xy(df2["lon_2"], df2["lat_2"]),
        crs="EPSG:4326"
    )

    dest_join = (
        gpd.sjoin(dest, poly[["STUSPS", "geometry"]], predicate="within", how="left")
        .drop(columns=["index_right"])
        .rename(columns={"STUSPS": "dest_STUSPS"})
    )

    # 3) Merge results back into the original rows
    out = (
        df2
        .merge(orig_join[["row_id", "orig_STUSPS"]], on="row_id", how="left")
        .merge(dest_join[["row_id", "dest_STUSPS"]], on="row_id", how="left")
        .drop(columns=["row_id"])
    )


    # Save the resulting dataset to CSV
    out.to_csv("Data/Interim/adjusted_airline_tickets.csv", index=False)

if __name__ == "__main__":
    process_airline_data()
    