import pandas as pd

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

    # Save the resulting dataset to CSV
    df2.to_csv("Data/Interim/adjusted_airline_tickets.csv", index=False)

if __name__ == "__main__":
    process_airline_data()