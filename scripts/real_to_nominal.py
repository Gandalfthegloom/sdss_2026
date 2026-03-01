#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv("Data/Interim/adjusted_airline_tickets.csv")
cpi = pd.read_excel("Data/Raw/CPI US.xlsx", sheet_name="Monthly")


# In[2]:


# def add_cpi(airline_ticket: pd.DataFrame, cpi:pd.DataFrame):
#     airline_ticket["fare_per_miles"] = airline_ticket["fare"]/df["nsmiles"]

#     cpi["Year"] = cpi["observation_date"].dt.year
#     cpi["month"] = cpi["observation_date"].dt.month
#     cpi["quarter"] = (cpi["month"] - 1) // 3 + 1

#     cpi_q = (
#         cpi.groupby(["Year", "quarter"], as_index=False)
#            .agg(cpi_q=("CPIAUCSL", "mean"),
#                 months_in_q=("CPIAUCSL", "count"))
#            .sort_values(["Year", "quarter"])
#     )

#     cpi_q["cpi_adj"] = cpi_q["cpi_q"]/284.905667 * 100
#     cpi_q.drop([14, 15, 16], axis=0, inplace=True)

#     airline_ticket = airline_ticket.merge(cpi_q[["Year", "quarter", "cpi_adj"]], on=["Year", "quarter"], how="right")
#     nom_price = ["fare", "fare_lg", "fare_low"]

#     for x in nom_price:
#         airline_ticket[x + "_real"] = airline_ticket[x] * (100 / airline_ticket["cpi_adj"])

#     return airline_ticket

# df = add_cpi(df, cpi)


# In[3]:


df.head()


# # Create real_to_nominal Function

# In[11]:


# def real_to_nominal(airline_ticket: pd.DataFrame, year: int, quarter: int):
#     """
#     Converts real prices back to nominal prices for a specific year and quarter.
#     Creates new columns with '_norm' suffix if they don't exist, fills only the specified rows,
#     and preserves existing values in other rows.

#     Parameters:
#     -----------
#     airline_ticket: DataFrame containing the airline ticket data with real prices
#     year: int - The year to convert
#     quarter: int - The quarter to convert (1, 2, 3, or 4)

#     Returns:
#     --------
#     DataFrame with nominal price columns added/updated for the specified period
#     """
#     # Create a copy to avoid modifying the original
#     df_result = airline_ticket.copy()

#     # Filter for the specified year and quarter
#     mask = (df_result["Year"] == year) & (df_result["quarter"] == quarter)

#     if not mask.any():
#         print(f"Warning: No data found for Year {year}, Quarter {quarter}")
#         return df_result

#     # Identify real price columns (ending with '_real')
#     real_price_cols = [col for col in df_result.columns if col.endswith('_real')]

#     # Convert each real price column back to nominal for the specified period
#     for col in real_price_cols:
#         # Remove '_real' suffix to get the original nominal column name
#         nominal_col = col.replace('_real', '')
#         # Create new column name with '_norm' suffix
#         new_col = nominal_col + '_norm'

#         # Check if the column already exists
#         if new_col not in df_result.columns:
#             # If it doesn't exist, create it with NaN for all rows
#             df_result[new_col] = float('nan')

#         # Fill only the rows matching the specified year and quarter
#         # This updates only the specified rows without affecting others
#         df_result.loc[mask, new_col] = df_result.loc[mask, col] * (df_result.loc[mask, "cpi_adj"] / 100)

#     return df_result



# In[15]:


def real_to_nominal_simple(real_value: float, year: int, quarter: int, cpi_df: pd.DataFrame) -> float:
    """
    Converts a real dollar value (base 2022 Q1) to nominal dollars for a given year and quarter.
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(cpi_df['observation_date']):
        cpi_df['observation_date'] = pd.to_datetime(cpi_df['observation_date'])

    # Extract Year and Quarter if they don't exist
    if 'Year' not in cpi_df.columns:
        cpi_df['Year'] = cpi_df['observation_date'].dt.year
    if 'quarter' not in cpi_df.columns:
        cpi_df['quarter'] = (cpi_df['observation_date'].dt.month - 1) // 3 + 1

    # Aggregate monthly CPI into quarterly averages
    cpi_q = cpi_df.groupby(['Year', 'quarter'])['CPIAUCSL'].mean().reset_index()

    # Get baseline CPI (2022 Q1)
    try:
        baseline_cpi = cpi_q[(cpi_q['Year'] == 2022) & (cpi_q['quarter'] == 1)]['CPIAUCSL'].values[0]
    except IndexError:
        raise ValueError("Baseline CPI for 2022 Q1 not found in the dataset.")

    # Get target CPI for the user-selected period
    try:
        target_cpi = cpi_q[(cpi_q['Year'] == year) & (cpi_q['quarter'] == quarter)]['CPIAUCSL'].values[0]
    except IndexError:
        raise ValueError(f"CPI data for Year {year}, Quarter {quarter} not found.")

    # Calculate and return the nominal value
    nominal_value = real_value * (target_cpi / baseline_cpi)
    
    return nominal_value

# In[ ]:




