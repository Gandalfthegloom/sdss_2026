#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_excel("airline_ticket_dataset.xlsx")
cpi = pd.read_excel("CPI US.xlsx", sheet_name="Monthly")


# # Create add_cpi Function
# This function is used to convert the monthly CPI into quarterly, then merge it to the Airline Ticket Dataset after adjusting the CPI so that Q1 2022 becomes the base year with CPI = 100.

# In[2]:


def add_cpi(df, cpi):
    df["fare_per_miles"] = df["fare"]/df["nsmiles"]

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

    df = df.merge(cpi_q[["Year", "quarter", "cpi_adj"]], on=["Year", "quarter"], how="right")
    nom_price = ["fare", "fare_lg", "fare_low"]

    for x in nom_price:
        df[x + "_real"] = df[x] * (100 / df["cpi_adj"])

    return df


# In[3]:


add_cpi(df, cpi)


# In[ ]:




