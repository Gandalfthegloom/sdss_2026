#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Read Airline Ticket Dataset
df = pd.read_excel("airline_ticket_dataset.xlsx")


# In[2]:


df["fare_per_miles"] = df["fare"]/df["nsmiles"]


# # Read CPI Dataset
# Add the Monthly CPI Dataset and convert it to quarterly, then adjust so that Q1 2022 becomes our base year.

# In[3]:


# Read CPI Dataset
cpi = pd.read_excel("CPI US.xlsx", sheet_name="Monthly")

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


# In[5]:


df2 = df.merge(cpi_q[["Year", "quarter", "cpi_adj"]], on=["Year", "quarter"], how="right")


# In[6]:


nom_price = ["fare", "fare_lg", "fare_low"]

for x in nom_price:
    df2[x + "_real"] = df2[x] * (100 / df2["cpi_adj"])


df2.head()


# In[ ]:




