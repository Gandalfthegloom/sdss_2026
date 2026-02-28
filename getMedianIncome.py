import requests, pandas as pd

def acs_state_median_income(year, product="acs1"):  # product: "acs1" or "acs5"
    url = f"https://api.census.gov/data/{year}/acs/{product}"
    params = {
        "get": "NAME,B19013_001E,B19013_001M",
        "for": "state:*"
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df["year"] = year
    df["median_household_income"] = pd.to_numeric(df["B19013_001E"], errors="coerce")
    df["median_household_income_moe"] = pd.to_numeric(df["B19013_001M"], errors="coerce")
    return df[["year","state","NAME","median_household_income","median_household_income_moe"]]

def getMedianIncome():
    out = pd.concat([acs_state_median_income(y, "acs1") for y in [2022, 2023, 2024]], ignore_index=True)

    STATE_ABBR_TO_NAME = {
        "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
        "CO":"Colorado","CT":"Connecticut","DE":"Delaware","DC":"District of Columbia",
        "FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana",
        "IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
        "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri",
        "MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey",
        "NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio",
        "OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina",
        "SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia",
        "WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming",
        # territories if you want:
        "PR":"Puerto Rico"
    }
    STATE_NAME_TO_ABBR = {v.upper(): k for k, v in STATE_ABBR_TO_NAME.items()}


    out["STATE"] = out["NAME"].str.upper().map(STATE_NAME_TO_ABBR)
    print(out.head(5))
    out.to_csv("Data/Median_Income.csv")

