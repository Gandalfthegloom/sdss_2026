def joinMedianIncome(origin, median, state_column_origin, state_column_median):
    """
    Merge median dataset into origin dataset
    :param origin: Original/Target Dataset
    :param median: Median Income Dataset
    :param state_column_origin: Column Name for State in Original Dataset
    :param state_column_median: Column Name for State in Median Dataset
    :return: New Dataset
    """

    return origin.merge(
        median,
        left_on=[state_column_origin, "Year"],
        right_on=[state_column_median, "year"],
        how="left"
    )


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
