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
        right_on=[state_column_median, "Year"],
        how="left"
    )

