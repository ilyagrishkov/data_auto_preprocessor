def nnd(approach='remove', rows_to_keep=[], rows_to_remove=[]):

    """A method to deal with missing values in the dataset.

    No-null-dataset(NND) is a method that accepts a dataset with null values
and replaces the rows with missing data with average (median, mean, 50th quantile)
or removes them.

    Parameters:
        :param approach: The way the method should handle missing values
        :param rows_to_keep: Specify the rows to keep in case 'remove' method was chosen,
otherwise it has no effect
        :param rows_to_remove: Specify the rows to remove. Can be used with all methods


    Returns:
        array: Dataset with no null values

    """

    return approach, rows_to_keep, rows_to_remove
