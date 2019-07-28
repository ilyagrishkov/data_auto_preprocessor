"""
Module that is used to remove or replace all rows with null values in them.
"""
# pylint: disable=R0913
# pylint: disable=W0612
import pandas as pd
import numpy as np


def __find_average(data_frame: pd.DataFrame) -> (dict, dict, dict):
    """
    A helper method to find average values: mean, median

    This method is a helper method for the main No-Null-dataset (NND) method that
    receives a DataFrame in which it find a mean and median for every numeric column.

    :param data_frame: DataFrame to find average of columns of
    :return: returns a tuple consisting of mean and median
    """

    mean = {}
    median = {}
    mode = {}

    for column in list(data_frame):
        if np.issubdtype(data_frame[column].dtype, np.number):
            mean[column] = round(data_frame.loc[:, column].mean(), 2)
            median[column] = round(data_frame.loc[:, column].median(), 2)

        mode[column] = data_frame.loc[:, column].mode()

    return mean, median, mode


def nnd(data_frame, strategy='median', keep_rows=None, remove_rows=None,
        reindex=True, drop=False) -> pd.DataFrame:
    """A method to deal with missing values in the dataset.

    No-null-dataset(NND) is a method that accepts a dataset with null values
    and replaces the rows with missing data with average (median, mean) or removes them.

        :param data_frame: The data to be preprocessed
        :param strategy: The way the method should handle rows with missing values (median,
        mean, mode)
        :param keep_rows: Specify the rows to keep in case 'remove' method was chosen,
        otherwise it has no effect. keep_rows has a priority over remove_rows.
        :param remove_rows: Specify the rows to remove. Can be used with all methods.
        :param reindex: A new dataset will create new indexes if True.
        :param drop: Removes all the rows that contain null values, except for those in keep_rows

        :return: Dataset with no null values
    """

    if keep_rows is None:
        keep_rows = []

    if remove_rows is None:
        remove_rows = []

    if drop:
        remove_rows = data_frame[data_frame.isnull().any(axis=1)].index.values

    remove_rows = [x for x in remove_rows if x not in keep_rows]
    data_after_drop = data_frame.drop(remove_rows, axis=0)

    mean, median, mode = __find_average(data_after_drop)

    for index, row in data_after_drop.iterrows():
        for column, value in row.items():
            if pd.isnull(row[column]):

                if np.issubdtype(data_frame[column].dtype, np.number):

                    if strategy == 'mean':
                        data_after_drop.at[index, column] = mean[column]
                    elif strategy == 'mode':
                        data_after_drop.at[index, column] = mode[column][0]
                    else:
                        data_after_drop.at[index, column] = median[column]

                else:
                    data_after_drop.at[index, column] = mode[column][0]

    no_null_dataset = data_after_drop.reset_index(drop=True) if reindex else data_after_drop

    return no_null_dataset
