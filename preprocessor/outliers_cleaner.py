"""
Module that is used to remove outliers from the dataset with no missing data.
"""
from scipy import stats
import pandas as pd
import numpy as np


def remove_outliers(dataset: pd.DataFrame, strategy='Z', reindex=True, threshold=3) -> \
        pd.DataFrame:
    """
    A method that removes outliers from a dataset that contains no null values. Two strategies
    can be used for outliers' removal: z-score and IQR. In case the dataset contains less than
    12 values only IQR strategy can be used.

    :param dataset: A dataset to remove outliers form containing no null values.
    :param strategy: A strategy for removal (Z or IQR).
    :param reindex: A new dataset will create new indexes if True.
    :param threshold: A threshold for a value to be considered outliers in case Z-score was chosen.
    :return: DataFrame containing no outliers.
    """

    if dataset.count()[0] < 12:
        strategy = 'IQR'

    if strategy.lower() == 'z':

        cols = list(dataset.columns)
        z_scores = pd.DataFrame()

        for col in cols:
            if np.issubdtype(dataset[col].dtype, np.number):
                col_zscore = col + '_zscore'
                z_scores[col_zscore] = np.abs(stats.zscore(dataset[col]))

        # noinspection PyTypeChecker
        no_outliers_dataset = dataset[(z_scores < threshold).all(axis=1)]

    else:

        first_quartile = dataset.quantile(0.25)
        third_quartile = dataset.quantile(0.75)
        iqr = third_quartile - first_quartile

        # noinspection PyTypeChecker
        no_outliers_dataset = dataset[~((dataset < (third_quartile - 1.5 * iqr))
                                        | (dataset > (third_quartile + 1.5 * iqr))).any(axis=1)]

    no_outliers_dataset = no_outliers_dataset.reset_index(
        drop=True) if reindex else no_outliers_dataset
    print(no_outliers_dataset)
    return no_outliers_dataset
