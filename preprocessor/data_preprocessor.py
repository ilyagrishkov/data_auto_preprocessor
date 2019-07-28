"""
A Preprocessor for data.
"""

# pylint: disable=W0511
# pylint: disable=R0903
# pylint: disable=R0913
# pylint: disable=R0914
# pylint: disable=W0632
# pylint: disable=W0612

import numpy as np
import pandas as pd

from preprocessor import null_value_cleaner as nvc
from preprocessor import outliers_cleaner as oc


class Preprocessor:
    """
    A class for data preprocessor.
    """

    def __init__(self, initial_data):
        """
        A constructor for data preprocessor.

        :param initial_data: dataset to be preprocessed.
        """
        self.data = initial_data

    def __standardize_data(self) -> np.ndarray:

        mean = self.data.mean().values
        std = self.data.std().values

        standardized_data = []
        for index, row in self.data.iterrows():

            standardized_row = []
            for i, value in enumerate(row):
                standardized_row.append((value - mean[i]) / std[i])

            standardized_data.append(standardized_row)

        return np.array(standardized_data)

    def preprocess(self, target_columns: list, input_columns=None, missing_value_strategy='median',
                   keep_rows=None, remove_rows=None, drop_missing_values=False,
                   remove_outliers=True, outliers_strategy='Z', z_score_threshold=3,
                   encode_categorical_data=True, encoding_drop_first=True, standardize=True,
                   split=True, test_size=0.2, reindex=True,):
        """
        Method to automatically preprocess data and split it into train - test datasets.

        :param input_columns: input columns.
        :param target_columns: target columns.
        :param missing_value_strategy: The way the method should handle rows with missing values
        (median,mean, mode).
        :param keep_rows: Specify the rows to keep in case 'remove' method was chosen, otherwise
        it has no effect. keep_rows has a priority over remove_rows.
        :param remove_rows: Specify the rows to remove. Can be used with all methods.
        :param drop_missing_values: Removes all the rows that contain null values, except for
        those in keep_rows
        :param remove_outliers: Whether to remove outliers.
        :param outliers_strategy: A strategy for removal (Z or IQR).
        :param z_score_threshold: A threshold for a value to be considered outliers in case
        Z-score was chosen.
        :param encode_categorical_data: Whether to encode categorical data.
        :param encoding_drop_first: Whether to get k-1 dummies out of k categorical levels by
        removing the first level.
        :param standardize: Whether to standardize dataset (only possible if categorical data is
        encoded).
        :param split: Whether to split dataset.
        :param test_size: size of test dataset.
        :param reindex: A new dataset will create new indexes if True.
        :return: Four datasets: target_train, target_test, input_train, input_test or a
        preprocessed dataset if split=False
        """

        if input_columns is None:
            input_columns = []

        # Missing values' replacement/removal
        self.data = nvc.nnd(self.data, strategy=missing_value_strategy, keep_rows=keep_rows,
                            remove_rows=remove_rows, reindex=reindex, drop=drop_missing_values)

        # Outliers' removal
        if remove_outliers:
            self.data = oc.remove_outliers(self.data, strategy=outliers_strategy,
                                           reindex=reindex, threshold=z_score_threshold)

        # Split dataset into targets, inputs
        if split:
            targets = self.data[target_columns]

            input_columns = [x for x in input_columns if x not in targets]

            if not input_columns:
                inputs = self.data.drop(target_columns, axis=1)

            else:
                inputs = self.data[input_columns]

        # Categorical data encoding
        if encode_categorical_data and split:
            self.data = pd.get_dummies(inputs, drop_first=encoding_drop_first)

        elif encode_categorical_data:
            self.data = pd.get_dummies(self.data, drop_first=encoding_drop_first)

        # Data standardization
        if standardize and encode_categorical_data:
            self.data = self.__standardize_data()

        # Data split into train - test
        if split:
            test_size = 0.2 if test_size not in (0, 1) else test_size

            targets_train, targets_test = np.split(targets.to_numpy(), [int((1-test_size) * len(
                targets))])
            inputs_train, inputs_test = np.split(self.data, [int((1-test_size) * len(targets))])
            return targets_train, targets_test, inputs_train, inputs_test

        return self.data
