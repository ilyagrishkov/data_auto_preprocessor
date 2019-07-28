"""
A Preprocessor for data.
"""

# pylint: disable=W0511
# pylint: disable=R0903

import pandas as pd

from preprocessor import null_value_cleaner as nvc
from preprocessor import outliers_cleaner as oc


class Preprocessor:
    """
    A class for data preprocessor.
    """

    def __init__(self, initial_data: pd.DataFrame):
        """
        A constructor for data preprocessor.

        :param initial_data: dataset to be preprocessed.
        """
        self.initial_data = initial_data

    def auto_clean(self, target_columns: list) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                   pd.DataFrame):
        """
        Method to automatically preprocess data and split it into train - test datasets.

        :param target_columns: target columns
        :return: Four datasets: target_train, input_train, target_test, input_test
        """

        print('Shape before cleaning')
        print(self.initial_data.shape)
        print('--------------------------')

        no_null_data = nvc.nnd(self.initial_data, drop=True)
        print('Missing values removed!')
        print('Shape after cleaning')
        print(no_null_data.shape)
        print('--------------------------')

        no_outlier_data = oc.remove_outliers(no_null_data)
        print('Outliers removed!')
        print('Shape after removing outliers')
        print(no_outlier_data.shape)
        print('--------------------------')

        # TODO: add more sophisticated method to encode categorical data
        encoded_categorical_data = pd.get_dummies(no_outlier_data, drop_first=True)
        print('Categorical data encoded!')
        print('Shape after encoding categorical datas')
        print(encoded_categorical_data.shape)
        print('--------------------------')

        # TODO: add data standardization

        # TODO: add data split

        data_preprocessed = encoded_categorical_data

        # TODO: do proper split. This is a temporary solution.
        targets = data_preprocessed[target_columns[0]]
        inputs = data_preprocessed.drop([target_columns[0]], axis=1)

        return targets, inputs, None, None
