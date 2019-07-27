"""
A unit test suite for remove_outliers method.
"""
import unittest
import pandas as pd
from preprocessor import outliers_cleaner as oc


class MyTestCase(unittest.TestCase):
    """
    A tests class for remove_outliers method.
    """

    numeric_initial = {'Feature1': [56, 53, 1, 54, 55, 52, 51, 50, 45, 56, 54, 56, 58],
                       'Feature2': [56, 51, 48, 59, 61, 53, 55, 56, 51, 55, 52, 51, 50],
                       'Feature3': [53, 50, 47, 60, 52, 58, 59, 51, 51, 50, 45, 56, 54]}

    def test_remove_outliers_iqr(self):
        """
        Remove outliers using IQR strategy.
        :return: True if the rows with outliers were removed.
        """
        numeric_correct = {'Feature1': [56, 53, 54, 55, 52, 51, 50, 56, 56, 58],
                           'Feature2': [56, 51, 59, 61, 53, 55, 56, 55, 51, 50],
                           'Feature3': [53, 50, 60, 52, 58, 59, 51, 50, 56, 54]}

        initial_data = pd.DataFrame(self.numeric_initial)
        correct_data = pd.DataFrame(numeric_correct)

        self.assertTrue(correct_data.equals(oc.remove_outliers(initial_data, strategy='iqr')))

    def test_remove_outliers_zscore(self):
        """
        Remove outliers using z-score strategy.
        :return: True if the rows with outliers were removed.
        """
        numeric_correct = {'Feature1': [56, 53, 54, 55, 52, 51, 50, 45, 56, 54, 56, 58],
                           'Feature2': [56, 51, 59, 61, 53, 55, 56, 51, 55, 52, 51, 50],
                           'Feature3': [53, 50, 60, 52, 58, 59, 51, 51, 50, 45, 56, 54]}

        initial_data = pd.DataFrame(self.numeric_initial)
        correct_data = pd.DataFrame(numeric_correct)

        self.assertTrue(correct_data.equals(oc.remove_outliers(initial_data, strategy='z')))

    def test_remove_outliers_iqr_non_numeric(self):
        """
        Remove outliers using IQR strategy given that not all columns are numeric.
        :return: True if the rows with outliers were removed.
        """
        non_numeric_initial = {'Feature1': [56, 1, 54, 55],
                               'Feature2': ['a', 'a', 'a', 'a'],
                               'Feature3': [53, 50, 53, 52]}

        non_numeric_correct = {'Feature1': [56, 54, 55],
                               'Feature2': ['a', 'a', 'a'],
                               'Feature3': [53, 53, 52]}

        initial_data = pd.DataFrame(non_numeric_initial)
        correct_data = pd.DataFrame(non_numeric_correct)

        self.assertTrue(correct_data.equals(oc.remove_outliers(initial_data, strategy='iqr')))

    def test_remove_outliers_zscore_non_numeric(self):
        """
        Remove outliers using z-score strategy given that not all columns are numeric.
        :return: True if the rows with outliers were removed.
        """
        non_num_initial = {'Feature1': [56, 1, 54, 55, 52, 51, 50, 45, 56, 54, 56, 58],
                           'Feature2': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'],
                           'Feature3': [53, 50, 60, 52, 58, 59, 51, 51, 50, 45, 56, 54]}

        non_num_correct = {'Feature1': [56, 54, 55, 52, 51, 50, 45, 56, 54, 56, 58],
                           'Feature2': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'],
                           'Feature3': [53, 60, 52, 58, 59, 51, 51, 50, 45, 56, 54]}

        initial_data = pd.DataFrame(non_num_initial)
        correct_data = pd.DataFrame(non_num_correct)

        self.assertTrue(correct_data.equals(oc.remove_outliers(initial_data, strategy='z')))

    def test_remove_outliers_zscore_non_numeric_diff_threshold(self):
        """
        Remove outliers using z-score strategy given a different threshold for a value to be
        considered an outlier.
        :return: True if the rows with outliers were removed.
        """
        numeric_correct = {'Feature1': [56, 53, 54, 52, 51, 50, 45, 56, 54, 56, 58],
                           'Feature2': [56, 51, 59, 53, 55, 56, 51, 55, 52, 51, 50],
                           'Feature3': [53, 50, 60, 58, 59, 51, 51, 50, 45, 56, 54]}

        initial_data = pd.DataFrame(self.numeric_initial)
        correct_data = pd.DataFrame(numeric_correct)

        self.assertTrue(correct_data.equals(oc.remove_outliers(initial_data, strategy='z',
                                                               threshold=2)))


if __name__ == '__main__':
    unittest.main()
