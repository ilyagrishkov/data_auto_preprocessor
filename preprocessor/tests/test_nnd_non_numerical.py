"""
A unit test suite for NND method, using only numeric values in the dataset.
"""
import unittest
import numpy as np
import pandas as pd
from preprocessor import null_value_cleaner as nvc


class NonNumericNNDTest(unittest.TestCase):
    """
    A tests class for NND method.
    """
    non_numeric_only_two_null = {'Feature1': ['a', 'b', np.nan, 'c'],
                                 'Feature2': ['a', np.nan, 'r', 'f'],
                                 'Feature3': ['d', 'd', 'q', 't']}

    mixed_two_null = {'Feature1': ['a', 'b', np.nan, 'c'],
                      'Feature2': [1, np.nan, 45, 23],
                      'Feature3': ['d', 'd', 'q', 't']}

    def test_nnd_remove_two_rows(self):
        """
        Two null rows remove approach.
        :return: True if both rows with null were removed.
        """
        correct_non_numeric_no_null = {'Feature1': ['a', 'c'], 'Feature2': ['a', 'f'],
                                       'Feature3': ['d', 't']}

        initial_data = pd.DataFrame(self.non_numeric_only_two_null)
        correct_data = pd.DataFrame(correct_non_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, drop=True)))

    def test_nnd_mode_two_rows(self):
        """
        Two null rows mode approach.
        :return: True if both rows with null were removed.
        """
        correct_non_numeric_no_null = {'Feature1': ['a', 'b', 'a', 'c'],
                                       'Feature2': ['a', 'a', 'r', 'f'],
                                       'Feature3': ['d', 'd', 'q', 't']}

        initial_data = pd.DataFrame(self.non_numeric_only_two_null)
        correct_data = pd.DataFrame(correct_non_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data)))

    def test_nnd_mixed_two_rows(self):
        """
        Two null rows mixed numeric and non mode approach.
        :return: True if both rows with null were replaced by median and mode.
        """
        correct_non_numeric_no_null = {'Feature1': ['a', 'b', 'a', 'c'],
                                       'Feature2': [1.0, 23.0, 45.0, 23.0],
                                       'Feature3': ['d', 'd', 'q', 't']}

        initial_data = pd.DataFrame(self.mixed_two_null)
        correct_data = pd.DataFrame(correct_non_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, strategy='mean')))

    def test_nnd_inner_mixed_two_rows(self):
        """
        Two null rows mixed numeric and non mode approach.
        :return: True if both rows with null were replaced by median and mode.
        """
        mixed_two_null = {'Feature1': ['a', 'b', np.nan, 'c'],
                          'Feature2': [1, np.nan, 45, 23],
                          'Feature3': [np.nan, 'd', '1', 'b']}

        correct_non_numeric_no_null = {'Feature1': ['a', 'b', 'a', 'c'],
                                       'Feature2': [1.0, 23.0, 45.0, 23.0],
                                       'Feature3': ['1', 'd', '1', 'b']}

        initial_data = pd.DataFrame(mixed_two_null)
        correct_data = pd.DataFrame(correct_non_numeric_no_null)

        print(nvc.nnd(initial_data, strategy='mean'))

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, strategy='mean')))


if __name__ == '__main__':
    unittest.main()
