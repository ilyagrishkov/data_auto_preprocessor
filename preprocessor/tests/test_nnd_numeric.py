"""
A unit test suite for NND method, using only numeric values in the dataset.
"""
import unittest
import pandas as pd
import numpy as np
from preprocessor import null_value_cleaner as nvc


class NumericNNDTest(unittest.TestCase):
    """
    A tests class for NND method.
    """

    numeric_two_null = {'Feature1': [12, 23, np.nan, 22], 'Feature2': [20, np.nan, 19, 18],
                        'Feature3': [34, 84, 10, 20]}

    numeric_three_null = {'Feature1': [12, 23, np.nan, 22], 'Feature2': [20, np.nan, 19, 18],
                          'Feature3': [34, 84, 10, np.nan]}

    def test_nnd_nothing_to_remove(self):
        """
        Dataset with no null values.
        :return: True if returns the same dataset.
        """
        correct_numeric_no_null = {'Feature1': [12, 23, 34, 22], 'Feature2': [20, 21, 19, 18],
                                   'Feature3': [34, 84, 10, 20]}
        data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(data.equals(nvc.nnd(data)))

    def test_nnd_remove_one_null_row(self):
        """
        One null row with remove approach.
        :return: True if the row with null was removed.
        """
        numeric_one_null = {'Feature1': [12, 23, 34, 22], 'Feature2': [20, np.nan, 19, 18],
                            'Feature3': [34, 84, 10, 20]}
        correct_numeric_no_null = {'Feature1': [12, 34, 22], 'Feature2': [20.0, 19.0, 18.0],
                                   'Feature3': [34, 10, 20]}

        initial_data = pd.DataFrame(numeric_one_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, drop=True)))

    def test_nnd_remove_two_null_rows(self):
        """
        Two null rows remove approach.
        :return: True if both rows with null were removed.
        """
        correct_numeric_no_null = {'Feature1': [12.0, 22.0], 'Feature2': [20.0, 18.0],
                                   'Feature3': [34, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, drop=True)))

    def test_nnd_mean_two_null_rows_mean_approach(self):
        """
        Two null rows mean approach.
        :return: True if both nulls were replaced by mean.
        """
        correct_numeric_no_null = {'Feature1': [12.0, 23.0, 19.0, 22.0],
                                   'Feature2': [20.0, 19.0, 19.0, 18.0],
                                   'Feature3': [34, 84, 10, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, strategy='mean')))

    def test_nnd_median_two_null_rows_median_approach(self):
        """
        Two null rows median approach.
        :return: True if both nulls were replaced by median.
        """
        correct_numeric_no_null = {'Feature1': [12.0, 23.0, 22.0, 22.0],
                                   'Feature2': [20.0, 19.0, 19.0, 18.0],
                                   'Feature3': [34, 84, 10, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, strategy='median')))

    def test_nnd_keep_one_row(self):
        """
        Two null rows with explicitly keeping one of them.
        :return: True if the specified row was kept and the null there was replaced by the median.
        """
        correct_numeric_no_null = {'Feature1': [12.0, 23.0, 22.0], 'Feature2': [20.0, 19.0, 18.0],
                                   'Feature3': [34, 84, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, drop=True, keep_rows=[1])))

    def test_nnd_keep_two_rows(self):
        """
        Three null rows with explicitly keeping two of them.
        :return: True if the specified rows were kept and the nulls there were replaced by the
        median.
        """
        correct_numeric_no_null = {'Feature1': [12.0, 23.0, 17.5], 'Feature2': [20.0, 19.5, 19.0],
                                   'Feature3': [34.0, 84.0, 10.0]}

        initial_data = pd.DataFrame(self.numeric_three_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, drop=True, keep_rows=[1, 2])))

    def test_nnd_remove_one_row(self):
        """
        Two null rows with explicitly removing one of them.
        :return: True if the specified row was removed.
        """
        correct_numeric_no_null = {'Feature1': [12.0, 17.0, 22.0], 'Feature2': [20.0, 19.0, 18.0],
                                   'Feature3': [34, 10, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, strategy='median',
                                                    remove_rows=[1])))

    def test_nnd_remove_two_rows(self):
        """
        Three null rows with explicitly removing two of them.
        :return: True if the specified rows were removed.
        """
        correct_numeric_no_null = {'Feature1': [12.0, 22.0], 'Feature2': [20.0, 18.0],
                                   'Feature3': [34.0, 34.0]}

        initial_data = pd.DataFrame(self.numeric_three_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, strategy='median',
                                                    remove_rows=[1, 2])))

    def test_nnd_keep_remove_one_row_keep_conflict(self):
        """
        Two null rows with explicitly keeping one of them and removing the same one. Keeping has
        a priority.
        :return: True if the specified row was kept and the null there was replaced by the median.
        """
        correct_numeric_no_null = {'Feature1': [12.0, 23.0, 22.0], 'Feature2': [20.0, 19.0, 18.0],
                                   'Feature3': [34, 84, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, drop=True, keep_rows=[1],
                                                    remove_rows=[1])))

    def test_nnd_keep_remove_two_row_keep_conflict(self):
        """
        Two null rows with explicitly keeping both of them and removing one of them. Keeping has
        a priority.
        :return: True if the specified rows were kept and the null there was replaced by the median.
        """
        correct_numeric_no_null = {'Feature1': [12.0, 23.0, 22.0, 22.0],
                                   'Feature2': [20.0, 19.0, 19.0, 18.0],
                                   'Feature3': [34, 84, 10, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, drop=True, keep_rows=[1, 2],
                                                    remove_rows=[1])))

    def test_nnd_keep_remove_two_row_remove_conflict(self):
        """
        Two null rows with explicitly keeping one of them and removing two of them. Keeping has
        a priority.
        :return: True if the specified rows were kept, and the null there was replaced by the
        median, and one row removed.
        """
        correct_numeric_no_null = {'Feature1': [12.0, 23.0, 22.0], 'Feature2': [20.0, 19.0, 18.0],
                                   'Feature3': [34, 84, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, drop=True, keep_rows=[1],
                                                    remove_rows=[1, 2])))

    def test_nnd_keep_one_row_approach_mean(self):
        """
        Two null rows with explicitly keeping one of them.
        :return: True if the specified row was kept and the null there was replaced by the mean.
        """
        correct_numeric_no_null = {'Feature1': [12.0, 23.0, 22.0], 'Feature2': [20.0, 19.0, 18.0],
                                   'Feature3': [34, 84, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, drop=True, keep_rows=[1],
                                                    strategy='mean')))

    def test_nnd_median_two_null_rows_no_reindex(self):
        """
        Two null rows with remove approach and without reindexing.
        :return: True if rows with null values were removed and indexing was preserved.
        """
        correct_numeric_no_null = {'Feature1': [12.0, 22.0], 'Feature2': [20.0, 18.0],
                                   'Feature3': [34, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        new_index = pd.Series([0, 3])
        correct_data = correct_data.set_index([new_index], drop=True)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, drop=True,
                                                    reindex=False)))


if __name__ == '__main__':
    unittest.main()
