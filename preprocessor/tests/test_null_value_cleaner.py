import unittest
import pandas as pd
import numpy as np
from preprocessor import null_value_cleaner as nvc


class MyTestCase(unittest.TestCase):

    numeric_two_null = {'Feature1': [12, 23, np.nan, 22], 'Feature2': [20, np.nan, 19, 18],
                        'Feature3': [34, 84, 10, 20]}

    numeric_three_null = {'Feature1': [12, 23, np.nan, 22], 'Feature2': [20, np.nan, 19, 18],
                          'Feature3': [34, 84, 10, np.nan]}

    def test_nnd_nothing_to_remove(self):
        correct_numeric_no_null = {'Feature1': [12, 23, 34, 22], 'Feature2': [20, 21, 19, 18],
                                   'Feature3': [34, 84, 10, 20]}
        data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(data.equals(nvc.nnd(data)))

    def test_nnd_remove_one_null_row(self):
        numeric_one_null = {'Feature1': [12, 23, 34, 22], 'Feature2': [20, np.nan, 19, 18],
                            'Feature3': [34, 84, 10, 20]}
        correct_numeric_no_null = {'Feature1': [12, 34, 22], 'Feature2': [20.0, 19.0, 18.0],
                                   'Feature3': [34, 10, 20]}

        initial_data = pd.DataFrame(numeric_one_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, approach='remove')))

    def test_nnd_remove_two_null_rows(self):

        correct_numeric_no_null = {'Feature1': [12.0, 22.0], 'Feature2': [20.0, 18.0],
                                   'Feature3': [34, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, approach='remove')))

    def test_nnd_mean_two_null_rows_mean_approach(self):
        correct_numeric_no_null = {'Feature1': [12.0, 23.0, 19.0, 22.0], 'Feature2': [20.0, 19.0, 19.0, 18.0],
                                   'Feature3': [34, 84, 10, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, approach='mean')))

    def test_nnd_median_two_null_rows_median_approach(self):
        correct_numeric_no_null = {'Feature1': [12.0, 23.0, 22.0, 22.0], 'Feature2': [20.0, 19.0, 19.0, 18.0],
                                   'Feature3': [34, 84, 10, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, approach='median')))

    def test_nnd_keep_one_row(self):
        correct_numeric_no_null = {'Feature1': [12.0, 23.0, 22.0], 'Feature2': [20.0, 19.0, 18.0],
                                   'Feature3': [34, 84, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, approach='remove', keep_rows=[1])))

    def test_nnd_keep_two_rows(self):
        correct_numeric_no_null = {'Feature1': [12.0, 23.0, 17.5], 'Feature2': [20.0, 19.5, 19.0],
                                   'Feature3': [34.0, 84.0, 10.0]}

        initial_data = pd.DataFrame(self.numeric_three_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, approach='remove', keep_rows=[1, 2])))

    def test_nnd_remove_one_row(self):
        correct_numeric_no_null = {'Feature1': [12.0, 17.0, 22.0], 'Feature2': [20.0, 19.0, 18.0],
                                   'Feature3': [34, 10, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, approach='median', remove_rows=[1])))

    def test_nnd_remove_two_rows(self):
        correct_numeric_no_null = {'Feature1': [12.0, 22.0], 'Feature2': [20.0, 18.0],
                                   'Feature3': [34.0, 34.0]}

        initial_data = pd.DataFrame(self.numeric_three_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, approach='median', remove_rows=[1, 2])))

    def test_nnd_keep_remove_one_row_keep_conflict(self):
        correct_numeric_no_null = {'Feature1': [12.0, 23.0, 22.0], 'Feature2': [20.0, 19.0, 18.0],
                                   'Feature3': [34, 84, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, approach='remove', keep_rows=[1], remove_rows=[1])))

    def test_nnd_keep_remove_two_row_keep_conflict(self):
        correct_numeric_no_null = {'Feature1': [12.0, 23.0, 22.0, 22.0], 'Feature2': [20.0, 19.0, 19.0, 18.0],
                                   'Feature3': [34, 84, 10, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, approach='remove', keep_rows=[1, 2],
                                                    remove_rows=[1])))

    def test_nnd_keep_remove_two_row_remove_conflict(self):
        correct_numeric_no_null = {'Feature1': [12.0, 23.0, 22.0], 'Feature2': [20.0, 19.0, 18.0],
                                   'Feature3': [34, 84, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, approach='remove', keep_rows=[1],
                                                    remove_rows=[1, 2])))

    def test_nnd_keep_one_row_approach_mean(self):
        correct_numeric_no_null = {'Feature1': [12.0, 23.0, 22.0], 'Feature2': [20.0, 19.0, 18.0],
                                   'Feature3': [34, 84, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, approach='remove', keep_rows=[1],
                                                    keep_rows_approach='mean')))

    def test_nnd_median_two_null_rows_no_reindex(self):
        correct_numeric_no_null = {'Feature1': [12.0, 22.0], 'Feature2': [20.0, 18.0],
                                   'Feature3': [34, 20]}

        initial_data = pd.DataFrame(self.numeric_two_null)
        correct_data = pd.DataFrame(correct_numeric_no_null)

        s = pd.Series([0, 3])
        correct_data = correct_data.set_index([s], drop=True)

        self.assertTrue(correct_data.equals(nvc.nnd(initial_data, approach='remove', reindex=False)))


if __name__ == '__main__':
    unittest.main()
