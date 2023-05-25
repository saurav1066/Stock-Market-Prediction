from unittest import TestCase
import numpy as np
import pandas as pd
import main

use_cols = ['stock_symbol', 'Total_Transactions', 'Total_Traded_Shares', 'Total_Traded_Amount',
            'Opening_Price', 'Max_Price', 'Min_Price', 'Closing_Price', 'Next_Day_Closing_Price']

df_all = pd.read_csv("stocks_data.csv", usecols=use_cols)
data_np = df_all.to_numpy()
df_sewapo = pd.read_csv("sewapo.csv", usecols=use_cols)
sewapo_stock_data = df_sewapo.to_numpy()
df_purged_sewapo = pd.read_csv("purged_sewapo.csv", usecols=use_cols)
purged_sewapo_stock_data = df_purged_sewapo.to_numpy()


class TestMain(TestCase):
    def test_normalize(self):

        input_column = np.array([[1], [2], [3], [4], [5]])
        expected_normalized_output = np.array([[0], [0.25], [0.5], [0.75], [1]])

        self.assertTrue(np.array_equal(main.normalize(input_column), expected_normalized_output))

    def test_get_data_of(self):
        input_symbol = "SEWAPO"
        expected_data_of_symbol = sewapo_stock_data
        print(expected_data_of_symbol, main.get_data_of(input_symbol))
        self.assertTrue(np.array_equal(main.get_data_of(input_symbol), expected_data_of_symbol))

    def test_purge(self):
        input_stock_data = sewapo_stock_data
        expected_purged_stock_data = purged_sewapo_stock_data
        self.assertTrue(np.array_equal(main.purge(input_stock_data), expected_purged_stock_data))