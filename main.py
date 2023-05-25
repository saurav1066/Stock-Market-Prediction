import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Form import *
from NeuralNetwork import NeuralNetwork

use_cols = ['stock_symbol', 'Total_Transactions', 'Total_Traded_Shares', 'Total_Traded_Amount',
            'Opening_Price', 'Max_Price', 'Min_Price', 'Closing_Price', 'Next_Day_Closing_Price']

df = pd.read_csv("stocks_data.csv", usecols=use_cols)
data_np = df.to_numpy()


def normalize(data_col):
    output = np.zeros(data_col.shape)

    max_of_row = data_col[0][0]
    min_of_row = data_col[0][0]

    for row in data_col:
        if row[0] > max_of_row:
            max_of_row = row[0]
        if row[0] < min_of_row:
            min_of_row = row[0]

    for index, data in enumerate(data_col):
        output[index] = (data[0] - min_of_row) / (max_of_row - min_of_row)

    return output


def get_data_of(stock_symbol):
    symbol = np.asarray(data_np[:, [0]])
    indices = [i for i, x in enumerate(symbol) if x == stock_symbol]
    return data_np[indices]


def purge(data_of_company):
    data = np.asarray(data_of_company)
    indices = [i for i, x in enumerate(data[:, [7]]) if x != 0]
    return data_of_company[indices]


def main():
    stock_symbol, training_data, output_graph = setData()

    data_of_symbol = get_data_of(stock_symbol)

    data_of_symbol = purge(data_of_symbol)

    input_x_tt = data_of_symbol[:, [1]]
    input_x_tts = data_of_symbol[:, [2]]
    input_x_tta = data_of_symbol[:, [3]]
    input_x_op = data_of_symbol[:, [4]]
    input_x_max_p = data_of_symbol[:, [5]]
    input_x_min_p = data_of_symbol[:, [6]]
    input_x_cp = data_of_symbol[:, [7]]

    closing_price = data_of_symbol[:, [7]]
    next_day_closing_price = data_of_symbol[:, [8]]
    diff = closing_price - next_day_closing_price
    y = (diff < 0).astype(int)

    normalized_x_tt = normalize(input_x_tt)
    normalized_x_tts = normalize(input_x_tts)
    normalized_x_tta = normalize(input_x_tta)
    normalized_x_op = normalize(input_x_op)
    normalized_x_max_p = normalize(input_x_max_p)
    normalized_x_min_p = normalize(input_x_min_p)
    normalized_x_cp = normalize(input_x_cp)

    normalized_x = np.concatenate((normalized_x_tt, normalized_x_tts, normalized_x_tta,
                                   normalized_x_op, normalized_x_max_p, normalized_x_min_p, normalized_x_cp), axis=1)

    split = int((training_data / 100) * len(y))
    train_x, train_y = normalized_x[:split, :], y[:split, :]
    test_x, test_y = normalized_x[split:, :], y[split:, :]

    neural_net = NeuralNetwork(train_x, train_y)
    for _ in range(150):
        neural_net.feedforward()
        neural_net.backprop()

    neural_net.evaluate(test_x, test_y)
    # print(NeuralNetwork.accuracy / test_y.size)
    output = (NeuralNetwork.accuracy / test_y.size)
    if output_graph == 1:
        plt.plot(input_x_cp)
        plt.show()
    popupmsg(output)


if __name__ == '__main__':
    main()
