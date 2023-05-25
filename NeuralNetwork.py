import numpy as np


class NeuralNetwork:
    accuracy = 0

    def __init__(self, x, y):
        self.input = x
        self.weights_1 = np.random.rand(self.input.shape[1], 5)
        self.weights_2 = np.random.rand(5, 5)
        self.weights_3 = np.random.rand(5, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights_1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights_2))
        self.output = sigmoid(np.dot(self.layer2, self.weights_3))

    def backprop(self):

        d_weights_3 = np.dot(self.layer2.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights_2 = np.dot(self.layer1.T, (d_weights_3 * sigmoid_derivative(np.dot(self.weights_2, self.layer1.T))).T)
        d_weights_1 = np.dot(self.input.T, np.dot(d_weights_2, sigmoid_derivative(np.dot(self.weights_1.T, self.input.T))).T)

        # d_weights_2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        # d_weights_1 = np.dot(self.input.T,
        #                      (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
        #                              self.weights_2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights_1 += d_weights_1
        self.weights_2 += d_weights_2
        self.weights_3 += d_weights_3

    def evaluate(self, testing_x, testing_y):
        self.input = testing_x
        self.y = testing_y
        self.feedforward()
        for index in range(self.y.size):
            if int(self.output[index]) == self.y[index]:
                NeuralNetwork.accuracy += 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)