import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def mean_normalization(data):
    """

    :param data: m by n matrix
    :return:
    """
    average = np.average(data, axis=0)
    std_dev = np.std(data, axis=0)
    return (data - average) / std_dev, average, std_dev


def hypothesis(weight, X):
    """

    :param weight: n by 1 matrix, with n the number of parameter
    :param X: m by n matrix
    :return: m by 1 matrix
    """
    return np.matmul(X, weight)


def cost(weight, X, Y):
    """
    :param X: m by n matrix with m=nb of experience and n=nb of params
    :param Y: m by 1 matrix
    :param weight: n by 1 matrix
    """
    return np.sum(hypothesis(weight, X) - Y) ** 2 / (2 * X.shape[0])


def update_weight(weight, X, Y, learning_rate):
    """
    :param X: m by n matrix with m=nb of experience and n=nb of params
    :param Y: m by 1 matrix
    :param weight: n by 1 matrix
    :param learning_rate: learning rate
    :return: updated weight as a n by 1 matrix

    tmp = np.matmul(X.transpose(), hypothesis(weight, X) - Y)
    grad = learning_rate / X.shape[0] * tmp
    print("grad : \n{}".format(grad))
    new = weight - grad
    print("weight : \n{}".format(weight))
    return new
    """
    return weight - learning_rate / X.shape[0] * np.matmul(X.transpose(), hypothesis(weight, X) - Y)


def linear_regression(X, Y, nb_iter, learning_rate):
    """
    :param X: m by n matrix with m=nb of experience and n=nb of params
    :param Y: m by 1 matrix
    :param nb_iter: number of iteration
    :param learning_rate: learning rate
    :return: tuple (weight, cost_history) with weight as a n by 1 matrix and cost_history as a list
    """
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    weight = np.random.random((X.shape[1], 1))
    cost_history = []
    for i in range(nb_iter):
        cost_history.append(cost(weight, X, Y))
        weight = update_weight(weight, X, Y, learning_rate)
    final_hyp = hypothesis(weight, X)
    X, avg, std_dev = mean_normalization(X)
    accuracy = np.average(abs(final_hyp - Y))
    return weight, cost_history, accuracy


def predict(mileage, weight, average, std_dev):
    x = (mileage - average) / std_dev
    price = weight[0] + x * weight[1]
    return price

class MeanNormScaler:

    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, data):
        """
        fit scaler to data per column (parameters are column, experience are lines)
        :param data: array with parameters in columns
        """
        self.mean = np.average(data, axis=0)
        self.scale = np.std(data, axis=0)

    def transform(self, data):
        if not self.mean or not self.scale:
            raise RuntimeError("Scaler must be fitted to the data before transform.")
        normalized_data = np.zeros(data.shape)
        for i, col in enumerate(data.T):
            normalized_data[:, i] = (col - self.mean) / self.scale
        return normalized_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class LinearRegression:

    def __init__(self):
        self.accuracy = 0
        self.weight = None
        self.scaler = None
        self.nb_iter = 0
        self.learning_rate = 0
        self.cost_history = []
        self.X = None
        self.y = None
        self.verbose = False

    def load_data_from_csv(self, csv_file, y_col="first", remove_header=False):
        """
        load data from a comma (',') separated file where columns are parameters and line experience
        :param csv_file:
        :param y_col: "first" or "last". Tell whether the column with the data we want to predict is in 1st or last position
        :param remove_header:
        :return:
        """
        if y_col != "first" and y_col != "last":
            raise ValueError("y_col shall be either 'first' or 'last'. Got {}".format(y_col))
        with Path(csv_file).open(mode='r', encoding='utf-8') as fd:
            reader = csv.reader(fd)
            data = list(reader)
        if remove_header:
            data = data[1:]
        df = np.array(data[1:], dtype="float64")
        if y_col == "last":
            self.X = df[:, 0:-1]
            self.y = df[:, -1:]
        else:
            self.X = df[:, 1:]
            self.y = df[:, 0:1]

    def _compute_hypothesis(self):
        """

        :param weight: n by 1 matrix, with n the number of parameter
        :param X: m by n matrix
        :return: m by 1 matrix
        """
        return np.matmul(self.X, self.weight)

    def _compute_cost(self):
        """
        self.X: m by n matrix with m=nb of experience and n=nb of params
        self.y: m by 1 matrix
        self.weight: n by 1 matrix
        """
        return np.sum(self._compute_hypothesis() - self.y) ** 2 / (2 * self.X.shape[0])

    def _update_weight(self):
        """
        self.X: m by n matrix with m=nb of experience and n=nb of params
        self.y: m by 1 matrix
        self.weight: n by 1 matrix
        """
        return self.weight - self.learning_rate / self.X.shape[0] * \
               np.matmul(self.X.transpose(), self._compute_hypothesis() - self.y)

    def train(self, nb_iter, learning_rate):
        """
        :param X: m by n matrix with m=nb of experience and n=nb of params
        :param Y: m by 1 matrix
        :param nb_iter: number of iteration
        :param learning_rate: learning rate
        :return: tuple (weight, cost_history) with weight as a n by 1 matrix and cost_history as a list
        """
        self.nb_iter = nb_iter
        self.learning_rate = learning_rate
        self.scaler = MeanNormScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.X = np.insert(self.X, 0, np.ones(self.X.shape[0]), axis=1)
        self.weight = np.random.random((self.X.shape[1], 1))
        # self.weight = np.zeros((self.X.shape[1], 1))
        self.cost_history.append(self._compute_cost())
        for i in range(nb_iter):
            self.weight = self._update_weight()
            self.cost_history.append(self._compute_cost())
        final_hyp = self._compute_hypothesis()
        self.accuracy = np.average(abs(final_hyp - self.y))

    def predict(self, x):
        if self.scaler:
            x = self.scaler.transform(x)
        x = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
        return np.matmul(x, self.weight)


if __name__ == "__main__":
    data_file = "../data.csv"
    with Path(data_file).open(mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = list(reader)
    df = np.array(data[1:], dtype="float64")
    X = df[:, 0:1]
    Y = df[:, 1:2]
    w, cost_hist, accuracy = linear_regression(X, Y, 200, 0.05)
    print("weight:")
    print(w)
    print("cost history:")
    print(cost_hist[-10:])
    print("accuracy")
    print(accuracy)
    plt.plot(cost_hist)
    plt.show()
    price = predict(60000, w, avg, std_dev)
    print("price = {}".format(price))
