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
    accuracy = np.average(abs(final_hyp - Y))
    return weight, cost_history, accuracy


def predict(mileage, weight, average, std_dev):
    x = (mileage - average) / std_dev
    price = weight[0] + x * weight[1]
    return price


if __name__ == "__main__":
    data_file = "../data.csv"
    with Path(data_file).open(mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = list(reader)
    df = np.array(data[1:], dtype="float64")
    X = df[:, 0:1]
    Y = df[:, 1:2]
    X, avg, std_dev = mean_normalization(X)
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
