import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import math

from . import scaler


class LinearRegression:

    @staticmethod
    def data_augmentation(x):
        return np.vstack((x[:, -1],
                          x[:, -1] ** 2
                          )).T
        # return np.vstack((x[:, -1], np.array(list(map(lambda val: math.exp(val / 10000), x[:, -1]))))).T

    def __init__(self, verbose=1):
        self.rmse = -1
        self.mae = -1
        self.weight = None
        self.scaler = None
        self.nb_iter = 0
        self.learning_rate = 0
        self.cost_history = []
        self.X = None
        self.X_original = None
        self.y = None
        self.verbose = verbose
        self.augmented_data = False
        self.y_pred = None

    def plot_training(self):
        fig = plt.figure("Training synthesis")
        plt.subplot(121)
        start = 500
        plt.plot(range(start, len(self.cost_history)), self.cost_history[start:])
        plt.title("Cost history", )
        plt.xlabel("nb of iterations")
        plt.ylabel("Cost")
        plt.subplot(122)
        pred = np.sort(np.hstack((self.X_original, self.y_pred)), axis=0)
        plt.plot(self.X_original, self.y, 'xr', pred[:, 0], pred[:, 1], 'b')
        plt.title("Training dataset")
        plt.xlabel("mileage")
        plt.ylabel("price")
        plt.show()

    def plot_prediction(self, mileage, prediction):
        if self.X_original is None:
            return False
        plt.scatter(self.X_original[:, 0], self.y, c='k', marker='.', label="training Dataset")
        pred = np.sort(np.hstack((self.X_original, self.y_pred)), axis=0)
        plt.plot(mileage, prediction, 'xg', pred[:, 0], pred[:, 1], 'r')
        plt.legend(("prediction", "polyfit line", "train dataset"))
        plt.show()
        return True

    def load_data_from_csv(self, csv_file, y_col="first", remove_header=False, data_augmentation=False):
        """
        load data from a comma (',') separated file where columns are parameters and line experience
        :param csv_file:
        :param y_col: "first" or "last". Tell whether the column with the data we want to predict is in 1st or last position
        :param remove_header:
        :param data_augmentation: if true, one column will be added to the data set, being the square of the last column
        :return:
        """
        if y_col != "first" and y_col != "last":
            raise ValueError("y_col shall be either 'first' or 'last'. Got {}".format(y_col))
        with Path(csv_file).open(mode='r', encoding='utf-8') as fd:
            reader = csv.reader(fd)
            try:
                data = list(reader)
                if remove_header:
                    data = data[1:]
                df = np.array(data, dtype="float64")
                if y_col == "last":
                    self.X = df[:, 0:-1]
                    self.y = df[:, -1:]
                else:
                    self.X = df[:, 1:]
                    self.y = df[:, 0:1]
            except (UnicodeDecodeError, ValueError, IndexError) as err:
                print("Error while reading data csv file ('{}') : {}".format(csv_file, err))
                exit(0)
        self.X_original = np.copy(self.X)
        if data_augmentation:
            self.X = LinearRegression.data_augmentation(self.X)
            self.augmented_data = True

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

    def train(self, nb_iter, learning_rate, scaler_type="standard", verbose=1):
        """
        :param X: m by n matrix with m=nb of experience and n=nb of params
        :param Y: m by 1 matrix
        :param nb_iter: number of iteration
        :param learning_rate: learning rate
        :param scaler_type: Define how to scale the train data set. Standard scaler by default (z = (x - avg) / std))
        :param verbose: Int. Level of verbosity: 0 = no print ; 1 = result print ; 2 = result plot
        :return: tuple (weight, cost_history) with weight as a n by 1 matrix and cost_history as a list
        """
        self.nb_iter = nb_iter
        self.learning_rate = learning_rate
        if scaler_type == "standard":
            self.scaler = scaler.MeanNormScaler()
        elif scaler_type == "minmax":
            self.scaler = scaler.MinMaxScaler()
        else:
            print("'{}' is not a valid scaler. Standard scaler will be used instead".format(scaler_type))
            self.scaler = scaler.MeanNormScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.X = np.insert(self.X, 0, np.ones(self.X.shape[0]), axis=1)
        self.weight = np.random.random((self.X.shape[1], 1))
        self.cost_history.append(self._compute_cost())
        for i in range(nb_iter):
            self.weight = self._update_weight()
            self.cost_history.append(self._compute_cost())
        self.y_pred = self._compute_hypothesis()
        self.rmse = math.sqrt(np.sum((self.y_pred - self.y) ** 2) / self.y.shape[0])
        self.mae = np.sum(abs(self.y_pred - self.y)) / self.y.shape[0]
        if verbose > 0:
            print("Training completed!")
            print("Model evaluation: RMSE = {} ; MAE = {}".format(self.rmse, self.mae))
        if verbose > 1:
            self.plot_training()

    def predict(self, x, verbose=1):
        """
        Make prediction based on x
        :param x: List or 1 by n numpy array with n = nb of parameter
        :return:
        """
        if not isinstance(x, np.ndarray):
            if not isinstance(x, list):
                raise TypeError("x shall be a list or a np array. Got {}".format(x))
            x_pred = np.array([x])
        else:
            x_pred = x
        if self.augmented_data:
            x_pred = LinearRegression.data_augmentation(x_pred)
        if self.scaler:
            x_pred = self.scaler.transform(x_pred)
        x_pred = np.insert(x_pred, 0, np.ones(x_pred.shape[0]), axis=1)
        if self.weight is None:
            self.weight = np.zeros((x_pred.shape[1], 1))
        prediction = np.matmul(x_pred, self.weight)
        if verbose == 2:
            self.plot_prediction(x[0], prediction)
        return prediction

    def plot_train_set(self):
        print(self.X_original)
        plt.scatter(self.X_original, self.y)
        plt.show()

    def save_model(self, file):
        with Path(file).open(mode='wb') as fd:
            pickle.dump(self.__dict__, fd)

    def load_model(self, file):
        with Path(file).open(mode='rb') as fd:
            try:
                model = pickle.load(fd)
            except (pickle.UnpicklingError, EOFError) as err:
                print("Can't load model from '{}' because : {}".format(file, err))
                return False
        if not isinstance(model, dict):
            print("Given file '{}' is not a valid LinearRegression model".format(file))
            return False
        for key in model.keys():
            if key not in self.__dict__.keys():
                print("Given file '{}' is not a valid LinearRegression model".format(file))
                return False
        self.__dict__.update(model)
        return True
