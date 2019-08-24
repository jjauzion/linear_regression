import numpy as np


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


class MinMaxScaler():

    def __init__(self):
        self.min = None
        self.max = None
        self.range = None

    def fit(self, data):
        """
        fit scaler to data per column (parameters are column, experience are lines)
        :param data: array with parameters in columns
        """
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        self.range = self.max - self.min

    def transform(self, data):
        if not self.min or not self.max:
            raise RuntimeError("Scaler must be fitted to the data before transform.")
        normalized_data = np.zeros(data.shape)
        for i, col in enumerate(data.T):
            normalized_data[:, i] = (col - self.min) / self.range
        return normalized_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
