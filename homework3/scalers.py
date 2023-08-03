import numpy as np


class MinMaxScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.minim = data.min(axis=0)
        self.max_min = data.max(axis=0) - self.minim

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)

        """
        self.max_min[np.where(self.max_min == 0)] = 1
        return (data - self.minim) / self.max_min


class StandardScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.var = np.average(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        self.std[np.where(self.std == 0)] = 1
        return (data - self.var) / self.std
