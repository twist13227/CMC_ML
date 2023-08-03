import numpy as np


class Preprocessor:
    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):
    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        # your code here
        self.features = [np.unique(X[column]) for column in X.columns]
        self.features_num = [len(feature) for feature in self.features]

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        # your code here
        res = np.zeros((X.shape[0], sum(self.features_num)), dtype=self.dtype)
        for obj in range(X.shape[0]):
            col = 0
            for i in range(len(self.features_num)):
                for j in range(self.features_num[i]):
                    if X.iloc[obj][X.columns[i]] == self.features[i][j]:
                        res[obj][col + j] = 1
                        break
                col += self.features_num[i]
        return res

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:
    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        # your code here
        self.data = []
        for column in X.columns:
            cur = {}
            for value in sorted(X[column].unique()):
                cur[value] = (np.mean(Y[X[column] == value]), len(Y[X[column] == value])/len(Y))
            self.data.append(cur)

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        # your code here
        res = np.zeros((X.shape[0], 3 * X.shape[1]), dtype=self.dtype)
        for i, info in enumerate(zip(X.columns, self.data)):
            for key, value in info[1].items():
                res[X[info[0]] == key, 3 * i] = value[0]
                res[X[info[0]] == key, 3 * i + 1] = value[1]
                res[X[info[0]] == key, 3 * i + 2] = (value[0] + a) / (value[1] + b)
        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_:(i + 1) * n_], np.hstack(
            (idx[:i * n_], idx[(i + 1) * n_:])
        )
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:
    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        # your code here
        self.objs = [SimpleCounterEncoder() for i in range(self.n_folds)]
        self.folds = [i for i in group_k_fold(len(X), self.n_folds, seed)]
        [self.objs[i].fit(X.iloc[fold[1]], Y.iloc[fold[1]]) for i, fold in enumerate(self.folds)]

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        # your code here
        res = np.zeros((X.shape[0], 3 * X.shape[1]), dtype=self.dtype)
        for i, fold in enumerate(self.folds):
            res[fold[0]] = self.objs[i].transform(X.iloc[fold[0]], a, b)
        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    # your code here
    return np.array([sum(y[x == obj]) / len(y[x == obj]) for obj in np.unique(x)])
