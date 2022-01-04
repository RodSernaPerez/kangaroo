import scipy
from sklearn.metrics import r2_score
import numpy as np


class OMP:
    def __init__(self, K: int):
        self.K = K

        self.coef_ = None
        self.indices = []
        self.res = None
        self.intercept = None

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("Number of rows in X and in Y must be the same")
        if len(y) < self.K:
            raise AttributeError("The number of not nulls coefficients must be less than the number of samples")
        self.indices = []
        self.res = None
        self.intercept = np.mean(y)

        indices, coefs, _ = self._compute_weights(X, y)
        self.coef_ = np.zeros(shape=(X.shape[1]))
        self.coef_[indices] = coefs

    def _estimate_weights(self, X, y):
        coefs = np.matmul(np.linalg.pinv(X[:, self.indices]), y)
        return coefs

    def _compute_weights(self, X, y):
        y = np.asarray(y) - self.intercept
        while len(self.indices) < self.K:
            res = y if not len(self.indices) else self.res

            c = np.matmul(X.T, res)
            c[self.indices] = 0

            i = self._atom_selection_function(c)
            self.indices.append(i)

            coefs = np.matmul(np.linalg.pinv(X[:, self.indices]), y)
            self.res = y - np.matmul(X[:, self.indices], coefs)
        return self.indices, coefs, np.linalg.norm(res)

    def _atom_selection_function(self, c):
        return np.argmax(np.abs(c)**2)

    def predict(self, X):
        return np.matmul(X, self.coef_) + self.intercept

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def improve_till(self, k, X, y):
        if k <= self.K:
            return None
        else:
            self.K = k
            indices, coefs, _ = self._compute_weights(X, y)
            self.coef_ = np.zeros(shape=(X.shape[1]))
            self.coef_[indices] = coefs



class ProbOMP(OMP):
    def __init__(self, K: int, n_fittings: int = 3):
        super(ProbOMP, self).__init__(K)
        self.n_fittings = n_fittings

    def _atom_selection_function(self, c):
        x = scipy.special.softmax(np.abs(c)**2)
        return np.random.choice(np.arange(0, len(c)), p=x)

    def fit(self, X, y):
        best_score = np.linalg.norm(y)
        best_weigths = np.inf
        best_indices = None

        for n in range(self.n_fittings):
            indices, weights, error = self._compute_weights(X, y)

            if error < best_score:
                best_score = error

                best_weigths = weights
                best_indices = indices

        self.coef_ = np.zeros(shape=(X.shape[1]))
        self.coef_[best_indices] = best_weigths

