import numpy as np
from sklearn.base import BaseEstimator
from scipy import stats

class KNearestNeighborsClassifier(BaseEstimator):
  def __init__(self, k=1):
    self.k = k

  def fit(self, X, y):
    # store X and y
    self.X = X
    self.y = y
    return self

  def score(self, X, y):
    y_pred = self.predict(X)
    return np.mean(y_pred == y)

  def predict(self, X):
    # useful numpy methods: np.argsort, np.unique, np.argmax, np.count_nonzero
    # pay close attention to the `axis` parameter of these methods
    # broadcasting is really useful for this task!
    # See https://numpy.org/doc/stable/user/basics.broadcasting.html
    diffs = np.reshape(X, (X.shape[0], 1, X.shape[1])) - np.reshape(self.X, (1, self.X.shape[0], self.X.shape[1]))
    distances = np.linalg.norm(diffs, axis=2)

    indices = np.argsort(distances)
    nearest_labels = self.y[indices]

    modes = stats.mode(nearest_labels[:, :self.k], axis=1)
    return modes[0].squeeze()
