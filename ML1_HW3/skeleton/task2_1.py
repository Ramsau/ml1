import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator

import plotting
from datasets import get_toy_dataset


def loss(w, b, C, X, y):
  # TODO: implement the loss function (eq. 1)
  # useful methods: np.sum, np.clip
  hinge_loss = np.maximum(0, 1 - y * (np.dot(X, np.transpose(w)) + b))
  regularization_term = 0.5 * w@w
  np.inf = regularization_term + C * np.sum(hinge_loss)

  return np.inf


def grad(w, b, C, X, y):
  # TODO: implement the gradients with respect to w and b.
  # useful methods: np.sum, np.where, numpy broadcasting

  raw_hinge_loss = 1 - y * (np.dot(X, w) + b)
  #mask_b = np.where(raw_hinge_loss > 0, C*(-1)*y, 0) # weird plot
  mask_b = np.where(raw_hinge_loss > 0, C*1, 0)
  grad_b = np.sum(mask_b)

  # mask_w = np.where(raw_hinge_loss > 0, w + (-1) * C *  y[:, np.newaxis] * X, w)
  # grad_w = np.sum(mask_w, axis=0)
  # doesnt work

  grad_w = w.copy()
  for i in range(X.shape[0]):

    if(1 - y[i] * (X[i].dot(w) + b) > 0):
      grad_w += w + (-1) * C*y[i]* X[i]
    else:
      grad_w += w

  return grad_w, grad_b


class LinearSVM(BaseEstimator):

  def __init__(self, C=1, eta=1e-3, max_iter=1000):
    self.C = C
    self.max_iter = max_iter
    self.eta = eta

  def fit(self, X, y):
    # TODO: initialize w and b. Does the initialization matter?
    # convert y: {0,1} -> -1, 1
    y = np.where(y == 0, -1, 1)
    self.w = np.random.normal(size=X.shape[1])
    self.b = 0.
    loss_list = []

    for j in range(self.max_iter):
      # TODO: compute the gradients, update the weights, compute the loss
      grad_w, grad_b = grad(self.w, self.b, self.C, X, y)
      self.w -= grad_w * self.eta # update weights
      self.b -= grad_b * self.eta
      loss_list.append(loss(self.w, self.b, self.C, X, y))

    return loss_list

  def predict(self, X):
    # TODO: assign class labels to unseen data
    y_pred = np.empty(len(X))  # to store the predicted class labels for each data point
    for i in range(len(X)):
      prediction_value = np.dot(self.w, X[i].T) + self.b
      if (prediction_value >= 0):
        y_pred[i] = 1
      else:
        y_pred[i] = -1
    # converting y_pred from {-1, 1} to {0, 1}
    return np.where(y_pred == -1, 0, 1)

  def score(self, X, y):
    # TODO: IMPLEMENT ME
    y_pred = self.predict(X)
    return np.mean(y_pred == y)
