import matplotlib.pyplot as plt

import plotting
from datasets import get_toy_dataset
from task2_1 import LinearSVM
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
  X_train, X_test, y_train, y_test = get_toy_dataset(1, remove_outlier=True)
  svm = LinearSVM()
  # TODO use grid search to find suitable parameters!

  parameters = {
        "max_iter": [1, 10, 100, 200, 500, 1000, 5000],
        "eta": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 200.0, 500.0],
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 200.0, 500.0]
    }

  print("Shape X_train: ")
  print(X_train.shape)
  print("Shape y_train: ")
  print(y_train.shape)

  clf = GridSearchCV(svm, parameters, n_jobs=-1)
  clf.fit(X_train, y_train)

  # TODO Use the parameters you have found to instantiate a LinearSVM.
  # the `fit` method returns a list of scores that you should plot in order
  # to monitor the convergence. When does the classifier converge?
  print(clf.best_params_)
  best_C = clf.best_params_.get('C')
  best_eta = clf.best_params_.get('eta')
  best_max_iter = clf.best_params_.get('max_iter')
  svm = LinearSVM(best_C, best_eta, best_max_iter)
  scores = svm.fit(X_train, y_train)
  plt.plot(scores)
  test_score = clf.score(X_test, y_test)
  print(f"Test Score: {test_score}")
  plt.show()

  # TODO plot the decision boundary!

  plotting.plot_decision_boundary(X_test, svm)
  plt.show()
