import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import plotting
from datasets import get_toy_dataset

if __name__ == '__main__':
  print("n_extimators = 1")
  for idx in [1, 2, 3]:
    X_train, X_test, y_train, y_test = get_toy_dataset(idx)
    # start with `n_estimators = 1`
    rf = RandomForestClassifier(n_estimators=1)
    clf = GridSearchCV(rf, {"max_depth": [i for i in range(1, 21)]})
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"Dataset {idx}: {clf.best_params_}")
    print("Test Score:", test_score)
    # plot decision boundary
    plotting.plot_decision_boundary(X_train, clf)
    plotting.plot_dataset(X_train, X_test, y_train, y_test)
    plt.savefig(f"../plots/randomtree_nest1_{idx}.png")
    plt.show()

print("\nn_estimators = 100")
for idx in [1, 2, 3]:
  X_train, X_test, y_train, y_test = get_toy_dataset(idx)
  # start with `n_estimators = 1`
  rf = RandomForestClassifier(n_estimators=100)
  clf = GridSearchCV(rf, {"max_depth": [i for i in range(1, 21)]})
  clf.fit(X_train, y_train)
  test_score = clf.score(X_test, y_test)
  print(f"Dataset {idx}: {clf.best_params_}")
  print("Test Score:", test_score)
  # plot decision boundary
  plotting.plot_decision_boundary(X_train, clf)
  plotting.plot_dataset(X_train, X_test, y_train, y_test)
  plt.savefig(f"../plots/randomtree_nest100_{idx}.png")
  plt.show()
