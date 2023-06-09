import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import plotting
from datasets import get_toy_dataset

if __name__ == '__main__':
  for idx in [1, 2, 3]:
    X_train, X_test, y_train, y_test = get_toy_dataset(idx)
    if idx == 1:
            param_grid = [{'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
    else:
        param_grid = [{'kernel': ['linear', 'rbf'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

        
    
    svc = SVC(tol=1e-4)
    # TODO perform grid search, decide on suitable parameter ranges and state sensible parameter ranges in your report
    clf = GridSearchCV(svc, param_grid, cv=5)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print("Test Score:", test_score)
    print(f"Dataset {idx}: {clf.best_params_}")
    # TODO plot and save decision boundaries

    print("Test Score:", test_score)
    print(f"Dataset {idx}: {clf.best_params_}")
    
    # Plot and save decision boundaries
    plotting.plot_decision_boundary(X_test, clf)
    plotting.plot_dataset(X_train, X_test, y_train, y_test)
    plt.show()
