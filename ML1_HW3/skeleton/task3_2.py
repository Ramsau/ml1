import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from datasets import get_heart_dataset, get_toy_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
import pickle as pkl

if __name__ == '__main__':
  X_train, X_test, y_train, y_test = get_toy_dataset(4)

  #TODO fit a random forest classifier and check how well it performs on the test set after tuning the parameters,
  # report your results
  rf = RandomForestClassifier()
  rf.fit(X_train, y_train)
  score = rf.score(X_test, y_test)
  print(f"Score of Random Forest: {score}")

  #TODO fit a SVC and find suitable parameters, report your results
  svc = SVC()
  svc.fit(X_train, y_train)
  score = svc.score(X_test, y_test)
  print(f"Score of SVC: {score}")

  # TODO create a bar plot (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html#matplotlib.pyplot.barh)
  # of the `feature_importances_` of the RF classifier.
  plt.bar([i for i in range(len(rf.feature_importances_))], rf.feature_importances_)
  plt.ylabel("importance")
  plt.xlabel("feature number")
  plt.savefig("../plots/randomtree_feat_importances.png")
  plt.show()

  # TODO create another RF classifier
  # Use recursive feature elimination (https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV)
  # to automatically choose the best number of parameters
  # set `scoring = 'accuracy'` to look for the feature subset with highest accuracy
  # and fit the RFECV to the training data
  rf = RandomForestClassifier()
  rfecv = RFECV(rf, scoring='accuracy')

  # TODO use the RFECV to transform the training and test dataset -- it automatically removes the least important
  # feature columns from the datasets. You don't have to change y_train or y_test
  # Fit a SVC classifier on the new dataset. Do you see a difference in performance?
  X_train_pruned = rfecv.fit_transform(X_train, y_train)
  X_test_pruned = rfecv.transform(X_test)
  svc = SVC()
  svc.fit(X_train_pruned, y_train)
  score = svc.score(X_test_pruned, y_test)
  print(f"Score of SVC on pruned data: {score}")
