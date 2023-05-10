import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def reduce_dimension(features, n_components):
    """
    :param features: Data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality. Shape: (n_samples, n_components)
    """

    pca = PCA(n_components=n_components, random_state=1)

    X_reduced = pca.fit(features).transform(features)

    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f'Explained variance: {explained_var}')
    return X_reduced

def train_nn(features, targets):
    """
    Train MLPClassifier with different number of neurons in one hidden layer.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    n_hidden_neurons = [2, 10, 100, 200]

    for n_hid in n_hidden_neurons:
        mlp = MLPClassifier(hidden_layer_sizes=(n_hid, ), max_iter=500, solver='adam', random_state=1)
        mlp.fit(X_train, y_train)

        train_acc = mlp.score(X_train, y_train)
        test_acc = mlp.score(X_test, y_test)
        loss = mlp.loss_
        print(f'Number of hidden neurons: {n_hid}')
        print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')

def train_nn_with_regularization(features, targets):
    """
    Train MLPClassifier using regularization.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    n_hidden_neurons = [2, 10, 100, 200]

    for n_hid in n_hidden_neurons:
        print(f'\nNumber of hidden neurons: {n_hid}')

        # alpha, early stopping, alpha + early stopping
        # prediction: third works best
        # result: first is best
        mlps = [
            MLPClassifier(hidden_layer_sizes=(n_hid, ), max_iter=500, solver='adam', random_state=1, alpha=0.1),
            MLPClassifier(hidden_layer_sizes=(n_hid, ), max_iter=500, solver='adam', random_state=1, early_stopping=True),
            MLPClassifier(hidden_layer_sizes=(n_hid, ), max_iter=500, solver='adam', random_state=1, alpha=0.1, early_stopping=True)
            ]

        for i in range(3):

            mlps[i].fit(X_train, y_train)

            train_acc = mlps[i].score(X_train, y_train)
            test_acc = mlps[i].score(X_test, y_test)
            loss = mlps[i].loss_
            print(f'Variation {i}:')
            print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
            print(f'Loss: {loss:.4f}')


def train_nn_with_different_seeds(features, targets):
    """
    Train MLPClassifier using different seeds.
    Print (mean +/- std) accuracy on the training and test set.
    Print confusion matrix and classification report.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    seeds = [12011965, 12004184, 1234, 7, 4321] # TODO create a list of different seeds of your choice
    mlps = [
        MLPClassifier(hidden_layer_sizes=(200, ), max_iter=500, solver='adam', random_state=seed, alpha=0.1)
        for seed in seeds
    ]

    train_acc_arr = np.zeros(len(seeds))
    test_acc_arr = np.zeros(len(seeds))

    for i in range(len(seeds)):
        mlps[i].fit(X_train, y_train)

        train_acc = mlps[i].score(X_train, y_train)
        test_acc =  mlps[i].score(X_test, y_test)
        loss = mlps[i].loss_
        train_acc_arr[i] = train_acc
        test_acc_arr[i] = test_acc
        print(f'Seed: {seeds[i]}')
        print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')


    train_acc_mean = np.mean(train_acc_arr)
    train_acc_std = np.std(train_acc_arr)
    train_acc_min = np.min(train_acc_arr)
    train_acc_max = np.max(train_acc_arr)
    
    test_acc_mean = np.mean(test_acc_arr)
    test_acc_std = np.std(test_acc_arr)
    test_acc_min = np.min(test_acc_arr)
    test_acc_max = np.max(test_acc_arr)

    print(f'Train accuracy overall:')
    print(f'On the train set: {train_acc_mean:.4f} +/- {train_acc_std:.4f} [{train_acc_min:.4f}:{train_acc_max:.4f}]')
    print(f'On the test set: {test_acc_mean:.4f} +/- {test_acc_std:.4f} [{test_acc_min:.4f}:{test_acc_max:.4f}]')
    # TODO: print min and max accuracy as well

    # TODO: plot the loss curve

    print("Predicting on the test set")
    plt.plot(mlps[1].loss_curve_, label=f'Loss curve for MLP with seed {seeds[1]}')
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.legend()
    plt.show()
    plt.savefig("plots/losscurve.jpg")
    y_pred = mlps[i].predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=range(10)))


def perform_grid_search(features, targets):
    """
    BONUS task: Perform GridSearch using GridSearchCV.
    Create a dictionary of parameters, then a MLPClassifier (e.g., nn, set default values as specified in the HW2 sheet).
    Create an instance of GridSearchCV with parameters nn and dict.
    Print the best score and the best parameter set.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    parameters = None # TODO create a dictionary of params

    # nn = # TODO create an instance of MLPClassifier. Do not forget to set parameters as specified in the HW2 sheet.
    # grid_search = # TODO create an instance of GridSearchCV from sklearn.model_selection (already imported) with
    # appropriate params. Set: n_jobs=-1, this is another parameter of GridSearchCV, in order to get faster execution of the code.

    # TODO call fit on the train data
    # TODO print the best score
    # TODO print the best parameters found by grid_search
