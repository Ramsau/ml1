from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV


def calculate_mse(targets, predictions):
    """
    :param targets:
    :param predictions: Predictions obtained by using the model
    :return:
    """
    mse = mean_squared_error(targets, predictions) # TODO Calculate MSE using mean_squared_error from sklearn.metrics (alrady imported)
    return mse


def solve_regression_task(features, targets):
    """
    :param features:
    :param targets:
    :return: 
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    n_hidden_neurons_list = [100, 200, 300, 400] # TODO (try at least 3 different numbers of neurons)

    parameters = {
        "alpha": [0.0, 0.1, 1.0],
        "solver": ["lbfgs", "adam"],
        "activation": ["logistic", "relu"],
        'early_stopping': [True, False],
        "hidden_layer_sizes": [(100, ), (200, ), (300, ), (400, )]
    }

    # TODO: MLPRegressor, choose the model yourself

    mlp_reg = MLPRegressor(random_state=1, max_iter=500)

    grid_search = GridSearchCV(mlp_reg, parameters, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    print(f'Best score: {grid_search.best_score_:.4f}')

    print(f'Best parameters: {grid_search.best_params_}')

    
    best_mlp = grid_search.best_estimator_
    

    # Calculate predictions
    y_pred_train = best_mlp.predict(X_train) # TODO
    y_pred_test = best_mlp.predict(X_test) # TODO
    print(f'Train MSE: {calculate_mse(y_train, y_pred_train):.4f}. Test MSE: {calculate_mse(y_test, y_pred_test):.4f}')
