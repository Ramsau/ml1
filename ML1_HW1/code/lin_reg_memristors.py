import numpy as np
from numpy.linalg import pinv  # if you decide to implement the equations using the matrix notation


def test_fit_zero_intercept_lin_model():
    # TODO: bonus task
    # Implement two test cases that test your implementation of the function fit_zero_intercept_lin_model. Use assert command for that.
    return 0


def test_fit_lin_model_with_intercept():
    # TODO: bonus task
    # Implement two test cases that test your implementation of the function fit_lin_model_with_intercept. Use assert command for that.
    return 0


def fit_zero_intercept_lin_model(x, y):
    """
    :param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    :param y: y coordinates of data points (i.e., \Delta R_i$)
    :return: theta 
    """

    upper = 0
    lower = 0

    for (x_e, y_e) in zip(x, y):
        upper += x_e * y_e
        lower += x_e * x_e

    theta = upper / lower
    return theta


def fit_lin_model_with_intercept(x, y):
    """
    :param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    :param y: y coordinates of data points (i.e., \Delta R_i$)
    :return: theta_0, theta_1 
    """

    sum_val_ideal = np.sum(x)
    sum_val = np.sum(y)
    m = len(x)
    sum_val_ideal_squared = 0
    sum_val_ideal_val = 0
    for (val_ideal, val) in zip(x, y):
        sum_val_ideal_squared += val_ideal * val_ideal
        sum_val_ideal_val += val_ideal * val

    theta_0 = ((sum_val_ideal_val / sum_val_ideal_squared) - (sum_val / sum_val_ideal)) /\
        ((sum_val_ideal / sum_val_ideal_squared) / (m / sum_val_ideal))

    theta_1 = (sum_val - m * theta_0) / sum_val_ideal
    return theta_0, theta_1 


