import numpy as np
from numpy.linalg import pinv  # if you decide to implement the equations using the matrix notation


def test_fit_zero_intercept_lin_model():
    # Implement two test cases that test your implementation of the function fit_zero_intercept_lin_model. Use assert command for that.`
    tests = np.array([
        [[100, 150], [200, 300], [300, 450], [400, 600]],
        [[100, -150], [200, -300], [300, -450], [400, -600]],
        [[100, 50], [100, -50], [200, 50], [200, -50]],
        [[100, 50], [100, 50], [200, 50], [200, 50]],
    ])
    correct_thetas = [1.5, -1.5, 0, 0.3]

    for i in range(len(correct_thetas)):
        x = tests[i, :, 0]
        y = tests[i, :, 1]
        theta = fit_zero_intercept_lin_model(x, y)
        assert(abs(theta - correct_thetas[i]) < 1e-10)
    return 0


def test_fit_lin_model_with_intercept():
    # Implement two test cases that test your implementation of the function fit_lin_model_with_intercept. Use assert command for that.
    # TODO: bonus task
    tests = np.array([
        [[100, 150], [200, 300], [300, 450], [400, 600]],
        [[100, 250], [200, 400], [300, 550], [400, 700]],
        [[100, -150], [200, -300], [300, -450], [400, -600]],
        [[100, 150], [100, 50], [200, 150], [200, 50]],
        [[100, 50], [100, 50], [200, 50], [200, 50]],
    ])
    correct_thetas = [[0, 1.5], [100, 1.5], [0, -1.5], [100, 0], [50, 0]]

    for i in range(len(correct_thetas)):
        x = tests[i, :, 0]
        y = tests[i, :, 1]
        theta_0, theta_1 = fit_lin_model_with_intercept(x, y)
        assert(abs(theta_0 - correct_thetas[i][0]) < 1e-10)
        assert(abs(theta_1 - correct_thetas[i][1]) < 1e-10)
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
        ((sum_val_ideal / sum_val_ideal_squared) - (m / sum_val_ideal))

    theta_1 = (sum_val - m * theta_0) / sum_val_ideal
    return theta_0, theta_1 


