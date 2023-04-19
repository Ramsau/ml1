import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_eggholder_function(f):
    '''
    Plotting the 3D surface of a given cost function f.
    :param f: The function to visualize
    :return:
    '''
    n = 1000
    bounds = [-512, 512]

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    x_ax = np.linspace(bounds[0], bounds[1], n)
    y_ax = np.linspace(bounds[0], bounds[1], n)
    XX, YY = np.meshgrid(x_ax, y_ax)

    ZZ = np.zeros(XX.shape)
    ZZ = f([XX, YY])

    ax.plot_surface(XX, YY, ZZ, cmap='jet')
    plt.show()


def gradient_descent(f, df, x, learning_rate, max_iter):
    """
    Find the optimal solution of the function f(x) using gradient descent:
    Until the max number of iteration is reached, decrease the parameter x by the gradient times the learning_rate.
    The function should return the minimal argument x and the list of errors at each iteration in a numpy array.

    :param f: function to minimize
    :param df: function representing the gradient of f
    :param x: vector, initial point
    :param learning_rate:
    :param max_iter: maximum number of iterations
    :return: x (solution, vector), E_list (array of errors over iterations)
    """

    E_list = np.zeros(max_iter)
    # Implement the gradient descent algorithm
    # E_list should be appended in each iteration, with the current value of the cost
    for i in range(max_iter):
        E_list[i] = f(x) # update E_list with the current value of the cost f(x)
        gradient = df(x)
        x[0] -= learning_rate * gradient[0] # learning_rate * gradient -> step size along the direction of steepest ascent. We subtract this quantity from x[0] to move x[0] in the direction of steepest descent
        x[1] -= learning_rate * gradient[1] # same as above
        
    
    return x, E_list


def eggholder(x):
    # Implement the cost function specified in the HW1 sheet
    # TODO: change me
    z = (-1) * (x[1] + 47) * np.sin(np.sqrt(np.abs(x[0] / 2 + x[1] + 47))) - x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))
    return z


def gradient_eggholder(x):
    # Implement gradients of the Eggholder function w.r.t. x and y
    sqrt = np.sqrt(x[0] - x[1] - 47)
    sqrt_abs = np.sqrt(np.abs(x[0] / 2 + x[1] + 47))
    cos_sqrt = np.cos(sqrt)
    cos_sqrt_abs = np.cos(sqrt_abs)
    sin_sqrt = np.sin(sqrt)
    sin_sqrt_abs = np.sin(sqrt_abs)

    first_block = -(((x[1] + 47) * (x[0]/2 + x[1] + 47) * cos_sqrt_abs) / (2 * np.float_power(np.abs(x[0]/2 + x[1] + 47), 3/2)))
    last_block = ((x[0] * cos_sqrt) / (2 * sqrt))

    grad_x = (first_block / 2) - sin_sqrt - last_block
    grad_y = first_block - sin_sqrt_abs + last_block

    return np.array([grad_x, grad_y])


def generic_GD_solver(x):
    # TODO: bonus task
    return 0
