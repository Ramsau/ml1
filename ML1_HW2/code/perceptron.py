import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs, make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron as SkPerceptron
from sklearn.metrics import mean_squared_error


class Perceptron:
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.w = None

    @staticmethod
    def f(a): # heaviside
        if a < 0:
            return 0
        else:
            return 1

    def fit(self, x_train, y_train):
        assert x_train.shape[0] == y_train.shape[0], "x and y should have the same number of rows"
        self._fit(x_train, y_train)
        assert self.w.shape == (x_train.shape[1], 1)
        return self
    

    def predict(self, x):
        assert x.shape[1] == self.w.shape[0]
        y_predictions = self._predict(x)
        y_predictions = np.array(y_predictions)
        assert y_predictions.shape[0] == x.shape[0], "Predictions should have the same number of rows as the input x"
        assert np.bitwise_or(y_predictions == 0, y_predictions == 1).all(), "predictions have to be 0 or 1"
        return y_predictions

    def _fit(self, x_train, y_train):
        ## TODO
        self.w = np.zeros((x_train.shape[1], 1)) # will result in two weights (for two features in x_train)

        #x_train_test = x_train # 100 samples in load_data x

        iter = 1
        z_vector = np.zeros(x_train.shape[0])

        while iter <= self.max_iter:

            for i in range(x_train.shape[0]):
                a = np.dot(np.transpose(self.w), x_train[i])
                z = self.f(a)
                z_vector[i] = z

                if (z != y_train[i]):
                    # misclassified: update weights according to w:=w+η(y(i) −z) * x(i)
                    for j in range(self.w.shape[0]):
                        self.w[j] += self.learning_rate * (y_train[i] - z) * x_train[i][j]


            
            if (z_vector == y_train).all():
                print(f"ALL CLASSIFIED CORRECTLY after {iter} iterations")
            if iter >= self.max_iter or (z_vector == y_train).all():
                break
            iter += 1
        
        incorrect_indices = np.where(z_vector != y_train)[0]
        incorrect_values = z_vector[incorrect_indices]
        correct_values = y_train[incorrect_indices]

        print(f"Incorrectly classified indices: {incorrect_indices}")
        print(f"Corresponding values in z: {incorrect_values}")
        print(f"Corresponding values in y_train: {correct_values}")

        # while iter <= self.max_iter:

        #     for i in range(x_train_test.shape[0]):
        #         a = np.dot(np.transpose(self.w), x_train_test[i])
        #         z = self.f(a)

        #         if (z != y_train[i]):
        #             # misclassified: update weights according to w:=w+η(y(i) −z) * x(i)
        #             for j in range(self.w.shape[0]):
        #                 self.w[j] += self.learning_rate * (y_train[i] - z) * x_train_test[i][j]
        #         elif (z == y_train[i]):
        #             # classified correctly: remove sample x_train_test[i] from x_train_test
        #             x_train_test = np.delete(x_train_test, i, axis=0)

                
        #         if iter > self.max_iter:
        #             break
        #     iter += 1



    def _predict(self, x):
        ## TODO
        predictions = []
        for i in range(x.shape[0]):
            a = np.dot(self.w.T, x[i])
            z = self.f(a)
            predictions.append(z)

        return predictions
    


def load_data():
    x, y = make_blobs(n_features=2, centers=2, random_state=3)
    assert np.bitwise_or(y == 0, y == 1).all()
    return x, y


def load_non_linearly_separable_data():
    """
    Generates non-linearly separable data and returns the samples and class labels
    :return:
    """
    x, y = make_gaussian_quantiles(n_features=2, n_classes=2, random_state=1)
    assert np.bitwise_or(y == 0, y == 1).all()
    return x, y


def plot_data(x, y):
    plt.figure()
    plt.title("Two linearly-separable classes", fontsize='small')
    plt.scatter(x[:, 0], x[:, 1], marker='o', c=y)
    plt.show()


def plot_decision_boundary(perceptron, x, y):
    dim1_max, dim1_min = np.max(x[:, 0]), np.min(x[:, 0])
    dim2_max, dim2_min = np.max(x[:, 1]), np.min(x[:, 1])
    dim1_vals, dim2_vals = np.meshgrid(np.arange(dim1_min, dim1_max, 0.1),
                                       np.arange(dim2_min, dim2_max, 0.1))
    y_vals = perceptron.predict(np.c_[dim1_vals.ravel(), dim2_vals.ravel()])
    y_vals = y_vals.reshape(dim1_vals.shape)

    plt.figure()
    plt.title("Two linearly-separable classes with decision boundary", fontsize='small')
    plt.contourf(dim1_vals, dim2_vals, y_vals, alpha=0.4)
    plt.scatter(x[:, 0], x[:, 1], marker='o', c=y) 

def classification_error(targets, predictions):
    misclassified = np.sum(targets != predictions)
    total_samples = len(targets)
    classification_error = misclassified / total_samples
    return classification_error


def main():
    #x, y = load_data()
    x, y = load_non_linearly_separable_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    #print(f'Shapes: {x_train.shape}, {y_train.shape}')

    learning_rate = 0.0001
    n_iter = 10000

    print("Sklearn perceptron\n")
    # Perceptron from sklearn
    perceptron = SkPerceptron(alpha=learning_rate, max_iter=n_iter, fit_intercept=False)
    perceptron.fit(x_train, y_train)
    train_mse = mean_squared_error(y_train, perceptron.predict(x_train))
    test_mse = mean_squared_error(y_test, perceptron.predict(x_test))
    print("Training MSE:", train_mse)
    print("Testing MSE: ", test_mse)
    plot_decision_boundary(perceptron, x, y)

    train_error = classification_error(y_train, perceptron.predict(x_train))
    test_error = classification_error(y_test, perceptron.predict(x_test))
    print(f"train error: {train_error}")
    print(f"test error: {test_error}")
    plt.savefig("plots/SKperceptron_LR0.0001_ITER10000_dataset_load_non_linearly_separable_data.png")

    print("\n\nOur perceptron\n")
    # Your own perceptron
    perceptron = Perceptron(learning_rate=learning_rate, max_iter=n_iter)
    perceptron.fit(x_train, y_train)
    train_mse = mean_squared_error(y_train, perceptron.predict(x_train))
    test_mse = mean_squared_error(y_test, perceptron.predict(x_test))
    print("Training MSE:", train_mse)
    print("Testing MSE: ", test_mse)
    plot_decision_boundary(perceptron, x, y)
    plt.savefig("plots/our_perceptron_LR0.0001_ITER10000_dataset_load_non_linearly_separable_data.png")
    plt.show()
    

    train_error = classification_error(y_train, perceptron.predict(x_train))
    test_error = classification_error(y_test, perceptron.predict(x_test))
    print(f"train error {train_error}")
    print(f"test error: {test_error}")


if __name__ == '__main__':
    main()
