import numpy as np
from lin_reg_memristors import test_fit_zero_intercept_lin_model, test_fit_lin_model_with_intercept, fit_zero_intercept_lin_model, fit_lin_model_with_intercept
from gradient_descent import eggholder, gradient_eggholder, gradient_descent, plot_eggholder_function

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
cm_blue_orange = ListedColormap(['blue', 'orange'])


def task_1():
    print('---- Task 1.1 ----')
    test_fit_zero_intercept_lin_model()
    test_fit_lin_model_with_intercept()

    # Load the data from 'data/memristor_measurements.npy'
    data = np.load('data/memristor_measurements.npy')
    print(data.shape)
    
    n_memristor = data.shape[0]

    ### --- Use Model 1 (zero-intercept lin. model, that is, fit the model using fit_zero_intercept_lin_model)    
    estimated_theta_per_memristor = np.zeros(n_memristor)
    for i in range(n_memristor):
        # Implement an approprate function call
        x = 0  # TODO: change me
        y = 0  # TODO: change me
        theta = 0  # TODO: change me

        # Visualize the data and the best fit for each memristor
        plt.figure()
        plt.plot(x, y, 'ko')
        x_line = np.array([np.min(x), np.max(x)])
        y_line = theta * x_line
        plt.xlabel('Delta_R_ideal') # Expected
        plt.ylabel('Delta_R') # Achieved
        plt.title(f'Memristor {i+1}')
        plt.plot(x_line, y_line, label=f'Delta_R = {theta:.2f} * Delta_R_ideal')
        plt.legend()
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(f'plots/model_1_memristor_{i+1}.jpg', dpi=120)
        # plt.show()
        plt.close()  # Comment/Uncomment
        
    print('\nModel 1 (zero-intercept linear model).')
    print(f'Estimated theta per memristor: {estimated_theta_per_memristor}')

    ### --- Use Model 2 (lin. model with intercept, that is, fit the model using fit_lin_model_with_intercept)    
    estimated_params_per_memristor = np.zeros((n_memristor, 2))
    for i in range(n_memristor):
        # Implement an approprate function call
        x = 0  # TODO: change me
        y = 0  # TODO: change me
        theta_0, theta_1 = 0, 0  # TODO: change me
        
        # Visualize the data and the best fit for each memristor
        plt.figure()
        plt.plot(x, y, 'ko')
        x_line = np.array([np.min(x), np.max(x)])
        y_line = theta_0 + theta_1 * x_line
        plt.xlabel('Delta_R_ideal') # Expected
        plt.ylabel('Delta_R') # Achieved
        plt.title(f'Memristor {i+1}')
        plt.plot(x_line, y_line, label=f'Delta_R = {theta_0:.2f} + {theta_1:.2f} * Delta_R_ideal')
        plt.legend()
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(f'plots/model_2_memristor_{i+1}.jpg', dpi=120)
        # plt.show()
        plt.close()  # Comment/Uncomment
    
    print('\nModel 2 (linear model with intercept).')
    print(f"Estimated params (theta_0, theta_1) per memristor: {estimated_params_per_memristor}")

    # TODO: Use either Model 1 or Model 2 for the decision on memristor fault type. 
    # This should be a piece of code with if-statements and thresholds on parameters (you have to decide which thresholds make sense).

    
def task_2():
    print('\n---- Task 2 ----')

    def plot_datapoints(X, y, title, fig_name='fig.png'):
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        fig.suptitle(title, y=0.93)

        p = axs.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_blue_orange)

        axs.set_xlabel('x1')
        axs.set_ylabel('x2')
        axs.legend(*p.legend_elements(), loc='best', bbox_to_anchor=(0.96, 1.15))    

        #fig.savefig(fig_name) # TODO: Uncomment if you want to save it
        plt.close()  # Comment/Uncomment
    
    for task in [0, 1, 2]:
        print(f'---- Logistic regression task {task + 1} ----')
        if task == 0:
            # Load the data set 1 (X-1-data.npy and targets-dataset-1.npy)
            X_data = np.load('data/X-1-data.npy') # TODO: change me
            y = np.load('data/targets-dataset-1.npy') # TODO: change me

            x1_feature = np.select([X_data[:,0] >= 10], [1], default=0) # 0-> blue dot, 1->yellow dot
            x2_feature = np.select([X_data[:,1] <= 20], [1], default=0) # 0-> blue dot, 1->yellow dot
            final_feature = np.where((x1_feature == 1) & (x2_feature == 1), 1, 0)


            X = np.concatenate((X_data, final_feature.reshape(-1, 1)), axis=1) # create the design matrix based on the features in X_data

        elif task == 1:
            # Load the data set 2 (X-1-data.npy and targets-dataset-2.npy)
            X_data = np.load('data/X-1-data.npy') # TODO: change me
            y = np.load('data/targets-dataset-2.npy') # TODO: change me

            #final_feature = np.where((X_data[:,0] + X_data[:,1] < 30), 1, 0) # cut square in half (diagonally)
            final_feature = np.where((X_data[:,1] < (-1) * 0.04 * X_data[:,0] + 25), 1, 0) # approximately parabola divide with help of parabola
            X = np.concatenate((X_data, final_feature.reshape(-1, 1)), axis=1) #TODO # create the design matrix based on the features in X_data

        
        elif task == 2: 
            # Load the data set 3 (X-2-data.npy and targets-dataset-3.npy)
            X_data = X_data = np.load('data/X-2-data.npy') # TODO: change me
            y = np.load('data/targets-dataset-3.npy') # TODO: change me

            final_feature = np.where((X_data[:,1] < 0.16 * X_data[:,0]**5 + 0.23 * X_data[:,0]**4 - 0.5 * X_data[:,0]**3 - 0.68 * X_data[:,0]**2 + 0.25 * X_data[:,0] - 0.52), 1, 0)
            # entered points into curve fitter and got approximately this 5th order polynomial: 0.16x^5 + 0.23x^4 - 0.5x^3 - 0.68x^2 + 0.25x - 0.52
            X = np.concatenate((X_data, final_feature.reshape(-1, 1)), axis=1) # TODO create the design matrix based on the features in X_data
   
        plot_datapoints(X, y, 'Targets', 'plots/targets_' + str(task) + '.png')  # Uncomment to generate plots as in the exercise sheet

        # Split the data into train and test sets, using train_test_split function that is already imported 
        # We want 20% of the data to be in the test set. Fix the random_state parameter (use value 0)).
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        print(f'Shapes of: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}')

        # Create a classifier, and fit the model to the data
        clf = LogisticRegression(penalty='none') # TODO use LogisticRegression from sklearn.linear_model (already imported)
        clf.fit(X_train, y_train)
        
        acc_train = clf.score(X_train, y_train)
        acc_test = clf.score(X_test, y_test) # TODO
        print(f'Train accuracy: {acc_train * 100:.2f}. Test accuracy: {100 * acc_test:.2f}.')
        
        # Calculating the loss.
        # Calculate PROBABILITIES of predictions. Output will be with the second dimension that equals 2, because we have 2 classes. 
        # (The returned estimates for all classes are ordered by the label of classes.)
        # When calculating log_loss, provide yhat_train and yhat_test of dimension (n_samples, ). That means, "reduce" the dimension, 
        # simply by selecting (indexing) the probabilities of the positive class. 

        yhat_train_prob = clf.predict_proba(X_train)[:, 1]
        yhat_test_prob = clf.predict_proba(X_test)[:, 1]
        loss_train = log_loss(y_train, yhat_train_prob)# TODO use log_loss from sklearn.metrics (already imported)
        loss_test = log_loss(y_test, yhat_test_prob)
        print(f'Train loss: {loss_train:.4f}. Test loss: {loss_test:.4f}.')

  
        # Calculate the predictions, we need them for the plots.
        yhat_train = clf.predict(X_train) # TODO
        yhat_test = clf.predict(X_test) # TODO

        plot_datapoints(X_train, yhat_train, 'Predictions on the train set', fig_name='logreg_train' + str(task + 1) + '.png')
        plot_datapoints(X_test, yhat_test, 'Predictions on the test set', fig_name='logreg_test' + str(task + 1) + '.png')

        # TODO: Print the theta vector (and also the bias term). Hint: check Attributes of the classifier
        print("Theta vector: ", clf.coef_)
        print("Bias term: ", clf.intercept_)


def task_3():
    print('\n---- Task 3 ----')
    # Plot the function, to see how it looks like
    plot_eggholder_function(eggholder)

    x0 = np.array([0, 0]) # TODO: choose a 2D random point from randint (-512, 512)
    print(f'Starting point: x={x0}')

    # Call the function gradient_descent. Choose max_iter, learning_rate.
    x, E_list = gradient_descent(eggholder, gradient_eggholder, x0, learning_rate=0.0, max_iter=0)

    # print(f'Minimum found: f({x}) = {eggholder(x)}')
    
   # TODO Make a plot of the cost over iteration. Do not forget to label the plot (xlabel, ylabel, title).

    x_min = np.array([512, 404.2319])
    print(f'Global minimum: f({x_min}) = {eggholder(x_min)}')

    # Test 1 - Problematic point 1. See HW1, Tasks 3.6 and 3.7.
    x, y = 0, 0 # TODO: change me
    print('A problematic point: ', gradient_eggholder([x, y]))
    
    # Test 2 - Problematic point 2. See HW1, Tasks 3.6 and 3.7.
    x, y = 0, 0 # TODO: change me
    print('Another problematic point: ', gradient_eggholder([x, y]))


def main():
    task_1()
    task_2()
    task_3()


if __name__ == '__main__':
    main()
