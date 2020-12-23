import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


class CustomAdaline(object):

    def __init__(self, n_iterations, random_state=1, learning_rate=1.5, bias=0.5):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate
        # where we store the bias
        self.bias = bias

    def fit(self, X, y):
        randomGenreator = np.random.RandomState(self.random_state)
        # generate random number for the weights
        self.weights = randomGenreator.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.weights[0] = self.bias
        scores = []
        for _ in range(self.n_iterations):
            count=0
            for idx in X:
                # Compute the predicted output value y^(i)
                activation_function_output = self.activation_function(self.net_input(idx))
                errors = y - activation_function_output
                # The accuracy test
                scores.append(1 if y[count] == self.predict(idx) else 0)
                # Update the "weight update" value
                self.weights[0] = self.weights[0] + self.learning_rate * errors.sum()
                # Update the weight coefficients by the accumulated "weight update" values
                self.weights[1:] = self.weights[1:] + self.learning_rate * X.T.dot(errors)
                count += 1
        print("performance of test data: ", np.sum(scores) / len(scores))

    def net_input(self, X):
        weighted_sum = np.dot(X, self.weights[1:]) + self.weights[0]
        return weighted_sum

    def activation_function(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation_function(self.net_input(X)) >= 0.0, 1, 0)

    def score(self, X, y):
        misclassified_data_count = 0
        for xi, target in zip(X, y):
            output = self.predict(xi)
            if (target != output):
                misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count) / total_data_count
        return self.score_


# Read data from file 'filename.csv'
data = pd.read_csv("wpbc.csv")

# Replace '?' to zero
data = data.replace(to_replace='?', value=0)

# Turn data to data frame and seperate between data and recurrent
Xdf = data.iloc[0:, 1:]
Ydf = data.iloc[:, 0:1]

# Turn matrices to arrays as well as convering the string to floats
X = Xdf.to_numpy().astype(np.float)
y = Ydf.to_numpy()

z = []
# Changing 'R' to 1 and 'N' to 0
for element in y:
    if element == 'R':
        z.append(1)
    else:
        z.append(0)

# converting list to array of floats
y = np.array(z).astype(np.float)


start = time.time()
# errors validation
scores = []
cv = KFold(n_splits=3, random_state=None, shuffle=True)
for train_index, test_index in cv.split(X):
    # Create training and test split
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
 # Instantiate CustomAdaline
    adaline = CustomAdaline(n_iterations=1000)
    # Fit the model
    adaline.fit(X_train, y_train)
    # Score the model
    scores.append(adaline.score(X_test,y_test))
    print("accuracy", adaline.score(X_test,y_test))
    print("%s seconds to train and test model" % (time.time() - start))

print("Adaline classification mean accuracy: %.2f%% standard deviation: (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))







