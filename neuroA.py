import random
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import time


def activation_func(x):
    return np.where(x >= 0, 1, 0)

def plo(weights,bias,X_train,y_train):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-1 * weights[0] * x0_1 - bias) / (weights[1] + 0.00001)
    x1_2 = (-1 * weights[0] * x0_2 - bias) / (weights[1] + 0.00001)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()

def fit(X, y):
    n_samples, n_features = X.shape
    learning_rate = 0.001
    # init parameters
    weights = np.zeros(n_features)
    bias = 0

    y_ = np.array([1 if i > 0 else 0 for i in y])
    scores = []
    for _ in range(n_iters):

        for idx, x_i in enumerate(X):

            linear_output = np.dot(x_i, weights) + bias
            y_predicted = activation_func(linear_output)

            # Perceptron update rule
            update = learning_rate * (y_[idx] - y_predicted)
            scores.append(1 if y_[idx] == y_predicted else 0)
            weights += update * x_i
            bias += update
    print("performance of test data: ", np.sum(scores)/len(scores))
    return weights


def predict(X,weights, bias):
    linear_output = np.dot(X, weights) + bias
    y_predicted = activation_func(linear_output)
    return y_predicted


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy




# Read data from file 'filename.csv'
data = pd.read_csv("wpbcErase.csv")

# Replace '?' to zero
data = data.replace(to_replace='?', value=0)

# Add bias column if needed in code below i used same bias for all dataset
# data[-1] = -1

# Turn data to data frame and separate between data and result of parameters
Xdf = data.iloc[0:, 1:]
Ydf = data.iloc[:, 0:1]

# Turn matrices to arrays as well as converting the X matrix string to floats
X = Xdf.to_numpy().astype(np.float)
y = Ydf.to_numpy()

# Changing 'R' to 1 and 'N' to 0
z = []
for element in y:
    if element == 'R':
        z.append(1)
    else:
        z.append(0)

# Override y matrix to numbers(Z matrix) and converting list to array of floats
y = np.array(z).astype(np.float)


start_time = time.time()
# cross validation
scores = []
cv = KFold(n_splits=3, random_state=None, shuffle=True)
for train_index, test_index in cv.split(X):

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    learning_rate = 0.001
    n_iters = 1000
    weights = np.zeros(len(X[0]))
    bias = -1
    w = fit(X_train, y_train)
    predictions = predict(X_test,w,-1)
    print("accuracy", accuracy(y_test, predictions))
    print("%s seconds to train and test model" % (time.time() - start_time))
    scores.append(accuracy(y_test, predictions))
print("Perceptron classification mean accuracy: %.2f%% standard deviation: (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))




fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X_train[:,0], X_train[:,1],marker='o', c=y_train)

x0_1 = np.amin(X_train[:,0])
x0_2 = np.amax(X_train[:,0])


x1_1 = (-1*weights[0] * x0_1 - bias) / (weights[1]+0.00001)
x1_2 = (-1*weights[0] * x0_2 - bias) / (weights[1]+0.00001)

ax.plot([x0_1, x0_2],[x1_1, x1_2], 'k')

ymin = np.amin(X_train[:,1])
ymax = np.amax(X_train[:,1])
ax.set_ylim([ymin-3,ymax+3])

plt.show()
