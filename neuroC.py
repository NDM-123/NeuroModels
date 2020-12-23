import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import time
from sklearn.model_selection import train_test_split, KFold

data = pd.read_csv("wpbc.csv")

# Replace '?' to zero
data = data.replace(to_replace='?', value=0)


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

# Init an array that equals to a row in X
weights = np.zeros(len(X[0]))

start = time.time()
# crros validation
scores = []
cv = KFold(n_splits=3, random_state=None, shuffle=True)
for train_index, test_index in cv.split(X):
    # define the keras model
    model = Sequential()
    model.add(Dense(128, input_dim=32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X, y, epochs=1500, batch_size=10, verbose=0)

    # summarize the first 5 cases
    acc = model.evaluate(X,y,verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], acc[1]))
    scores.append(acc[1])
    end = time.time()-start
    if end < 60:
        print("seconds to train and test model: ",end)
    else:
        print("minutes to train and test model: ", end/60.0)
print("Neural Network classification mean accuracy: %.2f%% standard deviation: (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))
