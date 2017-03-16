# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model, datasets

# import some data to play with
# The following is the infamous iris dataset
iris = datasets.load_iris()

# Create a DataFrame with the data and labels to visualize if needed
df = pd.DataFrame(np.c_[iris.data, iris.target], columns = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class"])


# Data Pre-processing

# Take the required data and throw it into a numpy array with 3 columns
# We are only taking the first 2 columns as features along with all the 
# correct classes (targets) as our 3rd column.
data = np.c_[iris.data[:, :2], iris.target] # we only take the first two features.

# shuffle all the data so we're training on random samples
np.random.shuffle(data)


# define a percentage split to cut out training and testing data into
test_training_split = 0.7


# Split the data into our features and labels
# X is features
# y is labels
X = data[:, :2] 
y = data[:, 2:]


# Split the data into training and testing data.
X_training = X[:int(X.shape[0]*test_training_split),:]
y_training = y[:int(y.shape[0]*test_training_split)]

X_testing = X[int(X.shape[0]*test_training_split):,:]
y_testing = y[int(y.shape[0]*test_training_split):]


# Initialize our logistic regression function from the sklearn library
logreg = linear_model.LogisticRegression()

# We use the initialized function and fit the data.
# NOTE: the ravel function turns the column based labels array into a 1d matrix.
# This type of array is needed to feed into the logistic regression function
logreg.fit(X_training, y_training.ravel())


# Predict data based on the testing data and store it into a variable
Z = logreg.predict(X_testing)


# This function will take an array of real outputs along with an array
# of predicted outputs and calculate the classification rate precentage
def classification_rate(y, Z):
    num_right = 0
    for i in range(len(Z)):
        if y[i] == Z[i]:
            num_right = num_right + 1
    return num_right/Z.shape[0]


# Compute the classification rate and print it out
print("Classification rate:", classification_rate(y_testing.ravel(), Z))


# Import library required for k-fold cross validation
from sklearn import metrics, cross_validation


# perform cross validation with logistic regression on the data set 10-fold times
# and store the resulted predictions in another vairables
Z_cross_validation = cross_validation.cross_val_predict(linear_model.LogisticRegression(), X, y.ravel(), cv=10)

# Compute the classification rate and print it out
print("Classification rate:", classification_rate(y.ravel(), Z_cross_validation))