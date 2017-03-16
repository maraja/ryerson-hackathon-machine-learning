# imports
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd


# read data into a DataFrame
# This DataFrame will allow us to view the information nicely.
data = pd.read_csv('./goog.csv')

# Data Pre-processing

# Because the file we have formats the dates in DD-Month-YY, 
# we're going to need something numeric to work with instead of a string
# this piece of code will go through each date, take everything
# before the FIRST hyphen and store it as an array of integers into 
# the dates array
dates = [int(i.split('-')[0]) for i in np.array(data)[:,0]]
# Get the prices and store in an array
prices = np.array(data)[:,1]

# Add another dimension to each array so we can take the transpose
# of it and treat it as a column of values
prices = np.array([prices]).T
dates = np.array([dates]).T


# Print the data out to understand it
# print(dates)
# print(prices)


# This function takes your dates as your independent X variable
# and your prices as your dependent Y variable.
# It also takes an x value as the value you'd like to predict the Y
# value for and spits out the predicted value along with 
# the slope and intercept for a regression line.
def predict_price(dates, prices, x):
    linear_mod = linear_model.LinearRegression()
    linear_mod.fit(dates, prices)
    predicted_price = linear_mod.predict(x)
    return predicted_price, linear_mod.coef_, linear_mod.intercept_


# This function will display your data along with the regression
# line using a library called matplotlib.
def show_plot(dates, prices):
    linear_mod = linear_model.LinearRegression()
    linear_mod.fit(dates, prices)
    plt.scatter(dates, prices, color='red')
    plt.plot(dates, linear_mod.predict(dates), color='blue', linewidth=3)
    plt.show()
    return


# predict the price at X=29 (day 29)
print(predict_price(dates, prices, 29))
# show the plot for visualization
show_plot(dates, prices)

