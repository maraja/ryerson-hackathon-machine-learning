# Imports
import numpy as np
import matplotlib.pyplot as plt


# Create a couple arrays with synthetic data to model
X = np.array([0,1,2,3,4,5])
y = np.array([0,0.8,0.9,0.1,-0.8,-1])

# fit the data with a straight line. 
# p1 will store an array of the slope and intercept of the line
p1 = np.polyfit(X, y, 1)

# Do the same for a quadratic and cubic line
# p2 and p3 will hold the co-efficients required to plot the lines
p2 = np.polyfit(X, y, 2)
p3 = np.polyfit(X, y, 3)


# Plot the data as dots first.
plt.plot(X,y,'o')

# Create a larger x axis with 100 steps in between -2 and 6 to visualize the data better
xp = np.linspace(-2,6,100)


# Plot the 3 lines
plt.plot(xp, np.polyval(p1,xp), 'r-')
plt.plot(xp, np.polyval(p2,xp), 'b--')
plt.plot(xp, np.polyval(p3,xp), 'm:')

plt.show()