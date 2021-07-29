import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn import linear_model
#import appropriate modules

#declare variables
d = load_diabetes()
d_X = d.data[:, np.newaxis, 2]
dx_train = d_X[:-20]
dy_train = d.target[:-20]
dx_test = d_X[-20:]
dy_test = d.target[-20:]

#define a gradient function
def grad_function(x,y):
    #squeese the x variable to reshape dimensions
    x = np.squeeze(x)
    #formula for the gradient
    m = (np.mean(x)*np.mean(y) - np.mean(x*y))/((np.mean(x))**2 - np.mean(x**2))
    return m

#define function for the y-intercept
def intercept(x,y,m):
    #using mean functions of numpy
    b = np.mean(y) - m*np.mean(x)
    return b

#define the regression function
def regression(m,x,c):
    Y = m*x + c
    return Y
    
gradient = grad_function(dx_train, dy_train)
y_intercept = intercept(dx_train, dy_train, gradient)
#define the prediction variable which is the regression function
y_prediction = regression(gradient,dx_test, y_intercept)

#plot the graphs
plt.scatter(dx_train, dy_train, c = "r")
plt.scatter(dx_test, dy_test, c = "g")
plt.plot(dx_test, y_prediction, c = "b")
plt.show()

