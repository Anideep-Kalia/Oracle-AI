import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import iteractive


#function to upadte and display the plot
def update plot(hidden_layer_size): #Generate synthetic data (circle)

#X, y make circlesin_samples-300, noiseme.1, factor-0.5, random statewil)

Create a multi-layer perceptron (MLP) classifier

clf MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), activation relu', max iter=3000, random state-1)

Fit the classifier to the data

clf.fit(X, y)

#Create a grad of points for visualization

#These are 10 arrays of 100 values each, representing the x and y coordinates of the grid.

x_vals np.linspace(X):, .min()0.1, X, .max() 0.1, 100)

y_vals np.linspace(X):, .min()0.1, X, 1.max() 0.1, 100)

#The resulting X plane and plane are both 100x100 arrays, #representing a grid of 10,000 points.

X_plane, Y_plane np.meshgrid(x_vals, y_vals)

#grid points is a single 20 array (grad points) of shape (10000, 2),

#where each row represents a point in the grid.

grid_points np.column_stack((X_plane.ravel(), Y_plane.ravel()))

#Predict class labels for the grid points (for decision boundary)

Z=clf.predict(grid_points)

#2.reshape(X_plane.shape) reshapes 2 into a 100x100 array. ZZ.reshape(X_plane.shape)

#Predict class labels for the original data points

y pred clf.predict(X)