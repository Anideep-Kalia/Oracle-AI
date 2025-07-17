import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import iteractive


#function to upadte and display the plot
def update_plot(hidden_layer_size): 
    # Generate synthetic data (circle)
    # X, y = make circles(n_samples=300, noise=0.1, factor=0.5, random_state=1)

    # Create a multi-layer perceptron (MLP) classifier

    clf = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), activation='relu', max_iter=3000, random state=1)

    # Fit the classifier to the data
    clf.fit(X, y)

    #Create a grid of points for visualization
    #These are 1D arrays of 100 values each, representing the x and y coordinates of the grid.
    x_vals = np.linspace(X[:,0].min()-0.1, X[:,0].max() + 0.1, 100)
    y_vals = np.linspace(X[:,1].min()-0.1, X[:,1].max() + 0.1, 100)

    #The resulting X_plane and Y_plane are both 100x100 arrays,
    #representing a grid of 10,000 points.

    X_plane, Y_plane = np.meshgrid(x_vals, y_vals)

    #grid points is a single 20 array (grad points) of shape (10000, 2),
    #where each row represents a point in the grid.
    grid_points = np.column_stack((X_plane.ravel(), Y_plane.ravel()))

    #Predict class labels for the grid points (for decision boundary)
    Z=clf.predict(grid_points)

    #Z.reshape(X_plane.shape) reshapes Z into a 100x100 array. 
    Z=Z.reshape(X_plane.shape)

    #Predict class labels for the original data points
    y_pred = clf.predict(X)

    # Clear previous plot
    plt.clf()

    # Plot the decision boundary
    # it is oftern used to visualise the decision boundary of a model by plotting
    # the predicted class probabilities or labels across a grid of points
    plt.contourf(X_plane, Y_plane,Z, level=[-0.5,0.5.1.5], cmap=plt.cm.RdYlGn, alpha=0.6)

    # plot the original data points with their predicted labels
    # Seperate points for eadch predicted class
    class_0=y_pred==0 # Indices of points predicted as class 0
    class_1=y_pred==1 # Indices of points predicted as class 1

    plt.scatter(X[class_0,0], X[class_0,1], c='red', edgecolors='k', marker='o',s=50,label='Predicted Class 0')
    plt.scatter(X[class_1,1],X[class_1,1], c='green',edgecolors='k', marker='o',s=50,label='Predicted Class 1')

    #Add labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Bounadry and Predicted labels(Hidden Layer Size={hidden_layer_size})')
    plt.legend()
    plt.show()

    # Create a slider fopr hidden layer size
    hidden_layer_size_slider=widgets.IntSlider(value=1,min=1,max=10,step=1,description='Hidden Layer Size')

    # Create an interactive widget
    interactive_plot=interactive(update_plot, hidden_layer_size=hidden_layer_size_slider)

    # Display the widgets
    display(interactive_plot)