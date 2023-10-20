import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

class ArtificialNeuron:
    def __init__(self, learning_rate=0.1, n_iter=100):
        """
       Initialize an Artificial Neuron.

       Parameters:
       - learning_rate (float, optional): The learning rate for gradient descent. Default is 0.1.
       - n_iter (int, optional): The number of iterations for training. Default is 100.
       """
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Train the Artificial Neuron on input data.

        Parameters:
        - X (numpy.ndarray): Input features.
        - y (numpy.ndarray): Target labels.
        """
        self.W, self.b = self._initialize(X)
        self.loss_history = []

        for _ in range(self.n_iter):
            A = self._model(X)
            self.loss_history.append(self._log_loss(A, y))
            dW, db = self._gradients(A, X, y)
            self.W, self.b = self._update(dW, db)

        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print("Performance: ", accuracy)

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X (numpy.ndarray): Input features.

        Returns:
        - numpy.ndarray: Predicted binary labels.
        """
        return self._model(X) >= 0.5

    def _initialize(self, X):
        """
        Initialize the weights and bias.

        Parameters:
        - X (numpy.ndarray): Input features.

        Returns:
        - numpy.ndarray: Initial weights.
        - numpy.ndarray: Initial bias.
        """
        W = np.random.randn(X.shape[1], 1)
        b = np.random.randn(1)
        return W, b

    def _model(self, X):
        """
        Compute the model's predictions.

        Parameters:
        - X (numpy.ndarray): Input features.

        Returns:
        - numpy.ndarray: Model predictions.
        """
        Z = X.dot(self.W) + self.b
        return 1 / (1 + np.exp(-Z))

    def _log_loss(self, A, y):
        """
        Compute the logistic loss.

        Parameters:
        - A (numpy.ndarray): Predicted probabilities.
        - y (numpy.ndarray): Target labels.

        Returns:
        - float: Logistic loss.
        """
        epsilon = 1e-15 # permet de ne pas avoir d'erreurs de calcul avec A = 0 ou = 1
        return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

    def _gradients(self, A, X, y):
        """
        Compute gradients of weights and bias.

        Parameters:
        - A (numpy.ndarray): Predicted probabilities.
        - X (numpy.ndarray): Input features.
        - y (numpy.ndarray): Target labels.

        Returns:
        - numpy.ndarray: Gradients of weights.
        - float: Gradient of bias.
        """
        dW = 1 / len(y) * np.dot(X.T, A - y)
        db = 1 / len(y) * np.sum(A - y)
        return dW, db

    def _update(self, dW, db):
        """
        Update weights and bias using gradients.

        Parameters:
        - dW (numpy.ndarray): Gradients of weights.
        - db (float): Gradient of bias.

        Returns:
        - numpy.ndarray: Updated weights.
        - float: Updated bias.
        """
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self.W, self.b

def main():
    # Generate datas
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape((y.shape[0], 1))
    print('Dimensions de X:', X.shape)
    print('Dimensions de y:', y.shape)

    # Model initialization
    model = ArtificialNeuron()

    # Model training
    model.fit(X, y)

    # You create a new_point data point with coordinates(2, 1).
    # This data point will be used to show how the model classifies this new data.
    new_point = np.array([2, 1])

    # You create a set of 100 equally spaced values in the range -1 to 4.
    # These x0 values will be used to draw the model's decision line.
    x0 = np.linspace(-1, 4, 100)

    # You use the weights and bias learned by the model (model.W and model.b) to calculate the corresponding values of
    # x1. These values represent the model's decision boundary, indicating how the model would classify the data in this
    # two-dimensional space.
    x1 = (-model.W[0] * x0 - model.b) / model.W[1]

    # Create a scatter plot of the training data (X). The data is distributed on the chart based on the first two
    # columns of X, that is, X[:, 0] on the x-axis and X[:, 1] on the y-axis. The color of the points is determined by
    # the y class labels (0 or 1) using the 'summer' colormap. The points corresponding to each class will have
    # different colors.
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')

    # Adds a red point ('r') on the scatter plot to represent the new_point. This allows you to visualize
    # where this point is in relation to the other data.
    plt.scatter(new_point[0], new_point[1], c='r')

    # Draws a decision line using the values x0 (x coordinates) and x1 (y coordinates) calculated previously.
    # The decision line is drawn in orange ('c') with a line width (lw) of 3 pixels. The decision line shows how the
    # model divides the space based on its learned weights and biases.
    plt.plot(x0, x1, c='orange', lw=3)
    plt.show()

    # Prediction
    # Calls the model's predict method (model.predict(new_point)) to make a prediction on the new_point data point.
    # The predict method will return True if the model classifies new_point as class 1 and False if it classifies it as
    # class 0 (using the default 0.5 threshold rule).
    prediction = model.predict(new_point)
    print("Prediction for new data point:", prediction)

if __name__ == "__main__":
    main()
