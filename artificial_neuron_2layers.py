import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from utilities import *
from tqdm import tqdm

class ArtificialNeuron:
    def __init__(self, n1, learning_rate=0.01, n_iter=1000):
        """
       Initialize an Artificial Neuron.

       Parameters:
       - learning_rate (float, optional): The learning rate for gradient descent. Default is 0.1.
       - n_iter (int, optional): The number of iterations for training. Default is 100.
       """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.train_loss_history = []  # Perte sur les données d'entraînement
        self.train_accuracy_history = []  # Performance sur les données d'entraînement
        self.test_loss_history = []  # Perte sur les données de test
        self.test_accuracy_history = []  # Performance sur les données de test
        self.n1 = n1

    def fit(self, X_train, y_train, X_test, y_test):
        """
        Train the Artificial Neuron on input data.

        Parameters:
        - X (numpy.ndarray): Input features.
        - y (numpy.ndarray): Target labels.
        """
        n0 = X_train.shape[0]
        n2 = y_train.shape[0]
        self._initialize(n0, self.n1, n2)

        for i in tqdm(range(self.n_iter)):
            self._forward_propagation(X_train)  # Activations
            self._back_propagation(X_train, y_train)
            self._update()

            if i % 10 == 0:  # On calcul la perf et la perte 1x/10 pour ne pas trop perdre en temps de calcul
                # Train
                self.train_loss_history.append(self._log_loss(self.A2, y_train))
                y_pred = self.predict(X_train)
                self.train_accuracy_history.append(accuracy_score(y_train.flatten(), y_pred.flatten())) # flatten() car tab 2 dims on force en 1 dim

                # Test
                A_test = self._forward_propagation(X_test)
                self.test_loss_history.append(self._log_loss(A_test, y_test))
                y_pred = self.predict(X_test)
                self.test_accuracy_history.append(accuracy_score(y_test, y_pred))

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X (numpy.ndarray): Input features.

        Returns:
        - numpy.ndarray: Predicted binary labels.
        """
        return self.A2 >= 0.5

    def _initialize(self, n0, n1, n2):
        """
        Initialize the weights and bias.

        Parameters:
        - X (numpy.ndarray): Input features.

        Returns:
        - numpy.ndarray: Initial weights.
        - numpy.ndarray: Initial bias.
        """
        self.W1 = np.random.randn(n1, n0)
        self.b1 = np.random.randn(n1, 1)
        self.W2 = np.random.randn(n2, n1)
        self.b2 = np.random.randn(n2, 1)

        parameters = {'W1': self.W1,
                      'b1': self.b1,
                      'W2': self.W2,
                      'b2': self.b2
                      }
        return parameters


    def _forward_propagation(self, X):
        """
        Compute the model's predictions.

        Parameters:
        - X (numpy.ndarray): Input features.

        Returns:
        - numpy.ndarray: Model predictions.
        """
        Z1 = self.W1.dot(X) + self.b1
        self.A1 = 1 / (1 + np.exp(-Z1))
        Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = 1 / (1 + np.exp(-Z2))



    def _log_loss(self, A, y):
        """
        Compute the logistic loss.

        Parameters:
        - A (numpy.ndarray): Predicted probabilities.
        - y (numpy.ndarray): Target labels.

        Returns:
        - float: Logistic loss.
        """
        epsilon = 1e-15  # permet de ne pas avoir d'erreurs de calcul avec A = 0 ou = 1
        return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

    def _back_propagation(self, X, y):
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

        m = y.shape[1]
        dZ2 = self.A2 - y
        self.dW2 = 1 / m * dZ2.dot(self.A1.T)
        self.db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(self.W2.T, dZ2) * self.A1 * (1 - self.A1)
        self.dW1 = 1 / m * dZ1.dot(X.T)
        self.db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)


    def _update(self):
        """
        Update weights and bias using gradients.

        Parameters:
        - dW (numpy.ndarray): Gradients of weights.
        - db (float): Gradient of bias.

        Returns:
        - numpy.ndarray: Updated weights.
        - float: Updated bias.
        """

        self.W1 = self.W1 - self.learning_rate * self.dW1
        self.b1 = self.b1 - self.learning_rate * self.db1
        self.W2 = self.W2 - self.learning_rate * self.dW2
        self.b2 = self.b2 - self.learning_rate * self.db2


    def plot_loss_curve(self):
        iterations = range(len(self.train_loss_history))
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(iterations, self.train_loss_history, label='Train Loss')
        plt.plot(iterations, self.test_loss_history, label='Test Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

    def plot_accuracy_curve(self):
        iterations = range(len(self.train_accuracy_history))
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(iterations, self.train_accuracy_history, label='Train Accuracy')
        plt.plot(iterations, self.test_accuracy_history, label='Test Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

    def plot_learning_and_accuracy_curve(self):
        iterations = range(len(self.train_loss_history))
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(iterations, self.train_loss_history, label='Train Loss')
        plt.plot(iterations, self.test_loss_history, label='Test Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(iterations, self.train_accuracy_history, label='Train Accuracy')
        plt.plot(iterations, self.test_accuracy_history, label='Test Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.tight_layout()
        plt.show()


def generate_data():
    # Generate synthetic data
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape((y.shape[0], 1))
    return X, y

def reshape_data(X_train, X_test):
    # Reshape the data
    X_train_reshape = X_train.reshape(X_train.shape[0], -1) / X_train.max()
    X_test_reshape = X_test.reshape(X_test.shape[0], -1) / X_train.max()
    return X_train_reshape, X_test_reshape

def plot_data(X, y):
    # Create a scatter plot of the data
    plt.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], c='blue', label='Class 0')
    plt.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], c='red', label='Class 1')
    plt.legend()



def plot_decision_boundary(model, X, y):
    # Plot the decision boundary of the model
    x0 = np.linspace(-1, 4, 100)
    x1 = (-model.W[0] * x0 - model.b) / model.W[1]
    plt.plot(x0, x1, c='orange', lw=3)




def main():
    X_train, y_train, X_test, y_test = load_data()

    X_train_reshape, X_test_reshape = reshape_data(X_train, X_test)


    print('Dimensions de X_train:', X_train_reshape.shape)
    print('Dimensions de y_train:', y_train.shape)
    print('Dimensions de X_test:', X_test_reshape.shape)
    print('Dimensions de y_test:', y_test.shape)

    model = ArtificialNeuron(n1=2, n_iter=10000, learning_rate=0.01)
    model.fit(X_train=X_train_reshape, y_train=y_train, X_test=X_test_reshape, y_test=y_test)

    model.plot_loss_curve()  # Affiche la courbe d'apprentissage
    model.plot_accuracy_curve()
    model.plot_learning_and_accuracy_curve()

    #prediction = model.predict(X_test_reshape)
    #print("Prediction for new data points:", prediction)

if __name__ == "__main__":
    main()
