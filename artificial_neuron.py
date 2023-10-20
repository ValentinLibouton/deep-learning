import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from utilities import *
from tqdm import tqdm

class ArtificialNeuron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
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

    def fit(self, X_train, y_train, X_test, y_test):
        """
        Train the Artificial Neuron on input data.

        Parameters:
        - X (numpy.ndarray): Input features.
        - y (numpy.ndarray): Target labels.
        """
        self.W, self.b = self._initialize(X_train)

        for i in tqdm(range(self.n_iter)):
            A = self._model(X_train)  # Activations

            if i % 10 == 0:  # On calcul la perf et la perte 1x/10 pour ne pas trop perdre en temps de calcul
                # Train
                self.train_loss_history.append(self._log_loss(A, y_train))
                y_pred = self.predict(X_train)
                self.train_accuracy_history.append(accuracy_score(y_train, y_pred))

                # Test
                A_test = self._model(X_test)
                self.test_loss_history.append(self._log_loss(A_test, y_test))
                y_pred = self.predict(X_test)
                self.test_accuracy_history.append(accuracy_score(y_test, y_pred))

            # Update
            dW, db = self._gradients(A, X_train, y_train)
            self.W, self.b = self._update(dW, db)


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
        epsilon = 1e-15  # permet de ne pas avoir d'erreurs de calcul avec A = 0 ou = 1
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

    model = ArtificialNeuron(n_iter=10000, learning_rate=0.01)
    model.fit(X_train=X_train_reshape, y_train=y_train, X_test=X_test_reshape, y_test=y_test)

    model.plot_loss_curve()  # Affiche la courbe d'apprentissage
    model.plot_accuracy_curve()
    model.plot_learning_and_accuracy_curve()

    #prediction = model.predict(X_test_reshape)
    #print("Prediction for new data points:", prediction)

if __name__ == "__main__":
    main()
