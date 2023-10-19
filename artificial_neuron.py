import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

class ArtificialNeuron:
    def __init__(self, learning_rate=0.1, n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
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
        return self._model(X) >= 0.5

    def _initialize(self, X):
        W = np.random.randn(X.shape[1], 1)
        b = np.random.randn(1)
        return W, b

    def _model(self, X):
        Z = X.dot(self.W) + self.b
        return 1 / (1 + np.exp(-Z))

    def _log_loss(self, A, y):
        return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

    def _gradients(self, A, X, y):
        dW = 1 / len(y) * np.dot(X.T, A - y)
        db = 1 / len(y) * np.sum(A - y)
        return dW, db

    def _update(self, dW, db):
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self.W, self.b

def main():
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape((y.shape[0], 1))

    print('Dimensions de X:', X.shape)
    print('Dimensions de y:', y.shape)

    model = ArtificialNeuron()
    model.fit(X, y)

    new_point = np.array([2, 1])
    x0 = np.linspace(-1, 4, 100)
    x1 = (-model.W[0] * x0 - model.b) / model.W[1]

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
    plt.scatter(new_point[0], new_point[1], c='r')
    plt.plot(x0, x1, c='orange', lw=3)
    plt.show()

    prediction = model.predict(new_point)
    print("Prediction for new data point:", prediction)

if __name__ == "__main__":
    main()
