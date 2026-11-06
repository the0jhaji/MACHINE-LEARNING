import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class LinearRegression:
    """Linear regression implemented from scratch using gradient descent."""

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.m, self.n = X.shape
        self.weights = np.zeros(self.n, dtype=float)
        self.bias = 0.0

        for _ in range(self.n_iters):
            self.update_weights(X, y)

        return self

    def update_weights(self, X, y):
        predictions = self.predict(X)
        error = predictions - y

        dw = (2 / self.m) * X.T.dot(error)
        db = (2 / self.m) * np.sum(error)

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X.dot(self.weights) + self.bias


def train_salary_model():
    data = pd.read_csv("salary_data.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    model = LinearRegression(learning_rate=0.02, n_iters=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)

    print("Weight =", model.weights[0])
    print("Bias =", model.bias)
    print("R2 score =", score)
    print("Test predictions:")
    print(predictions)

    return model, X_test, y_test


if __name__ == "__main__":
    train_salary_model()
