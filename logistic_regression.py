import random

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def batches(X, y, batch_size):
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


class MyLogisticRegression:
    def __init__(self):
        self.w = None
        self.l1 = 0.0001
        self.l2 = 0.1

    def logit(self, X):
        return np.dot(X, self.w)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_weights(self):
        return self.w

    def weight_to_csv(self):
        fd = open('logistic_regression.csv', 'w')
        fd.write(self.w)
        fd.close()

    def weight_from_csv(self):
        fd = open('logistic_regression.csv', 'r')
        self.w = np.load(fd)
        fd.close()

    def get_grad(self, X, y):
        pred = self.sigmoid(self.logit(X))
        pred = np.clip(pred, 1e-10, 1 - 1e-10)
        err = pred - y
        return (np.dot(X.T, err)) + 2 * self.l2 * self.w + self.l1 * np.sign(self.w)

    def fit(self, X, y, l1=None, l2=None,epochs=1000, batch_size=1000, learning_rate=0.00002):
        losses = []
        n, k = X.shape
        if self.w is None:
            np.random.seed(42)
            # Вектор столбец в качестве весов
            self.w = np.random.randn(k + 1)
        if l1 is not None:
            self.l1 = l1
        if l2 is not None:
            self.l2 = l2
        X_train = np.concatenate((np.ones((n, 1)), X), axis=1)
        for i in range(epochs):
            batch = batches(X_train, y, batch_size)
            for X_batch, y_batch in batch:
                grad = self.get_grad(X_batch, y_batch) / batch_size
                self.w -= grad * learning_rate
            loss = self.loss(X_train, y)
            losses.append(loss)
            # pred = self.predict(X)
            # print(balanced_accuracy_score(pred, y))
            # print(f'Epoch {i + 1} ended.')
        return losses

    def predict(self, X, threshold=0.51):
        n, k = X.shape
        X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        pred = self.sigmoid(self.logit(X_))
        print(pred)
        print(pred > threshold)
        return (pred > threshold).astype(int)

    def loss(self, X, y):
        pred = np.clip(self.sigmoid(self.logit(X)), 1e-10, 1 - 1e-10)
        return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
