import random

import numpy as np
import cupy as cp
from sklearn.metrics import balanced_accuracy_score


class MyLogisticRegression:
    def __init__(self, mode='cpu', tol=1e-5):
        self.w = None
        self.l1 = 0.0001
        self.l2 = 0.1
        self.tol = tol
        if mode == 'gpu':
            self.mv = cp
        else :
            self.mv = np

    def logit(self, X):
        return self.mv.dot(X, self.w)

    def sigmoid(self, x):
        return 1 / (1 + self.mv.exp(-x))

    def get_weights(self):
        return self.w



    def get_grad(self, X, y):
        n = X.shape[0]
        X_ = self.mv.concatenate((self.mv.ones((n,1)),X), axis=1)
        pred = self.sigmoid(self.logit(X_))
        pred = self.mv.clip(pred, 1e-10, 1 - 1e-10)
        err = pred - y
        return (self.mv.dot(X_.T, err)) / X_.shape[0]

    def  get_hess(self, X):
        n = X.shape[0]
        X_ = self.mv.concatenate((self.mv.ones((n,1)),X),axis=1)
        p = self.sigmoid(self.logit(X_))
        p_w = p*(1-p)
        weighted_X = X_ * self.mv.sqrt(p_w.reshape(-1, 1))
        hess = self.mv.dot(weighted_X.T, weighted_X) / n
        return hess

    def fit(self, X, y ,epochs=100000, batch_size=1000, learning_rate=0.00002):
        X = self.mv.asarray(X)
        y = self.mv.asarray(y)
        self.w = self.mv.random.random(X.shape[1] + 1)
        losses = []
        err = self.tol + 1
        losses.append(self.loss(X, y))
        while err > self.tol:
            grad = self.get_grad(X, y)
            hess = self.get_hess(X)
            epsilon = 1e-8
            diagonal_idx = self.mv.diag_indices(hess.shape[0])
            hess[diagonal_idx] += epsilon
            L = self.mv.linalg.cholesky(hess)
            temp = self.mv.linalg.solve(L, -grad)
            update = self.mv.linalg.solve(L.T, temp)
            self.w += update
            losses.append(self.loss(X, y))
            err = abs(losses[-1] - losses[-2])
        return losses

    def predict(self, X, threshold=0.51):
        pred = self.predict_prob(X)
        print(pred)
        print(pred > threshold)
        return (pred > threshold).astype(int)

    def predict_prob(self, X):
        X = self.mv.asarray(X)
        n, k = X.shape
        X_ = self.mv.concatenate((self.mv.ones((n,1)),X), axis=1)
        pred = self.sigmoid(self.logit(X_))
        return pred

    def loss(self, X, y):
        n = X.shape[0]
        X_ = self.mv.concatenate((self.mv.ones((n, 1)), X), axis=1)
        pred = self.mv.clip(self.sigmoid(self.logit(X_)), 1e-10, 1 - 1e-10)
        return -self.mv.mean(y * self.mv.log(pred) + (1 - y) * self.mv.log(1 - pred))
