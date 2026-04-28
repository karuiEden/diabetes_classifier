import importlib.util
import numpy as np


class MyLogisticRegression:
    def __init__(self, mode='cpu', tol=1e-5, l2=0, learning_rate=0.1):
        self.w = None
        self.l2 = l2
        self.tol = tol
        self.lr = learning_rate
        if mode == 'gpu' and importlib.util.find_spec('cupy') is not None:
            self.mv = importlib.import_module('cupy')
        else:
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
        grad = self.mv.dot(X_.T, err)
        reg = self.l2 * self.w
        reg[0] -= 0
        grad_reg = (grad + reg) / n
        return grad_reg

    def get_hess(self, X):
        n = X.shape[0]
        X_ = self.mv.concatenate((self.mv.ones((n,1)),X),axis=1)
        p = self.sigmoid(self.logit(X_))
        p_w = p*(1-p)
        weighted_x = X_ * self.mv.sqrt(p_w.reshape(-1, 1))
        hess = self.mv.dot(weighted_x.T, weighted_x) / n
        hess_reg = hess + (self.l2 / n) * self.mv.eye(hess.shape[0])
        hess_reg[0,0] -= self.l2 / n
        return hess_reg

    def fit(self, X, y):
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
            l = self.mv.linalg.cholesky(hess)
            temp = self.mv.linalg.solve(l, -grad)
            update = self.mv.linalg.solve(l.T, temp)
            self.w += self.lr * update
            losses.append(self.loss(X, y))
            err = abs(losses[-1] - losses[-2])
        return losses

    def predict(self, X, threshold=0.44):
        pred = self.predict_prob(X)
        return (pred > threshold).astype(int)

    def predict_prob(self, X):
        X = self.mv.asarray(X)
        n, k = X.shape
        X_ = self.mv.concatenate((self.mv.ones((n, 1)), X), axis=1)
        pred = self.sigmoid(self.logit(X_))
        return pred

    def loss(self, X, y):
        n = X.shape[0]
        X_ = self.mv.concatenate((self.mv.ones((n, 1)), X), axis=1)
        pred = self.mv.clip(self.sigmoid(self.logit(X_)), 1e-10, 1 - 1e-10)
        reg = (self.l2 / (2 * n)) * self.mv.sum(self.w[1:] ** 2)
        return -self.mv.mean(y * self.mv.log(pred) + (1 - y) * self.mv.log(1 - pred)) + reg
