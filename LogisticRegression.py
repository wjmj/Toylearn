#!/usr/bin/env python
# coding=utf-8

import numpy as np

class LogisticRegression(object):
    def __init__(self, alpha=0.01, max_iter=100):
        self.alpha = alpha
        self.max_iter = max_iter

    def _sigmoid(self, z):
        return 1. / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(y.shape[0], 1)
        
        self.w = np.random.randn(X.shape[1], 1)
        self.b = np.random.randn()
        self.m = X.shape[0]
        for index in range(self.max_iter):
            z = np.dot(X, self.w) + self.b
            a = self._sigmoid(z)
            dz = a - y
            dw = np.dot(X.T, dz) / self.m
            db = np.dot(np.ones((1, dz.shape[0])), dz) / self.m
            self.w -= self.alpha * dw
            self.b -= self.alpha * db

    def predict(self, X):
        X = np.array(X)
        z = np.dot(X, self.w) + self.b
        y_pred = self._sigmoid(z)
        return y_pred

if __name__ == '__main__':
    x = np.array([[0,1,2], [3,4,5], [6,7,8]])
    y = np.array([1,0,1])

    lr = LogisticRegression()
    lr.fit(x, y)
    
    z = np.array([[10,11,12], [13,14,15]])
    pred = lr.predict(z)
    print(pred)
