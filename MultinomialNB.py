#!/usr/bin/env python
# coding=utf-8
import numpy as np

class MultinomialNB(object):
    def __init__(self, alpha=1.0):
        self._alpha = alpha
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(y.size, 1)
        labels, counts = np.unique(y, return_counts = True)
        self._c_num = labels.size
        self._c = labels.copy()
        self._c_prob = counts / y.size
        self._x_c = {}
        for label in labels:
            self._x_c[label] = {}
            for key,col in enumerate(X.T):
                self._x_c[label][key] = {}
                u, ucounts = np.unique(col, return_counts = True)
                for (k,x) in enumerate(u):
                    self._x_c[label][key][x] = (ucounts[k] + 1) / (col.size + u.size)
        
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        yi = 0
        for x in X:
            probs = np.log(self._c_prob)
            for k,c in enumerate(self._c):
                for i in range(x.size):
                    probs[k] += np.log(self._x_c[c][i][x[i]])
            max_i = np.argmax(probs)
            y_pred[yi] = self._c[max_i]
        return y_pred

if __name__ == '__main__':
    x = np.array([[0,1,2], [3,4,5], [6,7,8]])
    y = np.array([1,0,1])

    clf = MultinomialNB()
    clf.fit(x, y)

    z = np.array([[0,4,2], [3,7,8]])

    pred = clf.predict(z)
    print(pred)
