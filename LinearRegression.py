#!/usr/bin/env python
# coding=utf-8

import numpy as np

class LinearRegression(object):
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        x = np.array(X)
        ones = np.ones((x.shape[0], 1))
        x = np.column_stack((x, ones))
        t_x = x.T
        y = np.array(y).reshape((y.shape[0],1))
        m_xx = np.dot(t_x,  x)
        m_xy = np.dot(t_x,  y)
        self.w = np.linalg.lstsq(m_xx, m_xy)[0]

    def predict(self, X):
        x = np.array(X)
        ones = np.ones((x.shape[0], 1))
        x = np.column_stack((x, ones))
        y_pred = np.dot(x, self.w)
        return y_pred

if __name__ == '__main__':
    x = np.array([[0,1,2], [3,4,5], [6,7,8]])
    y = np.array([1,2,3]).reshape((x.shape[0], 1))

    lr = LinearRegression()
    lr.fit(x, y)

    z = np.array([[10,11,12], [13,14,15]])
    pred = lr.predict(z)
    print(pred)
