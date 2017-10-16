#!/usr/bin/env python
# coding = utf-8
import numpy as np

class DecisionTreeClassifier(object):
    def __init__(self):
        self._tree = {}

    def _entropy(self, s):
        u, counts = np.unique(s, return_counts = True)
        ratio = counts / s.size
        return -np.sum(ratio * np.log2(ratio));

    def _gain(self, col, s):
        ent_s = self._entropy(s)
        u, counts = np.unique(col, return_counts = True)
        D = np.zeros(counts.size)
        for (k, v) in enumerate(u):
            D[k] = self._entropy(s[col == v])
        return ent_s - np.sum((counts / s.size) * D)

    def _chose_best_split(self, X, y, fea_indeies):
        gain_vec = np.zeros(fea_indeies.size)
        for (key, value) in enumerate(fea_indeies):
            gain_vec[key] = self._gain(X[:, value], y)
        max_i = np.argmax(gain_vec)
        return fea_indeies[max_i]

    def _most(self, y):
        counts = np.bincount(y)
        return np.argmax(counts) 

    def _build_tree(self, X, y, fea_indeies):
        if np.unique(y).size == 1:
            return y[0]
        a = self._chose_best_split(X, y, fea_indeies)
        fea_indeies = np.delete(fea_indeies, a)
        u, counts = np.unique(X[:, a], return_counts = True)
        V = counts.size
        tree = {}
        tree[a] = {}
        for v in u:
            if (y[X[:, a] == v].size == 0):
                tree[a][v] = self._most(y)
            else:
                tree[a][v] = self._build_tree(X[X[:,a] == v], y[X[:, a] == v], fea_indeies)
        return tree

    def fit(self, X, y):
        X = np.array(X)
        y = np.reshape(y , (y.size, 1))
        fea_indeies = np.arange(X.shape[1])
        self._tree = self._build_tree(X, y, fea_indeies)

    def _pred(self, x, tree):
        node = tree[0][x[tree[0]]]
        if node is dict:
            return self._pred(x, tree[0][x[tree[0]]])
        return node

    def predict(self, X):
        X = np.array(X)
        pred = np.zeros(X.shape[0])
        for (row, x) in enumerate(X):
            pred[row] = self._pred(x, self._tree)
        return pred

if __name__ == '__main__':
    x = np.array([[0,1,2], [3,4,5], [6,7,8]])
    y = np.array([1,0,1])
    
    clf = DecisionTreeClassifier()
    clf.fit(x, y)
    
    z = np.array([[10,11,12], [13, 14, 15]])
    pred =  clf.predict(z)
    print(pred)
