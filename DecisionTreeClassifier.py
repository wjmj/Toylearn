#!/usr/bin/env python
# coding = utf-8

class DecisionTreeClassifier(object):
    def __init__(self):
        pass

    def _entropy(self, s):
        u, counts = np.unique(s, return_counts = True)
        ratio = counts / s.size
        return -np.sum(ratio * np.log2(ratio));

    def _gain(self, col, s):
        ent_s = _entropy(s)
        u, counts = np.unique(col, return_counts = True)
        V = counts.size
        D = np.zeros(V)
        for v in range(V):
            D[v] = _entropy(s[col == u[v])
        return ent_s - np.sum((counts / s.shape) * D)

    def _chose_best_split(X, y):
        pass

    def _most(y):
        pass

    def _build_tree(self, X, y):
        if np.unique(y).size == 1:
            return y[0]
        a = _chose_best_split(X, y)
        u, counts = np.unique(X[:, a], return_counts = True)
        V = counts.size
        for v in range(V):
            if (y[X[:, a] == u[v].size == 0):
                return _most(y)
            else:
                
    def fit(self, X, y):
        X = np.array(X)
        y = np.reshape(y, (1, y.size))
        for index in range(X.shape[1]):
            best_fea_index = chose_best_fea(X, y)
    def predict(self, X):
        pass

if __name__ == '__main__':
    pass
