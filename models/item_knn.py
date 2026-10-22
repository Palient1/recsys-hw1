"""Item-based коллаборативная фильтрация.

Сходство — косинус со сжатием (shrinkage) по бинарной матрице взаимодействий X (users x items):
    s_ij = <x_i, x_j> / (||x_i|| * ||x_j|| + shrink)
Для каждого объекта оставляем top-k соседей. Скор пользователя:
    score(u, j) = sum_{i in I(u)} s_ij  (сумма сходств с объектами истории).
"""
import numpy as np
import scipy.sparse as sp


class ItemKNN:
    def __init__(self, k=100, shrink=10.0, idf=False):
        self.k, self.shrink, self.idf = k, shrink, idf

    def fit(self, stage):
        X = stage.seen.astype(np.float32).tocsc()
        if self.idf:  # понижаем вклад очень активных пользователей
            act = np.asarray(X.sum(axis=1)).ravel()
            X = sp.diags(1.0 / np.log1p(act + 1)).dot(X).tocsc()
        self.X = stage.seen
        G = (X.T @ X).toarray()
        norms = np.sqrt(np.diag(G))
        S = G / (np.outer(norms, norms) + self.shrink + 1e-8)
        np.fill_diagonal(S, 0.0)
        # оставляем top-k соседей в каждом столбце (j -> его соседи i)
        if self.k < S.shape[0]:
            thr = -np.partition(-S, self.k, axis=0)[self.k - 1]
            S[S < thr[None, :]] = 0.0
        self.S = sp.csr_matrix(S.astype(np.float32))
        return self

    def score(self, users):
        return (self.X[users] @ self.S).toarray()
