"""EASE (Steck, 2019): линейный item-item автоэнкодер с нулевой диагональю.

    min_B ||X - X B||_F^2 + lambda ||B||_F^2,   diag(B) = 0
Решение в замкнутой форме:
    P = (X^T X + lambda I)^{-1},   B_ij = -P_ij / P_jj (i != j),   B_jj = 0
Скор: score(u, .) = x_u B.
"""
import numpy as np


class EASE:
    def __init__(self, reg=500.0):
        self.reg = reg

    def fit(self, stage):
        X = stage.seen.astype(np.float64)
        G = (X.T @ X).toarray()
        G[np.diag_indices_from(G)] += self.reg
        P = np.linalg.inv(G)
        B = -P / np.diag(P)[None, :]
        np.fill_diagonal(B, 0.0)
        self.B = B.astype(np.float32)
        self.X = stage.seen
        return self

    def score(self, users):
        return self.X[users] @ self.B
