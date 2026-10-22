"""Факторизационная машина (Rendle, 2010) с метапризнаками, обучение на ранжирование (BPR).

Объект x = [признаки пользователя | признаки товара]:
  * пользователь: one-hot ID + профиль (нормированная сумма метапризнаков товаров из истории);
  * товар: one-hot ID + store, main_category, categories, ESRB, год появления, корзина популярности.
Модель второго порядка:
    y(x) = w0 + sum_f w_f x_f + sum_{f<g} <v_f, v_g> x_f x_g
Разделим признаки на пользовательские (U) и товарные (I), a_u = sum_{f in U} x_f v_f,
b_i = sum_{f in I} x_f v_f. Тогда
    y(u, i) = const(u) + sum_{f in I} w_f x_f + 1/2 (||b_i||^2 - sum_{f in I} x_f^2 ||v_f||^2) + <a_u, b_i>,
const(u) не влияет на ранжирование для пользователя, поэтому скор товара считается одним
матричным произведением. Функция потерь — BPR по парам (наблюдаемый i, случайный j):
    L = -mean log sigma(y(u, i) - y(u, j)) + lambda * (||a_u||^2 + ||b_i||^2 + ||b_j||^2).
Градиенты считает autograd PyTorch, оптимизатор Adam.
"""
import numpy as np
import scipy.sparse as sp
import torch


def _to_torch(X):
    X = X.tocoo()
    return torch.sparse_coo_tensor(np.vstack([X.row, X.col]), X.data, X.shape, dtype=torch.float32,
                                   check_invariants=False).coalesce()


class FM:
    def __init__(self, factors=64, lr=0.01, reg=1e-4, epochs=20, batch_size=4096, use_meta=True, seed=42):
        self.factors, self.lr, self.reg, self.epochs = factors, lr, reg, epochs
        self.batch_size, self.use_meta, self.seed = batch_size, use_meta, seed

    def _features(self, stage):
        Fu = sp.identity(stage.n_users, format="csr", dtype=np.float32)
        Fi = sp.identity(stage.n_items, format="csr", dtype=np.float32)
        if self.use_meta:
            Fu = sp.hstack([Fu, stage.user_features]).tocsr()
            Fi = sp.hstack([Fi, stage.item_features]).tocsr()
        return Fu, Fi

    def fit(self, stage):
        g = torch.Generator().manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Fu, Fi = self._features(stage)
        self.Fu, self.Fi = _to_torch(Fu), _to_torch(Fi)
        self.Fi_sq = _to_torch(Fi.multiply(Fi))
        self.Vu = torch.nn.Parameter(torch.randn(Fu.shape[1], self.factors, generator=g) * 0.01)
        self.Vi = torch.nn.Parameter(torch.randn(Fi.shape[1], self.factors, generator=g) * 0.01)
        self.wi = torch.nn.Parameter(torch.zeros(Fi.shape[1]))
        opt = torch.optim.Adam([self.Vu, self.Vi, self.wi], lr=self.lr)
        users = stage.fit_df["uid"].values
        items = stage.fit_df["iid"].values
        cand = np.flatnonzero(stage.candidates)
        seen = stage.seen
        for _ in range(self.epochs):
            perm = rng.permutation(len(users))
            for s in range(0, len(perm), self.batch_size):
                b = perm[s:s + self.batch_size]
                u, i = users[b], items[b]
                j = cand[rng.integers(0, len(cand), len(b))]
                # пересэмплируем негативы, попавшие в историю пользователя
                bad = np.asarray(seen[u, j]).ravel() > 0
                while bad.any():
                    j[bad] = cand[rng.integers(0, len(cand), bad.sum())]
                    bad = np.asarray(seen[u, j]).ravel() > 0
                a = torch.sparse.mm(torch.index_select(self.Fu, 0, torch.from_numpy(u)), self.Vu)  # a_u
                y, b = self._item_terms(torch.from_numpy(np.concatenate([i, j])), a.repeat(2, 1))
                yi, yj = y.chunk(2)
                loss = -torch.nn.functional.logsigmoid(yi - yj).mean()
                loss = loss + self.reg * (a.pow(2).sum(1).mean() + b.pow(2).sum(1).mean())
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            self.A = torch.sparse.mm(self.Fu, self.Vu).numpy()
            Bm = torch.sparse.mm(self.Fi, self.Vi)
            lin = torch.sparse.mm(self.Fi, self.wi[:, None]).squeeze(1)
            self_inter = 0.5 * (Bm.pow(2).sum(1) - torch.sparse.mm(self.Fi_sq, self.Vi.pow(2)).sum(1))
            self.B = Bm.numpy()
            self.item_bias = (lin + self_inter).numpy()
        return self

    def _item_terms(self, items, a):
        rows = torch.index_select(self.Fi, 0, items)
        rows_sq = torch.index_select(self.Fi_sq, 0, items)
        b = torch.sparse.mm(rows, self.Vi)
        lin = torch.sparse.mm(rows, self.wi[:, None]).squeeze(1)
        self_inter = 0.5 * (b.pow(2).sum(1) - torch.sparse.mm(rows_sq, self.Vi.pow(2)).sum(1))
        return lin + self_inter + (a * b).sum(1), b

    def score(self, users):
        return self.A[users] @ self.B.T + self.item_bias[None, :]
