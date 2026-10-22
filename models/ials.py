"""iALS для неявной обратной связи (Hu, Koren, Volinsky, 2008).

Предпочтение p_ui и уверенность c_ui:
    p_ui = 1, если было взаимодействие, иначе 0;    c_ui = 1 + alpha * [было взаимодействие]
Функция потерь по всем парам (u, i):
    sum_{u,i} c_ui (p_ui - x_u^T y_i)^2 + lambda (sum ||x_u||^2 + sum ||y_i||^2)
Попеременно решаем для пользователей и объектов в замкнутой форме:
    x_u = (Y^T Y + Y^T (C_u - I) Y + lambda I)^{-1} Y^T C_u p_u
Y^T Y считается один раз на шаг; вклад Y^T (C_u - I) Y идёт только по наблюдаемым объектам.
Решения считаются батчами в PyTorch (padding историй до одинаковой длины + batched solve).
Класс принимает произвольный целевой сигнал p_ui через метод targets() — этим пользуется
модель с кастомным таргетом (ials_custom.py).
"""
import numpy as np
import scipy.sparse as sp
import torch


class IALS:
    def __init__(self, factors=64, reg=10.0, alpha=10.0, iterations=15, seed=42):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed = iterations, seed

    def targets(self, stage):
        """Целевой сигнал p_ui для наблюдаемых пар (базовый iALS: 1)."""
        return np.ones(len(stage.fit_df), dtype=np.float32)

    def fit(self, stage):
        df = stage.fit_df
        p = self.targets(stage).astype(np.float32)
        # структура CSR хранит номер наблюдения, чтобы нулевые таргеты не потерялись
        pos = sp.csr_matrix((np.arange(1, len(df) + 1, dtype=np.float64), (df["uid"].values, df["iid"].values)),
                            shape=(stage.n_users, stage.n_items))
        posT = pos.T.tocsr()
        g = torch.Generator().manual_seed(self.seed)
        self.U = torch.randn(stage.n_users, self.factors, generator=g) * 0.01
        self.V = torch.randn(stage.n_items, self.factors, generator=g) * 0.01
        for _ in range(self.iterations):
            self.U = self._solve(pos, p, self.V)
            self.V = self._solve(posT, p, self.U)
        self.U_np, self.V_np = self.U.numpy(), self.V.numpy()
        return self

    def _solve(self, pos, p, Y, budget=300_000):
        """Решает задачи наименьших квадратов для всех строк pos (пользователей или объектов)."""
        n, f = pos.shape[0], Y.shape[1]
        YtY = Y.T @ Y + self.reg * torch.eye(f)
        out = torch.zeros(n, f)
        lens = np.diff(pos.indptr)
        order = np.argsort(lens, kind="stable")
        order = order[lens[order] > 0]
        s = 0
        while s < len(order):
            l_end = int(lens[order[min(s + 4096, len(order)) - 1]])
            rows = order[s:s + max(1, min(4096, budget // l_end))]
            B, L = len(rows), int(lens[rows].max())
            starts, ln = pos.indptr[rows], lens[rows]
            col = np.arange(L)[None, :]
            mask = col < ln[:, None]
            flat = (starts[:, None] + col)[mask]
            idx = np.zeros((B, L), dtype=np.int64)
            pv = np.zeros((B, L), dtype=np.float32)
            idx[mask] = pos.indices[flat]
            pv[mask] = p[pos.data[flat].astype(np.int64) - 1]
            w = torch.from_numpy(mask.astype(np.float32) * self.alpha)   # c_ui - 1 (0 на паддинге)
            Yb = Y[torch.from_numpy(idx)]                                # B x L x f
            A = YtY + torch.einsum("blf,bl,blg->bfg", Yb, w, Yb)
            rhs = torch.einsum("blf,bl->bf", Yb, (1.0 + w) * torch.from_numpy(pv))
            out[torch.from_numpy(rows)] = torch.linalg.solve(A, rhs)
            s += B
        return out

    def score(self, users):
        return self.U_np[users] @ self.V_np.T
