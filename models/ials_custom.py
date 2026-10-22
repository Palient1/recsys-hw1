"""iALS с самостоятельно разработанным целевым сигналом (модель 5).

В базовом iALS p_ui = 1 для любого взаимодействия: покупка с оценкой 1 и покупка с оценкой 5
одинаково «нравятся», а покупка десятилетней давности так же важна, как вчерашняя. Метрика
задачи — попадание в будущие объекты с оценкой >= 4, поэтому таргет строим из двух множителей:

    p_ui = g(r_ui) * d(t_ui)

  * тональность оценки:  g(r) = (r - r0) / (5 - r0)
        r0 = 3: g in {-1, -0.5, 0, 0.5, 1} — плохая оценка становится отрицательным сигналом
        («купил и не понравилось» хуже, чем «не покупал»); r0 = None: g = 1 (без учёта оценки);
  * актуальность:  d(t) = gamma + (1 - gamma) * 2^(-(T - t) / h)
        T — конец обучающего периода, h — период полураспада в днях, gamma — нижняя граница
        (старые покупки остаются положительными, но описывают вкус слабее); h = None: d = 1.

Уверенность c_ui = 1 + alpha оставлена такой же, как в базовом iALS, — меняется именно значение
предпочтения, которое аппроксимирует x_u^T y_i. Обучение и формулы шагов ALS совпадают с ials.py.
"""
import numpy as np

from .ials import IALS


class IALSCustomTarget(IALS):
    def __init__(self, factors=64, reg=10.0, alpha=10.0, iterations=15, seed=42,
                 r0=3.0, half_life=730.0, gamma=0.3):
        super().__init__(factors, reg, alpha, iterations, seed)
        self.r0, self.half_life, self.gamma = r0, half_life, gamma

    def targets(self, stage):
        df = stage.fit_df
        g = np.ones(len(df)) if self.r0 is None else (df["rating"].values - self.r0) / (5.0 - self.r0)
        if self.half_life is None:
            d = np.ones(len(df))
        else:
            age_days = (df["timestamp"].max() - df["timestamp"].values) / (1000 * 86400)
            d = self.gamma + (1 - self.gamma) * np.power(2.0, -age_days / self.half_life)
        return (g * d).astype(np.float32)
