"""Общие функции: загрузка и подготовка данных, разбиение, матрицы, оценка top-K."""
import json
import os
import random
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HF_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"
RATINGS_URL = f"{HF_BASE}/benchmark/5core/rating_only/Video_Games.csv"
META_URL = f"{HF_BASE}/raw/meta_categories/meta_Video_Games.jsonl"

# параметры подвыборки и разбиения
START_DATE = "2015-01-01"
MIN_ITEM_INTERACTIONS = 10
MIN_USER_INTERACTIONS = 5
VAL_START = "2021-01-01"
TEST_START = "2022-01-01"
RELEVANT_RATING = 4  # релевантный объект в отложенном периоде: оценка >= 4


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def download_data(data_dir=DATA_DIR):
    """Скачивает рейтинги (5-core) и метаданные Video_Games из Amazon Reviews 2023."""
    import urllib.request
    os.makedirs(data_dir, exist_ok=True)
    for url in (RATINGS_URL, META_URL):
        path = os.path.join(data_dir, url.rsplit("/", 1)[1])
        if not os.path.exists(path):
            print("downloading", url)
            urllib.request.urlretrieve(url, path)
    return data_dir


def load_raw(data_dir=DATA_DIR):
    ratings = pd.read_csv(os.path.join(data_dir, "Video_Games.csv"))
    keep = set(ratings.parent_asin)
    rows = []
    with open(os.path.join(data_dir, "meta_Video_Games.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["parent_asin"] not in keep:
                continue
            details = r.get("details") or {}
            rows.append({
                "parent_asin": r["parent_asin"],
                "title": r.get("title"),
                "main_category": r.get("main_category"),
                "store": r.get("store"),
                "categories": r.get("categories") or [],
                "price": r.get("price"),
                "rated": details.get("Rated"),
                "date_first_available": details.get("Date First Available"),
            })
    meta = pd.DataFrame(rows)
    return ratings, meta


def filter_interactions(df, start=START_DATE, min_item=MIN_ITEM_INTERACTIONS, min_user=MIN_USER_INTERACTIONS):
    """Временное окно + итеративный k-core: товары >= min_item, пользователи >= min_user взаимодействий."""
    df = df[df["dt"] >= start]
    while True:
        n = len(df)
        df = df[df.groupby("parent_asin")["user_id"].transform("size") >= min_item]
        df = df[df.groupby("user_id")["parent_asin"].transform("size") >= min_user]
        if len(df) == n:
            return df.reset_index(drop=True)


def encode(df):
    """Кодирует идентификаторы в индексы 0..n-1."""
    users = np.sort(df["user_id"].unique())
    items = np.sort(df["parent_asin"].unique())
    df = df.copy()
    df["uid"] = np.searchsorted(users, df["user_id"])
    df["iid"] = np.searchsorted(items, df["parent_asin"])
    return df, users, items


def temporal_split(df, val_start=VAL_START, test_start=TEST_START):
    train = df[df["dt"] < val_start]
    val = df[(df["dt"] >= val_start) & (df["dt"] < test_start)]
    test = df[df["dt"] >= test_start]
    return train, val, test


def interaction_matrix(df, n_users, n_items, values=None):
    data = np.ones(len(df), dtype=np.float32) if values is None else np.asarray(values, dtype=np.float32)
    return sp.csr_matrix((data, (df["uid"].values, df["iid"].values)), shape=(n_users, n_items))


# ---------------------------------------------------------------- метаданные

def _clean(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def build_item_features(meta, items, fit_df, top_stores=300):
    """Разреженная матрица признаков товаров (без ID) и имена признаков.

    Используются только статические атрибуты товара (store, main_category, categories, ESRB, год
    появления) и популярность товара, посчитанная по обучающему периоду. Цена и average_rating
    не используются: это снимок на 2023 год, т.е. информация из будущего.
    """
    meta = meta.set_index("parent_asin").reindex(items)
    feats = []  # список (item_idx, feature_name, weight)

    stores = meta["store"].fillna("unknown").map(_clean)
    top = set(stores.value_counts().index[:top_stores])
    for i, s in enumerate(stores):
        feats.append((i, "store=" + (s if s in top else "other"), 1.0))
    for i, s in enumerate(meta["main_category"].fillna("unknown")):
        feats.append((i, "main=" + _clean(s), 1.0))
    for i, cats in enumerate(meta["categories"]):
        cats = [c for c in (cats if isinstance(cats, list) else []) if c != "Video Games"]
        for c in cats:
            feats.append((i, "cat=" + _clean(c), 1.0 / np.sqrt(len(cats))))
    for i, r in enumerate(meta["rated"]):
        feats.append((i, "esrb=" + (_clean(r) if isinstance(r, str) else "unknown"), 1.0))
    years = pd.to_datetime(meta["date_first_available"], errors="coerce", format="mixed").dt.year
    for i, y in enumerate(years):
        feats.append((i, "year=" + ("unknown" if pd.isna(y) else str(int(min(max(y, 2004), 2023)))), 1.0))
    # популярность в обучающем периоде (логарифмические корзины)
    cnt = np.bincount(fit_df["iid"].values, minlength=len(items))
    bucket = np.floor(np.log2(cnt + 1)).astype(int)
    for i, b in enumerate(bucket):
        feats.append((i, f"pop={b}", 1.0))

    names = sorted({f for _, f, _ in feats})
    idx = {f: k for k, f in enumerate(names)}
    rows, cols, vals = zip(*[(i, idx[f], w) for i, f, w in feats])
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(items), len(names)), dtype=np.float32)
    X.sum_duplicates()
    return X, names


def build_user_features(fit_df, item_features, n_users):
    """Профиль пользователя: нормированная сумма признаков товаров из его истории (только прошлое)."""
    R = interaction_matrix(fit_df, n_users, item_features.shape[0])
    U = (R @ item_features).tocsr()
    norm = np.sqrt(np.asarray(U.multiply(U).sum(axis=1)).ravel()) + 1e-8
    return sp.diags(1.0 / norm).dot(U).tocsr().astype(np.float32)


# ---------------------------------------------------------------- оценка

def eval_targets(fit_df, eval_df, relevant_rating=RELEVANT_RATING):
    """Пользователи для оценки и их релевантные объекты.

    Оцениваются «тёплые» пользователи (есть история в fit_df) с хотя бы одним релевантным
    (оценка >= relevant_rating) объектом в eval_df, которого нет в их истории.
    """
    seen = fit_df.groupby("uid")["iid"].agg(set)
    pos = eval_df[eval_df["rating"] >= relevant_rating]
    pos = pos[pos["uid"].isin(seen.index)]
    rel = pos.groupby("uid")["iid"].agg(set)
    rel = {u: s - seen[u] for u, s in rel.items()}
    rel = {u: s for u, s in rel.items() if s}
    users = np.array(sorted(rel), dtype=np.int64)
    return users, rel


def recommend(score_fn, users, seen_matrix, candidate_mask, k=10, batch=2048):
    """top-k для пользователей: score_fn(users_batch) -> [batch, n_items].

    Исключаются объекты из истории пользователя и объекты вне множества кандидатов.
    """
    out = np.empty((len(users), k), dtype=np.int64)
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        scores = np.asarray(score_fn(ub), dtype=np.float32).copy()
        scores[:, ~candidate_mask] = -np.inf
        sub = seen_matrix[ub]
        scores[np.repeat(np.arange(len(ub)), np.diff(sub.indptr)), sub.indices] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        out[s:s + batch] = np.take_along_axis(top, order, axis=1)
    return out


def evaluate(recs, users, rel, n_candidates, k=10):
    """NDCG@k (основная), Recall@k, HitRate@k, MRR@k и покрытие каталога Coverage@k."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ndcg, recall, hit, mrr = [], [], [], []
    for u, r in zip(users, recs):
        relset = rel[u]
        hits = np.array([i in relset for i in r[:k]], dtype=float)
        idcg = discounts[:min(len(relset), k)].sum()
        ndcg.append((hits * discounts).sum() / idcg)
        recall.append(hits.sum() / len(relset))
        hit.append(float(hits.any()))
        first = np.flatnonzero(hits)
        mrr.append(1.0 / (first[0] + 1) if len(first) else 0.0)
    return {
        f"NDCG@{k}": float(np.mean(ndcg)),
        f"Recall@{k}": float(np.mean(recall)),
        f"HitRate@{k}": float(np.mean(hit)),
        f"MRR@{k}": float(np.mean(mrr)),
        f"Coverage@{k}": len(np.unique(recs[:, :k])) / n_candidates,
    }


class Stage:
    """Всё, что нужно для одного этапа (validation или test): данные обучения и цели оценки."""

    def __init__(self, fit_df, eval_df, n_users, n_items, meta, items, k=10):
        self.fit_df, self.eval_df = fit_df, eval_df
        self.n_users, self.n_items, self.k = n_users, n_items, k
        self.seen = interaction_matrix(fit_df, n_users, n_items)
        self.seen.data[:] = 1.0
        self.candidates = np.zeros(n_items, dtype=bool)
        self.candidates[fit_df["iid"].unique()] = True  # кандидаты: объекты, встречавшиеся в обучении
        self.users, self.rel = eval_targets(fit_df, eval_df)
        self.item_features, self.item_feature_names = build_item_features(meta, items, fit_df)
        self.user_features = build_user_features(fit_df, self.item_features, n_users)

    def run(self, model):
        recs = recommend(model.score, self.users, self.seen, self.candidates, k=self.k)
        return evaluate(recs, self.users, self.rel, int(self.candidates.sum()), k=self.k), recs


class Popular:
    """Бейзлайн: популярность в обучающем периоде."""

    def fit(self, stage):
        self.pop = np.bincount(stage.fit_df["iid"].values, minlength=stage.n_items).astype(np.float32)
        return self

    def score(self, users):
        return np.broadcast_to(self.pop, (len(users), len(self.pop)))


def ndcg_per_user(recs, users, rel, k=10):
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    out = np.empty(len(users))
    for n, (u, r) in enumerate(zip(users, recs)):
        hits = np.array([i in rel[u] for i in r[:k]], dtype=float)
        out[n] = (hits * discounts).sum() / discounts[:min(len(rel[u]), k)].sum()
    return out


def paired_bootstrap(a, b, n=2000, seed=SEED):
    """95% доверительный интервал для разности средних mean(a) - mean(b) по пользователям."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    diff = (a[idx] - b[idx]).mean(axis=1)
    return float(np.mean(a - b)), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


class RecentPopular(Popular):
    """Бейзлайн: популярность за последние `days` дней периода обучения (тай-брейк — общая популярность)."""

    def __init__(self, days=180):
        self.days = days

    def fit(self, stage):
        f = stage.fit_df
        recent = f[f["timestamp"] >= f["timestamp"].max() - self.days * 86400 * 1000]
        total = np.bincount(f["iid"].values, minlength=stage.n_items)
        self.pop = (np.bincount(recent["iid"].values, minlength=stage.n_items) + 1e-6 * total).astype(np.float32)
        return self
