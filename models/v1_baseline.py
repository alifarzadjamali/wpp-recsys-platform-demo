"""V1: a small hybrid of latent factors, co-visits, categories and popularity."""

from collections import Counter, defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from .common import ModelData


class Baseline:
    name = "v1_baseline"

    def __init__(self, factors: int = 32, weights=None):
        self.factors = factors
        self.weights = weights or {"latent": 0.45, "co_visit": 0.2, "session": 0.1,
                                   "category": 0.1, "popular": 0.1, "recency": 0.05}

    def fit(self, data: ModelData):
        self.data = data
        rows, cols = [], []
        for user, history in data.histories.items():
            rows.extend([data.user_index[user]] * len(history))
            cols.extend(history)
        matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(data.users), len(data.items)))
        factors = min(self.factors, min(matrix.shape) - 1)
        if factors >= 1:
            svd = TruncatedSVD(n_components=factors, random_state=42)
            self.user_factors = svd.fit_transform(matrix)
            self.item_factors = svd.components_.T
        else:
            self.user_factors = np.zeros((len(data.users), 1))
            self.item_factors = np.zeros((len(data.items), 1))
        self.neighbors = defaultdict(Counter)
        for history in data.histories.values():
            recent = list(dict.fromkeys(history[-50:]))
            for item in recent:
                for other in recent:
                    if item != other:
                        self.neighbors[item][other] += 1
        self.session_neighbors = defaultdict(Counter)
        for session in data.sessions:
            items = list(dict.fromkeys(session))
            for item in items:
                for other in items:
                    if item != other:
                        self.session_neighbors[item][other] += 1
        return self

    def score(self, user: str):
        data = self.data
        scores = np.zeros(len(data.items))
        index = data.user_index.get(user)
        if index is not None:
            latent = self.user_factors[index] @ self.item_factors.T
            latent = np.maximum(latent, 0)
            if latent.max() > 0:
                latent /= latent.max()
            scores += self.weights["latent"] * latent
        co = np.zeros(len(data.items))
        session = np.zeros(len(data.items))
        for item in set(data.histories.get(user, [])):
            for other, count in self.neighbors[item].items():
                co[other] += count
            for other, count in self.session_neighbors[item].items():
                session[other] += count
        if co.max() > 0:
            co /= co.max()
        if session.max() > 0:
            session /= session.max()
        scores += self.weights["co_visit"] * co
        scores += self.weights["session"] * session
        scores += self.weights["category"] * data.category_scores(user)
        scores += self.weights["popular"] * data.popularity
        scores += self.weights["recency"] * data.recency
        return scores
