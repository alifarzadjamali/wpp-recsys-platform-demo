from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd


class ModelData:
    def __init__(self, train: pd.DataFrame, catalog: pd.DataFrame):
        self.catalog = catalog.copy()
        self.items = self.catalog["item_id"].astype(str).tolist()
        self.item_index = {item: i for i, item in enumerate(self.items)}
        self.users = sorted(train["user_id"].astype(str).unique())
        self.user_index = {user: i for i, user in enumerate(self.users)}
        self.categories = self.catalog["category"].fillna("").astype(str).tolist()
        self.category_names = sorted(set(self.categories) | {""})
        self.category_index = {cat: i for i, cat in enumerate(self.category_names)}
        self.item_categories = np.array([self.category_index[x] for x in self.categories])
        self.histories = defaultdict(list)
        counts = np.zeros(len(self.items), dtype=float)
        for row in train.sort_values("timestamp").itertuples():
            item = self.item_index.get(str(row.item_id))
            if item is not None:
                self.histories[str(row.user_id)].append(item)
                counts[item] += 1
        self.popularity = np.log1p(counts)
        if self.popularity.max() > 0:
            self.popularity /= self.popularity.max()
        self.seen = {user: set(items) for user, items in self.histories.items()}
        self.recency = np.zeros(len(self.items))
        latest = train.groupby("item_id")["timestamp"].max()
        end = train["timestamp"].max()
        for item, stamp in latest.items():
            index = self.item_index.get(str(item))
            if index is not None:
                self.recency[index] = np.exp(-0.03 * (end - stamp).total_seconds() / 86400)
        self.sessions = []
        if "session_id" in train:
            for _, frame in train.groupby("session_id"):
                self.sessions.append([self.item_index[str(item)] for item in frame["item_id"] if str(item) in self.item_index])

    def category_scores(self, user: str) -> np.ndarray:
        history = self.histories.get(user, [])
        if not history:
            return np.zeros(len(self.items))
        counts = np.bincount(self.item_categories[history], minlength=len(self.category_names))
        return counts[self.item_categories] / len(history)


def rank(scores: np.ndarray, seen: set[int], k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).copy()
    if seen:
        scores[list(seen)] = -np.inf
    return np.argsort(-scores, kind="stable")[: min(k, len(scores) - len(seen))]
