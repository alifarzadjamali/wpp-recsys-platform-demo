from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd
from models.common import ModelData, rank


def evaluate(model, data: ModelData, events: pd.DataFrame, k=10, max_users=1000, seed=42):
    """Full-catalog ranking against held-out interactions, with cold-start slices."""
    truth = defaultdict(set)
    ordered = defaultdict(list)
    unavailable = 0
    for row in events.sort_values("timestamp", kind="stable").itertuples():
        item = data.item_index.get(str(row.item_id))
        if item is None:
            unavailable += 1
        else:
            truth[str(row.user_id)].add(item)
            ordered[str(row.user_id)].append(item)
    users = sorted(truth)
    if len(users) > max_users:
        users = sorted(np.random.default_rng(seed).choice(users, max_users, replace=False))
    groups = defaultdict(list)
    recommended = defaultdict(set)
    cold_items = set(np.flatnonzero(data.popularity == 0))
    for user in users:
        targets = truth[user] - data.seen.get(user, set())
        if not targets:
            continue
        pred = rank(model.score(user), data.seen.get(user, set()), k)
        labels = [("overall", targets)]
        if user not in data.user_index:
            labels.append(("new_user", targets))
        elif len(data.histories[user]) <= 3:
            labels.append(("short_history", targets))
        else:
            labels.append(("warm_user", targets))
        cold_truth = targets & cold_items
        if cold_truth:
            labels.append(("cold_item", cold_truth))
        if user not in data.user_index and len(ordered[user]) >= 2:
            first = ordered[user][0]
            data.histories[user] = [first]
            later = set(ordered[user][1:]) - {first}
            if later:
                one_shot = rank(model.score(user), {first}, k)
                labels.append(("new_user_after_one", later, one_shot))
            del data.histories[user]
        for label in labels:
            group, group_truth = label[:2]
            group_pred = label[2] if len(label) == 3 else pred
            hits = [int(item in group_truth) for item in group_pred]
            precision = sum(hits) / k
            recall = sum(hits) / len(group_truth)
            ap = sum(sum(hits[:i]) / i for i, hit in enumerate(hits, 1) if hit) / min(k, len(group_truth))
            dcg = sum(hit / np.log2(i + 1) for i, hit in enumerate(hits, 1))
            ideal = sum(1 / np.log2(i + 1) for i in range(1, min(k, len(group_truth)) + 1))
            groups[group].append((precision, recall, ap, dcg / ideal if ideal else 0))
            recommended[group].update(group_pred.tolist())
    rows = []
    for group, values in groups.items():
        mean = np.mean(values, axis=0)
        rows.append({"model": model.name, "group": group, "users": len(values),
                     "precision_at_k": mean[0], "recall_at_k": mean[1],
                     "map_at_k": mean[2], "ndcg_at_k": mean[3],
                     "catalog_coverage": len(recommended[group]) / len(data.items),
                     "unavailable_events": unavailable, "k": k})
    return pd.DataFrame(rows)

