"""Tiny made-up event log for checking the pipeline without private data."""

import numpy as np
import pandas as pd
from .load import normalize


def make_demo(seed=42):
    rng = np.random.default_rng(seed)
    items = [str(i) for i in range(60)]
    categories = [str(i % 6) for i in range(60)]
    catalog = pd.DataFrame({"item_id": items, "category": categories,
                            "name": [f"Demo item {i}" for i in range(60)]})
    rows = []
    for day in range(400):
        user = f"u{rng.integers(0, 35 if day < 280 else 40)}"
        choice = rng.choice(55 if day < 280 else 60)
        item = str((choice // 6) * 6 + (int(user[1:]) % 6)) if rng.random() < 0.6 else str(choice)
        rows.append((user, item, pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day)))
    return normalize(pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"]), catalog)
