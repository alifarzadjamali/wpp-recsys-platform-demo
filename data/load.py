from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


REQUIRED = {"user_id", "item_id", "timestamp"}


def normalize(events: pd.DataFrame, catalog: pd.DataFrame | None = None):
    """Return one event schema for both the private and public data."""
    missing = REQUIRED - set(events)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    frame = events.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["user_id", "item_id", "timestamp"])
    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if catalog is None:
        catalog = pd.DataFrame({"item_id": frame["item_id"].unique()})
    catalog = catalog.copy()
    catalog["item_id"] = catalog["item_id"].astype(str)
    for col in ("category", "name"):
        if col not in catalog:
            catalog[col] = ""
        catalog[col] = catalog[col].fillna("").astype(str)
    return frame, catalog.drop_duplicates("item_id").reset_index(drop=True)


def load_wpp(path: str | Path):
    raw = pd.read_parquet(path)
    needed = {"member_id_str", "merchant", "createdAt"}
    if needed - set(raw):
        raise ValueError(f"WPP file needs {sorted(needed)}")
    events = raw.rename(columns={"member_id_str": "user_id", "merchant": "item_id", "createdAt": "timestamp"})
    if "is_anonymous" in events:
        events = events.loc[~events["is_anonymous"].fillna(True).astype(bool)]
    events = events[["user_id", "item_id", "timestamp"] + (["session_id"] if "session_id" in events else [])]
    # WPP has no independent catalog snapshot. Use only pre-cutoff rows for metadata.
    cutoff = events["timestamp"].sort_values().iloc[int(len(events) * 0.7) - 1]
    cols = [c for c in ("merchant", "name", "categories") if c in raw]
    catalog = raw.loc[pd.to_datetime(raw["createdAt"], utc=True) <= cutoff, cols].drop_duplicates("merchant").rename(columns={"merchant": "item_id"})
    if "categories" in catalog:
        catalog["category"] = catalog["categories"].map(
            lambda x: str(x[0]) if isinstance(x, (list, tuple, np.ndarray)) and len(x) else ""
        )
    return normalize(events, catalog)


def load_retailrocket(directory: str | Path, max_events: int | None = None):
    """Read the original Retailrocket CSVs. No data is shipped with this repo."""
    directory = Path(directory)
    events = pd.read_csv(directory / "events.csv", usecols=["timestamp", "visitorid", "event", "itemid"], nrows=max_events)
    events = events.loc[events["event"].isin(["view", "addtocart", "transaction"])]
    events = events.rename(columns={"visitorid": "user_id", "itemid": "item_id"})
    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms", utc=True)
    events = events[["user_id", "item_id", "timestamp", "event"]]
    events = events.sort_values(["user_id", "timestamp"])
    gap = events.groupby("user_id")["timestamp"].diff().gt(pd.Timedelta(minutes=30)).fillna(True)
    events["session_id"] = events["user_id"].astype(str) + ":" + gap.groupby(events["user_id"]).cumsum().astype(str)
    parts = [directory / "item_properties_part1.csv", directory / "item_properties_part2.csv"]
    available = [p for p in parts if p.exists()]
    if not available:
        raise FileNotFoundError("Retailrocket needs item_properties_part1.csv or item_properties_part2.csv")
    properties = pd.concat([pd.read_csv(p, usecols=["timestamp", "itemid", "property", "value"]) for p in available], ignore_index=True)
    category = properties.loc[properties["property"] == "categoryid"].copy()
    category["timestamp"] = pd.to_datetime(category["timestamp"], unit="ms", utc=True)
    # Catalog metadata is available only if published before the evaluation boundary.
    cutoff = events["timestamp"].sort_values().iloc[int(len(events) * 0.7) - 1]
    category = category.loc[category["timestamp"] <= cutoff].sort_values("timestamp").drop_duplicates("itemid", keep="last")
    catalog = category.rename(columns={"itemid": "item_id", "value": "category"})[["item_id", "category"]]
    catalog["name"] = "Item " + catalog["item_id"].astype(str)
    return normalize(events, catalog)


def split_time(events: pd.DataFrame, train_fraction: float = 0.7, valid_fraction: float = 0.15):
    if not 0 < train_fraction < 1 or not 0 < valid_fraction < 1 - train_fraction:
        raise ValueError("Split fractions must leave a test set")
    ordered = events.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if len(ordered) < 10:
        raise ValueError("At least ten events are needed for a split")
    a, b = int(len(ordered) * train_fraction), int(len(ordered) * (train_fraction + valid_fraction))
    return ordered.iloc[:a].copy(), ordered.iloc[a:b].copy(), ordered.iloc[b:].copy()
