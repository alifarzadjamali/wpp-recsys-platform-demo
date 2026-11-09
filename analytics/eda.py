from __future__ import annotations

import pandas as pd


def summary(events: pd.DataFrame, catalog: pd.DataFrame):
    return {
        "events": len(events), "users": events["user_id"].nunique(),
        "items_seen": events["item_id"].nunique(), "catalog_items": len(catalog),
        "first_event": str(events["timestamp"].min()), "last_event": str(events["timestamp"].max()),
    }


def daily(events: pd.DataFrame):
    return events.assign(day=events["timestamp"].dt.date).groupby("day", as_index=False).size().rename(columns={"size": "events"})


def top_items(events: pd.DataFrame, catalog: pd.DataFrame, n=20):
    counts = events["item_id"].value_counts().head(n).rename_axis("item_id").reset_index(name="events")
    return counts.merge(catalog[["item_id", "name", "category"]], on="item_id", how="left")
