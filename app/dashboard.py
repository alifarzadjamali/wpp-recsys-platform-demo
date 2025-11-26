import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import engine as eng 


@st.cache_data
def get_overview_data():
    """Collect basic stats and global metrics from the engine."""
    num_users = eng.num_users
    num_items = eng.num_items

    train_interactions = len(eng.df_train)
    test_interactions = len(eng.df_test)

    # global_metrics should have been computed in PART 4 of the engine
    global_metrics = getattr(eng, "global_metrics", None)
    if global_metrics is None:
        # fallback if not precomputed
        global_metrics = eng.evaluate_model(verbose=False)

    # cluster info if available
    cluster_info = None
    if hasattr(eng, "cluster_labels"):
        labels = eng.cluster_labels
        unique, counts = np.unique(labels, return_counts=True)
        cluster_info = pd.DataFrame({
            "cluster": unique,
            "n_users": counts
        }).sort_values("n_users", ascending=False)

    return {
        "num_users": num_users,
        "num_items": num_items,
        "train_interactions": train_interactions,
        "test_interactions": test_interactions,
        "global_metrics": global_metrics,
        "cluster_info": cluster_info,
    }


def main():
    st.set_page_config(
        page_title="WPP Hybrid Recommender Dashboard",
        page_icon="🎯",
        layout="wide"
    )

    st.title("🎯 WPP Hybrid Recommender Dashboard")
    st.write(
        "This dashboard sits on top of WPP_Hybrid_Engine_v1 model. "
        "It reads the ALS + IBCF + Category + Recency + Popularity hybrid, "
        "and exposes a friendly UI."
    )

    data = get_overview_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active members in ALS", f"{data['num_users']:,}")
    col2.metric("Merchants", f"{data['num_items']:,}")
    col3.metric("Train interactions", f"{data['train_interactions']:,}")
    col4.metric("Test interactions", f"{data['test_interactions']:,}")

    st.markdown("---")

    gm = data["global_metrics"]
    st.subheader("Global evaluation")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Precision@{eng.TOP_K_RECS}", f"{gm['Precision@K']:.3f}")
    c2.metric(f"MAP@{eng.TOP_K_RECS}", f"{gm['MAP@K']:.3f}")
    c3.metric(f"NDCG@{eng.TOP_K_RECS}", f"{gm['NDCG@K']:.3f}")

    st.markdown("---")

    # Category distribution across all interactions
    st.subheader("Global category distribution (train set)")

    # Build category distribution once
    @st.cache_data
    def compute_global_category_distribution():
        from collections import Counter

        profile = Counter()
        total = 0
        for row in eng.df_train[["merchant", "categories"]].itertuples():
            cats = row.categories or []
            for c in cats:
                profile[c] += 1
                total += 1

        if total == 0:
            return pd.DataFrame(columns=["category_id", "category_name", "share"])

        rows = []
        for cid, count in profile.items():
            share = count / total
            name = eng.category_id_to_name.get(cid, f"Category {cid}")
            rows.append({
                "category_id": cid,
                "category_name": name,
                "share": share
            })
        df_cat = pd.DataFrame(rows).sort_values("share", ascending=False)
        return df_cat

    df_cat = compute_global_category_distribution()
    if df_cat.empty:
        st.info("No category information found in train set.")
    else:
        fig = px.bar(
            df_cat,
            x="category_name",
            y="share",
            labels={"category_name": "Category", "share": "Share of interactions"},
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Cluster info
    st.subheader("User clusters (ALS embedding space)")

    if data["cluster_info"] is None:
        st.info("Clustering is disabled or cluster_labels not available in the engine.")
    else:
        df_clusters = data["cluster_info"].copy()
        df_clusters["share"] = df_clusters["n_users"] / df_clusters["n_users"].sum()

        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(df_clusters.reset_index(drop=True))

        with c2:
            fig_c = px.pie(
                df_clusters,
                names="cluster",
                values="n_users",
                title="Cluster sizes",
            )
            st.plotly_chart(fig_c, use_container_width=True)

    st.markdown(
        """
        Use the sidebar or the pages menu to:
        - Explore individual users
        - Adjust hybrid weights and diversity rules
        - Inspect clusters in more detail
        """
    )


if __name__ == "__main__":
    main()

