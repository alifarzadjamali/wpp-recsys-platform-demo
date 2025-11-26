import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import engine as eng


st.set_page_config(page_title="Cluster Viewer", page_icon="🧩", layout="wide")


def main():
    st.title("🧩 Cluster Viewer")

    if not hasattr(eng, "cluster_labels"):
        st.info("Clustering is disabled or cluster_labels not found in engine.")
        return

    labels = eng.cluster_labels
    unique, counts = np.unique(labels, return_counts=True)

    df_clusters = pd.DataFrame({
        "cluster": unique,
        "n_users": counts,
    }).sort_values("n_users", ascending=False)
    df_clusters["share"] = df_clusters["n_users"] / df_clusters["n_users"].sum()

    st.subheader("Cluster sizes")
    st.dataframe(df_clusters, use_container_width=True)

    fig = px.bar(
        df_clusters,
        x="cluster",
        y="n_users",
        labels={"cluster": "Cluster", "n_users": "Number of users"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Inspect a cluster")

    cluster_id = st.selectbox("Cluster id", df_clusters["cluster"].tolist())

    # users in this cluster
    users_in_cluster = np.where(labels == cluster_id)[0]

    st.write(f"Users in cluster {cluster_id}: {len(users_in_cluster):,}")

    # evaluate on a small sample of this cluster
    sample_size = st.slider(
        "Users to evaluate inside this cluster",
        100,
        min(len(users_in_cluster), 3000),
        min(len(users_in_cluster), 1000),
        100,
    )

    if st.button("Evaluate this cluster"):
        rng = np.random.default_rng(99)
        if len(users_in_cluster) > sample_size:
            sample_users = rng.choice(users_in_cluster, size=sample_size, replace=False)
        else:
            sample_users = users_in_cluster

        metrics = eng.evaluate_model(user_indices=sample_users, verbose=False)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Precision@{eng.TOP_K_RECS}", f"{metrics['Precision@K']:.3f}")
        c2.metric(f"MAP@{eng.TOP_K_RECS}", f"{metrics['MAP@K']:.3f}")
        c3.metric(f"NDCG@{eng.TOP_K_RECS}", f"{metrics['NDCG@K']:.3f}")

    st.markdown("---")

    st.subheader("Sample users from this cluster")

    sample_users_to_show = min(20, len(users_in_cluster))
    sample_indices = np.random.choice(users_in_cluster, size=sample_users_to_show, replace=False)

    demo_ids = [eng.idx_to_user[u] for u in sample_indices]
    st.write("Example member_id_str in this cluster:")
    st.write(", ".join(str(x) for x in demo_ids))


if __name__ == "__main__":
    main()

