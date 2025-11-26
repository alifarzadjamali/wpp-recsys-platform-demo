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
from collections import Counter


st.set_page_config(page_title="User Explorer", page_icon="🧑‍💻", layout="wide")


def mid_to_name(mid: int) -> str:
    return eng.merchant_name_map.get(mid, f"Merchant_{mid}")


def compute_category_profile(item_indices):
    profile = Counter()
    total = 0
    for iid in item_indices:
        mid = eng.idx_to_item[iid]
        cats = eng.merchant_categories.get(mid, [])
        for c in cats:
            profile[c] += 1
            total += 1

    if total == 0:
        return pd.DataFrame(columns=["category_id", "category_name", "share"])

    rows = []
    for cid, count in profile.items():
        rows.append({
            "category_id": cid,
            "category_name": eng.category_id_to_name.get(cid, f"Category {cid}"),
            "share": count / total,
        })
    df = pd.DataFrame(rows).sort_values("share", ascending=False)
    return df


#@st.cache_data
def get_demo_users():
    # pick_demo_users is defined in your PART 5
    if hasattr(eng, "pick_demo_users"):
        return eng.pick_demo_users(min_train=5, max_train=20)
    return []


def get_user_indices_from_id(member_id_str):
    if member_id_str not in eng.user_to_idx:
        return None, None, None
    u = eng.user_to_idx[member_id_str]
    train_items = eng.train_ui[u].indices
    test_items = eng.test_ui[u].indices
    return u, train_items, test_items


def main():
    st.title("🧑‍💻 User Explorer")
    if st.session_state.get("weights_updated", False):
        st.session_state["weights_updated"] = False
        st.rerun()

    demo_users = get_demo_users()

    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.subheader("Select user")

        mode = st.radio(
            "Selection mode",
            ["Pick from demo list", "Search by member_id_str"],
        )

        member_id = None
        if mode == "Pick from demo list":
            if not demo_users:
                st.info("No demo users returned by pick_demo_users.")
            else:
                member_id = st.selectbox("Demo users", demo_users)
        else:
            member_id = st.text_input("Enter member_id_str exactly as in data")

        run_button = st.button("Show user profile")

    if not run_button or not member_id:
        st.info("Pick a user and click the button.")
        return

    u_idx, train_items, test_items = get_user_indices_from_id(member_id)
    if u_idx is None:
        st.error(f"User {member_id} not found in ALS model.")
        return

    # build names
    train_names = [mid_to_name(eng.idx_to_item[i]) for i in train_items]
    test_names = [mid_to_name(eng.idx_to_item[i]) for i in test_items]

    # cluster
    cluster_label = None
    if hasattr(eng, "cluster_labels"):
        cluster_label = int(eng.cluster_labels[u_idx])

    st.markdown("\n" + "=" * 90)
    st.markdown(f"### 👤 USER DEMO — `{member_id}`")
    st.markdown("=" * 90)

    if cluster_label is not None:
        st.markdown(f"**Cluster:** `{cluster_label}`")
    else:
        st.markdown("**Cluster:** not available")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Past behaviour (train)")
        if not train_names:
            st.write("(no train interactions)")
        else:
            st.write(", ".join(train_names[:40]))

    with col_b:
        st.subheader("Recent activity (test)")
        if not test_names:
            st.write("(no test interactions)")
        else:
            st.write(", ".join(test_names))

    # recommendations
    rec_items = eng.recommend_items_for_user(u_idx, top_n=eng.TOP_K_RECS)
    rec_names = [mid_to_name(eng.idx_to_item[i]) for i in rec_items]

    st.subheader("Model recommendations")

    df_recs = pd.DataFrame({
        "rank": list(range(1, len(rec_items) + 1)),
        "merchant_id": [eng.idx_to_item[i] for i in rec_items],
        "merchant_name": rec_names,
        "is_in_test": [i in test_items for i in rec_items],
    })

    st.dataframe(df_recs, use_container_width=True)

    # metrics
    prec = eng.precision_at_k(rec_items, test_items, eng.TOP_K_RECS)
    mapk = eng.map_at_k(rec_items, test_items, eng.TOP_K_RECS)
    ndcg = eng.ndcg_at_k(rec_items, test_items, eng.TOP_K_RECS)

    st.subheader("User level metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Precision@{eng.TOP_K_RECS}", f"{prec:.3f}")
    c2.metric(f"MAP@{eng.TOP_K_RECS}", f"{mapk:.3f}")
    c3.metric(f"NDCG@{eng.TOP_K_RECS}", f"{ndcg:.3f}")

    # category profiles
    st.markdown("---")
    st.subheader("Category profiles")

    df_train_cat = compute_category_profile(train_items)
    df_rec_cat = compute_category_profile(rec_items)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("**Train category distribution**")
        if df_train_cat.empty:
            st.write("(no category data)")
        else:
            fig_train = px.bar(
                df_train_cat,
                x="category_name",
                y="share",
                labels={"category_name": "Category", "share": "Share"},
            )
            fig_train.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_train, use_container_width=True)

    with col_d:
        st.markdown("**Recommendation category distribution**")
        if df_rec_cat.empty:
            st.write("(no category data)")
        else:
            fig_rec = px.bar(
                df_rec_cat,
                x="category_name",
                y="share",
                labels={"category_name": "Category", "share": "Share"},
            )
            fig_rec.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_rec, use_container_width=True)

    # export simple CSV for this user
    st.markdown("---")
    st.subheader("Export")

    csv_buf = df_recs.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download recommendations for this user (CSV)",
        data=csv_buf,
        file_name=f"wpp_recs_{member_id}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

