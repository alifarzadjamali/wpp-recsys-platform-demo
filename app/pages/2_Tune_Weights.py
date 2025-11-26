import streamlit as st
import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import engine as eng


st.set_page_config(page_title="Tune Hybrid Weights", page_icon="🎚", layout="wide")


def main():
    st.title("🎚 Tune Hybrid Weights")

    st.write(
        "This page lets you adjust the hybrid weights and diversity rules "
        ". It simply updates the global weights in "
        "WPP_Hybrid_Engine_v1 and calls the same recommend_items_for_user."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hybrid signal weights")

        w_als = st.slider("ALS (latent factors)", 0.0, 1.0, float(eng.W_ALS), 0.05)
        w_ibcf = st.slider("IBCF (item similarity)", 0.0, 1.0, float(eng.W_IBCF), 0.05)
        w_cat = st.slider("Category signal", 0.0, 1.0, float(eng.W_CAT), 0.05)
        w_pop = st.slider("Popularity", 0.0, 1.0, float(eng.W_POP), 0.05)
        w_rec = st.slider("Recency", 0.0, 1.0, float(eng.W_REC), 0.05)

    with col2:
        st.subheader("Diversity rules")

        max_same_merchant = st.slider(
            "Max times to show the same merchant",
            1,
            5,
            int(eng.MAX_SAME_MERCHANT),
        )

        max_per_category = st.slider(
            "Max items per category",
            1,
            10,
            int(eng.MAX_PER_CATEGORY),
        )

        explore_ratio = st.slider(
            "Exploration ratio placeholder (currently not used in code)",
            0.0,
            0.5,
            float(eng.EXPLORATION_RATIO),
            0.05,
        )

    st.markdown("---")

    # Apply button
    if st.button("Apply changes"):
        eng.W_ALS = w_als
        eng.W_IBCF = w_ibcf
        eng.W_CAT = w_cat
        eng.W_POP = w_pop
        eng.W_REC = w_rec

        eng.MAX_SAME_MERCHANT = max_same_merchant
        eng.MAX_PER_CATEGORY = max_per_category
        eng.EXPLORATION_RATIO = explore_ratio
        st.session_state["weights_updated"] = True

        st.success("Updated global weights and diversity rules in the engine.")

    st.markdown("---")
    st.subheader("Evaluate on a sample of users")

    sample_size = st.slider(
        "Number of users to evaluate",
        500,
        min(eng.num_users, 5000),
        min(eng.num_users, 2000),
        500,
    )

    if st.button("Run sample evaluation"):
        rng = np.random.default_rng(123)
        user_indices = rng.choice(eng.num_users, size=sample_size, replace=False)

        metrics = eng.evaluate_model(user_indices=user_indices, verbose=False)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Precision@{eng.TOP_K_RECS}", f"{metrics['Precision@K']:.3f}")
        c2.metric(f"MAP@{eng.TOP_K_RECS}", f"{metrics['MAP@K']:.3f}")
        c3.metric(f"NDCG@{eng.TOP_K_RECS}", f"{metrics['NDCG@K']:.3f}")

        st.info(
            "Keep in mind this is a sampled evaluation to stay fast. "
            "For full evaluation we should run the offline script as before."
        )


if __name__ == "__main__":
    main()

