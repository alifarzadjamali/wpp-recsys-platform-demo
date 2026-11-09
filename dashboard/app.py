"""Local dashboard for data checks and saved model comparisons."""

import json
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.load import load_retailrocket, load_wpp, split_time
from data.demo import make_demo
from analytics.eda import summary, daily, top_items
from analytics.evaluate import evaluate
from models.common import ModelData, rank
from models.v1_baseline import Baseline


st.set_page_config(page_title="Recommendation lab", layout="wide")
st.title("Recommendation lab")
st.caption("Explore the data, check a baseline recommendation, and read the saved model comparison.")


@st.cache_data(show_spinner="Reading events...")
def read_data(source, path, max_events):
    return make_demo() if source == "Demo" else load_retailrocket(path, max_events) if source == "Retailrocket" else load_wpp(path)


@st.cache_resource(show_spinner="Training baseline...")
def baseline(source, path, max_events):
    events, catalog = read_data(source, path, max_events)
    train, valid, _ = split_time(events)
    catalog = pd.concat([catalog, pd.DataFrame({"item_id": train["item_id"].unique(), "name": "", "category": ""})]).drop_duplicates("item_id")
    data = ModelData(train, catalog)
    return Baseline().fit(data), data, valid


source = st.sidebar.selectbox("Data source", ["Demo", "Retailrocket", "WPP"])
path = st.sidebar.text_input("Dataset path", "data/retailrocket" if source == "Retailrocket" else "data/wpp.parquet")
max_events = st.sidebar.number_input("Event limit (0 means all)", min_value=0, value=100000, step=10000)
limit = max_events or None
tab_data, tab_compare, tab_explore, tab_tune = st.tabs(["Data and EDA", "Model comparison", "Explore", "Tune baseline"])

with tab_data:
    if st.button("Load data"):
        try:
            events, catalog = read_data(source, path, limit)
            st.session_state["dataset"] = (events, catalog)
        except (OSError, ValueError) as error:
            st.error(str(error))
    if "dataset" in st.session_state:
        events, catalog = st.session_state["dataset"]
        st.json(summary(events, catalog))
        st.line_chart(daily(events).set_index("day"))
        st.subheader("Most viewed items")
        st.dataframe(top_items(events, catalog), hide_index=True)

with tab_compare:
    report_path = Path("artifacts/comparison.csv")
    if report_path.exists():
        report = pd.read_csv(report_path)
        st.caption("All models use the same held-out events and full catalog ranking. Larger is better for every metric shown.")
        st.dataframe(report, hide_index=True)
        st.bar_chart(report.loc[(report["split"] == "test") & (report["group"] == "overall")].set_index("model")[["ndcg_at_k", "recall_at_k"]])
        run_path = report_path.with_name("run.json")
        if run_path.exists():
            st.json(json.loads(run_path.read_text(encoding="utf-8")))
    else:
        st.info("Run `python run.py --source retailrocket --path data/retailrocket` first.")

with tab_explore:
    if st.button("Train baseline for browsing"):
        try:
            st.session_state["baseline"] = baseline(source, path, limit)
        except (OSError, ValueError) as error:
            st.error(str(error))
    if "baseline" in st.session_state:
        model, data, _ = st.session_state["baseline"]
        user = st.selectbox("User", data.users[:1000])
        recs = rank(model.score(user), data.seen.get(user, set()), 10)
        st.dataframe(data.catalog.iloc[recs][["item_id", "name", "category"]], hide_index=True)

with tab_tune:
    if "baseline" not in st.session_state:
        st.info("Train the baseline in Explore first.")
    else:
        original, data, valid = st.session_state["baseline"]
        st.caption("Weights here only affect this evaluation. The saved comparison stays fixed.")
        weights = {key: st.slider(key.replace("_", " ").title(), 0.0, 1.0, float(value), 0.05)
                   for key, value in original.weights.items()}
        if st.button("Evaluate these weights on validation"):
            tuned = Baseline(weights=weights)
            tuned.__dict__.update({key: value for key, value in original.__dict__.items() if key != "weights"})
            result = evaluate(tuned, data, valid, max_users=500)
            st.dataframe(result, hide_index=True)
