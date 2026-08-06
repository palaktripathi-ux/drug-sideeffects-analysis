import streamlit as st
import pandas as pd
import plotly.express as px
import json

st.set_page_config(page_title="Side-Effect Clusters", page_icon="🧬", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_pickle("data/drugs_with_clusters.pkl")
    with open("dashboard/cluster_summary.json") as f:
        cluster_info = json.load(f)
    return df, cluster_info

df, cluster_info = load_data()

st.title("🧬 Side-Effect Profile Clusters (NLP)")
st.markdown("""
Every drug's free-text `side_effects` description was cleaned, vectorized with **TF-IDF**
(unigrams + bigrams), and grouped into **6 clusters** using **KMeans**. Drugs in the same
cluster tend to describe a similar *type* of adverse reaction — e.g. allergic/breathing
reactions vs. GI upset vs. topical skin irritation — even if they treat completely
different conditions.
""")

labeled = df[df["side_effect_cluster"] >= 0].copy()

st.subheader("Cluster overview")
cols = st.columns(3)
for i, (cid, info) in enumerate(cluster_info.items()):
    with cols[i % 3]:
        st.markdown(f"**Cluster {cid}** ({info['size']} drugs)")
        st.caption(", ".join(info["keywords"]))

st.divider()
st.subheader("2D projection of side-effect similarity")
fig = px.scatter(
    labeled, x="cluster_x", y="cluster_y", color=labeled["side_effect_cluster"].astype(str),
    hover_data=["drug_name", "medical_condition"],
    title="Drugs positioned by side-effect text similarity (TF-IDF + SVD)",
    labels={"color": "Cluster"},
)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Browse a cluster")
selected_cluster = st.selectbox("Choose a cluster", sorted(labeled["side_effect_cluster"].unique()))
subset = labeled[labeled["side_effect_cluster"] == selected_cluster][
    ["drug_name", "medical_condition", "primary_drug_class", "rating", "side_effect_count"]
].sort_values("side_effect_count", ascending=False)
st.dataframe(subset.head(30), use_container_width=True)
