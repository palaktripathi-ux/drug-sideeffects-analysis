import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Drug Recommender", page_icon="🔄", layout="wide")

@st.cache_resource
def load_artifacts():
    return joblib.load("models/recommender_artifacts.pkl")

artifacts = load_artifacts()
df = artifacts["df"]
feature_matrix = artifacts["feature_matrix"]

st.title("🔄 Drug Recommender (Content-Based)")
st.markdown("""
Pick a drug and this tool finds other drugs that **treat the same medical condition**
and have the **most similar side-effect / classification profile**, using cosine
similarity over a combined TF-IDF (side-effect text) + one-hot (drug class, Rx/OTC,
pregnancy category) feature space.

This mirrors what a "similar alternatives" feature on a pharmacy app would do.
""")
st.warning("⚠️ For academic demonstration only — not a substitute for a doctor's or pharmacist's advice.")

drug_list = sorted(df["drug_name"].unique())
selected = st.selectbox("Select a drug", drug_list, index=drug_list.index("ibuprofen") if "ibuprofen" in drug_list else 0)
top_n = st.slider("Number of alternatives to show", 3, 15, 5)

idx_matches = df.index[df["drug_name"] == selected].tolist()
if not idx_matches:
    st.error("Drug not found.")
    st.stop()

idx = idx_matches[0]
row = df.loc[idx]
condition = row["medical_condition"]

st.subheader(f"Selected: **{selected}**")
c1, c2, c3 = st.columns(3)
c1.metric("Treats", condition)
c2.metric("Rating", f"{row['rating']:.1f}/10" if pd.notna(row["rating"]) else "N/A")
c3.metric("Drug class", row["primary_drug_class"])

same_condition_idx = df.index[(df["medical_condition"] == condition) & (df.index != idx)]

if len(same_condition_idx) == 0:
    st.info("No other drugs in the dataset treat this exact condition.")
else:
    target_vec = feature_matrix[idx]
    candidate_vecs = feature_matrix[same_condition_idx]
    sims = cosine_similarity(target_vec, candidate_vecs).flatten()

    result = df.loc[same_condition_idx, [
        "drug_name", "primary_drug_class", "rating", "no_of_reviews",
        "side_effect_count", "rx_otc"
    ]].copy()
    result["similarity"] = sims
    result = result.sort_values("similarity", ascending=False).head(top_n)
    result["similarity"] = (result["similarity"] * 100).round(1).astype(str) + "%"

    st.subheader(f"Top {top_n} alternatives for '{selected}' ({condition})")
    st.dataframe(result.reset_index(drop=True), use_container_width=True)

    st.caption("Similarity combines side-effect text closeness and shared drug class / "
               "Rx-OTC / pregnancy-category metadata. Higher = more similar profile.")
