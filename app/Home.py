import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Drug, Side Effects & Medical Condition Analysis",
    page_icon="💊",
    layout="wide",
)

@st.cache_data
def load_data():
    df = pd.read_pickle("data/drugs_with_clusters.pkl")
    return df

df = load_data()

st.title("💊 Drug, Side Effects & Medical Condition Analysis")
st.caption("Advanced analytics platform | Major Project | Source: drugs.com dataset (Kaggle)")

st.markdown("""
This platform analyzes **2,900+ drugs** across **47 medical conditions**, combining
exploratory data analysis, NLP on side-effect text, machine learning, and a
content-based drug recommender system.
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Drugs", f"{df['drug_name'].nunique():,}")
col2.metric("Medical Conditions", df['medical_condition'].nunique())
col3.metric("Drug Classes", df['primary_drug_class'].nunique())
col4.metric("Avg. Rating", f"{df['rating'].mean():.2f} / 10")

st.divider()

st.subheader("What's in this app")
st.markdown("""
Use the sidebar to navigate:

- **🔍 Drug Explorer** — search any drug, see its conditions, side effects, brand names, and safety flags
- **📊 Insights Dashboard** — the key EDA findings: top conditions, rating patterns, side-effect frequency, correlations
- **🧬 Side-Effect Clusters (NLP)** — drugs grouped by *type* of side-effect profile using TF-IDF + KMeans
- **🤖 ML Predictions** — try the trained rating-band and drug-class prediction models live
- **🔄 Drug Recommender** — pick a drug, get similar alternatives that treat the same condition

---
**⚠️ Disclaimer:** This is an academic/analytical project built on a public dataset.
It is **not** medical advice. Always consult a licensed healthcare professional
before making any decisions about medication.
""")

if st.checkbox("Show raw dataset sample"):
    st.dataframe(df.drop(columns=[
        "cluster_x", "cluster_y", "side_effects_clean_text"
    ], errors="ignore").head(50), use_container_width=True)
