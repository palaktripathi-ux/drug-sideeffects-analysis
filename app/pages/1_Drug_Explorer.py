import streamlit as st
import pandas as pd

st.set_page_config(page_title="Drug Explorer", page_icon="🔍", layout="wide")

@st.cache_data
def load_data():
    return pd.read_pickle("data/drugs_with_clusters.pkl")

df = load_data()

st.title("🔍 Drug Explorer")

col_a, col_b = st.columns([2, 1])
with col_a:
    search = st.text_input("Search a drug by name", "")
with col_b:
    condition_filter = st.selectbox(
        "...or filter by medical condition",
        ["All"] + sorted(df["medical_condition"].dropna().unique().tolist())
    )

filtered = df.copy()
if condition_filter != "All":
    filtered = filtered[filtered["medical_condition"] == condition_filter]
if search:
    filtered = filtered[filtered["drug_name"].str.contains(search.lower(), na=False)]

st.write(f"**{filtered['drug_name'].nunique()}** matching drugs")

if len(filtered) == 0:
    st.warning("No drugs match your search/filter.")
    st.stop()

selected_drug = st.selectbox("Select a drug to view details", sorted(filtered["drug_name"].unique()))
row = filtered[filtered["drug_name"] == selected_drug].iloc[0]

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Rating", f"{row['rating']:.1f}/10" if pd.notna(row['rating']) else "N/A")
c2.metric("Reviews", int(row['no_of_reviews']) if pd.notna(row['no_of_reviews']) else 0)
c3.metric("Reported Side Effects", int(row['side_effect_count']))

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Generic name:** {row['generic_name']}")
    st.markdown(f"**Medical condition:** {row['medical_condition']}")
    st.markdown(f"**Drug class:** {row['primary_drug_class']}")
    st.markdown(f"**Rx/OTC status:** {row['rx_otc']}")
    st.markdown(f"**Pregnancy category:** {row['pregnancy_risk_label']}")
    st.markdown(f"**Alcohol interaction flagged:** {'⚠️ Yes' if row['alcohol']=='X' else 'No / Not flagged'}")
with col2:
    st.markdown("**Brand names:**")
    st.write(", ".join(row["brand_names_list"]) if row["brand_names_list"] else "—")
    st.markdown("**Related / alternative drugs (per source):**")
    st.write(", ".join(row["related_drugs_list"][:10]) if row["related_drugs_list"] else "—")

st.divider()
st.subheader("Reported Side Effects")
st.write(row["side_effects"] if pd.notna(row["side_effects"]) else "No side-effect text available.")

st.subheader("Medical Condition Description")
with st.expander("Read description"):
    st.write(row["medical_condition_description"] if pd.notna(row["medical_condition_description"]) else "N/A")
