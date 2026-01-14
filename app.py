import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/raw/drugs_side_effects.csv")

# App title
st.title("💊 Drug, Side Effects & Medical Condition Dashboard")

st.markdown("""
This dashboard analyzes drugs, medical conditions, and reported side effects.
""")

# Show dataset
if st.checkbox("Show Raw Dataset"):
    st.dataframe(df)

# -----------------------------
# Top Medical Conditions
# -----------------------------
st.subheader("📊 Top Medical Conditions")

condition_counts = df['medical_condition'].value_counts().head(10)

fig, ax = plt.subplots()
condition_counts.plot(kind='bar', ax=ax)
plt.xlabel("Medical Condition")
plt.ylabel("Count")

st.pyplot(fig)

# -----------------------------
# Drug Selection
# -----------------------------
st.subheader("🔍 Explore Drug Details")

selected_drug = st.selectbox(
    "Select a drug",
    df['drug_name'].unique()
)

drug_data = df[df['drug_name'] == selected_drug]

st.write("### Medical Conditions Treated:")
st.write(drug_data['medical_condition'].unique())

st.write("### Reported Side Effects:")
st.write(drug_data['side_effects'].unique())
