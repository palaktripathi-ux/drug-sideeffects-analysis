import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("drugs_side_effects.csv")

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
st.markdown("""
This dashboard analyzes drugs, medical conditions,
and reported side effects using real-world data.
""")
condition_counts = df["medical_condition"].value_counts().head(10)
fig, ax = plt.subplots()
condition_counts.plot(kind="bar", ax=ax)
st.pyplot(fig)
side_effects = df["side_effects"].str.split(",").explode()
top_side_effects = side_effects.value_counts().head(10)
top_conditions = df["medical_condition"].value_counts().head(5)
ax.pie(top_conditions, labels=top_conditions.index, autopct="%1.1f%%")
condition = st.selectbox(
    "Select a Medical Condition",
    df["medical_condition"].unique()
)
filtered_df = df[df["medical_condition"] == condition]
st.dataframe(filtered_df)
st.write("""
• Pain is the most treated condition  
• Some side effects repeat across drugs  
• Certain drugs treat multiple conditions  
""")
