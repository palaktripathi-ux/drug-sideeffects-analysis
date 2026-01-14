# ================================
# STEP 1: Import Required Libraries
# ================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# This line makes plots look better
sns.set_style("whitegrid")


# ================================
# STEP 2: App Title & Description
# ================================

st.title("💊 Drug, Side Effects & Medical Condition Analysis")

st.markdown("""
This interactive dashboard analyzes drugs, the medical conditions they treat,
and their reported side effects using real-world healthcare data.
""")


# ================================
# STEP 3: Load the Dataset
# ================================

# IMPORTANT:
# The CSV file is in the SAME folder as app.py on GitHub
# So we use only the filename (no folder path)

@st.cache_data
def load_data():
    return pd.read_csv("drugs_side_effects.csv")

df = load_data()


# ================================
# STEP 4: Show Raw Data (Optional)
# ================================

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())


# ================================
# STEP 5: Visual 1 – Top Medical Conditions
# ================================

st.subheader("📊 Top Medical Conditions Treated")

condition_counts = df["medical_condition"].value_counts().head(10)

fig1, ax1 = plt.subplots()
condition_counts.plot(kind="bar", ax=ax1)
ax1.set_xlabel("Medical Condition")
ax1.set_ylabel("Number of Drugs")

st.pyplot(fig1)


# ================================
# STEP 6: Visual 2 – Most Common Side Effects
# ================================

st.subheader("⚠️ Most Common Side Effects")

# Split comma-separated side effects into individual values
side_effects = df["side_effects"].str.split(",").explode()
side_effects = side_effects.str.strip()

top_side_effects = side_effects.value_counts().head(10)

fig2, ax2 = plt.subplots()
top_side_effects.plot(kind="barh", ax=ax2)
ax2.set_xlabel("Frequency")

st.pyplot(fig2)


# ================================
# STEP 7: Visual 3 – Pie Chart (Top Conditions)
# ================================

st.subheader("🥧 Distribution of Top Medical Conditions")

top_conditions = df["medical_condition"].value_counts().head(5)

fig3, ax3 = plt.subplots()
ax3.pie(
    top_conditions,
    labels=top_conditions.index,
    autopct="%1.1f%%",
    startangle=90
)
ax3.axis("equal")

st.pyplot(fig3)


# ================================
# STEP 8: Interactive Filter (KEY FEATURE)
# ================================

st.subheader("🔍 Explore Drugs by Medical Condition")

selected_condition = st.selectbox(
    "Select a Medical Condition",
    sorted(df["medical_condition"].unique())
)

filtered_df = df[df["medical_condition"] == selected_condition]

st.write(f"Showing drugs for **{selected_condition}**")
st.dataframe(filtered_df)


# ================================
# STEP 9: Key Insights Section
# ================================

st.subheader("🧠 Key Insights")

st.markdown("""
- Pain and diabetes-related conditions appear frequently in the dataset  
- Certain drugs are linked to multiple side effects  
- Some medical conditions have limited drug options  
""")


# ================================
# STEP 10: Footer
# ================================

st.markdown("---")
st.markdown("👩‍💻 **Project by Palak Tripathi**")
