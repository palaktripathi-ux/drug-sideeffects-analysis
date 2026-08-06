import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Insights Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    return pd.read_pickle("data/drugs_with_clusters.pkl")

df = load_data()

st.title("📊 Insights Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Conditions & Classes", "Ratings", "Side Effects", "Safety Flags"]
)

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        top_cond = df["medical_condition"].value_counts().head(12).reset_index()
        top_cond.columns = ["medical_condition", "count"]
        fig = px.bar(top_cond, x="count", y="medical_condition", orientation="h",
                     title="Top 12 Medical Conditions by Number of Drugs")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        top_class = df["primary_drug_class"].value_counts().head(12).reset_index()
        top_class.columns = ["drug_class", "count"]
        fig = px.bar(top_class, x="count", y="drug_class", orientation="h",
                     title="Top 12 Drug Classes by Number of Drugs", color_discrete_sequence=["#f97316"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Key finding:** the most-treated condition is far more common than the rest, "
                "showing the dataset (and the drugs.com catalogue it's from) is weighted toward "
                "high-prevalence conditions rather than rare diseases.")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df.dropna(subset=["rating"]), x="rating", nbins=20,
                            title="Distribution of Drug Ratings (0-10)", color_discrete_sequence=["#22c55e"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        band_counts = df["rating_band"].value_counts().reset_index()
        band_counts.columns = ["rating_band", "count"]
        fig = px.pie(band_counts, names="rating_band", values="count", title="Rating Band Breakdown", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Which conditions have the best/worst-rated treatments?")
    cond_rating = (
        df.groupby("medical_condition")
        .agg(avg_rating=("rating", "mean"), drug_count=("drug_name", "nunique"))
        .dropna().query("drug_count >= 5").sort_values("avg_rating", ascending=False)
    )
    c1, c2 = st.columns(2)
    c1.markdown("**Best-treated (highest avg. rating)**")
    c1.dataframe(cond_rating.head(8).round(2), use_container_width=True)
    c2.markdown("**Worst-treated (lowest avg. rating)**")
    c2.dataframe(cond_rating.tail(8).round(2), use_container_width=True)

    corr = df.dropna(subset=["rating", "side_effect_count"])["rating"].corr(
        df.dropna(subset=["rating", "side_effect_count"])["side_effect_count"]
    )
    st.info(f"Correlation between number of reported side effects and rating: **{corr:.3f}** "
            f"({'weak' if abs(corr) < 0.3 else 'moderate'} relationship - more side effects "
            f"listed doesn't strongly predict a worse rating, likely because thorough labeling "
            f"and drug efficacy are independent).")

with tab3:
    from collections import Counter
    import re
    counter = Counter()
    for lst in df["side_effects_list"]:
        for s in lst:
            s_clean = re.sub(r"\s+", " ", s.strip().lower())
            if len(s_clean) >= 3:
                counter[s_clean] += 1
    top_se = pd.DataFrame(counter.most_common(15), columns=["side_effect", "count"])
    fig = px.bar(top_se, x="count", y="side_effect", orientation="h",
                 title="Top 15 Most Frequently Reported Side Effects", color_discrete_sequence=["#ef4444"])
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Drugs with the most reported side effects")
    most_se = df.sort_values("side_effect_count", ascending=False).drop_duplicates("drug_name")[
        ["drug_name", "medical_condition", "side_effect_count"]
    ].head(10)
    st.dataframe(most_se, use_container_width=True)

with tab4:
    c1, c2, c3 = st.columns(3)
    with c1:
        rx = df["rx_otc"].value_counts().reset_index()
        rx.columns = ["status", "count"]
        fig = px.pie(rx, names="status", values="count", title="Prescription vs OTC")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        preg = df["pregnancy_category"].value_counts().reset_index()
        preg.columns = ["category", "count"]
        fig = px.pie(preg, names="category", values="count", title="Pregnancy Risk Category")
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        alc = df["alcohol"].value_counts().reset_index()
        alc.columns = ["flag", "count"]
        fig = px.pie(alc, names="flag", values="count", title="Alcohol Interaction Flag ('X' = warning)")
        st.plotly_chart(fig, use_container_width=True)

    alcohol_pct = (df["alcohol"] == "X").mean() * 100
    st.warning(f"**{alcohol_pct:.1f}%** of drugs in the dataset carry an explicit alcohol-interaction warning.")
