import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="ML Predictions", page_icon="🤖", layout="wide")

@st.cache_resource
def load_models():
    rating_clf = joblib.load("models/rating_band_classifier.pkl")
    text_clf = joblib.load("models/drug_class_from_text.pkl")
    return rating_clf, text_clf

@st.cache_data
def load_data():
    return pd.read_pickle("data/drugs_with_clusters.pkl")

@st.cache_data
def load_results():
    with open("outputs/ml_results.json") as f:
        return json.load(f)

df = load_data()
rating_clf, text_clf = load_models()
results = load_results()

st.title("🤖 Machine Learning Predictions")

tab1, tab2 = st.tabs(["Rating-Band Predictor", "Drug-Class-from-Text Predictor"])

with tab1:
    st.subheader("Predict a drug's likely rating band from its clinical profile")
    r = results["rating_band_model"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", r["accuracy"])
    c2.metric("Macro Precision", r["macro_precision"])
    c3.metric("Macro Recall", r["macro_recall"])
    c4.metric("Macro F1", r["macro_f1"])
    st.caption(f"Random Forest classifier, trained on {r['n_train']} drugs, tested on {r['n_test']} held-out drugs.")

    st.markdown("**Try it:** fill in a hypothetical drug's profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        condition = st.selectbox("Medical condition", sorted(df["medical_condition"].dropna().unique()))
        drug_class = st.selectbox("Primary drug class", sorted(df["primary_drug_class"].dropna().unique()))
    with col2:
        rx_otc = st.selectbox("Rx/OTC status", sorted(df["rx_otc"].dropna().unique()))
        pregnancy = st.selectbox("Pregnancy category", sorted(df["pregnancy_category"].dropna().unique()))
    with col3:
        alcohol = st.selectbox("Alcohol interaction flag", sorted(df["alcohol"].dropna().unique()))
        csa = st.selectbox("Controlled substance schedule (CSA)", sorted(df["csa"].dropna().astype(str).unique()))

    col4, col5, col6 = st.columns(3)
    side_effect_count = col4.slider("Number of reported side effects", 0, 80, 20)
    brand_count = col5.slider("Number of brand names", 0, 30, 3)
    no_of_reviews = col6.slider("Number of user reviews", 0, 2000, 50)

    if st.button("Predict rating band", type="primary"):
        input_df = pd.DataFrame([{
            "medical_condition": condition, "primary_drug_class": drug_class,
            "rx_otc": rx_otc, "pregnancy_category": pregnancy, "alcohol": alcohol,
            "csa": str(csa), "side_effect_count": side_effect_count,
            "brand_count": brand_count, "no_of_reviews": no_of_reviews,
        }])
        pred = rating_clf.predict(input_df)[0]
        proba = rating_clf.predict_proba(input_df)[0]
        classes = rating_clf.named_steps["model"].classes_
        st.success(f"Predicted rating band: **{pred}**")
        proba_df = pd.DataFrame({"rating_band": classes, "probability": proba}).sort_values(
            "probability", ascending=False)
        st.bar_chart(proba_df.set_index("rating_band"))

    with st.expander("Top predictive features (model interpretability)"):
        st.json(r["top_features"])

with tab2:
    st.subheader("Predict a drug's class purely from its side-effect description")
    r2 = results["drug_class_text_model"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", r2["accuracy"])
    c2.metric("Macro Precision", r2["macro_precision"])
    c3.metric("Macro Recall", r2["macro_recall"])
    c4.metric("Macro F1", r2["macro_f1"])
    st.caption(f"Logistic Regression + TF-IDF, {r2['n_classes']} drug classes with sufficient examples, "
               f"trained on {r2['n_train']}, tested on {r2['n_test']} drugs.")

    st.markdown("**Try it:** paste a side-effect description and see the predicted drug class")
    example = df.dropna(subset=["side_effects"]).sample(1, random_state=None)["side_effects"].values[0]
    user_text = st.text_area("Side-effect text", value=example, height=150)

    if st.button("Predict drug class"):
        import re
        stopwords = {"may","call","doctor","signs","sign","symptoms","symptom","include",
                     "including","common","serious","severe","if","you","your","have","occur",
                     "other","seek","medical","treatment","at","once","or","and","of","the","a",
                     "an","in","to","with","this","that","reaction","effects","effect","side",
                     "not","check","some","any","professional","get","more","than","worse",
                     "immediately","away","such","these","also","can","for","are","who","use","using"}
        cleaned = re.sub(r"[^a-z\s]", " ", user_text.lower())
        cleaned = " ".join(t for t in cleaned.split() if t not in stopwords and len(t) > 2)
        vec = text_clf["vectorizer"].transform([cleaned])
        pred = text_clf["model"].predict(vec)[0]
        proba = text_clf["model"].predict_proba(vec)[0]
        top_idx = proba.argsort()[::-1][:5]
        classes = text_clf["model"].classes_
        st.success(f"Predicted drug class: **{pred}**")
        top_df = pd.DataFrame({
            "drug_class": [classes[i] for i in top_idx],
            "probability": [proba[i] for i in top_idx],
        })
        st.bar_chart(top_df.set_index("drug_class"))
