# Drug, Side Effects & Medical Condition Analysis — Advanced Analytics Platform

A major-project-grade analytics platform built on the drugs.com "Drugs, Side Effects and
Medical Condition Dataset" (2,931 cleaned drug records, 47 medical conditions, 230 drug
classes). Extends the original EDA + Streamlit deployment with NLP, machine learning,
and a content-based recommender system.

## What's included

| Component | Location | Description |
|---|---|---|
| Data cleaning | `src/clean_data.py` | Detects/removes corrupted rows, splits multi-valued fields, engineers new features |
| EDA & insights | `src/eda_insights.py` | Computes all summary stats, saves charts + `insights.json` |
| NLP clustering | `src/nlp_clustering.py` | TF-IDF + KMeans on side-effect text → 6 side-effect profile clusters |
| ML models | `src/ml_models.py` | Rating-band classifier (Random Forest) + drug-class-from-text classifier (Logistic Regression) |
| Recommender | `src/recommender.py` | Content-based "similar drugs" system, cosine similarity |
| Streamlit app | `app/` | 6-page interactive web app |
| Dashboard | `dashboard/dashboard.html` | Standalone, self-contained analytics dashboard (open directly in any browser, no server needed) |
| Report | `report/Project_Report.docx` | Full written report: methodology, results, limitations, future work |


## Key results at a glance

- **2,912 unique drugs** across **47 medical conditions** and **230 drug classes**
- Average rating: **6.83 / 10**; **47%** of drugs carry an alcohol-interaction warning
- Side-effect count has almost **no correlation with rating** (r = 0.046) — more listed
  side effects doesn't mean a worse-perceived drug
- **6 NLP-derived side-effect clusters**: allergic/breathing reactions, cardiac/fainting
  reactions, GI upset, and two skin-irritation profiles
- Rating-band classifier: **56.7% accuracy** (4-class problem)
- Drug-class-from-text classifier: **81.2% accuracy** (47-class problem)
- Recommender validated against known equivalents (e.g. ibuprofen → Advil at 99.8% similarity)

## Disclaimer

This is an academic project for demonstration purposes only. It is **not** medical
advice. Always consult a licensed healthcare professional for any medication decisions.

## Data source

"Drugs, Side Effects and Medical Condition Dataset", originally scraped from drugs.com
and distributed via Kaggle.
