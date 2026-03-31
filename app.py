import streamlit as st
import numpy as np
import pickle
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Judicial AI System",
    page_icon="⚖️",
    layout="wide"
)

# ==============================
# STYLING
# ==============================
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    text-align: center;
    color: #1f77b4;
}
.card {
    padding: 1rem;
    border-radius: 10px;
    background-color: #ffffff;
    margin-bottom: 1rem;
    color: #000000;   
}
.priority-critical {background:#dc3545;color:white;padding:8px;border-radius:5px;}
.priority-high {background:#fd7e14;color:white;padding:8px;border-radius:5px;}
.priority-medium {background:#ffc107;color:black;padding:8px;border-radius:5px;}
.priority-low {background:#28a745;color:white;padding:8px;border-radius:5px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>⚖️ Judicial AI Decision Support System</h1>", unsafe_allow_html=True)

# ==============================
# LOAD MODELS
# ==============================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_embeddings():
    emb = np.load("data/embeddings/all_embeddings.npy")
    with open("data/embeddings/chunk_index.pkl", "rb") as f:
        index = pickle.load(f)
    return emb, index

@st.cache_resource
def load_priority_model():
    try:
        model = joblib.load("data/prioritization/models/prioritizer.pkl")
        encoder = joblib.load("data/prioritization/models/encoders.pkl")
        return model, encoder
    except:
        return None, None

@st.cache_data
def load_cases():
    try:
        return pd.read_csv("data/prioritization/labeled_cases.csv")
    except:
        return None

# ==============================
# INIT
# ==============================
model = load_embedding_model()
embeddings, chunk_index = load_embeddings()
priority_model, encoder = load_priority_model()
cases_df = load_cases()

# ==============================
# SEARCH FUNCTION
# ==============================
def search_cases(query, top_k=5):
    query_emb = model.encode([query])
    sims = cosine_similarity(query_emb, embeddings)[0]

    idxs = sims.argsort()[-top_k:][::-1]

    results = []
    for i in idxs:
        meta = chunk_index[i]
        results.append({
            "case_id": meta["case_id"],
            "score": float(sims[i]),
            "date": meta.get("date", "Unknown")
        })

    return results

# ==============================
# PRIORITY PREDICTION
# ==============================
def predict_priority(features):
    X = pd.DataFrame([{
        "case_type": encoder.transform([features["case_type"]])[0],
        "num_ipc_sections": features["num_ipc_sections"],
        "num_cpc_sections": features["num_cpc_sections"],
        "num_precedents": features["num_precedents"],
        "total_words": features["total_words"],
        "case_age_days": features["case_age_days"]
    }])

    score = priority_model.predict(X)[0]
    return round(float(score), 2)

def get_priority_class(score):
    if score >= 8:
        return "Critical", "priority-critical"
    elif score >= 6:
        return "High", "priority-high"
    elif score >= 4:
        return "Medium", "priority-medium"
    else:
        return "Low", "priority-low"

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("⚙️ System Overview")
st.sidebar.markdown("""
- 🔍 Semantic Search (RAG)
- ⚖️ Priority Prediction (ML)
- 📊 Case Analytics
""")

# ==============================
# TABS
# ==============================
tab1, tab2, tab3 = st.tabs(["🔍 Search", "⚖️ Prioritization", "📊 Analytics"])

# =========================================================
# TAB 1 — SEARCH
# =========================================================
with tab1:
    st.header("🔍 Legal Precedent Search")

    query = st.text_input("Enter query", placeholder="contract breach, murder, arbitration...")

    if st.button("Search"):
        results = search_cases(query)

        for r in results:
            st.markdown(f"""
            <div class='card'>
            <b>{r['case_id']}</b><br>
            Similarity: {r['score']:.3f}<br>
            Date: {r['date']}
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# TAB 2 — PRIORITY
# =========================================================
with tab2:
    st.header("⚖️ Case Prioritization")

    if priority_model is None:
        st.warning("Train model first")
    else:
        st.subheader("📋 Existing Cases")

        st.dataframe(cases_df.head(15), use_container_width=True)

        st.markdown("---")

        st.subheader("🎯 Predict New Case")

        col1, col2 = st.columns(2)

        with col1:
            case_type = st.selectbox("Case Type", ["criminal", "civil"])
            case_age = st.number_input("Case Age (days)", 0, 10000, 365)

        with col2:
            ipc = st.number_input("IPC Sections", 0, 20, 0)
            cpc = st.number_input("CPC Sections", 0, 20, 0)
            precedents = st.number_input("Precedents", 0, 50, 5)

        if st.button("Predict Priority"):
            features = {
                "case_type": case_type,
                "num_ipc_sections": ipc,
                "num_cpc_sections": cpc,
                "num_precedents": precedents,
                "case_age_days": case_age,
                "total_words": 3000
            }

            score = predict_priority(features)
            label, css = get_priority_class(score)

            st.markdown("### Result")

            col1, col2 = st.columns(2)
            col1.metric("Score", f"{score}/10")
            col2.markdown(f"<div class='{css}'>{label}</div>", unsafe_allow_html=True)

# =========================================================
# TAB 3 — ANALYTICS
# =========================================================
with tab3:
    st.header("📊 Analytics")

    if cases_df is not None:
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Cases", len(cases_df))
        col2.metric("Avg Score", round(cases_df["priority_score"].mean(), 2))
        col3.metric("Critical Cases", len(cases_df[cases_df["priority_category"] == "Critical"]))

        st.subheader("Priority Distribution")
        st.bar_chart(cases_df["priority_category"].value_counts())

        st.subheader("Case Type Distribution")
        st.bar_chart(cases_df["case_type"].value_counts())