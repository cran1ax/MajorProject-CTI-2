"""
Cybersecurity Threat Intelligence Dashboard
============================================
Inference-only Streamlit app.  No training code.

Loads pre-trained ML models (.pkl) and the ChromaDB + llama.cpp RAG engine,
then lets the user submit incident metrics, get a risk prediction, and receive
an LLM-generated explanation grounded in the knowledge base.

Run:
    streamlit run ui/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ---------------------------------------------------------------------------
# Make sure the `src` package is importable regardless of how Streamlit
# is launched (from project root or from the ui/ directory).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rag.engine import (
    init_chroma,
    get_or_create_collection,
    ingest_knowledge_file,
    init_llm,
    generate_threat_explanation,
    answer_question,
    LLMLoadError,
    DEFAULT_KNOWLEDGE_PATH,
)
from src.ml.predictor import predict_risk          # RF-based predictor
from src.dl.predictor import DeepRiskPredictor      # DL-based predictor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = _PROJECT_ROOT / "models"   # pkl files live here

ATTACK_TYPES = ["DDoS", "Insider Threat", "Malware", "Phishing", "Ransomware"]
INDUSTRIES = ["Education", "Finance", "Government", "Healthcare", "Technology"]
COUNTRIES = ["Germany", "India", "Japan", "UK", "USA"]

FEATURE_COLUMNS = [
    "financial_loss",
    "affected_users",
    "response_time",
    "vulnerability_score",
    "attack_enc",
    "industry_enc",
    "country_enc",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Cached resource loaders  (@st.cache_resource)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading ML models …")
def load_ml_models():
    """
    Load the pre-trained Random Forest, StandardScaler, and LabelEncoders
    from .pkl files stored in the ``models/`` directory.

    Returns a dict with keys: rf_model, scaler, le_attack, le_industry, le_country
    """
    required = {
        "rf_model":    MODELS_DIR / "rf_model.pkl",
        "scaler":      MODELS_DIR / "scaler.pkl",
        "le_attack":   MODELS_DIR / "le_attack.pkl",
        "le_industry": MODELS_DIR / "le_industry.pkl",
        "le_country":  MODELS_DIR / "le_country.pkl",
    }

    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        st.error(
            f"**Missing model files in `models/`:** {', '.join(missing)}\n\n"
            f"Run `python src/ml/train.py` first to generate them."
        )
        st.stop()

    return {name: joblib.load(str(path)) for name, path in required.items()}


@st.cache_resource(show_spinner="Loading deep-learning model …")
def load_dl_model():
    """
    Attempt to load the PyTorch MLP from models/dl_model.pt.

    Returns ``(predictor, error_msg)``:
    - If loaded OK  → ``(DeepRiskPredictor_instance, None)``
    - If missing    → ``(None, description_str)``
    """
    try:
        predictor = DeepRiskPredictor(models_dir=MODELS_DIR)
        predictor.load()
        return predictor, None
    except FileNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, f"Unexpected error loading DL model: {exc}"


@st.cache_resource(show_spinner="Initialising RAG engine …")
def load_rag_engine():
    """
    Initialise ChromaDB, ingest the knowledge base (idempotent), and
    attempt to load the local LLM.

    Returns (collection, llm_available: bool, llm_error: str | None).
    """
    client = init_chroma()
    collection = get_or_create_collection(client)

    # Ingest knowledge file if it exists (upsert = safe to re-run)
    try:
        ingest_knowledge_file(collection=collection)
    except FileNotFoundError:
        pass  # knowledge.txt not yet in place — retrieval returns empty

    # Try loading the LLM (may fail if .gguf is missing — that's OK)
    llm_available = True
    llm_error = None
    try:
        init_llm()
    except LLMLoadError as exc:
        llm_available = False
        llm_error = str(exc)

    return collection, llm_available, llm_error


# ═══════════════════════════════════════════════════════════════════════════
# 2.  ML prediction helper  (implementation lives in src/ml/predictor.py)
# ═══════════════════════════════════════════════════════════════════════════

# predict_risk() is imported from src.ml.predictor above — no local definition needed.


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Streamlit page
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Cybersecurity Threat Intelligence",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ Cybersecurity Threat Intelligence Dashboard")
    st.caption(
        "Predict incident risk with ML, then get an explanation "
        "grounded in the knowledge base via RAG."
    )

    # --- Load resources (cached after first run) ---
    models = load_ml_models()
    dl_model, dl_error = load_dl_model()
    collection, llm_available, llm_error = load_rag_engine()

    # --- LLM status banner ---
    if not llm_available:
        st.warning(
            "**Local LLM not loaded** — RAG explanations are unavailable.  \n"
            f"Reason: `{llm_error}`",
            icon="⚠️",
        )

    # --- DL model status (non-blocking) ---
    if dl_error:
        st.info(
            f"**Deep-learning model not loaded** — only Random Forest predictions shown.  \n"
            f"Run `python -m src.dl.train` to generate `models/dl_model.pt`.  \n"
            f"Reason: `{dl_error}`",
            icon="ℹ️",
        )

    # ────────────────────────────────────────────────────────────
    # Tabs
    # ────────────────────────────────────────────────────────────
    tab_predict, tab_url, tab_qa = st.tabs(
        ["📊 Incident Prediction", "🌐 URL Checker", "🤖 Knowledge Assistant"]
    )

    # ==================== TAB 1: Incident Prediction ====================
    with tab_predict:
        st.header("Incident Risk Prediction")

        with st.form("incident_form"):
            col1, col2 = st.columns(2)

            with col1:
                financial_loss = st.slider(
                    "Estimated Financial Loss ($ millions)",
                    min_value=0.0, max_value=200.0, value=30.0, step=1.0,
                )
                affected_users = st.slider(
                    "Number of Affected Users",
                    min_value=0, max_value=50_000, value=10_000, step=500,
                )

            with col2:
                response_time = st.slider(
                    "Response Time (hours)",
                    min_value=0.0, max_value=24.0, value=5.0, step=0.5,
                )
                vulnerability_score = st.slider(
                    "Vulnerability Score (1–10)",
                    min_value=1.0, max_value=10.0, value=5.0, step=0.5,
                )

            col3, col4, col5 = st.columns(3)
            with col3:
                attack_type = st.selectbox("Attack Type", ATTACK_TYPES)
            with col4:
                industry = st.selectbox("Industry", INDUSTRIES)
            with col5:
                country = st.selectbox("Country", COUNTRIES)

            user_query = st.text_input(
                "Ask a question about this prediction (optional)",
                placeholder="Why is this rated high risk? What should we do?",
            )

            submitted = st.form_submit_button("🔍 Predict & Explain", use_container_width=True)

        # --- Results (shown after form submission) ---
        if submitted:
            features = {
                "financial_loss": financial_loss,
                "affected_users": affected_users,
                "response_time": response_time,
                "vulnerability_score": vulnerability_score,
                "attack_type": attack_type,
                "industry": industry,
                "country": country,
            }

            # ---- RF prediction ----
            prob_rf = predict_risk(models, features)

            # ---- DL prediction (if model is loaded) ----
            prob_dl = None
            if dl_model is not None:
                # Build the encoded feature dict expected by DeepRiskPredictor
                dl_features = {
                    "financial_loss":     features["financial_loss"],
                    "affected_users":     features["affected_users"],
                    "response_time":      features["response_time"],
                    "vulnerability_score": features["vulnerability_score"],
                    "attack_enc":   models["le_attack"].transform([features["attack_type"]])[0],
                    "industry_enc": models["le_industry"].transform([features["industry"]])[0],
                    "country_enc":  models["le_country"].transform([features["country"]])[0],
                }
                try:
                    prob_dl = dl_model.predict(dl_features)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"DL prediction failed: {exc}")

            # Use RF probability as the canonical score for RAG
            prob = prob_rf

            st.divider()
            st.subheader("📊 Model Predictions")

            if prob_dl is not None:
                # Side-by-side comparison
                col_rf, col_dl, col_badge = st.columns([1, 1, 2])
                with col_rf:
                    st.metric("🌳 Random Forest", f"{prob_rf:.0%}")
                with col_dl:
                    st.metric("🧠 Deep Learning", f"{prob_dl:.0%}")
                avg_prob = (prob_rf + prob_dl) / 2
                with col_badge:
                    st.caption(f"Ensemble average: **{avg_prob:.0%}**")
                    if avg_prob < 0.3:
                        st.success("🟢 **Low Risk** — Routine monitoring recommended.")
                    elif avg_prob < 0.6:
                        st.warning("🟡 **Medium Risk** — Elevated alerting advised.")
                    else:
                        st.error("🔴 **High Risk** — Immediate response recommended.")
            else:
                # Only RF available
                metric_col, badge_col = st.columns([1, 2])
                with metric_col:
                    st.metric("🌳 Random Forest Probability", f"{prob_rf:.0%}")
                with badge_col:
                    if prob_rf < 0.3:
                        st.success("🟢 **Low Risk** — Routine monitoring recommended.")
                    elif prob_rf < 0.6:
                        st.warning("🟡 **Medium Risk** — Elevated alerting advised.")
                    else:
                        st.error("🔴 **High Risk** — Immediate response recommended.")

            # ---- RAG explanation ----
            if llm_available:
                query = user_query or "Explain this risk prediction and suggest mitigations."
                with st.spinner("Generating RAG explanation …"):
                    explanation = generate_threat_explanation(
                        prediction_prob=prob,
                        features=features,
                        user_query=query,
                        collection=collection,
                    )
                st.subheader("🤖 RAG Explanation")
                st.markdown(explanation)
            else:
                st.info(
                    "RAG explanation skipped — no local LLM loaded. "
                    "Place a `.gguf` model in the `models/` directory to enable it."
                )

    # ==================== TAB 2: URL Checker ====================
    with tab_url:
        st.header("🌐 URL Threat Checker")

        user_url = st.text_input(
            "Enter a URL to analyse",
            value="http://secure-login.bank-update.com/verify",
        )

        if st.button("Check URL Risk"):
            risk_score = 0
            if "@" in user_url:
                risk_score += 1
            if user_url.count("-") > 2:
                risk_score += 1
            if user_url.count(".") > 4:
                risk_score += 1
            if any(w in user_url.lower() for w in ["login", "verify", "secure", "bank"]):
                risk_score += 1
            if user_url.startswith("http://"):
                risk_score += 1

            url_prob = min(risk_score / 5, 1.0)
            st.metric("Malicious Probability", f"{url_prob:.0%}")

            if url_prob > 0.6:
                st.error("🚨 This URL is likely **malicious**.")
            elif url_prob > 0.3:
                st.warning("⚠️ This URL looks **suspicious**.")
            else:
                st.success("✅ This URL appears **safe**.")

    # ==================== TAB 3: Knowledge Assistant ====================
    with tab_qa:
        st.header("🤖 Cybersecurity Knowledge Assistant")
        st.caption(
            "Ask any cybersecurity question. Answers are retrieved instantly from the "
            "knowledge base — **no LLM required**. " +
            ("🟢 LLM also available for richer explanations in the Prediction tab."
             if llm_available else
             "_Activate an LLM for full RAG explanations in the Prediction tab._")
        )

        # Suggested questions for quick exploration
        QUICK_QUESTIONS = [
            "What is ransomware and how does it spread?",
            "How does phishing work?",
            "What is MITRE ATT&CK?",
            "Explain DDoS attacks.",
            "What is an APT (Advanced Persistent Threat)?",
            "How does SQL injection work?",
            "What is the NIST Incident Response framework?",
            "What are insider threats?",
        ]

        col_q, col_btn = st.columns([4, 1])
        with col_q:
            question = st.text_input(
                "Ask a cybersecurity question",
                placeholder="What is ransomware and how does it spread?",
                label_visibility="collapsed",
            )
        with col_btn:
            n_results = st.selectbox(
                "Passages", [3, 4, 5, 6], index=1, label_visibility="collapsed"
            )

        st.caption("Quick questions:")
        cols = st.columns(4)
        for idx, q in enumerate(QUICK_QUESTIONS):
            if cols[idx % 4].button(q, key=f"qq_{idx}", use_container_width=True):
                question = q

        if question:
            with st.spinner("Searching knowledge base …"):
                answer = answer_question(
                    question=question,
                    collection=collection,
                    n_results=int(n_results),
                )
            st.markdown(answer)


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
