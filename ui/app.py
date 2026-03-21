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
    LLMLoadError,
    DEFAULT_KNOWLEDGE_PATH,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = _PROJECT_ROOT / "models"

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
    from .pkl files in the project root.

    Returns a dict with keys: rf_model, scaler, le_attack, le_industry, le_country
    """
    root = _PROJECT_ROOT

    required = {
        "rf_model": root / "rf_model.pkl",
        "scaler": root / "scaler.pkl",
        "le_attack": root / "le_attack.pkl",
        "le_industry": root / "le_industry.pkl",
        "le_country": root / "le_country.pkl",
    }

    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        st.error(
            f"**Missing model files:** {', '.join(missing)}\n\n"
            f"Run `model_training.py` (or `src/ml/train.py`) first to generate them."
        )
        st.stop()

    return {name: joblib.load(str(path)) for name, path in required.items()}


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
# 2.  ML prediction helper
# ═══════════════════════════════════════════════════════════════════════════

def predict_risk(models: dict, features: dict) -> float:
    """
    Encode + scale the user-provided features and return the predicted
    probability of high risk (0.0 – 1.0).
    """
    input_df = pd.DataFrame(
        [[
            features["financial_loss"],
            features["affected_users"],
            features["response_time"],
            features["vulnerability_score"],
            models["le_attack"].transform([features["attack_type"]])[0],
            models["le_industry"].transform([features["industry"]])[0],
            models["le_country"].transform([features["country"]])[0],
        ]],
        columns=FEATURE_COLUMNS,
    )
    scaled = models["scaler"].transform(input_df)
    return float(models["rf_model"].predict_proba(scaled)[0][1])


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
    collection, llm_available, llm_error = load_rag_engine()

    # --- LLM status banner ---
    if not llm_available:
        st.warning(
            "**Local LLM not loaded** — RAG explanations are unavailable.  \n"
            f"Reason: `{llm_error}`",
            icon="⚠️",
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

            # ---- ML prediction ----
            prob = predict_risk(models, features)

            st.divider()
            metric_col, badge_col = st.columns([1, 2])

            with metric_col:
                st.metric("High Risk Probability", f"{prob:.0%}")

            with badge_col:
                if prob < 0.3:
                    st.success("🟢 **Low Risk** — Routine monitoring recommended.")
                elif prob < 0.6:
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

        if not llm_available:
            st.info("The Knowledge Assistant requires a local LLM. See the warning above.")
        else:
            question = st.text_input(
                "Ask a cybersecurity question",
                placeholder="What is ransomware and how does it spread?",
            )
            if question:
                with st.spinner("Thinking …"):
                    answer = generate_threat_explanation(
                        prediction_prob=0.0,
                        features={
                            "attack_type": "N/A",
                            "industry": "N/A",
                            "country": "N/A",
                            "financial_loss": 0,
                            "affected_users": 0,
                            "vulnerability_score": 0,
                            "response_time": 0,
                        },
                        user_query=question,
                        collection=collection,
                    )
                st.markdown(answer)


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
