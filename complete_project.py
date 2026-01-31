# ==================== CYBERSECURITY THREAT DETECTION WITH RAG ====================

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
import re
import urllib.parse as urlparse

warnings.filterwarnings("ignore")

# ==================== DATA GENERATION ====================

def generate_cybersecurity_data(n_samples=2000):
    attack_types = ['Phishing', 'Ransomware', 'DDoS', 'Malware', 'Insider Threat']
    industries = ['Finance', 'Healthcare', 'Government', 'Education', 'Technology']
    countries = ['USA', 'UK', 'Germany', 'India', 'Japan']

    df = pd.DataFrame({
        'financial_loss': np.random.exponential(50, n_samples),
        'affected_users': np.random.poisson(10000, n_samples),
        'response_time': np.random.gamma(2, 2, n_samples),
        'vulnerability_score': np.random.uniform(1, 10, n_samples),
        'attack_type': np.random.choice(attack_types, n_samples),
        'industry': np.random.choice(industries, n_samples),
        'country': np.random.choice(countries, n_samples)
    })

    df['high_risk'] = (
        (df.financial_loss > 80) |
        (df.affected_users > 30000) |
        (df.response_time > 8)
    ).astype(int)

    return df

df = generate_cybersecurity_data()

# ==================== FEATURE ENGINEERING ====================

le_attack = LabelEncoder()
le_industry = LabelEncoder()
le_country = LabelEncoder()

df['attack_enc'] = le_attack.fit_transform(df['attack_type'])
df['industry_enc'] = le_industry.fit_transform(df['industry'])
df['country_enc'] = le_country.fit_transform(df['country'])

X = df[
    ['financial_loss', 'affected_users', 'response_time',
     'vulnerability_score', 'attack_enc', 'industry_enc', 'country_enc']
]
y = df['high_risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save models
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_attack, "le_attack.pkl")
joblib.dump(le_industry, "le_industry.pkl")
joblib.dump(le_country, "le_country.pkl")

# ==================== STREAMLIT APP ====================

def run_dashboard():
    st.set_page_config(
        page_title="Cybersecurity Threat Detection with RAG",
        layout="wide"
    )

    st.title("🛡️ Cybersecurity Threat Detection with RAG")
    st.caption(
        "This system predicts cybersecurity incident risk using Machine Learning "
        "and explains results using Retrieval-Augmented Generation (RAG)."
    )

    tab1, tab2, tab3 = st.tabs(
        ["📊 Incident Prediction", "🌐 URL Checker", "🤖 RAG Assistant"]
    )

    # ==================== TAB 1 ====================
    with tab1:
        st.header("Incident Risk Prediction")

        col1, col2 = st.columns(2)

        with col1:
            financial_loss = st.slider(
                "Estimated Financial Loss (in millions)",
                0.0, 200.0, 30.0
            )
            affected_users = st.slider(
                "Number of Affected Users",
                0, 50000, 10000
            )

        with col2:
            response_time = st.slider(
                "Response Time (hours)",
                0.0, 24.0, 5.0
            )
            vulnerability_score = st.slider(
                "Vulnerability Score (1–10)",
                1.0, 10.0, 5.0
            )

        attack_type = st.selectbox("Attack Type", le_attack.classes_)
        industry = st.selectbox("Industry", le_industry.classes_)
        country = st.selectbox("Country", le_country.classes_)

        input_df = pd.DataFrame([[
            financial_loss,
            affected_users,
            response_time,
            vulnerability_score,
            le_attack.transform([attack_type])[0],
            le_industry.transform([industry])[0],
            le_country.transform([country])[0]
        ]], columns=X.columns)

        scaled_input = scaler.transform(input_df)
        prob = rf_model.predict_proba(scaled_input)[0][1]

        st.metric("High Risk Probability", f"{prob:.2%}")

        if prob < 0.3:
            st.success("🟢 Low Risk Incident")
        elif prob < 0.6:
            st.warning("🟡 Medium Risk Incident")
        else:
            st.error("🔴 High Risk Incident")

        # ---------- RAG EXPLANATION ----------
        from rag_engine import build_vector_store, ask_rag

        @st.cache_resource
        def load_rag():
            with open("knowledge.txt", "r", encoding="utf-8") as f:
                return build_vector_store(f.read())

        vector_db = load_rag()

        with st.expander("Why is this risk predicted? (RAG Explanation)"):
            explanation = ask_rag(
                vector_db,
                f"Explain why high response time {response_time} hours, "
                f"vulnerability score {vulnerability_score}, "
                f"and financial loss {financial_loss} million increase cyber risk."
            )
            st.write(explanation)

    # ==================== TAB 2 ====================
    with tab2:
        st.header("🌐 URL Threat Checker")

        user_url = st.text_input(
            "Enter URL to analyze",
            value="http://secure-login.bank-update.com/verify"
        )

        if st.button("Check URL Risk"):
            risk_score = 0

            if "@" in user_url:
                risk_score += 1
            if user_url.count("-") > 2:
                risk_score += 1
            if user_url.count(".") > 4:
                risk_score += 1
            if any(word in user_url.lower() for word in ["login", "verify", "secure", "bank"]):
                risk_score += 1
            if user_url.startswith("http://"):
                risk_score += 1

            probability = min(risk_score / 5, 1.0)

            st.metric("Malicious Probability", f"{probability:.2%}")

            if probability > 0.6:
                st.error("🚨 This URL is likely malicious.")
            elif probability > 0.3:
                st.warning("⚠️ This URL looks suspicious.")
            else:
                st.success("✅ This URL appears safe.")

    # ==================== TAB 3 ====================
    with tab3:
        st.header("🤖 Cybersecurity Knowledge Assistant (RAG)")

        question = st.text_input(
            "Ask a cybersecurity question",
            placeholder="What is phishing?"
        )

        if question:
            answer = ask_rag(vector_db, question)
            st.write(answer)


# ==================== RUN ====================

if __name__ == "__main__":
    run_dashboard()
