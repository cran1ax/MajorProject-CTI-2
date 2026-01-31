# ==================== COMPLETE CYBERSECURITY PROJECT (v4.0 - With Gemini Bot) ====================
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import joblib
import re
import urllib.parse as urlparse
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import requests # Added for API calls
import json # Added for API calls

warnings.filterwarnings('ignore')

# Helper function for generating synthetic URL data
def generate_mock_url_data(n_samples=500):
    """Generates synthetic URLs and their binary risk labels."""
    safe_domains = ['google.com', 'microsoft.com', 'wikipedia.org', 'streamlit.io', 'github.com']
    mal_patterns = ['login', 'verify', 'update', 'docs', 'secure', 'bank']
    
    urls = []
    risks = []
    
    for i in range(n_samples):
        is_malicious = np.random.rand() < 0.3 # 30% malicious
        if is_malicious:
            # Generate malicious-looking URL
            domain = np.random.choice(safe_domains)
            path = np.random.choice(mal_patterns) + str(np.random.randint(100, 999)) + ".html"
            query_length = np.random.randint(5, 15)
            url = f"http://{domain}-{np.random.randint(10, 99)}/{path}?id={query_length}"
            if np.random.rand() < 0.4: url = url.replace('http', 'https')
            if np.random.rand() < 0.2: url = "http://192.168.1.1/" + path # Use IP
            if np.random.rand() < 0.2: url = url.replace('.', '..')
            risks.append(1)
        else:
            # Generate safe-looking URL
            domain = np.random.choice(safe_domains)
            path = np.random.choice(['/search', '/about', '/info', '/blog'])
            url = f"https://www.{domain}{path}/{np.random.randint(1000, 9999)}"
            risks.append(0)
        urls.append(url)
        
    return urls, np.array(risks)


# ==================== 1. DATA GENERATION (Incident Metrics) ====================
def generate_cybersecurity_data(n_samples=2000):
    np.random.seed(42)
    attack_types = ['Phishing', 'Ransomware', 'DDoS', 'Malware', 'Insider Threat']
    industries = ['Finance', 'Healthcare', 'Government', 'Education', 'Technology']
    countries = ['USA', 'UK', 'Germany', 'India', 'Japan']
    
    data = {
        'timestamp': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'attack_type': np.random.choice(attack_types, n_samples),
        'target_industry': np.random.choice(industries, n_samples),
        'country': np.random.choice(countries, n_samples), 
        'financial_loss': np.random.exponential(50, n_samples),
        'affected_users': np.random.poisson(10000, n_samples),
        'response_time': np.random.gamma(2, 2, n_samples),
        'data_breach_size': np.random.lognormal(8, 2, n_samples),
        'network_traffic': np.random.normal(1000, 300, n_samples),
        'vulnerability_score': np.random.uniform(1, 10, n_samples),
    }

    risk_conditions = (
        (data['financial_loss'] > 80) |
        (data['affected_users'] > 30000) |
        (data['response_time'] > 8)
    )
    data['high_risk_incident'] = risk_conditions.astype(int)
    df = pd.DataFrame(data)
    return df

df = generate_cybersecurity_data()


# ==================== 2. FEATURE ENGINEERING (Incident Metrics) ====================
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['loss_per_user'] = df['financial_loss'] / (df['affected_users'] + 1)

feature_columns = [
    'financial_loss', 'affected_users', 'response_time',
    'data_breach_size', 'network_traffic', 'vulnerability_score', 'loss_per_user'
]

le_attack = LabelEncoder()
le_industry = LabelEncoder()
le_country = LabelEncoder()

df['attack_type_encoded'] = le_attack.fit_transform(df['attack_type'])
df['industry_encoded'] = le_industry.fit_transform(df['target_industry'])
df['country_encoded'] = le_country.fit_transform(df['country'])

feature_columns.extend(['attack_type_encoded', 'industry_encoded', 'country_encoded', 'year', 'month'])
X = df[feature_columns]
y = df['high_risk_incident']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==================== 3. MAIN INCIDENT MODEL TRAINING ====================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression
baseline_model = LogisticRegression(random_state=42)
baseline_model.fit(X_train, y_train)
baseline_accuracy = accuracy_score(y_test, baseline_model.predict(X_test))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_accuracy = accuracy_score(y_test, gb_model.predict(X_test))

# Deep Neural Network (Incident Data)
dnn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
dnn_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
dnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
dnn_pred_prob = dnn_model.predict(X_test, verbose=0)
dnn_pred = (dnn_pred_prob > 0.5).astype(int)
dnn_accuracy = accuracy_score(y_test, dnn_pred)


# ==================== 4. URL FEATURE EXTRACTOR & DEDICATED MODEL ====================
def extract_url_features(url):
    """Extracts structural features from a URL."""
    features = {}
    features['url_length'] = len(url)
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_special_chars'] = len(re.findall(r'[\W_]', url))
    features['has_https'] = int(url.startswith("https"))
    
    netloc = urlparse.urlparse(url).netloc
    features['has_ip'] = int(bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', netloc))) 
    
    features['num_subdirs'] = url.count('/')
    features['num_dots'] = url.count('.')
    features['has_at_symbol'] = int('@' in url)
    features['has_hyphen'] = int('-' in url)
    features['domain_length'] = len(netloc)
    return pd.DataFrame([features])

# Generate mock data for URL classifier
urls, y_url_risk = generate_mock_url_data()
X_url_features_raw = pd.concat([extract_url_features(u) for u in urls], ignore_index=True)
url_feature_cols = X_url_features_raw.columns

# Scale URL features
url_scaler = StandardScaler()
X_url_scaled = url_scaler.fit_transform(X_url_features_raw)

# Split URL data
X_url_train, X_url_test, y_url_train, y_url_test = train_test_split(
    X_url_scaled, y_url_risk, test_size=0.2, random_state=42, stratify=y_url_risk
)

# Train dedicated URL Deep Neural Network
url_dnn_model = Sequential([
    Dense(16, activation='relu', input_shape=(X_url_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
url_dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
url_dnn_model.fit(X_url_train, y_url_train, epochs=15, batch_size=16, verbose=0)

# Evaluate URL Model (for logging)
url_pred_prob = url_dnn_model.predict(X_url_test, verbose=0)
url_accuracy = accuracy_score(y_url_test, (url_pred_prob > 0.5).astype(int))
url_auc = roc_auc_score(y_url_test, url_pred_prob)


# ==================== 5. MODEL COMPARISON & SAVING ====================
y_pred_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]
dnn_auc = roc_auc_score(y_test, dnn_pred_prob)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Deep Neural Network (Incident)'],
    'Accuracy': [baseline_accuracy, rf_accuracy, gb_accuracy, dnn_accuracy],
    'AUC Score': [
        roc_auc_score(y_test, y_pred_proba_baseline),
        roc_auc_score(y_test, y_pred_proba_rf),
        roc_auc_score(y_test, y_pred_proba_gb),
        dnn_auc
    ]
})

# Save all models and encoders/scalers
joblib.dump(baseline_model, 'baseline_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(gb_model, 'gradient_boosting_model.pkl')
dnn_model.save('incident_dl_model.h5')
joblib.dump(scaler, 'incident_scaler.pkl')
joblib.dump(le_attack, 'label_encoder_attack.pkl')
joblib.dump(le_industry, 'label_encoder_industry.pkl')
joblib.dump(le_country, 'label_encoder_country.pkl')
url_dnn_model.save('url_dl_model.h5')
joblib.dump(url_scaler, 'url_scaler.pkl')
joblib.dump(url_feature_cols, 'url_feature_cols.pkl')


# ==================== LLM BOT LOGIC ====================

def get_gemini_response(prompt, history):
    """Calls the Gemini API to get a grounded response."""
    
    # 1. Define the API configuration
    system_prompt = "You are a Senior Cybersecurity Consultant named 'Gemini Defense Bot'. Your goal is to provide clear, actionable, and up-to-date advice on cybersecurity threats, incident response, and best practices. Use a professional, helpful, and concise tone. Always cite your sources if you use real-time information. Do not mention your internal system name or model version."
    
    API_KEY = "" # Must be empty string
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

    # 2. Format the chat history for the API payload
    chat_contents = [
        {"role": "user" if msg["role"] == "user" else "model", "parts": [{"text": msg["content"]}]}
        for msg in history
    ]
    chat_contents.append({"role": "user", "parts": [{"text": prompt}]})

    # 3. Construct the API payload
    payload = {
        "contents": chat_contents,
        "tools": [{"google_search": {}}], # Enable search grounding
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }

    # 4. Make the request with exponential backoff
    max_retries = 3
    delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL, 
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=20
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            
            # Extract content and grounding metadata
            candidate = result.get('candidates', [{}])[0]
            text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'Error: Could not retrieve text.')
            
            sources = []
            grounding_metadata = candidate.get('groundingMetadata')
            if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                sources = grounding_metadata['groundingAttributions']
            
            # Format output text with citations
            if sources:
                citations = "\n\n**Sources:**\n"
                for i, source in enumerate(sources):
                    title = source.get('web', {}).get('title', 'Untitled')
                    uri = source.get('web', {}).get('uri', '#')
                    citations += f"[{i+1}] [{title}]({uri})\n"
                text += citations
                
            return text

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                # print(f"API request failed: {e}. Retrying in {delay}s...") # Suppress debug logging as per instructions
                time.sleep(delay)
                delay *= 2
            else:
                return f"Sorry, the connection failed after multiple retries. Please check your network or try again later. ({e})"
        except Exception as e:
            return f"An unexpected error occurred while processing the response: {e}"

    return "An unknown error occurred during API communication."


# ==================== 6. STREAMLIT DASHBOARD ====================
def run_dashboard():
    # Ensure time is imported for backoff sleep
    import time
    
    st.set_page_config(page_title="Cybersecurity Threat Predictor", layout="wide")
    st.title("🛡️ Integrated Cybersecurity Prediction Dashboard")
    st.markdown("---")

    # Initialize chat history for the bot
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "model", "content": "Hello! I am Gemini Defense Bot. I can answer your questions about the latest cyber threats, incident response, and security best practices."}
        ]

    # Load all models and components
    @st.cache_resource
    def load_all_models():
        try:
            # Main Incident Models
            models = {
                'Logistic Regression': joblib.load('baseline_model.pkl'),
                'Random Forest': joblib.load('random_forest_model.pkl'),
                'Gradient Boosting': joblib.load('gradient_boosting_model.pkl'),
                'Deep Neural Network (Incident)': load_model('incident_dl_model.h5')
            }
            incident_scaler = joblib.load('incident_scaler.pkl')
            
            # Label Encoders
            le_attack = joblib.load('label_encoder_attack.pkl')
            le_industry = joblib.load('label_encoder_industry.pkl')
            le_country = joblib.load('label_encoder_country.pkl')
            
            # URL Detection Models
            url_model = load_model('url_dl_model.h5')
            url_scaler = joblib.load('url_scaler.pkl')
            url_feature_cols = joblib.load('url_feature_cols.pkl')
            
            return models, incident_scaler, le_attack, le_industry, le_country, url_model, url_scaler, url_feature_cols
        except Exception as e:
            # If models don't exist, Streamlit fails gracefully
            # User will be prompted to run python complete_project.py first
            st.warning("⚠️ Waiting for model files. Run the script once with `python complete_project.py` to generate models.")
            return None, None, None, None, None, None, None, None

    models, incident_scaler, le_attack, le_industry, le_country, url_model, url_scaler, url_feature_cols = load_all_models()
    
    # Create three tabs
    tab1, tab2, tab3 = st.tabs(["📊 Incident Risk Analyzer", "🌐 URL Threat Checker", "🤖 Ask the Bot"])

    # ========== TAB 1: INCIDENT METRIC PREDICTION ==========
    with tab1:
        st.header("Incident Risk Analysis")
        if models:
            st.subheader("Input Threat Parameters")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                financial_loss = st.slider("Financial Loss ($M)", 0.0, 200.0, 50.0, key='fl')
                affected_users = st.slider("Affected Users", 0, 50000, 10000, key='au')
                
            with col2:
                response_time = st.slider("Response Time (hours)", 0.0, 24.0, 5.0, key='rt')
                data_breach_size = st.slider("Data Breach Size (MB)", 0, 5000, 1000, key='dbs')
                
            with col3:
                network_traffic = st.slider("Network Traffic (GB)", 0.0, 2000.0, 800.0, key='nt')
                vulnerability_score = st.slider("Vulnerability Score", 1, 10, 5, key='vs')
            
            with col4:
                attack_type = st.selectbox("Attack Type", le_attack.classes_, key='at')
                industry = st.selectbox("Target Industry", le_industry.classes_, key='ti')
                country = st.selectbox("Country", le_country.classes_, key='ct') 
                year = st.slider("Year", 2020, 2024, 2023, key='yr')
                month = st.slider("Month", 1, 12, 6, key='mo')
            
            # Prepare input
            loss_per_user = financial_loss / (affected_users + 1)
            
            input_data = pd.DataFrame({
                'financial_loss': [financial_loss],
                'affected_users': [affected_users],
                'response_time': [response_time],
                'data_breach_size': [data_breach_size],
                'network_traffic': [network_traffic],
                'vulnerability_score': [vulnerability_score],
                'loss_per_user': [loss_per_user],
                'attack_type_encoded': [le_attack.transform([attack_type])[0]],
                'industry_encoded': [le_industry.transform([industry])[0]],
                'country_encoded': [le_country.transform([country])[0]], 
                'year': [year],
                'month': [month]
            })
            
            input_scaled = incident_scaler.transform(input_data)
            
            st.markdown("---")
            st.subheader("Model Predictions")
            
            cols = st.columns(4)
            
            for i, (name, model) in enumerate(models.items()):
                if "Deep Neural Network" in name:
                    prob = model.predict(input_scaled, verbose=0)[0, 0]
                else:
                    prob = model.predict_proba(input_scaled)[0, 1]
                
                risk = "HIGH RISK" if prob > 0.5 else "LOW RISK"
                
                with cols[i]:
                    st.metric(f"Risk Probability ({name})", f"{prob:.2%}")
                    st.markdown(f"**Prediction:** {'<span style=\\"color:red; font-weight:bold;\\">'+risk+'</span>' if risk == 'HIGH RISK' else '<span style=\\"color:green; font-weight:bold;\\">'+risk+'</span>'}", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Model Performance Summary")
            st.dataframe(comparison_df.style.format({'Accuracy': '{:.3f}', 'AUC Score': '{:.3f}'}), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.bar(comparison_df, x='Model', y='Accuracy', color='Model', title="Model Accuracy")
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.bar(comparison_df, x='Model', y='AUC Score', color='Model', title="AUC Comparison")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("Model files are not yet available. Please run the script in bare mode first.")

    # ========== TAB 2: URL THREAT CHECKER (Dedicated DL Model) ==========
    with tab2:
        st.header("🌐 Dedicated URL Threat Detection")
        st.markdown("This feature uses a separate Deep Learning model trained specifically on URL structural characteristics to predict risk.")

        user_url = st.text_input("🔗 Enter URL to check:", value="http://secure-login.bank-update.com/verify-account", key='url_input')

        if st.button("🚨 Check URL Risk"):
            if user_url:
                try:
                    # 1. Extract features using the dedicated function
                    features = extract_url_features(user_url)
                    
                    # 2. Reorder features and scale using the dedicated URL scaler
                    features_aligned = features.reindex(columns=url_feature_cols, fill_value=0)
                    X_url_scaled_input = url_scaler.transform(features_aligned)
                    
                    # 3. Predict using the dedicated URL model
                    prob = url_model.predict(X_url_scaled_input, verbose=0)[0, 0]
                    prediction = 1 if prob > 0.5 else 0

                    st.subheader("Analysis Result")
                    
                    if prediction == 1:
                        st.error(f"🚨 **MALICIOUS URL DETECTED** (Risk Probability: {prob:.2%})")
                        st.markdown("**Recommendation:** Do not click. This URL exhibits structural characteristics common to phishing or malware links.")
                    else:
                        st.success(f"✅ **URL Appears SAFE** (Risk Probability: {prob:.2%})")
                        st.markdown("**Recommendation:** This URL's structure appears normal.")
                        
                    st.markdown("---")
                    with st.expander("View Extracted URL Features"):
                        st.dataframe(features)

                except Exception as e:
                    st.error(f"⚠️ An internal error occurred during URL checking. Error: {e}")
            else:
                st.warning("Please enter a URL to check.")

    # ========== TAB 3: LLM CHAT BOT ==========
    with tab3:
        st.header("🤖 Cybersecurity Consultant Bot")
        st.markdown("Ask the bot about the latest cyber threats, incident response protocols, or best security practices. (Powered by Gemini with Google Search grounding.)")

        # Display chat messages from history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input handling
        if prompt := st.chat_input("Ask a security question..."):
            # 1. Display user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Display thinking message (placeholder for bot)
            with st.chat_message("model"):
                with st.spinner("Gemini Defense Bot is analyzing the threat..."):
                    # 3. Get response from Gemini API
                    response_text = get_gemini_response(prompt, st.session_state.chat_history)
                
                    # 4. Display and save bot response
                    st.markdown(response_text)
                    st.session_state.chat_history.append({"role": "model", "content": response_text})


# ==================== 7. RUN APP ====================
if __name__ == "__main__":
    # The training phase runs here when the script is executed
    
    # We guide the user on how to run the Streamlit dashboard.
    # Note: In a Streamlit environment, run_dashboard() is executed automatically.
    # We include the call structure for completeness in a single-file script.
    
    run_dashboard()
