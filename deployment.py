import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

class CybersecurityPredictor:
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        try:
            self.models['baseline'] = joblib.load('baseline_model.pkl')
            self.models['random_forest'] = joblib.load('random_forest_model.pkl')
            self.models['gradient_boosting'] = joblib.load('gradient_boosting_model.pkl')
            self.models['best_model'] = joblib.load('best_model.pkl')
        except FileNotFoundError:
            st.error("Model files not found. Please train models first.")
    
    def predict_threat(self, input_data):
        """Make predictions using all models"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                pred_proba = model.predict_proba(input_data)[0]
                predictions[model_name] = {
                    'risk_probability': pred_proba[1],
                    'prediction': model.predict(input_data)[0],
                    'confidence': max(pred_proba)
                }
            except Exception as e:
                st.error(f"Error in {model_name}: {str(e)}")
        
        return predictions

def main():
    st.set_page_config(
        page_title="Cybersecurity Threat Predictor",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.title("🛡️ Cybersecurity Threat Prediction Dashboard")
    st.markdown("""
    This dashboard predicts high-risk cybersecurity incidents using machine learning models.
    Compare baseline vs advanced model performance in real-time.
    """)
    
    # Initialize predictor
    predictor = CybersecurityPredictor()
    
    # Sidebar for input
    st.sidebar.header("🔧 Input Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        financial_loss = st.slider("Financial Loss ($M)", 0.0, 500.0, 50.0)
        affected_users = st.slider("Affected Users", 0, 100000, 10000)
        response_time = st.slider("Response Time (hours)", 0.0, 48.0, 5.0)
    
    with col2:
        data_breach_size = st.slider("Data Breach Size (MB)", 0, 10000, 1000)
        network_traffic = st.slider("Network Traffic (GB)", 0.0, 2000.0, 800.0)
        vulnerability_score = st.slider("Vulnerability Score", 1, 10, 5)
    
    attack_type = st.sidebar.selectbox("Attack Type", 
                                     ['Phishing', 'Ransomware', 'DDoS', 'Malware', 'Insider Threat'])
    industry = st.sidebar.selectbox("Target Industry", 
                                  ['Finance', 'Healthcare', 'Government', 'Education', 'Technology'])
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'financial_loss': [financial_loss],
        'affected_users': [affected_users],
        'response_time': [response_time],
        'data_breach_size': [data_breach_size],
        'network_traffic': [network_traffic],
        'vulnerability_score': [vulnerability_score],
        'attack_type': [attack_type],
        'target_industry': [industry],
        'loss_per_user': [financial_loss / (affected_users + 1)],
        'efficiency_ratio': [affected_users / (response_time + 1)],
        'risk_score': [financial_loss * 0.4 + affected_users * 0.3 + response_time * 0.3],
        'year': [2024],
        'month': [1]
    })
    
    # Encode categorical variables (simplified)
    attack_mapping = {'Phishing': 0, 'Ransomware': 1, 'DDoS': 2, 'Malware': 3, 'Insider Threat': 4}
    industry_mapping = {'Finance': 0, 'Healthcare': 1, 'Government': 2, 'Education': 3, 'Technology': 4}
    
    input_data['attack_type'] = input_data['attack_type'].map(attack_mapping)
    input_data['target_industry'] = input_data['target_industry'].map(industry_mapping)
    
    # Make predictions
    if st.sidebar.button("🔮 Predict Threat Risk"):
        predictions = predictor.predict_threat(input_data)
        
        # Display results
        st.header("📊 Prediction Results")
        
        # Create comparison chart
        model_names = list(predictions.keys())
        risk_probabilities = [predictions[model]['risk_probability'] for model in model_names]
        
        fig = go.Figure(data=[
            go.Bar(name='Risk Probability', x=model_names, y=risk_probabilities,
                  marker_color=['red' if prob > 0.5 else 'green' for prob in risk_probabilities])
        ])
        
        fig.update_layout(
            title='Threat Risk Prediction: Model Comparison<br><sub>Before vs After Model Enhancement</sub>',
            xaxis_title='Models',
            yaxis_title='Risk Probability',
            yaxis=dict(range=[0, 1]),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Baseline Model")
            baseline_pred = predictions.get('baseline', {})
            st.metric("Risk Probability", f"{baseline_pred.get('risk_probability', 0):.2%}")
            st.metric("Prediction", "HIGH RISK" if baseline_pred.get('prediction', 0) == 1 else "LOW RISK")
        
        with col2:
            st.subheader("Advanced Model")
            rf_pred = predictions.get('random_forest', {})
            st.metric("Risk Probability", f"{rf_pred.get('risk_probability', 0):.2%}")
            st.metric("Prediction", "HIGH RISK" if rf_pred.get('prediction', 0) == 1 else "LOW RISK")
        
        with col3:
            st.subheader("Best Model")
            best_pred = predictions.get('best_model', {})
            st.metric("Risk Probability", f"{best_pred.get('risk_probability', 0):.2%}")
            st.metric("Prediction", "HIGH RISK" if best_pred.get('prediction', 0) == 1 else "LOW RISK")
        
        # Improvement analysis
        st.subheader("📈 Performance Improvement")
        if 'baseline' in predictions and 'best_model' in predictions:
            baseline_prob = predictions['baseline']['risk_probability']
            best_prob = predictions['best_model']['risk_probability']
            improvement = ((best_prob - baseline_prob) / baseline_prob) * 100
            
            st.metric(
                "Model Improvement", 
                f"{improvement:+.1f}%",
                delta=f"From {baseline_prob:.2%} to {best_prob:.2%}"
            )

    # Model performance comparison section
    st.markdown("---")
    st.header("📋 Model Performance Summary")
    
    try:
        with open('training_results.json', 'r') as f:
            training_results = json.load(f)
        
        # Create performance comparison
        performance_data = []
        for model_name, metrics in training_results.items():
            performance_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': metrics['accuracy'],
                'AUC Score': metrics['auc']
            })
        
        perf_df = pd.DataFrame(performance_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Accuracy Comparison")
            fig_acc = px.bar(perf_df, x='Model', y='Accuracy', 
                           color='Accuracy', color_continuous_scale='Viridis')
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            st.subheader("AUC Score Comparison")
            fig_auc = px.bar(perf_df, x='Model', y='AUC Score',
                           color='AUC Score', color_continuous_scale='Plasma')
            st.plotly_chart(fig_auc, use_container_width=True)
    
    except FileNotFoundError:
        st.warning("Training results not available. Please run model training first.")

if __name__ == "__main__":
    main()