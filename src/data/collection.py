import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time

class CybersecurityDataCollector:
    def __init__(self):
        self.dataset = None
    
    def generate_synthetic_data(self, n_samples=5000):
        """Generate realistic cybersecurity threat data"""
        np.random.seed(42)
        
        # Feature parameters
        attack_types = ['Phishing', 'Ransomware', 'DDoS', 'Malware', 'Insider Threat', 'SQL Injection']
        industries = ['Finance', 'Healthcare', 'Government', 'Education', 'Technology', 'Retail']
        countries = ['USA', 'UK', 'Germany', 'India', 'Japan', 'Brazil', 'Australia']
        defense_mechanisms = ['Firewall', 'IDS', 'Encryption', 'Multi-Factor Auth', 'Security Training']
        
        data = {
            'timestamp': pd.date_range('2015-01-01', periods=n_samples, freq='D'),
            'attack_type': np.random.choice(attack_types, n_samples),
            'target_industry': np.random.choice(industries, n_samples),
            'country': np.random.choice(countries, n_samples),
            'financial_loss': np.random.exponential(50, n_samples),  # in million $
            'affected_users': np.random.poisson(10000, n_samples),
            'response_time': np.random.gamma(2, 2, n_samples),  # in hours
            'defense_mechanism': np.random.choice(defense_mechanisms, n_samples),
            'data_breach_size': np.random.lognormal(8, 2, n_samples),  # in MB
            'network_traffic': np.random.normal(1000, 300, n_samples),  # in GB
            'vulnerability_score': np.random.uniform(1, 10, n_samples),
            'employee_training_hours': np.random.poisson(20, n_samples)
        }
        
        # Create target variable (high_risk_incident)
        conditions = (
            (data['financial_loss'] > 100) |
            (data['affected_users'] > 50000) |
            (data['response_time'] > 10)
        )
        data['high_risk_incident'] = conditions.astype(int)
        
        self.dataset = pd.DataFrame(data)
        return self.dataset
    
    def save_data(self, filename='cybersecurity_threats.csv'):
        """Save collected data to CSV"""
        if self.dataset is not None:
            self.dataset.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        else:
            print("No data to save. Generate data first.")

# Usage
if __name__ == "__main__":
    collector = CybersecurityDataCollector()
    df = collector.generate_synthetic_data(5000)
    collector.save_data()
    print("Data collection completed!")
    print(f"Dataset shape: {df.shape}")
    print(f"High-risk incidents: {df['high_risk_incident'].sum()}")