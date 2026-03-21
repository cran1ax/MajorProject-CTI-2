import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, data_path='cybersecurity_threats.csv'):
        self.data = pd.read_csv(data_path)
        self.features = None
        self.target = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def identify_parameters(self):
        """Identify and analyze key parameters"""
        print("=== PARAMETER IDENTIFICATION ===")
        print(f"Dataset Shape: {self.data.shape}")
        print("\nData Types:")
        print(self.data.dtypes)
        
        # Numerical parameters
        numerical_params = self.data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\nNumerical Parameters ({len(numerical_params)}): {numerical_params}")
        
        # Categorical parameters
        categorical_params = self.data.select_dtypes(include=['object']).columns.tolist()
        print(f"Categorical Parameters ({len(categorical_params)}): {categorical_params}")
        
        # Target parameter
        if 'high_risk_incident' in self.data.columns:
            print(f"\nTarget Parameter: high_risk_incident")
            print(f"Class Distribution:\n{self.data['high_risk_incident'].value_counts()}")
        
        return numerical_params, categorical_params
    
    def feature_engineering(self):
        """Create new features and transform existing ones"""
        df = self.data.copy()
        
        # Create time-based features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Create composite features
        df['loss_per_user'] = df['financial_loss'] / (df['affected_users'] + 1)
        df['efficiency_ratio'] = df['affected_users'] / (df['response_time'] + 1)
        
        # Risk score based on multiple factors
        df['risk_score'] = (
            df['financial_loss'] * 0.4 +
            df['affected_users'] * 0.3 +
            df['response_time'] * 0.3
        )
        
        self.data = df
        return df
    
    def preprocess_data(self):
        """Preprocess data for model training"""
        df = self.data.copy()
        
        # Handle categorical variables
        categorical_cols = ['attack_type', 'target_industry', 'country', 'defense_mechanism']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Select features and target
        feature_cols = [
            'financial_loss', 'affected_users', 'response_time', 'data_breach_size',
            'network_traffic', 'vulnerability_score', 'employee_training_hours',
            'attack_type', 'target_industry', 'year', 'month', 'loss_per_user',
            'efficiency_ratio', 'risk_score'
        ]
        
        # Only use available columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        self.features = df[available_features]
        self.target = df['high_risk_incident']
        
        # Scale numerical features
        numerical_cols = self.features.select_dtypes(include=[np.number]).columns
        self.features[numerical_cols] = self.scaler.fit_transform(self.features[numerical_cols])
        
        print(f"Final feature set: {list(self.features.columns)}")
        return self.features, self.target
    
    def analyze_feature_importance(self):
        """Analyze feature importance"""
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = self.features, self.target
        
        # Train a quick model for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Plot feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance for Cybersecurity Risk Prediction')
        plt.tight_layout()
        plt.show()
        
        return importance_df

# Usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    num_params, cat_params = preprocessor.identify_parameters()
    preprocessor.feature_engineering()
    X, y = preprocessor.preprocess_data()
    importance_df = preprocessor.analyze_feature_importance()