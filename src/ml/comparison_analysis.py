import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
import json

class ComparisonAnalyzer:
    def __init__(self, results_file='training_results.json'):
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.models = list(self.results.keys())
    
    def performance_comparison(self):
        """Compare model performance before vs after"""
        print("=== PERFORMANCE COMPARISON: BEFORE VS AFTER ===")
        
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'AUC Score': metrics['auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df)
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        sns.barplot(data=comparison_df, x='Model', y='Accuracy', ax=ax1)
        ax1.set_title('Model Accuracy Comparison\n(Before vs After)')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # AUC comparison
        sns.barplot(data=comparison_df, x='Model', y='AUC Score', ax=ax2)
        ax2.set_title('Model AUC Score Comparison\n(Before vs After)')
        ax2.set_ylabel('AUC Score')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, metrics in self.results.items():
            # Load model
            model = joblib.load(f"{model_name}_model.pkl")
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves: Model Comparison\n(Before vs After)')
        plt.legend()
        plt.grid(True)
        plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def improvement_analysis(self):
        """Calculate improvement percentages"""
        baseline_acc = self.results['baseline']['accuracy']
        baseline_auc = self.results['baseline']['auc']
        
        print("\n=== IMPROVEMENT ANALYSIS ===")
        for model_name, metrics in self.results.items():
            if model_name != 'baseline':
                acc_improvement = ((metrics['accuracy'] - baseline_acc) / baseline_acc) * 100
                auc_improvement = ((metrics['auc'] - baseline_auc) / baseline_auc) * 100
                
                print(f"{model_name}:")
                print(f"  Accuracy Improvement: {acc_improvement:+.2f}%")
                print(f"  AUC Improvement: {auc_improvement:+.2f}%")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive comparison report"""
        report = {
            'project_summary': {
                'title': 'Cybersecurity Threat Prediction - Model Comparison Report',
                'description': 'Comparison of Baseline vs Advanced ML Models',
                'total_models': len(self.models),
                'models_tested': self.models
            },
            'performance_metrics': self.results,
            'recommendations': self._generate_recommendations()
        }
        
        with open('comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("Comprehensive report generated: comparison_report.json")
        return report
    
    def _generate_recommendations(self):
        """Generate recommendations based on comparison"""
        best_model = max(self.results.items(), key=lambda x: x[1]['auc'])[0]
        best_accuracy = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        
        recommendations = [
            f"Best overall model (AUC): {best_model}",
            f"Best accuracy model: {best_accuracy}",
            "Advanced models show significant improvement over baseline",
            "Consider ensemble methods for production deployment"
        ]
        
        return recommendations

# Usage
if __name__ == "__main__":
    analyzer = ComparisonAnalyzer()
    comparison_df = analyzer.performance_comparison()
    analyzer.improvement_analysis()
    report = analyzer.generate_comprehensive_report()