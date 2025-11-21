# src/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

class ResultVisualizer:
    def __init__(self):
        pass
    
    def plot_attack_distribution(self, df_before, df_after, attack_cat_column):
        """Plot attack category distribution before and after balancing"""
        counts_before = df_before[attack_cat_column].value_counts()
        counts_after = df_after[attack_cat_column].value_counts()
        
        df_compare = pd.DataFrame({
            'Before': counts_before,
            'After': counts_after
        }).fillna(0)
        
        plt.figure(figsize=(12, 6))
        df_compare.plot(kind='bar', figsize=(12,6))
        plt.title("Attack Category Counts: Before vs After Oversampling")
        plt.ylabel("Number of Samples")
        plt.xlabel("Attack Category")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels if labels else ['Normal', 'Attack'],
                   yticklabels=labels if labels else ['Normal', 'Attack'])
        plt.title('Confusion Matrix - Hybrid CNN+LSTM Autoencoder')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_roc_curves(self, results_dict):
        """Plot ROC curves for different models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, (fpr, tpr, auc_score) in results_dict.items():
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()