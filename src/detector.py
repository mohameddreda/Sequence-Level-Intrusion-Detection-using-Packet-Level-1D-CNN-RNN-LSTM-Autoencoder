import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve
)
from config import Config
from sklearn.metrics import f1_score, accuracy_score

class AnomalyDetector:
    def __init__(self, model, threshold_percentile=95):
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.threshold = None
    
    def calculate_reconstruction_error(self, data):
        """Calculate reconstruction error for sequences"""
        reconstructed = self.model.predict(data)
        reconstruction_error = np.mean(np.square(data - reconstructed), axis=(1, 2))
        return reconstruction_error
    
    def set_threshold(self, normal_data):
        """Set anomaly threshold based on normal data"""
        normal_errors = self.calculate_reconstruction_error(normal_data)
        self.threshold = np.percentile(normal_errors, self.threshold_percentile)
        return self.threshold

    def tune_threshold(self, X_val, y_val, metric='accuracy'):
        """Tune threshold on a validation set (X_val, y_val).

        metric: 'accuracy' or 'f1' to maximize.
        """
        if X_val is None or y_val is None:
            raise ValueError('Validation set not provided for threshold tuning')

        errors = self.calculate_reconstruction_error(X_val)

        # Search percentile thresholds from 50 to 99.5
        percentiles = np.linspace(50, 99.5, 100)
        best_score = -1
        best_thr = None

        for p in percentiles:
            thr = np.percentile(errors, p)
            preds = (errors > thr).astype(int)
            if metric == 'f1':
                score = f1_score(y_val, preds)
            else:
                score = accuracy_score(y_val, preds)

            if score > best_score:
                best_score = score
                best_thr = thr

        self.threshold = best_thr
        print(f"🔎 Tuned threshold (percentile approx): {np.mean(errors<=best_thr)*100:.2f}th percentile -> {best_thr:.4f} | best {metric}: {best_score:.4f}")
        return self.threshold
    
    def detect_anomalies(self, data):
        """Detect anomalies in data"""
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        
        reconstruction_error = self.calculate_reconstruction_error(data)
        predictions = (reconstruction_error > self.threshold).astype(int)
        
        return predictions, reconstruction_error
    
    def evaluate_performance(self, y_true, y_pred, reconstruction_errors):
        """Evaluate model performance"""
        print("Anomaly Detection Results:")
        print("=" * 50)
        print(f"Threshold ({self.threshold_percentile}th percentile): {self.threshold:.4f}")
        print(f"Accuracy: {np.mean(y_pred == y_true):.4f}")
        print(f"ROC AUC: {roc_auc_score(y_true, reconstruction_errors):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Attack']))
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'], 
                   yticklabels=['Normal', 'Attack'])
        plt.title('Confusion Matrix - Hybrid CNN+LSTM Autoencoder')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        return plt.gcf()
    
    def plot_error_distribution(self, normal_errors, attack_errors):
        """Plot distribution of reconstruction errors"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Histogram
        ax1.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue')
        ax1.hist(attack_errors, bins=50, alpha=0.7, label='Attack', color='red')
        ax1.axvline(self.threshold, color='black', linestyle='--', 
                   label=f'Threshold: {self.threshold:.4f}')
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Reconstruction Errors')
        ax1.legend()
        
        # Box plot
        error_data = [normal_errors, attack_errors]
        ax2.boxplot(error_data, labels=['Normal', 'Attack'])
        ax2.set_ylabel('Reconstruction Error')
        ax2.set_title('Box Plot of Reconstruction Errors')
        
        # ROC Curve
        all_errors = np.concatenate([normal_errors, attack_errors])
        all_labels = np.concatenate([np.zeros_like(normal_errors), np.ones_like(attack_errors)])
        fpr, tpr, _ = roc_curve(all_labels, all_errors)
        ax3.plot(fpr, tpr, linewidth=2)
        ax3.plot([0, 1], [0, 1], 'k--')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title(f'ROC Curve (AUC = {roc_auc_score(all_labels, all_errors):.4f})')
        ax3.grid(True)
        
        plt.tight_layout()
        return fig