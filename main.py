# main.py
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import json
warnings.filterwarnings('ignore')

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model import HybridAutoencoder
from src.trainer import ModelTrainer
from src.detector import AnomalyDetector
from src.visualizer import ResultVisualizer
from src import unsupervised_extras
from src import supervised
from config import Config

def main():
    print("🚀 Starting Hybrid CNN+LSTM Autoencoder for Intrusion Detection")
    print("=" * 60)
    
    # Load configuration
    config = Config()
    visualizer = ResultVisualizer()
    
    try:
        # Step 1: Load data
        print("📊 Step 1: Loading data...")
        data_loader = DataLoader()
        df = data_loader.load_data()
        
        if df is None:
            print("❌ Failed to load data")
            return
        
        basic_info = data_loader.get_basic_info(df)
        print(f"Dataset shape: {basic_info['shape']}")
        print(f"Label distribution:\n{basic_info['label_distribution']}")
        
        # Step 2: Preprocess data
        print("\n🔧 Step 2: Preprocessing and balancing data...")
        preprocessor = DataPreprocessor()
        
        df_original = df.copy()
        
        data_dict = preprocessor.prepare_data(df)
        
        if data_dict is None:
            print("❌ Preprocessing failed")
            return
            
        print(f"✅ Training sequences: {data_dict['X_train_normal'].shape}")
        print(f"✅ Test sequences: {data_dict['X_test'].shape}")
        
        # Step 3: Build model
        print("\n🧠 Step 3: Building hybrid CNN+LSTM autoencoder...")
        sequence_length, n_features = data_dict['sequence_shape'][1], data_dict['sequence_shape'][2]
        
        model_builder = HybridAutoencoder()
        autoencoder, encoder = model_builder.build_model(sequence_length, n_features)
        model_builder.get_model_summary()
        
        # Step 4: Train model
        print("\n🏋️ Step 4: Training model...")
        trainer = ModelTrainer()
        history = trainer.train(autoencoder, data_dict['X_train_normal'])
        
        if history is not None:
            trainer.plot_training_history()
        
        # Step 5: Detect anomalies
        print("\n🔍 Step 5: Anomaly detection...")
        detector = AnomalyDetector(autoencoder, config.THRESHOLD_PERCENTILE)

        # If validation set is available, tune threshold on validation set (maximize accuracy)
        if data_dict.get('X_val') is not None and data_dict.get('y_val') is not None:
            print("🔧 Tuning threshold on validation set to maximize accuracy...")
            # Use validation sequences (they contain labels) to tune threshold
            detector.tune_threshold(data_dict['X_val'], data_dict['y_val'], metric='accuracy')
            print(f"Anomaly threshold tuned to: {detector.threshold:.4f}")
        else:
            # Set threshold using normal training data
            threshold = detector.set_threshold(data_dict['X_train_normal'])
            print(f"Anomaly threshold set to: {threshold:.4f}")
        
        # Detect anomalies in test set
        test_predictions, test_errors = detector.detect_anomalies(data_dict['X_test'])
        
        # Evaluate performance
        print("\n📈 Evaluation Results:")
        print("=" * 40)
        detector.evaluate_performance(data_dict['y_test'], test_predictions, test_errors)
        
        # Plot results
        print("\n📊 Generating visualizations...")
        # Ensure results directories
        os.makedirs(os.path.join('results', 'plots'), exist_ok=True)
        os.makedirs(os.path.join('results', 'reports'), exist_ok=True)

        cm_fig = visualizer.plot_confusion_matrix(data_dict['y_test'], test_predictions)
        # Save confusion matrix
        cm_path = os.path.join('results', 'plots', 'confusion_matrix.png')
        try:
            cm_fig.savefig(cm_path)
            print(f"Saved confusion matrix to {cm_path}")
        except Exception as e:
            print(f"Failed to save confusion matrix: {e}")
        
        # Plot error distribution
        normal_test_indices = np.where(data_dict['y_test'] == 0)[0]
        attack_test_indices = np.where(data_dict['y_test'] == 1)[0]
        
        if len(normal_test_indices) > 0 and len(attack_test_indices) > 0:
            normal_errors = test_errors[normal_test_indices]
            attack_errors = test_errors[attack_test_indices]
            
            detector.plot_error_distribution(normal_errors, attack_errors)
            # Save error distribution figure
            try:
                err_fig = detector.plot_error_distribution(normal_errors, attack_errors)
                err_path = os.path.join('results', 'plots', 'error_distribution.png')
                err_fig.savefig(err_path)
                print(f"Saved error distribution to {err_path}")
            except Exception as e:
                print(f"Failed to save error distribution: {e}")
        
        plt.show()

        # ----------------------
        # Unsupervised anomaly detection: fixed per-feature thresholding (primary method)
        # ----------------------
        try:
            print("\n🔬 Unsupervised anomaly detection: Per-feature thresholding (fixed percentile)...")
            # If a saved fixed-95 thresholds file exists, load and use it
            thresholds_path = os.path.join('models', 'saved_models', 'thresholds_fixed95.json')
            if os.path.exists(thresholds_path):
                with open(thresholds_path, 'r') as f:
                    th_data = json.load(f)
                pf_thresholds = np.array(th_data.get('thresholds'))
                # Compute per-feature MSE errors on test set
                recon_test = autoencoder.predict(data_dict['X_test'])
                pf_errors = np.mean(np.square(data_dict['X_test'] - recon_test), axis=1)
                pf_preds = (pf_errors > pf_thresholds).any(axis=1).astype(int)
                print(f"Loaded fixed thresholds from {thresholds_path}")
            else:
                # Fall back to computing thresholds from training-normal data using config percentile
                pf_preds, pf_errors, pf_thresholds = unsupervised_extras.per_feature_thresholding(
                    autoencoder, data_dict['X_test'], data_dict['X_train_normal'], percentile=config.THRESHOLD_PERCENTILE)

            pf_acc = np.mean(pf_preds == data_dict['y_test'])
            print(f"Per-feature thresholding accuracy: {pf_acc:.4f}")
            print("Classification report:")
            from sklearn.metrics import classification_report
            print(classification_report(data_dict['y_test'], pf_preds, target_names=['Normal', 'Attack']))
        except Exception as e:
            print(f"⚠️ Error in unsupervised anomaly detection: {e}")
            import traceback
            traceback.print_exc()
        # Write a short text report with metrics
        try:
            from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

            global_acc = accuracy_score(data_dict['y_test'], test_predictions)
            global_roc = roc_auc_score(data_dict['y_test'], test_errors)
            global_clf = classification_report(data_dict['y_test'], test_predictions, target_names=['Normal', 'Attack'])

            per_feature_acc = np.mean(pf_preds == data_dict['y_test'])
            per_feature_clf = classification_report(data_dict['y_test'], pf_preds, target_names=['Normal', 'Attack'])

            report_lines = []
            report_lines.append('Anomaly Detection Report')
            report_lines.append('=' * 60)
            report_lines.append('\n-- Global (tuned) threshold results --')
            report_lines.append(f'Threshold: {detector.threshold:.6f}')
            report_lines.append(f'Accuracy: {global_acc:.6f}')
            report_lines.append(f'ROC AUC: {global_roc:.6f}')
            report_lines.append('\nClassification Report:\n')
            report_lines.append(global_clf)
            report_lines.append('\n-- Per-feature thresholding (fixed) results --')
            report_lines.append(f'Per-feature accuracy: {per_feature_acc:.6f}')
            report_lines.append('\nClassification Report:\n')
            report_lines.append(per_feature_clf)

            report_text = '\n'.join(report_lines)
            report_path = os.path.join('results', 'reports', 'report.txt')
            with open(report_path, 'w') as rf:
                rf.write(report_text)

            print(f"Saved textual report to {report_path}")
        except Exception as e:
            print(f"Failed to write report: {e}")

        print("\n✅ Hybrid CNN+LSTM Autoencoder completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()