import json
import os
import sys
import numpy as np

# Ensure project root is on sys.path so `src` can be imported when running from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model import HybridAutoencoder
from src.trainer import ModelTrainer
from src import unsupervised_extras
from config import Config


def main():
    cfg = Config()
    os.makedirs('models/saved_models', exist_ok=True)

    print('Loading data...')
    dl = DataLoader()
    df = dl.load_data()
    pre = DataPreprocessor()
    data = pre.prepare_data(df)

    seq_len, n_feat = data['sequence_shape'][1], data['sequence_shape'][2]
    builder = HybridAutoencoder()
    autoencoder, encoder = builder.build_model(seq_len, n_feat)

    # Try to load saved best model if present
    checkpoint = 'models/saved_models/autoencoder_best.h5'
    if os.path.exists(checkpoint):
        try:
            print('Loading saved model from', checkpoint)
            autoencoder.load_weights(checkpoint)
        except Exception:
            # If full model saved (not weights-only), try load_model
            from tensorflow.keras.models import load_model
            autoencoder = load_model(checkpoint)
            print('Loaded full model file.')
    else:
        print('No saved model found — training a fresh model (this may take time)')
        trainer = ModelTrainer()
        trainer.train(autoencoder, data['X_train_normal'])

    print('Tuning per-feature percentiles on validation set...')
    # Ensure X_val and y_val exist
    if data.get('X_val') is None or data.get('y_val') is None:
        raise RuntimeError('Validation set (X_val, y_val) not available for tuning.')

    percentiles, thresholds, val_preds = unsupervised_extras.tune_per_feature_percentiles(
        autoencoder, data['X_val'], data['y_val'], data['X_train_normal'], percentiles=np.arange(80,100), metric='f1')

    # Evaluate on test set
    recon_test = autoencoder.predict(data['X_test'])
    errors_test = np.mean(np.square(data['X_test'] - recon_test), axis=1)
    test_preds = (errors_test > thresholds).any(axis=1).astype(int)

    from sklearn.metrics import classification_report, accuracy_score
    acc = accuracy_score(data['y_test'], test_preds)
    print('Per-feature tuned accuracy on test set: {:.4f}'.format(acc))
    print('Classification report:')
    print(classification_report(data['y_test'], test_preds, target_names=['Normal', 'Attack']))

    # Save thresholds and percentiles
    out = {
        'percentiles': percentiles.tolist(),
        'thresholds': thresholds.tolist(),
        'accuracy': float(acc)
    }
    out_path = 'models/saved_models/thresholds.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print('Saved tuned thresholds to', out_path)


if __name__ == '__main__':
    main()
