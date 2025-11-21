import os
import sys
import json
import numpy as np

# ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model import HybridAutoencoder
from config import Config


def main():
    cfg = Config()
    os.makedirs('models/saved_models', exist_ok=True)

    dl = DataLoader()
    df = dl.load_data()
    pre = DataPreprocessor()
    data = pre.prepare_data(df)

    seq_len, n_feat = data['sequence_shape'][1], data['sequence_shape'][2]
    builder = HybridAutoencoder()
    autoencoder, encoder = builder.build_model(seq_len, n_feat)

    # load saved best model if exists
    checkpoint = 'models/saved_models/autoencoder_best.h5'
    if os.path.exists(checkpoint):
        try:
            autoencoder.load_weights(checkpoint)
            print('Loaded model weights from', checkpoint)
        except Exception:
            from tensorflow.keras.models import load_model
            autoencoder = load_model(checkpoint)
            print('Loaded full model file from', checkpoint)
    else:
        raise RuntimeError('No trained model found at models/saved_models/autoencoder_best.h5 — train model first')

    # compute normal errors (per-feature mean squared error across sequence)
    recon_norm = autoencoder.predict(data['X_train_normal'])
    errors_norm = np.mean(np.square(data['X_train_normal'] - recon_norm), axis=1)  # (n_seq, n_feat)

    p = cfg.THRESHOLD_PERCENTILE
    thresholds = np.percentile(errors_norm, p, axis=0)

    out = {
        'percentile': int(p),
        'thresholds': thresholds.tolist()
    }
    out_path = 'models/saved_models/thresholds_fixed95.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    # quick eval on test set
    recon_test = autoencoder.predict(data['X_test'])
    errors_test = np.mean(np.square(data['X_test'] - recon_test), axis=1)
    preds = (errors_test > thresholds).any(axis=1).astype(int)

    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(data['y_test'], preds)
    print(f'Fixed {p}th-percentile per-feature thresholds saved to {out_path}')
    print(f'Test accuracy using fixed {p}th-percentile thresholds: {acc:.4f}')
    print('Classification report:')
    print(classification_report(data['y_test'], preds, target_names=["Normal","Attack"]))


if __name__ == '__main__':
    main()
