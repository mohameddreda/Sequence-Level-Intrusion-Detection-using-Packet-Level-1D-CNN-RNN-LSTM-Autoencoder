import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


def extract_error_features(model, X):
    """Extract per-sequence reconstruction-error features from an autoencoder.

    Returns a 2D array of shape (n_sequences, n_features_extracted).
    Features include per-feature mean squared error across the sequence and global stats.
    """
    # X: (n_seq, seq_len, n_feat)
    reconstructed = model.predict(X, verbose=0)
    se = np.square(X - reconstructed)  # (n_seq, seq_len, n_feat)

    # per-feature mean error across time
    per_feat_mean = se.mean(axis=1)  # (n_seq, n_feat)
    per_feat_std = se.std(axis=1)    # (n_seq, n_feat)

    # global stats
    global_mean = per_feat_mean.mean(axis=1).reshape(-1, 1)
    global_std = per_feat_mean.std(axis=1).reshape(-1, 1)
    global_max = per_feat_mean.max(axis=1).reshape(-1, 1)

    # Concatenate: per-feature means + per-feature stds + global stats
    features = np.concatenate([per_feat_mean, per_feat_std, global_mean, global_std, global_max], axis=1)
    return features


def train_random_forest(X_train_feats, y_train, X_val_feats=None, y_val=None, n_estimators=100, random_state=42):
    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', random_state=random_state, n_jobs=-1)
    clf.fit(X_train_feats, y_train)

    if X_val_feats is not None and y_val is not None:
        preds = clf.predict(X_val_feats)
        acc = accuracy_score(y_val, preds)
        try:
            auc = roc_auc_score(y_val, clf.predict_proba(X_val_feats)[:, 1])
        except Exception:
            auc = None
        print(f"Supervised classifier validation accuracy: {acc:.4f}")
        if auc is not None:
            print(f"Supervised classifier validation AUC: {auc:.4f}")
        print(classification_report(y_val, preds, target_names=['Normal', 'Attack']))

    return clf


def evaluate_classifier(clf, X_test_feats, y_test):
    preds = clf.predict(X_test_feats)
    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, clf.predict_proba(X_test_feats)[:, 1])
    except Exception:
        auc = None

    print("\nSupervised Classifier Test Results:")
    print("=" * 40)
    print(f"Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"ROC AUC: {auc:.4f}")
    print(classification_report(y_test, preds, target_names=['Normal', 'Attack']))

    return acc, auc, preds
