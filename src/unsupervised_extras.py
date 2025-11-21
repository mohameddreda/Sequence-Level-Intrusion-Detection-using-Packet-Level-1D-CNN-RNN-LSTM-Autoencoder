import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
import numpy as np

def per_feature_thresholding(model, data, normal_data, percentile=95):
    """
    Compute per-feature mean squared error and threshold each feature independently.
    Returns anomaly predictions (1=anomaly, 0=normal) for each sequence.
    """
    reconstructed = model.predict(data)
    errors = np.mean(np.square(data - reconstructed), axis=1)  # shape: (n_seq, n_feat)
    normal_reconstructed = model.predict(normal_data)
    normal_errors = np.mean(np.square(normal_data - normal_reconstructed), axis=1)
    thresholds = np.percentile(normal_errors, percentile, axis=0)  # shape: (n_feat,)
    anomaly_flags = (errors > thresholds).astype(int)  # shape: (n_seq, n_feat)
    # If any feature is anomalous, flag the sequence
    sequence_anomaly = (anomaly_flags.sum(axis=1) > 0).astype(int)
    return sequence_anomaly, errors, thresholds


def tune_per_feature_percentiles(model, X_val, y_val, normal_data, percentiles=None, metric='f1'):
    """
    Tune per-feature percentile thresholds on a validation set (uses labels).
    percentiles: list or array of percentiles to search (e.g., range 80..99)
    Returns tuned thresholds (array of length n_features) and predictions on X_val.
    """
    if percentiles is None:
        percentiles = np.arange(80, 100)

    # Errors shape: (n_seq, n_feat)
    recon_val = model.predict(X_val)
    errors_val = np.mean(np.square(X_val - recon_val), axis=1)

    recon_norm = model.predict(normal_data)
    errors_norm = np.mean(np.square(normal_data - recon_norm), axis=1)

    n_feat = errors_val.shape[1]
    best_percentiles = np.zeros(n_feat)
    best_thresholds = np.zeros(n_feat)

    # For each feature, find percentile that gives best metric on validation set
    from sklearn.metrics import f1_score, accuracy_score
    for f in range(n_feat):
        best_score = -1
        best_p = percentiles[0]
        for p in percentiles:
            thr = np.percentile(errors_norm[:, f], p)
            preds = (errors_val[:, f] > thr).astype(int)
            # Sequence is anomalous if any feature flagged => approximate final preds by OR across features
            # But here we tune per-feature individually by using preds directly and evaluating
            if metric == 'f1':
                score = f1_score(y_val, preds)
            else:
                score = accuracy_score(y_val, preds)

            if score > best_score:
                best_score = score
                best_p = p

        best_percentiles[f] = best_p
        best_thresholds[f] = np.percentile(errors_norm[:, f], best_p)

    # Build final predictions on validation by OR-ing per-feature flags using tuned thresholds
    val_preds = (errors_val > best_thresholds).any(axis=1).astype(int)
    return best_percentiles.astype(int), best_thresholds, val_preds


def mahalanobis_thresholding(model, data, normal_data, percentile=95):
    """
    Compute per-feature mean errors for each sequence, then compute Mahalanobis distance
    of each sequence error vector from the distribution of normal sequence errors.
    Threshold by percentile of distances on normal data.
    Returns anomaly flags, distances, and chosen threshold.
    """
    recon = model.predict(data)
    errors = np.mean(np.square(data - recon), axis=1)  # (n_seq, n_feat)
    recon_norm = model.predict(normal_data)
    errors_norm = np.mean(np.square(normal_data - recon_norm), axis=1)

    # Estimate mean and covariance from normal errors
    mu = np.mean(errors_norm, axis=0)
    cov = np.cov(errors_norm, rowvar=False)
    # Regularize covariance
    cov += np.eye(cov.shape[0]) * 1e-6

    # Compute Mahalanobis distance
    inv_cov = np.linalg.inv(cov)
    dists = np.array([distance.mahalanobis(e, mu, inv_cov) for e in errors])
    dists_norm = np.array([distance.mahalanobis(e, mu, inv_cov) for e in errors_norm])

    thr = np.percentile(dists_norm, percentile)
    flags = (dists > thr).astype(int)
    return flags, dists, thr


def gmm_thresholding(model, data, normal_data, n_components=2):
    """
    Fit a Gaussian Mixture Model to the reconstruction errors of normal data.
    Use the GMM to compute anomaly scores for all data.
    Returns anomaly predictions (1=anomaly, 0=normal) for each sequence.
    """
    reconstructed = model.predict(data)
    errors = np.mean(np.square(data - reconstructed), axis=(1, 2))  # shape: (n_seq,)
    normal_reconstructed = model.predict(normal_data)
    normal_errors = np.mean(np.square(normal_data - normal_reconstructed), axis=(1, 2))
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(normal_errors.reshape(-1, 1))
    # Compute log-likelihood for each error
    log_probs = gmm.score_samples(errors.reshape(-1, 1))
    # Use a threshold on log-likelihood (e.g., 5th percentile of normal log-probs)
    normal_log_probs = gmm.score_samples(normal_errors.reshape(-1, 1))
    threshold = np.percentile(normal_log_probs, 5)
    anomaly_flags = (log_probs < threshold).astype(int)
    return anomaly_flags, errors, log_probs, threshold


def ensemble_thresholding(model, data, normal_data, percentile=95):
    """
    Use mean, max, and std of reconstruction error per sequence and combine their anomaly flags.
    Returns anomaly predictions (1=anomaly, 0=normal) for each sequence.
    """
    reconstructed = model.predict(data)
    errors = np.square(data - reconstructed)  # shape: (n_seq, seq_len, n_feat)
    mean_error = errors.mean(axis=(1, 2))
    max_error = errors.max(axis=(1, 2))
    std_error = errors.std(axis=(1, 2))
    normal_reconstructed = model.predict(normal_data)
    normal_errors = np.square(normal_data - normal_reconstructed)
    mean_thr = np.percentile(normal_errors.mean(axis=(1, 2)), percentile)
    max_thr = np.percentile(normal_errors.max(axis=(1, 2)), percentile)
    std_thr = np.percentile(normal_errors.std(axis=(1, 2)), percentile)
    flags = np.stack([
        (mean_error > mean_thr).astype(int),
        (max_error > max_thr).astype(int),
        (std_error > std_thr).astype(int)
    ], axis=1)  # shape: (n_seq, 3)
    # If any flag is set, mark as anomaly
    sequence_anomaly = (flags.sum(axis=1) > 0).astype(int)
    return sequence_anomaly, mean_error, max_error, std_error, (mean_thr, max_thr, std_thr)
