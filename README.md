# Sequence-Level Intrusion Detection (Hybrid CNN+LSTM Autoencoder)

This repository implements a sequence-level intrusion detection pipeline using a hybrid 1D-CNN + LSTM autoencoder trained on the UNSW-NB15 dataset. The primary detection method is unsupervised per-feature reconstruction-error thresholding (fixed 95th percentile by default).

Quick highlights

- Trains a hybrid Conv1D + LSTM autoencoder on sequences of packet/flow features.
- Unsupervised detection: per-feature MSE thresholds (fixed 95th percentile) saved to `models/saved_models/thresholds_fixed95.json`.
- Optional tuning: `scripts/tune_per_feature.py` finds per-feature percentiles via validation (uses labels) and saves to `models/saved_models/thresholds.json`.

Repository layout

- `main.py` — run full pipeline (train, detect, visualize). By default the main flow uses validation tuning if `X_val` is available and then runs per-feature fixed detection as a final evaluation.
- `src/` — source code modules (data loader, preprocessor, model, trainer, detector, utilities).
- `data/raw/` — raw UNSW-NB15 CSV files (large — not removed by scripts).
- `models/saved_models/` — trained model and thresholds saved here.
- `scripts/` — helper scripts:
  - `save_fixed_thresholds.py` — compute & save 95th-percentile per-feature thresholds to `thresholds_fixed95.json` and evaluate on test set.
  - `tune_per_feature.py` — tune per-feature percentiles using validation labels and save results to `thresholds.json`.

How to run (Windows PowerShell)

1. Activate the project's virtual environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass;
.\tf-env\Scripts\Activate.ps1
```

2. Run the main pipeline (train and evaluate):

```powershell
python main.py
```

3. Compute & save fixed 95th-percentile thresholds (if you already trained the model):

```powershell
python .\scripts\save_fixed_thresholds.py
```

4. Tune per-feature percentiles using validation labels (optional):

```powershell
python .\scripts\tune_per_feature.py
```

Notes and recommendations

- The repository currently includes a local virtual environment `tf-env/`. This folder is large and is typically excluded from source control. Consider removing it from the repo and recreating the venv locally using `python -m venv tf-env` and `pip install -r requirements.txt`.
- If you want me to delete generated/unused files (e.g., `tf-env/`, `__pycache__/`, `.vscode/`), tell me exactly which items to remove. I will not delete anything without explicit confirmation.

Contact

- If you want me to run further experiments (Mahalanobis, weighted voting, constrained tuning ranges, or add a CLI mode), tell me which experiment to run next.
