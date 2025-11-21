# Sequence-Level Intrusion Detection (Hybrid CNN+LSTM Autoencoder)

This repository implements a sequence-level intrusion detection pipeline using a hybrid 1D-CNN + LSTM autoencoder trained on the UNSW-NB15 dataset. The primary detection method is unsupervised per-feature reconstruction-error thresholding (fixed 95th percentile by default).

Quick highlights

- Trains a hybrid Conv1D + LSTM autoencoder on sequences of packet/flow features.
- Unsupervised detection: per-feature MSE thresholds (fixed 95th percentile) saved to `models/saved_models/thresholds_fixed95.json`.
- Optional tuning: `scripts/tune_per_feature.py` finds per-feature percentiles via validation (uses labels) and saves to `models/saved_models/thresholds.json`.

Repository layout

- `main.py` — run full pipeline (train, detect, visualize). By default the main flow uses validation tuning if `X_val` is available and then runs per-feature fixed detection as a final evaluation.
- `src/` — source code modules (data loader, preprocessor, model, trainer, detector, utilities).
- `data/raw/` — raw UNSW-NB15 CSV files (large — not removed by scripts). These are intentionally not included in the public repo by default to keep the repository small; place the CSV files in this folder if you plan to run the pipeline locally.
- `models/saved_models/` — trained model and thresholds saved here. Example files: `autoencoder_best.h5`, `thresholds_fixed95.json`, `thresholds.json`.
- `scripts/` — helper scripts:
  - `save_fixed_thresholds.py` — compute & save 95th-percentile per-feature thresholds to `thresholds_fixed95.json` and evaluate on test set.
  - `tune_per_feature.py` — tune per-feature percentiles using validation labels and save results to `thresholds.json`.

How to run (Windows PowerShell)

Follow these exact steps so your instructor can reproduce the results:

1. Create and activate a virtual environment, then install requirements:

```powershell
python -m venv tf-env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass;
.\tf-env\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Place the UNSW-NB15 CSV files in `data/raw/` (if they are not already present). The expected filenames used by the loader are in `config.py`.

3. Train and evaluate the model (this will train the autoencoder and run the unsupervised evaluation):

```powershell
python main.py
```

4. If you already have a trained model and want to recompute the fixed thresholds only (faster):

```powershell
python .\scripts\save_fixed_thresholds.py
```

5. To tune per-feature percentiles using labels on a validation set (optional, requires labels):

```powershell
python .\scripts\tune_per_feature.py
```

Notes and recommendations

- The repository intentionally excludes the raw `data/` files by default to keep the repo small for submission. If you want the data in the repo, consider using Git LFS or attaching the dataset separately.
- `models/saved_models/thresholds_fixed95.json` is loaded automatically by `main.py` if present; this allows instructors to reproduce the per-feature detection results without re-training the model.
- If you want me to remove generated/unused files (e.g., `tf-env/`, `__pycache__/`, `.vscode/`), tell me exactly which items to remove. I will not delete anything without explicit confirmation.

Contact

- If you want me to run further experiments (Mahalanobis, weighted voting, constrained tuning ranges, or add a CLI mode), tell me which experiment to run next.

CI and License

- A lightweight CI workflow (`.github/workflows/ci.yml`) runs on each push and pull-request to `master` and checks that all Python files compile (syntax check). This avoids shipping heavy dependencies in CI while catching syntax errors early.
- This project is licensed under the MIT License (see `LICENSE`).
