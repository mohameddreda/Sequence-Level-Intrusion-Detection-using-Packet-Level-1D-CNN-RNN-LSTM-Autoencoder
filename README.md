# Sequence-Level Intrusion Detection (Hybrid CNN + LSTM Autoencoder)

Short, reproducible implementation of a sequence-level intrusion detection pipeline using a hybrid 1D-CNN + LSTM autoencoder trained on UNSWNB15. The primary (unsupervised) detection method is per-feature reconstruction-error thresholding (fixed 95th percentile by default).

--

## What this repo contains

- `main.py`  full pipeline: load data, preprocess, train autoencoder, detect anomalies, and visualize results.
- `src/`  implementation modules (data loader, preprocessor, model, trainer, detector, visualizer, utilities).
- `scripts/`  helper scripts:
  - `save_fixed_thresholds.py`  compute & save 95th-percentile per-feature thresholds using a saved model.
  - `tune_per_feature.py`  tune per-feature percentiles on a labeled validation split (optional).
  - `generate_results_from_saved_model.py`  create plots and a text report using the saved model + thresholds (no retraining).
  - `smoke_check.py`  lightweight checks run by CI to validate repository readiness.
- `models/saved_models/`  trained model and thresholds (if present): `autoencoder_best.h5`, `thresholds_fixed95.json`, `thresholds.json`.
- `results/`  generated plots and textual report (created by scripts or `main.py`).
- `notebooks/`  Jupyter notebooks used for exploration and evaluation.

--

## Quick Start (recommended for instructors)

1. Clone the repository and change directory:

```powershell
git clone https://github.com/mohameddreda/Sequence-Level-Intrusion-Detection-using-Packet-Level-1D-CNN-RNN-LSTM-Autoencoder.git
cd "Sequence-Level-Intrusion-Detection-using-Packet-Level-1D-CNN-RNN-LSTM-Autoencoder"
```

2. Create & activate a Python virtual environment and install dependencies:

```powershell
python -m venv tf-env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass;
.\tf-env\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Place the UNSWNB15 CSV files in `data/raw/` if not present. See `config.py` for expected names.

4. Run the full pipeline (this will train the model if a saved model is not present):

```powershell
python main.py
```

Notes:
- If a trained model (`models/saved_models/autoencoder_best.h5`) and `thresholds_fixed95.json` are present, `main.py` will load them and skip retraining where possible.
- After a run you can find results in `results/plots/` and `results/reports/report.txt`.

--

## Quick evaluation without retraining

If you want to avoid retraining (fast):

- Ensure `models/saved_models/autoencoder_best.h5` and `models/saved_models/thresholds_fixed95.json` exist.
- Run:

```powershell
python .\scripts\generate_results_from_saved_model.py
```

This generates `results/plots/confusion_matrix.png`, `results/plots/error_distribution.png`, and `results/reports/report.txt` and does not retrain the model.

--

## What to look for in results

- Console output: data shapes, training progress (loss / MAE), threshold tuning logs.
- `results/reports/report.txt`: human-readable classification reports and overall metrics (accuracy, ROC AUC).
- Plots in `results/plots/`: confusion matrix and reconstruction-error distributions.

## About the results file

- **Location:** `results/reports/report.txt` (plain text).
- **Contents:** concise classification report (precision/recall/F1), overall metrics (accuracy, AUC), which thresholds and model were used, and a short summary of runtime and data shapes.
- **How to regenerate:** run `python main.py` to retrain/evaluate or `python .\scripts\generate_results_from_saved_model.py` to recreate the report and plots from the saved model without retraining.

--

## Notes & Recommendations

- The raw dataset (`data/raw/`) is intentionally not included in the public repo to keep the repository small. If you need to include large artifacts, use Git LFS or attach them separately.
- CI includes a lightweight syntax check and smoke-checks that run on each push  they do not run heavy training.
- If evaluation is slow, reduce `EPOCHS` or `BATCH_SIZE` in `config.py`.

--

## License & Contributing

- Licensed under the MIT License  see `LICENSE`.
- See `CONTRIBUTING.md` for developer setup and notes.

--

If you want, I can add a one-page "Instructor Quick View" script that opens the report and images automatically. Say `yes` to add it.
