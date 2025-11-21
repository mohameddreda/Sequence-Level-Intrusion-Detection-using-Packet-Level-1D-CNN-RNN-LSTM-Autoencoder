# Contributing

Thank you for contributing. For this project we aim to keep the repository easy to run and review by your instructor.

Development workflow (recommended)

- Create a virtual environment and install dependencies:

```powershell
python -m venv tf-env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\tf-env\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Run the main pipeline:

```powershell
python main.py
```

- Use the scripts in `scripts/` to compute or tune per-feature thresholds:

```powershell
python .\scripts\save_fixed_thresholds.py
python .\scripts\tune_per_feature.py
```

Notes

- The `data/raw/` directory contains the UNSW-NB15 CSVs — these are large and not tracked by the default push workflow unless you explicitly add them.
- If you are submitting this repo to an instructor, they can run `python main.py` after creating a venv and placing data files in `data/raw/`.
