"""Lightweight smoke checks for the repository.

This script performs a minimal set of checks that are safe to run in CI
without installing heavy dependencies (no TensorFlow). It verifies:
- `requirements.txt` exists
- data files referenced in `config.py` exist (warns if missing)
- model & thresholds files presence and basic info
- loads thresholds JSON if present and reports shape

Exits with code 0 even if warnings are present (so CI won't fail the build),
but prints clear messages for any missing or suspicious items.
"""

import os
import sys
import json
from pathlib import Path

# make sure project root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config


def human_size(path: Path):
    try:
        s = path.stat().st_size
        for unit in ['B','KB','MB','GB']:
            if s < 1024.0:
                return f"{s:.1f}{unit}"
            s /= 1024.0
        return f"{s:.1f}TB"
    except Exception:
        return 'n/a'


def main():
    cfg = Config()
    print('Running lightweight smoke checks')
    print('Project root:', ROOT)

    # requirements
    req = ROOT / 'requirements.txt'
    if req.exists():
        print('OK: requirements.txt found')
    else:
        print('WARN: requirements.txt not found')

    # Data files
    print('\nData file checks:')
    if cfg.USE_SPLIT_FILES:
        files = [Path(cfg.TRAIN_PATH), Path(cfg.TEST_PATH)]
    else:
        files = [Path(p) for p in cfg.DATA_PATHS]

    for p in files:
        p_abs = (ROOT / p).resolve() if not p.is_absolute() else p
        if p_abs.exists():
            print(f'  OK: {p} -> exists ({human_size(p_abs)})')
        else:
            print(f'  WARN: {p} -> MISSING')

    # Model & thresholds
    print('\nModel / thresholds checks:')
    models_dir = ROOT / 'models' / 'saved_models'
    if not models_dir.exists():
        print('  WARN: models/saved_models/ directory does not exist')
    else:
        files = list(models_dir.glob('*'))
        if files:
            for f in files:
                print(f'  {f.name} ({human_size(f)})')
        else:
            print('  WARN: models/saved_models/ is empty')

    # If thresholds_fixed95.json present, load and report
    thr_path = models_dir / 'thresholds_fixed95.json'
    if thr_path.exists():
        try:
            with open(thr_path, 'r') as fh:
                j = json.load(fh)
            thr = j.get('thresholds')
            print(f"\nLoaded {thr_path.name}: {len(thr)} thresholds")
        except Exception as e:
            print(f"Failed to read {thr_path}: {e}")
    else:
        print(f"\nNote: {thr_path.name} not found — main.py will compute or fall back as needed")

    print('\nSmoke checks completed — warnings do not fail CI by design.')


if __name__ == '__main__':
    main()
