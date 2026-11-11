"""Paths and shared signal / window settings.

The EMG conditioning (causal 4th-order 20 Hz high-pass + 60 Hz notch with harmonics) and the window
definition are the same as in the previous project (G:/내 드라이브/EMG2PoseForce/ML, Unity EmgFilterBank),
so the previous Random-Forest model can run on the live stream of this program.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("EMGMON_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
CALIB_DIR = DATA_DIR / "calibrations"
LOG_DIR = DATA_DIR / "logs"
MODELS_DIR = ROOT / "models"
RUNS_DIR = MODELS_DIR / "runs"
LEGACY_MODEL_DIR = MODELS_DIR / "legacy_20261103"
CONFIG_DIR = ROOT / "config"

# previous project (read-only!)
LEGACY_PROJECT = Path(os.environ.get("EMGMON_LEGACY", r"G:\내 드라이브\EMG2PoseForce"))
LEGACY_ML = LEGACY_PROJECT / "ML"

N_EMG = 8
N_IMU = 6
EMG_COLS = [f"emg_ch{i}_uv" for i in range(1, N_EMG + 1)]
IMU_COLS = ["acc_x_g", "acc_y_g", "acc_z_g", "gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]
FS_EMG_DEFAULT = 500.0
IMU_HZ_NOMINAL = 50.0
GYRO_SATURATION_DPS = 499.0

FILTER = {"highpass_hz": 20.0, "highpass_order": 4, "notch_hz": 60.0, "notch_q": 30.0, "notch_harmonics": True}

WINDOW_MS = 200.0
STRIDE_MS = 50.0
WINDOW_CHOICES_MS = (200.0, 300.0, 500.0)
MIN_EMG_COVERAGE = 0.8      # windows that lost > 20 % of their EMG samples are invalid
MIN_IMU_FRACTION = 0.5      # >= 50 % of the expected IMU samples (legacy: 5 of 10 for 200 ms)

REST = "Rest"
EFFORTS = ["Low", "High"]

SEED = 0


def ensure_dirs():
    for d in (SESSIONS_DIR, CALIB_DIR, LOG_DIR, RUNS_DIR, CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=json_default)
    os.replace(tmp, path)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
