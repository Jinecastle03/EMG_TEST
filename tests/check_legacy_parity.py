"""Parity of the new pipeline with the previous project (read-only):
 1. causal filter == legacy filtered columns (sessions.pkl f_ch*)
 2. new windows/features + legacy RF/LR models reproduce legacy results/offline_predictions.csv
Run: .venv/Scripts/python tests/check_legacy_parity.py
"""
import sys, time
sys.dont_write_bytecode = True   # do not write .pyc into the legacy project
from pathlib import Path
import numpy as np, pandas as pd, joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emgmonitor import config as C
from emgmonitor.session import load_session
from emgmonitor.dsp import WindowConfig, extract_window, window_features, ALL_FEATURES

t = time.time()
s = load_session(next(C.SESSIONS_DIR.glob("S01/*_legacy")))
print(f"load {time.time()-t:.1f}s", s.stats)
sys.path.insert(0, str(C.LEGACY_ML / "src"))
L = pd.read_pickle(C.LEGACY_ML / "data" / "sessions.pkl")[0]
lf = L.emg[[f"f_ch{i}" for i in range(1, 9)]].to_numpy()
err = np.abs(lf - s.emg_filt).max()
print("1. filter max abs diff vs legacy (uV):", err)
assert err < 1e-6
cl = (L.emg.package_num - L.emg.package_num.iloc[0]).to_numpy()
assert np.array_equal(cl, s.clock - s.clock[0]), "clock differs"

pred = pd.read_csv(C.LEGACY_ML / "results" / "offline_predictions.csv")
pose = joblib.load(C.LEGACY_MODEL_DIR / "pose_model.joblib"); eff = joblib.load(C.LEGACY_MODEL_DIR / "effort_model.joblib")
cfg = WindowConfig(200, 50, 500)
W = pd.read_pickle(C.LEGACY_ML / "data" / "windows.pkl")
W = W[W.valid & (W.split == "test")].reset_index(drop=True)
assert len(W) == len(pred)
X, ok = [], 0
pkg0 = L.emg.package_num.iloc[0]
for _, r in W.iterrows():
    w = extract_window(s.clock, s.emg_filt, s.imu_clock, s.imu, int(r.end_pkg - pkg0) + int(s.clock[0]), cfg)
    X.append(window_features(w))
X = pd.DataFrame(X, columns=ALL_FEATURES)
fd = np.abs(X.to_numpy() - W[ALL_FEATURES].to_numpy()) / np.maximum(np.abs(W[ALL_FEATURES].to_numpy()), 1e-6)
print("2a. feature max rel diff vs legacy windows.pkl:", fd.max())
pp = pose.predict_proba(X[list(pose.feature_names_in_)])
pp_cols = [f"p_pose_{c.replace(' ', '_')}" for c in pose.classes_]
print("2b. pose proba max diff vs offline_predictions.csv:", np.abs(pp - pred[pp_cols].to_numpy()).max())
print("    pred pose agreement:", (pose.classes_[pp.argmax(1)] == pred.pred_pose).mean())
