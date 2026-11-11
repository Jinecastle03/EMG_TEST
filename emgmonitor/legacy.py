"""Import of the previous project's recordings into the new session format (READ-ONLY on the source).

Two sources are supported:
  * the Unity recording folder  Session_YYYYMMDD_HHMMSS/{emg,imu,trials}.csv + metadata.json
  * the cached pickle  EMG2PoseForce/ML/data/sessions.pkl  (contains the raw columns of every session; used
    here because the original Recordings folder lives on the other PC)
The original labels (Rest, Grip, Twist CW, Twist CCW) are kept as they are. A legacy trial is one 5 s hold
(the subject was already holding the pose when the trial key was pressed), so hold = whole trial and
the inter-trial periods are NOT used as Rest (`inter_trial_rest: false`).

Usage: python -m emgmonitor.legacy [--pickle PATH | --unity-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .session import TRIAL_COLUMNS


def _write(out_dir: Path, meta: dict, emg: pd.DataFrame, imu: pd.DataFrame, trials: pd.DataFrame) -> Path:
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"already imported: {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    e = pd.DataFrame({"sample_index": emg.emg_index.astype(np.int64), "package_num": emg.package_num.astype(np.int64),
                      "device_ts": emg.device_ts, "host_ts": emg.host_ts, "trial_id": emg.trial_id.astype(int)})
    for c in C.EMG_COLS:
        e[c] = emg[c]
    pkg_of = pd.Series(emg.package_num.values, index=emg.emg_index.values)
    i = pd.DataFrame({"imu_index": imu.imu_index.astype(np.int64), "sample_index": imu.emg_index.astype(np.int64),
                      "package_num": pkg_of.reindex(imu.emg_index.values).values.astype(np.int64),
                      "device_ts": imu.device_ts, "host_ts": imu.host_ts, "trial_id": imu.trial_id.astype(int)})
    for c in C.IMU_COLS:
        i[c] = imu[c]
    rows = []
    for _, t in trials.iterrows():
        rows.append({"trial_id": int(t.trial_id), "pose": t.pose, "effort": "" if t.pose == "Rest" else t.effort,
                     "repetition": int(t.repetition), "condition": f"{t.pose}|{t.effort}" if t.pose != "Rest" else "Rest",
                     "status": t.status, "prep_start_idx": int(t.start_emg_index), "hold_start_idx": int(t.start_emg_index),
                     "hold_end_idx": int(t.end_emg_index), "rest_end_idx": int(t.end_emg_index),
                     "hold_start_host": float(t.start_host_unix), "hold_end_host": float(t.end_host_unix),
                     "quality": "", "note": "" if pd.isna(t.get("note", "")) else str(t.get("note", ""))})
    e.to_csv(out_dir / "emg.csv", index=False, float_format="%.6f")
    i.to_csv(out_dir / "imu.csv", index=False, float_format="%.6f")
    pd.DataFrame(rows, columns=TRIAL_COLUMNS).to_csv(out_dir / "trials.csv", index=False)
    (out_dir / "events.csv").write_text("host_unix,event,detail\n", encoding="utf-8")
    C.save_json(meta, out_dir / "metadata.json")
    print(f"imported {meta['session_id']} -> {out_dir}  ({len(e)} EMG rows, {len(i)} IMU rows, {len(rows)} trials)")
    return out_dir


def _meta(lm: dict, source: str) -> dict:
    rc = lm.get("recording_conditions", {})
    return {
        "format": "emgmonitor-session-1", "session_id": lm["session_id"] + "_legacy", "original_session_id": lm["session_id"],
        "participant_id": rc.get("subject_id", "S01"), "arm": rc.get("arm", ""), "kind": "training",
        "data_source": "IMPORTED_LEGACY_DEVICE", "imported_from": source, "status": "completed",
        "start_local": lm.get("session_start_local", ""), "fs": float(lm["board_layout"]["sampling_rate"]),
        "board_layout": lm.get("board_layout"), "device": lm.get("device_type", ""), "filter": dict(C.FILTER),
        "protocol": {"name": "legacy_unity", "inter_trial_rest": False, "onset_margin_s": 0.0, "hold_s": 5.0,
                     "note": "legacy: pose was held for the whole trial; inter-trial periods unlabeled"},
        "legacy_calibration": lm.get("calibration"), "legacy_metadata": {k: lm.get(k) for k in ("units", "sampling_rates", "processing", "labels")},
    }


def import_unity_dir(d: Path, out_root: Path = C.SESSIONS_DIR) -> Path:
    d = Path(d)
    lm = C.load_json(d / "metadata.json")
    if lm.get("is_simulated") or lm.get("data_source", "DEVICE") != "DEVICE":
        raise ValueError("simulated / replay sessions are not imported")
    meta = _meta(lm, str(d))
    return _write(out_root / meta["participant_id"] / meta["session_id"], meta, pd.read_csv(d / "emg.csv"),
                  pd.read_csv(d / "imu.csv"), pd.read_csv(d / "trials.csv"))


def import_pickle(pkl: Path = C.LEGACY_ML / "data" / "sessions.pkl", out_root: Path = C.SESSIONS_DIR) -> list[Path]:
    src = str(C.LEGACY_ML / "src")
    added = src not in sys.path
    if added:
        sys.path.insert(0, src)   # the pickle references the legacy module `preprocessing.Session`
    dwb = sys.dont_write_bytecode
    sys.dont_write_bytecode = True    # never write __pycache__ files into the (read-only) legacy project
    try:
        sessions = pd.read_pickle(pkl)
    finally:
        sys.dont_write_bytecode = dwb
        if added:
            sys.path.remove(src)
    out = []
    for s in sessions:
        meta = _meta(s.meta, f"{pkl} (cached copy of {s.path})")
        out.append(_write(out_root / meta["participant_id"] / meta["session_id"], meta, s.emg, s.imu, s.trials))
    return out


def import_legacy_calibration(session_dir: Path) -> Path | None:
    """Converts the Unity Rest/MAX calibration stored in a legacy session (same filter, 200 ms RMS) into a
    Calibration file. Legacy MAX was a Grip squeeze (95th percentile RMS) -> used as reference RMS."""
    from .calibration import Calibration
    m = C.load_json(Path(session_dir) / "metadata.json")
    lc = m.get("legacy_calibration")
    if not lc or not lc.get("has_baseline"):
        return None
    cal = Calibration(participant_id=m["participant_id"], session_id=m["session_id"],
                      calibration_id="Cal_legacy_" + m.get("original_session_id", "")[8:], created=lc.get("baseline_captured_local", ""),
                      device={"kind": "DEVICE", "layout": m.get("board_layout"), "fs": m.get("fs")},
                      signal={"filter": dict(C.FILTER), "rms_window_ms": 200.0, "stride_ms": 50.0},
                      rest={"rms_mean": lc["baseline_rms"], "rms_std": lc["baseline_rms_std"], "gyro_bias": lc.get("gyro_bias", [0, 0, 0]),
                            "duration_s": lc.get("baseline_duration_s"), "frames": lc.get("baseline_frames")},
                      notes="converted from the legacy Unity calibration (Rest 3 s + MAX grip 3 s); no per-pose Low/High references")
    if lc.get("has_max"):
        cal.norm = {"rest_rms": lc["baseline_rms"], "ref_rms": lc["max_rms"],
                    "method": "legacy Unity MAX capture (95th percentile RMS of a strong grip)",
                    "activation": "mean_c max(0, (rms_c - rest_c) / (ref_c - rest_c)) on 200 ms windows of filtered EMG"}
    out = C.CALIB_DIR / cal.participant_id / f"{cal.calibration_id}.json"
    if not out.exists():
        cal.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", type=Path, default=None)
    ap.add_argument("--unity-dir", type=Path, default=None)
    a = ap.parse_args()
    C.ensure_dirs()
    if a.unity_dir:
        print("calibration:", import_legacy_calibration(import_unity_dir(a.unity_dir)))
    else:
        for d in import_pickle(a.pickle or C.LEGACY_ML / "data" / "sessions.pkl"):
            print("calibration:", import_legacy_calibration(d))


if __name__ == "__main__":
    main()
