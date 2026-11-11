"""Per-user calibration (about 20 s): REST + MAX.

  1. REST 5 s        relaxed arm -> per-channel rest RMS (noise floor); the device check (rates, flat / noisy
                     channels) is computed from the same 5 s, no separate step.
  2. MAX  3 s (+3 s) isometric wrist flexion (and extension) against resistance -> per-channel max RMS
                     (95th percentile of the 200 ms RMS). Flexion activates the palm-side channels, extension the
                     back-side channels, so both together give a maximum for all 8 electrodes.

Use: normalized waveform view (filtered EMG / max RMS) and the relative activation index
        activation = mean_c max(0, (rms_c - rest_c) / (max_c - rest_c))     (0 = rest, 1 = calibrated max)
It is a relative EMG level, NOT force (N / kgf). The models do not use it (they are trained per user and
normalize with statistics of their own training data). Calibration is not model training.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import config as C

NOISY_REST_RMS_UV = 60.0        # filtered rest RMS above this -> poor contact / noise (legacy rest RMS ~8-14 uV)
FLAT_RAW_STD_UV = 1e-3

MAX_TASKS = {
    "flexion": ("손목 굽히기 (최대)", "손바닥이 위를 향하게 팔을 두고, 다른 손으로 손바닥을 눌러 막은 채 손목을 위로 굽히려고 "
                                  "최대한 힘을 줍니다. 손목은 움직이지 않게(버티기만). 무리하지 마세요."),
    "extension": ("손목 펴기 (최대)", "손등이 위를 향하게 팔을 두고, 다른 손으로 손등을 눌러 막은 채 손목을 위로 젖히려고 "
                                   "최대한 힘을 줍니다. 손목은 움직이지 않게(버티기만). 무리하지 마세요."),
}


@dataclass
class Calibration:
    participant_id: str = ""
    session_id: str = ""
    calibration_id: str = ""
    created: str = ""
    device: dict = field(default_factory=dict)
    signal: dict = field(default_factory=dict)
    device_check: dict = field(default_factory=dict)
    rest: dict = field(default_factory=dict)
    max: dict = field(default_factory=dict)          # task -> stats
    norm: dict = field(default_factory=dict)         # rest_rms, ref_rms (= max RMS), method
    poses: dict = field(default_factory=dict)        # unused since v2 (kept so older files still load)
    notes: str = ""

    @property
    def has_rest(self):
        return bool(self.rest.get("rms_mean"))

    @property
    def has_norm(self):
        return bool(self.norm.get("ref_rms")) and bool(self.norm.get("rest_rms"))

    def activation(self, rms: np.ndarray) -> float:
        if not self.has_norm:
            return float("nan")
        r0, r1 = np.asarray(self.norm["rest_rms"]), np.asarray(self.norm["ref_rms"])
        return float(np.mean(np.maximum(0, (np.asarray(rms) - r0) / np.maximum(r1 - r0, 1e-3))))

    def channel_activation(self, rms: np.ndarray) -> np.ndarray:
        r0, r1 = np.asarray(self.norm["rest_rms"]), np.asarray(self.norm["ref_rms"])
        return np.maximum(0, (np.asarray(rms) - r0) / np.maximum(r1 - r0, 1e-3))

    @property
    def max_rms(self):
        return np.asarray(self.norm["ref_rms"]) if self.has_norm else None

    def status_text(self):
        if not self.has_rest:
            return "Not calibrated"
        return f"{self.participant_id} {self.created} (rest{' + max' if self.has_norm else ' only'})"

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or C.CALIB_DIR / self.participant_id / f"{self.calibration_id}.json")
        C.save_json(asdict(self), path)
        return path

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        d = C.load_json(path)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def latest_calibration(participant: str) -> Path | None:
    files = sorted((C.CALIB_DIR / participant).glob("*.json"))
    return files[-1] if files else None


def rest_stats(frames: list) -> dict:
    rms = np.array([f.rms for f in frames])
    gyro = np.array([f.gyro_mean for f in frames if np.isfinite(f.gyro_mean).all()])
    acc = np.array([f.acc_mean for f in frames if np.isfinite(f.acc_mean).all()])
    return {"frames": len(frames), "rms_mean": rms.mean(0).tolist(), "rms_std": rms.std(0).tolist(),
            "rms_p95": np.percentile(rms, 95, axis=0).tolist(),
            "gyro_bias": gyro.mean(0).tolist() if len(gyro) else [0, 0, 0],
            "acc_mean": acc.mean(0).tolist() if len(acc) else [0, 0, 0],
            "gyro_mag_p95": float(np.percentile(np.linalg.norm(gyro, axis=1), 95)) if len(gyro) else 0.0}


def max_stats(frames: list) -> dict:
    rms = np.array([f.rms for f in frames])
    return {"frames": len(frames), "rms_p95": np.percentile(rms, 95, axis=0).tolist(), "rms_median": np.median(rms, 0).tolist()}


def finalize(cal: Calibration) -> Calibration:
    """max RMS per channel = largest 95th-percentile RMS over the MAX tasks."""
    if not cal.has_rest or not cal.max:
        return cal
    rest = np.asarray(cal.rest["rms_mean"])
    ref = np.max(np.stack([np.asarray(s["rms_p95"]) for s in cal.max.values()]), 0)
    weak = [c + 1 for c in range(C.N_EMG) if ref[c] < rest[c] * 2]
    ref = np.maximum(ref, rest + 1.0)
    cal.norm = {"rest_rms": rest.tolist(), "ref_rms": ref.tolist(),
                "method": "channel-wise max of the 95th-percentile 200 ms RMS over " + " + ".join(cal.max),
                "activation": "mean_c max(0, (rms_c - rest_c) / (max_c - rest_c)) on 200 ms windows of filtered EMG",
                "weak_channels": weak}
    return cal


def device_check(frames: list, raw_std: np.ndarray, emg_rate: float, imu_rate: float, fs_nominal: float) -> dict:
    rms = np.array([f.rms for f in frames]) if frames else np.zeros((1, C.N_EMG))
    m = rms.mean(0)
    ch = []
    for c in range(C.N_EMG):
        st = "OK"
        if raw_std[c] < FLAT_RAW_STD_UV:
            st = "FLAT (no signal)"
        elif m[c] > NOISY_REST_RMS_UV:
            st = "NOISY (check contact)"
        ch.append({"channel": c + 1, "status": st, "rest_rms_uv": float(m[c]), "raw_std_uv": float(raw_std[c])})
    ok_rate = abs(emg_rate - fs_nominal) < 0.05 * fs_nominal
    ok_imu = 30 <= imu_rate <= 70
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "emg_rate_hz": emg_rate, "imu_rate_hz": imu_rate,
            "emg_rate_ok": bool(ok_rate), "imu_rate_ok": bool(ok_imu), "channels": ch,
            "all_ok": bool(ok_rate and ok_imu and all(c["status"] == "OK" for c in ch))}
