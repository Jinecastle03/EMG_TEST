"""Run-time predictors. Every model type exposes the same interface:

    p = load_predictor(path)
    pose_proba, p_high = p.predict(window)     # window = dsp.Window from extract_window()
    p.pose_classes, p.active_poses, p.cfg (WindowConfig), p.needs_imu, p.fs, p.filter

p_high[k] = P(High | pose = active_poses[k]); the GUI shows the entry of the predicted pose
(hierarchical Pose -> Effort). The trainer calls exactly these classes to check train/real-time parity.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .dsp import ALL_FEATURES, EMG_FEATURES, WindowConfig, emg_features, imu_features


class Predictor:
    name = "?"
    kind = "?"
    needs_imu = True
    path: Path | None = None
    meta: dict = {}

    def predict(self, w):
        raise NotImplementedError

    @property
    def fs(self):
        return self.cfg.fs

    def describe(self):
        return f"{self.name} [{self.kind}] {self.cfg.window_ms:.0f} ms, classes {self.pose_classes}"


def _feature_frame(w, names):
    v = np.concatenate([emg_features(w.emg_actual), imu_features(w.imu_actual)])
    return pd.DataFrame([v], columns=ALL_FEATURES)[names]


def _proba_in_order(model, X, classes):
    p = model.predict_proba(X)
    cl = [str(c) for c in model.classes_]
    out = np.zeros((len(p), len(classes)))
    for j, c in enumerate(cl):
        out[:, classes.index(c)] = p[:, j]
    return out


class FeaturePredictor(Predictor):
    """Feature-based baseline: RF (pose) + per-pose Logistic Regression (effort). Also used for the legacy model."""
    kind = "feature RF/LR"

    def __init__(self, pose_pipe, effort_pipes: dict, pose_features, effort_features, pose_classes, active_poses,
                 cfg: WindowConfig, filt: dict, name="Feature RF/LR", meta=None):
        self.pose_pipe, self.effort_pipes = pose_pipe, effort_pipes
        self.pose_features, self.effort_features = list(pose_features), list(effort_features)
        self.pose_classes, self.active_poses = list(pose_classes), list(active_poses)
        self.cfg, self.filter, self.name, self.meta = cfg, filt, name, meta or {}
        self.needs_imu = any(not f.startswith("emg_") for f in self.pose_features + self.effort_features)
        try:
            self.pose_pipe.named_steps["clf"].n_jobs = 1     # single-window latency: 1 thread is fastest
        except Exception:
            pass

    def predict(self, w):
        x = _feature_frame(w, ALL_FEATURES)
        pp = _proba_in_order(self.pose_pipe, x[self.pose_features], self.pose_classes)[0]
        xe = x[self.effort_features]
        ph = np.full(len(self.active_poses), np.nan)
        for k, p in enumerate(self.active_poses):
            m = self.effort_pipes.get(p)
            if m is not None:
                ph[k] = m.predict_proba(xe)[0, [str(c) for c in m.classes_].index("High")]
        return pp, ph

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump({"type": "feature", "pose_pipe": self.pose_pipe, "effort_pipes": self.effort_pipes,
                     "pose_features": self.pose_features, "effort_features": self.effort_features,
                     "pose_classes": self.pose_classes, "active_poses": self.active_poses,
                     "window": self.cfg.to_dict(), "filter": self.filter, "name": self.name}, d / "model.joblib")


class DeepPredictor(Predictor):
    kind = "1D-CNN"

    def __init__(self, pose_net, effort_net, norm, pose_classes, active_poses, cfg: WindowConfig, filt: dict,
                 name="CNN", meta=None, arch=None):
        import torch
        self.torch = torch
        self.pose_net, self.effort_net = pose_net.eval(), (effort_net.eval() if effort_net is not None else None)
        self.norm = norm
        self.pose_classes, self.active_poses = list(pose_classes), list(active_poses)
        self.cfg, self.filter, self.name, self.meta, self.arch = cfg, filt, name, meta or {}, arch or {}
        nets = [pose_net] + ([effort_net] if effort_net is not None else [])
        self.needs_imu = any(n.imu is not None for n in nets)
        self.kind = "1D-CNN " + "/".join("+".join(n.streams) for n in nets)
        self._s = np.asarray(norm["emg_scale"], np.float32)[:, None]
        self._im = np.asarray(norm["imu_mean"], np.float32)[:, None]
        self._is = np.asarray(norm["imu_std"], np.float32)[:, None]

    def predict(self, w):
        torch = self.torch
        e = torch.from_numpy((w.emg_grid.astype(np.float32) / self._s)[None])
        i = torch.from_numpy(((w.imu_grid.astype(np.float32) - self._im) / self._is)[None])
        with torch.no_grad():
            pl, el = self.pose_net(e if self.pose_net.emg is not None else None, i if self.pose_net.imu is not None else None)
            if self.effort_net is not None:
                _, el = self.effort_net(e if self.effort_net.emg is not None else None,
                                        i if self.effort_net.imu is not None else None)
            pp = torch.softmax(pl, 1)[0].numpy().astype(float)
            ph = torch.sigmoid(el)[0].numpy().astype(float) if el is not None else np.full(len(self.active_poses), np.nan)
        return pp, ph

    def save(self, d: Path):
        torch = self.torch
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"type": "deep", "pose": {"streams": self.pose_net.streams, "state": self.pose_net.state_dict(),
                                             "n_pose": len(self.pose_classes),
                                             "n_effort": self.pose_net.effort_head.out_features if self.pose_net.effort_head is not None else 0},
                    "effort": None if self.effort_net is None else
                    {"streams": self.effort_net.streams, "state": self.effort_net.state_dict(), "n_effort": len(self.active_poses)},
                    "norm": self.norm, "pose_classes": self.pose_classes, "active_poses": self.active_poses,
                    "window": self.cfg.to_dict(), "filter": self.filter, "name": self.name, "arch": self.arch},
                   d / "model.pt")


def input_spec(p: Predictor) -> dict:
    """What the model reads per prediction and how often (shown in the GUI, saved in meta.json / report)."""
    cfg = p.cfg
    d = {"window_ms": cfg.window_ms, "stride_ms": cfg.stride_ms, "predictions_per_s": 1000.0 / cfg.stride_ms,
         "overlap_ms": max(0.0, cfg.window_ms - cfg.stride_ms),
         "emg": {"channels": C.N_EMG, "rate_hz": cfg.fs, "samples": cfg.n_emg,
                 "preprocessing": "causal HP 20 Hz (4th order) + notch 60 Hz & harmonics; no rectification"},
         "imu": {"channels": "acc x,y,z (g) + gyro x,y,z (deg/s)", "rate_hz": cfg.imu_hz, "samples": cfg.n_imu}}
    if isinstance(p, FeaturePredictor):
        def feat(names):
            ne = sum(f.startswith("emg_") for f in names)
            ni = len(names) - ne
            parts = []
            if ne:
                parts.append(f"EMG features {ne} (RMS, MAV, WL, STD x 8 ch)" if ne == 32 else f"EMG features {ne}")
            if ni:
                parts.append(f"IMU features {ni} (gyro mean/std x3, |gyro| mean, acc mean/std x3)")
            return {"uses": ["emg"] * bool(ne) + ["imu"] * bool(ni), "input": " + ".join(parts),
                    "model": None}
        d["pose"] = feat(p.pose_features) | {"model": "Random Forest on log1p + z-scored features"}
        d["effort"] = feat(p.effort_features) | {"model": "Logistic Regression, one per pose"}
    elif isinstance(p, DeepPredictor):
        def net(n, head):
            uses = list(n.streams)
            shp = []
            if "emg" in uses:
                shp.append(f"EMG ({C.N_EMG}, {cfg.n_emg})")
            if "imu" in uses:
                shp.append(f"IMU ({C.N_IMU}, {cfg.n_imu})")
            return {"uses": uses, "input": " + ".join(shp) + " time series (scaled per channel, no hand-crafted features)",
                    "model": f"1D-CNN {'+'.join(uses)} -> {head}"}
        d["pose"] = net(p.pose_net, "pose softmax")
        d["effort"] = net(p.effort_net, "P(High) per pose") if p.effort_net is not None else \
            net(p.pose_net, "P(High) per pose (shared encoder, multi-task)")
    return d


def input_spec_text(p: Predictor) -> str:
    s = input_spec(p)
    L = [f"예측 1회 = 최근 {s['window_ms']:.0f} ms 구간 · {s['stride_ms']:.0f} ms마다 예측 = {s['predictions_per_s']:.0f} Hz "
         f"(연속 예측끼리 {s['overlap_ms']:.0f} ms 겹침)",
         f"  EMG 8ch @ {s['emg']['rate_hz']:.0f} Hz → {s['emg']['samples']} 샘플/ch (필터: HP 20 Hz + notch 60 Hz)",
         f"  IMU 6축 @ {s['imu']['rate_hz']:.0f} Hz → {s['imu']['samples']} 샘플/축 (acc xyz g, gyro xyz °/s)"]
    for head, k in (("Pose", "pose"), ("Effort", "effort")):
        h = s.get(k)
        if h:
            L.append(f"  {head}: {' + '.join(u.upper() for u in h['uses'])} 사용 → {h['input']} → {h['model']}")
    return "\n".join(L)


def legacy_predictor(d: Path = C.LEGACY_MODEL_DIR) -> FeaturePredictor:
    meta = C.load_json(d / "model_metadata.json")
    pose = joblib.load(d / "pose_model.joblib")
    eff = joblib.load(d / "effort_model.joblib")
    classes = ["Rest"] + [c for c in ["Grip", "Twist CW", "Twist CCW"] if c in list(pose.classes_)]
    cfg = WindowConfig(meta["window"]["size_ms"], meta["window"]["increment_ms"], meta["signal"]["emg_sampling_rate_hz"],
                       min_imu_fraction=meta["window"]["min_imu_samples"] / 10)
    p = FeaturePredictor(pose, eff["models"], meta["pose_model"]["features_in_order"], eff["features"], classes,
                         list(eff["models"].keys()), cfg, meta["signal"]["emg_filter"],
                         name="Legacy RF/LR (S01 2026-11-03)", meta={"source": str(d), "selected": meta["pose_model"]["selected"]})
    p.kind = "legacy feature RF + per-pose LR"
    p.path = d
    return p


def load_predictor(path: Path) -> Predictor:
    path = Path(path)
    if (path / "pose_model.joblib").exists():
        return legacy_predictor(path)
    meta = C.load_json(path / "meta.json") if (path / "meta.json").exists() else {}
    if (path / "model.joblib").exists():
        b = joblib.load(path / "model.joblib")
        p = FeaturePredictor(b["pose_pipe"], b["effort_pipes"], b["pose_features"], b["effort_features"], b["pose_classes"],
                             b["active_poses"], WindowConfig.from_dict(b["window"]), b["filter"], b["name"], meta)
    elif (path / "model.pt").exists():
        import torch
        from .deep import StreamNet
        b = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        pn = StreamNet(b["pose"]["streams"], b["pose"]["n_pose"], b["pose"].get("n_effort", 0))
        pn.load_state_dict(b["pose"]["state"])
        en = None
        if b["effort"] is not None:
            en = StreamNet(b["effort"]["streams"], 0, b["effort"]["n_effort"])
            en.load_state_dict(b["effort"]["state"])
        p = DeepPredictor(pn, en, b["norm"], b["pose_classes"], b["active_poses"], WindowConfig.from_dict(b["window"]),
                          b["filter"], b["name"], meta, b.get("arch"))
    else:
        raise FileNotFoundError(f"no model in {path}")
    p.path = path
    return p


def list_models() -> list[dict]:
    """All selectable models: legacy + every trained bundle (newest run first)."""
    out = []
    if (C.LEGACY_MODEL_DIR / "pose_model.joblib").exists():
        out.append({"label": "Legacy RF/LR (S01, 2026-11-03, 4 classes)", "path": str(C.LEGACY_MODEL_DIR), "recommended": False})
    for mp in sorted(C.RUNS_DIR.glob("*/*/*/meta.json"), reverse=True):
        try:
            m = C.load_json(mp)
        except Exception:
            continue
        tag = " ★recommended" if m.get("recommended") else ""
        out.append({"label": f"{m.get('display', mp.parent.name)} | {m.get('window_ms', '?'):.0f} ms | run {m.get('run_id', '?')}"
                             f" | pose F1 {m.get('test_pose_macro_f1', float('nan')):.3f}{tag}",
                    "path": str(mp.parent), "recommended": bool(m.get("recommended"))})
    return out
