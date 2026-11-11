"""Train and compare Pose x Effort models.

Models (families)
  A_feature        legacy features (RMS/MAV/WL/STD x 8 + 13 IMU stats) -> RF (pose) + per-pose LR (effort)
  B_emg_cnn        RAW-EMG 1D-CNN (pose net + effort net)
  C_imu_cnn        IMU-only 1D-CNN
  D_twostream      EMG + IMU two-stream CNN (pose); effort net EMG-only or EMG+IMU, chosen on CV
  D_twostream_mt   two-stream multi-task (shared encoder, pose + effort heads, effort loss masked for Rest)

Protocol (per window size)
  1. trial-level split BEFORE windowing (dataset.assign_splits); test = held-out trials, evaluated once
  2. dev: label-stratified, trial-grouped CV folds -> model / effort-input selection (never the test set)
  3. final model refit on all dev trials (CNN: early stopping on 20 % of the dev trials)
  4. test metrics, trial-level parity check of the run-time predictor, CPU latency, stability, Rest FP

Usage:  python -m emgmonitor.trainer [--windows 200 300 500] [--families A_feature B_emg_cnn ...]
                                     [--split trial|session|participant] [--quick] [--sessions DIR ...]
"""
from __future__ import annotations

import argparse
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import config as C
from . import deep as D
from .dataset import assign_splits, attach_splits, build_windows, split_availability
from .dsp import ALL_FEATURES, EMG_FEATURES, WindowConfig, extract_window
from .predictors import DeepPredictor, FeaturePredictor, input_spec, input_spec_text
from .session import list_sessions, load_session

FAMILIES = {
    "A_feature": {"display": "Feature-based RF/LR (baseline)", "effort_options": ["emg", "emg+imu"]},
    "B_emg_cnn": {"display": "RAW EMG 1D-CNN", "pose": ("emg",), "effort_options": ["emg"]},
    "C_imu_cnn": {"display": "IMU-only 1D-CNN", "pose": ("imu",), "effort_options": ["imu"]},
    "D_twostream": {"display": "EMG+IMU Two-Stream CNN", "pose": ("emg", "imu"), "effort_options": ["emg", "emg+imu"]},
    "D_twostream_mt": {"display": "EMG+IMU Two-Stream CNN (multi-task)", "pose": ("emg", "imu"), "multitask": True,
                       "effort_options": ["emg+imu"]},
}
DEFAULT_FAMILIES = ["A_feature", "B_emg_cnn", "C_imu_cnn", "D_twostream"]
EFFORT_IMU_MARGIN = 0.01     # EMG+IMU effort input only if better than EMG-only by more than this (CV)
CONFOUND_MARGIN = 0.10       # ... and if it does not raise the speed confound (mean rho) by more than this
BASELINE_MARGIN = 0.01       # a deep model must beat the baseline by more than this (CV) to be recommended
SMOOTH_K = 5                 # 5 x 50 ms causal probability averaging for the stability metrics


@dataclass
class TrainConfig:
    sessions: list = field(default_factory=list)       # session dirs; empty = all training sessions found
    windows_ms: list = field(default_factory=lambda: [C.WINDOW_MS])
    families: list = field(default_factory=lambda: list(DEFAULT_FAMILIES))
    split: str = "trial"
    test_reps: int = 2
    n_folds: int = 4
    quick: bool = False
    max_epochs: int = 100
    run_id: str = ""
    include_synthetic: bool = False


# ============================================================================================ metrics
def cls_metrics(y, yp, labels):
    y, yp = np.asarray(y, dtype=object), np.asarray(yp, dtype=object)
    if len(y) == 0:
        return None
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
            "macro_f1": float(f1_score(y, yp, labels=labels, average="macro", zero_division=0)),
            "labels": list(labels), "confusion_matrix": confusion_matrix(y, yp, labels=labels).tolist(),
            "per_class_f1": dict(zip(labels, f1_score(y, yp, labels=labels, average=None, zero_division=0).round(4).tolist()))}


def effort_metrics(meta, ph_true, active):
    """Effort with the TRUE pose (isolates the Low/High question). ph_true: P(High) from the head of the true pose."""
    out, bas = {}, []
    for p in active:
        m = (meta.pose == p).to_numpy() & np.isfinite(ph_true)
        if m.sum() == 0:
            continue
        y = meta.effort.to_numpy()[m]
        r = cls_metrics(y, np.where(ph_true[m] >= 0.5, "High", "Low"), C.EFFORTS)
        out[p] = r
        bas.append(r["balanced_accuracy"])
    out["mean_balanced_accuracy"] = float(np.mean(bas)) if bas else float("nan")
    return out


def cascade_metrics(meta, pose_proba, ph_all, classes, active):
    """Deployable cascade: effort taken from the head of the PREDICTED pose; a wrong / Rest pose prediction
    counts as an effort error for that window."""
    pred_pose = np.array(classes)[pose_proba.argmax(1)]
    out, bas = {}, []
    for p in active:
        m = (meta.pose == p).to_numpy()
        if m.sum() == 0:
            continue
        pe = []
        for i in np.where(m)[0]:
            pp = pred_pose[i]
            if pp in active and np.isfinite(ph_all[i, active.index(pp)]):
                pe.append("High" if ph_all[i, active.index(pp)] >= 0.5 else "Low")
            else:
                pe.append("None")
        y = meta.effort.to_numpy()[m]
        pe = np.array(pe, dtype=object)
        ba = float(np.mean([np.mean(pe[y == e] == e) for e in C.EFFORTS if (y == e).any()]))
        out[p] = {"balanced_accuracy": ba, "pose_correct_fraction": float(np.mean(pred_pose[m] == p))}
        bas.append(ba)
    out["mean_balanced_accuracy"] = float(np.mean(bas)) if bas else float("nan")
    # joint label (pose|effort, Rest) macro-F1
    joint_true = meta.label.to_numpy()
    joint_pred = []
    for i in range(len(meta)):
        pp = pred_pose[i]
        if pp in active and np.isfinite(ph_all[i, active.index(pp)]):
            joint_pred.append(f"{pp}|{'High' if ph_all[i, active.index(pp)] >= 0.5 else 'Low'}")
        else:
            joint_pred.append(pp)
    labels = sorted(set(joint_true))
    out["joint_macro_f1"] = float(f1_score(joint_true, joint_pred, labels=labels, average="macro", zero_division=0))
    return out


def stability_metrics(meta, proba, classes, stride_s, k=SMOOTH_K):
    """Sequential predictions inside every test segment (ordered by time): prediction switches per second and
    the Rest false-positive rate, raw and with causal k-window probability averaging."""
    res = {"raw": {"switches": 0, "dur": 0.0}, "smooth": {"switches": 0, "dur": 0.0}}
    rest_fp = {"raw": [], "smooth": []}
    acc_s, y_all, p_all_s = [], [], []
    rest_i = classes.index(C.REST) if C.REST in classes else -1
    for _, g in meta.groupby(["trial_key", "seg"]):
        g = g.sort_values("end_clock")
        P = proba[g.index.to_numpy()]
        cs = np.cumsum(P, 0)
        Ps = (cs - np.vstack([np.zeros((k, P.shape[1])), cs[:-k]])[:len(P)]) / np.minimum(np.arange(1, len(P) + 1), k)[:, None]
        for name, Q in (("raw", P), ("smooth", Ps)):
            a = Q.argmax(1)
            res[name]["switches"] += int((a[1:] != a[:-1]).sum())
            res[name]["dur"] += len(a) * stride_s
            if g.pose.iloc[0] == C.REST and rest_i >= 0:
                rest_fp[name] += list(a != rest_i)
        y_all += list(g.pose)
        p_all_s += list(np.array(classes)[Ps.argmax(1)])
    return {"switches_per_s_raw": res["raw"]["switches"] / max(res["raw"]["dur"], 1e-9),
            "switches_per_s_smoothed": res["smooth"]["switches"] / max(res["smooth"]["dur"], 1e-9),
            "rest_false_positive_raw": float(np.mean(rest_fp["raw"])) if rest_fp["raw"] else float("nan"),
            "rest_false_positive_smoothed": float(np.mean(rest_fp["smooth"])) if rest_fp["smooth"] else float("nan"),
            "pose_macro_f1_smoothed": float(f1_score(y_all, p_all_s, labels=classes, average="macro", zero_division=0)),
            "smoothing_windows": k, "smoothing_added_delay_ms": (k - 1) / 2 * stride_s * 1000}


def speed_confound(meta, ph_true, active):
    """Spearman correlation between gyro magnitude (rotation speed) and P(High) WITHIN each effort level.
    A strong positive correlation inside 'Low' means the effort output follows movement speed, not effort."""
    out = {}
    for p in active:
        for e in C.EFFORTS:
            m = ((meta.pose == p) & (meta.effort == e)).to_numpy() & np.isfinite(ph_true)
            if m.sum() < 10:
                continue
            g, q = meta.gyro_mag_mean.to_numpy()[m], ph_true[m]
            if np.std(g) == 0 or np.std(q) == 0:
                continue
            rho, pv = sstats.spearmanr(g, q)
            out[f"{p}|{e}"] = {"n": int(m.sum()), "spearman_rho": float(rho), "p": float(pv),
                               "median_gyro_dps": float(np.median(g))}
    return out


# ============================================================================================ families
def _folds(meta_dev: pd.DataFrame, rows=None):
    """Dev-relative (train, val) index pairs; only folds with a non-empty val set."""
    f = meta_dev.fold.to_numpy()
    idx = np.arange(len(meta_dev)) if rows is None else np.asarray(rows)
    out = []
    for k in sorted(set(f[idx])):
        tr, va = idx[f[idx] != k], idx[f[idx] == k]
        if len(va) and len(tr):
            out.append((tr, va))
    return out


def _feature_pipe(model, features):
    emg = [f for f in features if f.startswith("emg_")]
    other = [f for f in features if not f.startswith("emg_")]
    parts = []
    if emg:
        parts.append(("emg_log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), emg))
    if other:
        parts.append(("imu", "passthrough", other))
    return Pipeline([("pre", ColumnTransformer(parts, verbose_feature_names_out=False)), ("scale", StandardScaler()), ("clf", model)])


RF_GRID = {"clf__max_depth": [None, 12], "clf__min_samples_leaf": [1, 5]}
LR_GRID = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
EFFORT_FEATS = {"emg": EMG_FEATURES, "emg+imu": ALL_FEATURES}


def _proba(model, X, classes):
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(classes)))
    for j, c in enumerate(model.classes_):
        out[:, classes.index(str(c))] = p[:, j]
    return out


def family_feature(ws, dev, test, classes, active, tc: TrainConfig, log):
    md, mt = ws.meta.iloc[dev].reset_index(drop=True), ws.meta.iloc[test].reset_index(drop=True)
    Fd = pd.DataFrame(ws.feats[dev], columns=ALL_FEATURES)
    Ft = pd.DataFrame(ws.feats[test], columns=ALL_FEATURES)
    folds = _folds(md)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=C.SEED, n_jobs=-1)
    pipe = _feature_pipe(rf, ALL_FEATURES)
    oof = np.full((len(md), len(classes)), np.nan)
    if tc.quick or len(folds) < 2:
        best = pipe.set_params(clf__min_samples_leaf=5).fit(Fd, md.pose)
        bp = {"clf__min_samples_leaf": 5}
    else:
        gs = GridSearchCV(pipe, RF_GRID, scoring="f1_macro", cv=folds, refit=True, n_jobs=1).fit(Fd, md.pose)
        best, bp = gs.best_estimator_, gs.best_params_
        for tr, va in folds:
            m = clone(best).fit(Fd.iloc[tr], md.pose.iloc[tr])
            oof[va] = _proba(m, Fd.iloc[va], classes)
    log(f"    RF pose params {bp}")
    pose_test = _proba(best, Ft, classes)

    eff = {}
    for opt in FAMILIES["A_feature"]["effort_options"]:
        feats = EFFORT_FEATS[opt]
        models, oof_ph, test_ph = {}, np.full(len(md), np.nan), np.full((len(mt), len(active)), np.nan)
        for k, p in enumerate(active):
            rows = np.where(md.pose == p)[0]
            if md.effort.iloc[rows].nunique() < 2:
                continue
            lr = _feature_pipe(LogisticRegression(max_iter=5000, class_weight="balanced"), feats)
            pf = [(tr, va) for tr, va in _folds(md, rows) if md.effort.iloc[tr].nunique() == 2]
            X, y = Fd[feats], md.effort
            if tc.quick or len(pf) < 2:
                m = lr.set_params(clf__C=0.1).fit(X.iloc[rows], y.iloc[rows])
            else:
                rel = {r: i for i, r in enumerate(rows)}
                pf_rel = [(np.array([rel[i] for i in tr]), np.array([rel[i] for i in va])) for tr, va in pf]
                m = GridSearchCV(lr, LR_GRID, scoring="balanced_accuracy", cv=pf_rel, refit=True).fit(X.iloc[rows], y.iloc[rows]).best_estimator_
                for tr, va in pf:
                    mm = clone(m).fit(X.iloc[tr], y.iloc[tr])
                    oof_ph[va] = mm.predict_proba(X.iloc[va])[:, list(mm.classes_).index("High")]
            models[p] = m
            test_ph[:, k] = m.predict_proba(Ft[feats])[:, list(m.classes_).index("High")]
        eff[opt] = {"models": models, "oof_ph_true": oof_ph, "test_ph_all": test_ph, "features": feats}
    return {"pose_oof": oof, "pose_test": pose_test, "pose_model": best, "effort": eff, "info": {"rf_params": bp}}


def _deep_arrays(ws, idx, norm, streams):
    e, i = D.apply_norm(norm, ws.emg[idx] if "emg" in streams else None, ws.imu[idx] if "imu" in streams else None)
    return e, i


def family_deep(ws, dev, test, classes, active, tc: TrainConfig, log, spec):
    meta = ws.meta
    pose_y = np.array([classes.index(p) for p in meta.pose])
    eff_pose = np.array([active.index(p) if (p in active and e in C.EFFORTS) else -1 for p, e in zip(meta.pose, meta.effort)])
    eff_y = (meta.effort.to_numpy() == "High").astype(np.float32)
    cfg = type("Cfg", (D.TrainCfg,), {"max_epochs": tc.max_epochs})
    multitask = spec.get("multitask", False)

    def fit(train_abs, streams, pose=True, effort=True, norm_abs=None):
        # one normalization per training fold (all its windows), shared by the pose and the effort net
        na = train_abs if norm_abs is None else norm_abs
        norm = D.fit_norm(ws.emg[na], ws.imu[na])
        e, i = _deep_arrays(ws, train_abs, norm, streams)
        val = D.inner_val_mask(meta, train_abs)
        net, info = D.train_net(streams, e, i, pose_y[train_abs] if pose else None, eff_pose[train_abs],
                                eff_y[train_abs] if effort else None, len(classes), len(active), val, cfg=cfg, log=log)
        return net, norm, info

    def pred(net, norm, idx):
        e, i = _deep_arrays(ws, idx, norm, net.streams)
        return D.predict_net(net, e, i)

    md = meta.iloc[dev].reset_index(drop=True)
    folds = [] if tc.quick else _folds(md)
    dev = np.asarray(dev)
    pose_streams = spec["pose"]
    out = {"pose_oof": np.full((len(dev), len(classes)), np.nan), "effort": {}, "info": {}}
    if multitask:
        opt = spec["effort_options"][0]
        oof_ph = np.full(len(dev), np.nan)
        for tr, va in folds:
            net, norm, _ = fit(dev[tr], pose_streams, True, True)
            P, E = pred(net, norm, dev[va])
            out["pose_oof"][va] = P
            ep = eff_pose[dev[va]]
            oof_ph[va] = np.where(ep >= 0, E[np.arange(len(va)), np.maximum(ep, 0)], np.nan)
        net, norm, info = fit(dev, pose_streams, True, True)
        P, E = pred(net, norm, test)
        out.update(pose_test=P, pose_net=net, pose_norm=norm)
        out["effort"][opt] = {"net": None, "norm": norm, "oof_ph_true": oof_ph, "test_ph_all": E}
        out["info"]["final"] = info
        return out
    for tr, va in folds:
        net, norm, _ = fit(dev[tr], pose_streams, True, False)
        out["pose_oof"][va] = pred(net, norm, dev[va])[0]
    net, norm, info = fit(dev, pose_streams, True, False)
    out.update(pose_test=pred(net, norm, test)[0], pose_net=net, pose_norm=norm)
    out["info"]["pose_final"] = info
    for opt in spec["effort_options"]:
        streams = tuple(opt.split("+"))
        oof_ph = np.full(len(dev), np.nan)
        for tr, va in folds:
            tr_e = tr[eff_pose[dev[tr]] >= 0]
            net_e, norm_e, _ = fit(dev[tr_e], streams, False, True, norm_abs=dev[tr])
            E = pred(net_e, norm_e, dev[va])[1]
            ep = eff_pose[dev[va]]
            oof_ph[va] = np.where(ep >= 0, E[np.arange(len(va)), np.maximum(ep, 0)], np.nan)
        dev_e = dev[eff_pose[dev] >= 0]
        net_e, norm_e, info_e = fit(dev_e, streams, False, True, norm_abs=dev)
        out["effort"][opt] = {"net": net_e, "norm": norm_e, "oof_ph_true": oof_ph, "test_ph_all": pred(net_e, norm_e, test)[1]}
        out["info"][f"effort_final_{opt}"] = info_e
    return out


# ============================================================================================ orchestration
def _ph_true(meta, ph_all, active):
    k = np.array([active.index(p) if p in active else -1 for p in meta.pose])
    return np.where(k >= 0, ph_all[np.arange(len(meta)), np.maximum(k, 0)], np.nan)


def _confound_score(md, ph_true, active):
    """Mean within-effort-level Spearman rho (rotation speed vs P(High)) over all pose x effort cells (CV, dev)."""
    sc = speed_confound(md, ph_true, active)
    return float(np.mean([v["spearman_rho"] for v in sc.values()])) if sc else 0.0


def _choose_effort(options: dict, md, active, quick):
    """EMG-only by default. EMG+IMU is used for the effort head only if, on CV, it (1) improves the mean balanced
    accuracy by > EFFORT_IMU_MARGIN and (2) does not increase the speed confound by > CONFOUND_MARGIN
    (i.e. it does not merely learn that 'High' trials were moved faster)."""
    scores = {o: effort_metrics(md, v["oof_ph_true"], active)["mean_balanced_accuracy"] for o, v in options.items()}
    conf = {o: _confound_score(md, v["oof_ph_true"], active) for o, v in options.items()}
    reason = ""
    if "emg" in options:
        chosen = "emg"
        for o, sc in scores.items():
            if o == "emg" or quick or not np.isfinite(sc):
                continue
            if sc <= scores["emg"] + EFFORT_IMU_MARGIN:
                reason = f"{o} not better than emg by > {EFFORT_IMU_MARGIN} (CV BA {sc:.3f} vs {scores['emg']:.3f})"
            elif conf[o] > conf["emg"] + CONFOUND_MARGIN:
                reason = (f"{o} rejected: higher CV BA ({sc:.3f} vs {scores['emg']:.3f}) but speed confound rho "
                          f"{conf[o]:+.2f} vs {conf['emg']:+.2f} (follows movement speed)")
            else:
                chosen, reason = o, f"{o}: CV BA {sc:.3f} > emg {scores['emg']:.3f}, confound {conf[o]:+.2f} vs {conf['emg']:+.2f}"
    else:
        chosen = next(iter(options))
        reason = "only option"
    return chosen, {"cv_ba": scores, "cv_speed_confound_rho": conf, "reason": reason}


def _make_predictor(fam, r, eff_opt, classes, active, cfg, name):
    if fam == "A_feature":
        e = r["effort"][eff_opt]
        return FeaturePredictor(r["pose_model"], e["models"], ALL_FEATURES, e["features"], classes, active, cfg, dict(C.FILTER), name)
    e = r["effort"][eff_opt]
    assert e["net"] is None or e["norm"] == r["pose_norm"], "pose/effort normalization must be identical"
    return DeepPredictor(r["pose_net"], e["net"], r["pose_norm"], classes, active, cfg, dict(C.FILTER), name)


def parity_and_latency(pred, sessions, ws, test, expected_pose, expected_ph, n=150):
    """Re-extract test windows from the raw session with the run-time code path, run the run-time predictor
    and compare with the batch evaluation; time extraction + prediction on CPU (1 window)."""
    rng = np.random.default_rng(0)
    pick = rng.choice(len(test), size=min(n, len(test)), replace=False)
    m = ws.meta.iloc[test].reset_index(drop=True)
    wins = []
    for j in pick:
        s = sessions[m.session_id.iloc[j]]
        wins.append((s, int(m.end_clock.iloc[j])))
    diffs = []
    for (s, end), j in zip(wins, pick):
        w = extract_window(s.clock, s.emg_filt, s.imu_clock, s.imu, end, ws.cfg)
        pp, ph = pred.predict(w)
        d = np.abs(pp - expected_pose[j]).max()
        fin = np.isfinite(expected_ph[j]) & np.isfinite(ph)
        if fin.any():
            d = max(d, np.abs(ph[fin] - expected_ph[j][fin]).max())
        diffs.append(d)

    def run(k):
        s, end = wins[k % len(wins)]
        a = max(0, np.searchsorted(s.clock, end) - 1500)          # ~3 s ring buffer like the live engine
        b = np.searchsorted(s.clock, end, side="right")
        ia, ib = np.searchsorted(s.imu_clock, s.clock[a]), np.searchsorted(s.imu_clock, end, side="right")
        w = extract_window(s.clock[a:b], s.emg_filt[a:b], s.imu_clock[ia:ib], s.imu[ia:ib], end, ws.cfg)
        pred.predict(w)
    med, p95 = D.cpu_latency_ms(run, n=200)
    return {"max_abs_diff": float(np.max(diffs)), "n_checked": len(diffs)}, {"median_ms": med, "p95_ms": p95}


def _plot_confusion(path, cm, labels, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 1.0 * len(labels) + 1.5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def select_sessions(tc: TrainConfig) -> list[str]:
    if tc.sessions:
        return [str(s) for s in tc.sessions]
    out = []
    for s in list_sessions():
        if s["kind"] != "training" or s["n_completed"] == 0:
            continue
        if "SYNTHETIC" in s["data_source"] and not tc.include_synthetic:
            continue
        out.append(s["path"])
    return out


def run(tc: TrainConfig, log=print) -> dict:
    t_start = time.time()
    tc.run_id = tc.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = C.RUNS_DIR / tc.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    paths = select_sessions(tc)
    if not paths:
        raise SystemExit("no training sessions found (collect data or import legacy data first)")
    log(f"run {tc.run_id}: loading {len(paths)} session(s)")
    sessions = {}
    for p in paths:
        s = load_session(Path(p))
        if "SYNTHETIC" in s.meta.get("data_source", "") and not tc.include_synthetic:
            log(f"  skip synthetic {s.session_id}")
            continue
        sessions[s.session_id] = s
        log(f"  {s.participant}/{s.session_id}: {s.stats['duration_s']:.0f} s, lost {s.stats['lost_samples']} samples, "
            f"IMU {s.stats['imu_rate_hz']:.1f} Hz")
    parts = sorted({s.participant for s in sessions.values()})
    if len(parts) > 1:
        log(f"  WARNING: sessions of {len(parts)} participants are pooled into one model: {parts}")
    fs_set = {s.fs for s in sessions.values()}
    if len(fs_set) != 1:
        raise SystemExit(f"sessions with different sampling rates: {fs_set}")
    fs = fs_set.pop()

    results = {"run_id": tc.run_id, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": tc.__dict__ | {"sessions": paths},
               "sessions": {k: {"participant": s.participant, "path": str(s.path), "stats": s.stats,
                                "data_source": s.meta.get("data_source")} for k, s in sessions.items()},
               "windows": {}, "torch_device": str(D._device())}
    rows = []
    for wms in tc.windows_ms:
        cfg = WindowConfig(float(wms), C.STRIDE_MS, fs)
        log(f"== window {wms:.0f} ms ({cfg.n_emg} EMG x {cfg.n_imu} IMU samples), stride {cfg.stride_ms:.0f} ms")
        ws = build_windows(list(sessions.values()), cfg)
        avail = split_availability(ws.meta)
        if not avail[tc.split]["available"]:
            raise SystemExit(f"split '{tc.split}' not possible: {avail[tc.split]['reason']}")
        splits = assign_splits(ws.meta, tc.split, tc.test_reps, tc.n_folds)
        attach_splits(ws, splits)
        poses_present = sorted(set(ws.meta.pose))
        classes = [C.REST] + [p for p in poses_present if p != C.REST] if C.REST in poses_present else poses_present
        active = sorted(p for p in poses_present if p != C.REST and
                        set(ws.meta[ws.meta.pose == p].effort) >= set(C.EFFORTS))
        dev = np.where(ws.meta.split == "dev")[0]
        test = np.where(ws.meta.split == "test")[0]
        md, mt = ws.meta.iloc[dev].reset_index(drop=True), ws.meta.iloc[test].reset_index(drop=True)
        wres = {"window": cfg.to_dict(), "split_mode": tc.split, "split_availability": avail,
                "pose_classes": classes, "effort_poses": active,
                "n_windows": {"dev": int(len(dev)), "test": int(len(test))}, "dropped_windows": ws.dropped,
                "trials": {"dev": int(md.trial_key.nunique()), "test": int(mt.trial_key.nunique())},
                "test_trials": sorted(mt.trial_key.unique().tolist()),
                "windows_per_label": ws.meta.groupby(["split", "label"]).size().unstack(0).fillna(0).astype(int).to_dict(),
                "gyro_mag_median_by_label": ws.meta.groupby("label").gyro_mag_mean.median().round(1).to_dict(),
                "families": {}}
        log(f"  classes {classes}; effort poses {active}; dev {len(dev)} windows / {wres['trials']['dev']} trials, "
            f"test {len(test)} windows / {wres['trials']['test']} trials; dropped {ws.dropped['invalid']}")
        for fam in tc.families:
            spec = FAMILIES[fam]
            log(f"  -- {fam}: {spec['display']}")
            t0 = time.time()
            try:
                r = family_feature(ws, dev, test, classes, active, tc, log) if fam == "A_feature" else \
                    family_deep(ws, dev, test, classes, active, tc, log, spec)
            except Exception as e:
                log(f"     FAILED: {e}\n{traceback.format_exc()}")
                wres["families"][fam] = {"error": str(e)}
                continue
            chosen, eff_cv = _choose_effort(r["effort"], md, active, tc.quick)
            fr = {"display": spec["display"], "train_time_s": time.time() - t0, "info": r.get("info", {})}
            oof_ok = np.isfinite(r["pose_oof"]).all(1)
            fr["cv_pose"] = cls_metrics(md.pose[oof_ok], np.array(classes)[r["pose_oof"][oof_ok].argmax(1)], classes) if oof_ok.any() else None
            fr["test_pose"] = cls_metrics(mt.pose, np.array(classes)[r["pose_test"].argmax(1)], classes)
            fr["effort_options"] = {}
            for opt, e in r["effort"].items():
                ph_true_test = _ph_true(mt, e["test_ph_all"], active)
                fr["effort_options"][opt] = {
                    "cv_true_pose": effort_metrics(md, e["oof_ph_true"], active) if np.isfinite(e["oof_ph_true"]).any() else None,
                    "test_true_pose": effort_metrics(mt, ph_true_test, active),
                    "test_cascade": cascade_metrics(mt, r["pose_test"], e["test_ph_all"], classes, active),
                    "speed_confound_test": speed_confound(mt, ph_true_test, active),
                    "speed_confound_cv": speed_confound(md, e["oof_ph_true"], active)}
            fr["effort_input"] = chosen
            fr["effort_input_cv_scores"] = eff_cv
            fr["stability_test"] = stability_metrics(mt, r["pose_test"], classes, cfg.stride_ms / 1000)
            pred = _make_predictor(fam, r, chosen, classes, active, cfg, spec["display"])
            parity, lat = parity_and_latency(pred, sessions, ws, test, r["pose_test"], r["effort"][chosen]["test_ph_all"])
            fr["runtime_parity"], fr["latency_cpu"] = parity, lat
            out_dir = run_dir / f"w{int(wms)}" / fam
            pred.save(out_dir)
            try:      # a figure must never abort a training run
                _plot_confusion(out_dir / "confusion_test.png", fr["test_pose"]["confusion_matrix"], classes,
                                f"{spec['display']} - {wms:.0f} ms - test pose")
            except Exception as e:
                log(f"     (confusion figure not saved: {type(e).__name__}: {e})")
            eo = fr["effort_options"][chosen]
            summary = {"run_id": tc.run_id, "family": fam, "display": spec["display"], "window_ms": float(wms),
                       "pose_classes": classes, "effort_poses": active, "effort_input": chosen,
                       "test_pose_macro_f1": fr["test_pose"]["macro_f1"],
                       "cv_pose_macro_f1": fr["cv_pose"]["macro_f1"] if fr["cv_pose"] else None,
                       "test_effort_ba_true_pose": eo["test_true_pose"]["mean_balanced_accuracy"],
                       "test_effort_ba_cascade": eo["test_cascade"]["mean_balanced_accuracy"],
                       "cv_effort_ba_true_pose": eo["cv_true_pose"]["mean_balanced_accuracy"] if eo["cv_true_pose"] else None,
                       "latency_median_ms": lat["median_ms"], "latency_p95_ms": lat["p95_ms"],
                       "rest_fp_raw": fr["stability_test"]["rest_false_positive_raw"],
                       "rest_fp_smoothed": fr["stability_test"]["rest_false_positive_smoothed"],
                       "switches_per_s_raw": fr["stability_test"]["switches_per_s_raw"],
                       "switches_per_s_smoothed": fr["stability_test"]["switches_per_s_smoothed"],
                       "parity_max_abs_diff": parity["max_abs_diff"], "train_sessions": list(sessions),
                       "split": tc.split, "quick": tc.quick, "recommended": False,
                       "input_spec": input_spec(pred), "input_spec_text": input_spec_text(pred)}
            C.save_json(summary, out_dir / "meta.json")
            fr["summary"] = summary
            fr["bundle"] = str(out_dir)
            wres["families"][fam] = fr
            rows.append(summary)
            log(f"     pose macro-F1 test {summary['test_pose_macro_f1']:.3f} (CV {_f(summary['cv_pose_macro_f1'])}) | "
                f"effort BA [{chosen}] test {summary['test_effort_ba_true_pose']:.3f} (CV {_f(summary['cv_effort_ba_true_pose'])}), "
                f"cascade {summary['test_effort_ba_cascade']:.3f} | latency {lat['median_ms']:.2f} ms | parity {parity['max_abs_diff']:.1e}")
        results["windows"][str(int(wms))] = wres

    rec = recommend(rows, tc.quick)
    results["recommendation"] = rec
    if rec.get("bundle_key"):
        fam, w = rec["bundle_key"]
        mp = run_dir / f"w{int(w)}" / fam / "meta.json"
        m = C.load_json(mp)
        m["recommended"] = True
        C.save_json(m, mp)
    results["elapsed_s"] = time.time() - t_start
    pd.DataFrame(rows).to_csv(run_dir / "comparison.csv", index=False)
    C.save_json(results, run_dir / "results.json")
    (run_dir / "report.md").write_text(make_report(results, rows), encoding="utf-8")
    log(f"done in {results['elapsed_s']:.0f} s -> {run_dir}")
    log(rec["text"])
    return {"run_dir": str(run_dir), "rows": rows, "recommendation": rec, "results": results}


def _f(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def recommend(rows, quick) -> dict:
    ok = [r for r in rows if r.get("cv_pose_macro_f1") is not None]
    if not rows:
        return {"text": "no models trained"}
    if quick or not ok:
        base = [r for r in rows if r["family"] == "A_feature"]
        r = base[0] if base else rows[0]
        return {"bundle_key": (r["family"], r["window_ms"]), "rule": "quick mode (no CV): baseline used by default",
                "text": f"Quick mode: no cross-validation -> no data-driven selection; default = {r['display']} {r['window_ms']:.0f} ms"}

    def score(r):
        e = r["cv_effort_ba_true_pose"] if r["cv_effort_ba_true_pose"] is not None else 0.5
        return (r["cv_pose_macro_f1"] + e) / 2

    best = max(ok, key=score)
    base = [r for r in ok if r["family"] == "A_feature" and r["window_ms"] == best["window_ms"]] or \
           [r for r in ok if r["family"] == "A_feature"]
    note = ""
    if best["family"] != "A_feature" and base:
        b = max(base, key=score)
        if score(best) - score(b) <= BASELINE_MARGIN:
            note = f" (deep model {best['display']} not better than the baseline by > {BASELINE_MARGIN}; baseline kept)"
            best = b
    smaller = [r for r in ok if r["family"] == best["family"] and r["window_ms"] < best["window_ms"]
               and score(best) - score(r) <= 0.01]
    if smaller:
        best = min(smaller, key=lambda r: r["window_ms"])
        note += " (shorter window within 0.01 preferred: less delay)"
    return {"bundle_key": (best["family"], best["window_ms"]),
            "rule": "max mean(CV pose macro-F1, CV effort BA with true pose); baseline kept unless a deep model is "
                    f"better by > {BASELINE_MARGIN}; shorter window preferred within 0.01. Test set never used for selection.",
            "text": f"Recommended: {best['display']} @ {best['window_ms']:.0f} ms (CV pose F1 {_f(best['cv_pose_macro_f1'])}, "
                    f"CV effort BA {_f(best['cv_effort_ba_true_pose'])}){note}"}


def make_report(res, rows) -> str:
    L = [f"# Pose x Effort model comparison - run {res['run_id']}", "",
         f"Created {res['created']} | elapsed {res.get('elapsed_s', 0):.0f} s | torch device {res['torch_device']}", "",
         "## Data", ""]
    for k, s in res["sessions"].items():
        L.append(f"- {s['participant']} / {k} ({s['data_source']}): {s['stats']['duration_s']:.0f} s, "
                 f"lost samples {s['stats']['lost_samples']}, IMU {s['stats']['imu_rate_hz']:.1f} Hz")
    parts = {s["participant"] for s in res["sessions"].values()}
    L += ["", f"Split mode: **{res['config']['split']}**" + (" (quick mode - no CV)" if res["config"]["quick"] else ""), ""]
    w0 = next(iter(res["windows"].values()))
    for m, a in w0["split_availability"].items():
        L.append(f"- {m}-independent test: {'available' if a['available'] else 'NOT available'} - {a['reason']}")
    if len(parts) < 2:
        L.append("- **Participant-independent evaluation is not possible with data from a single participant.** "
                 "All numbers below are subject- and session-specific.")
    L += ["", "## Comparison (test = held-out trials, evaluated once)", "",
          "| Model | Window | Pose Macro-F1 test (CV) | Effort BA true pose, test (CV) | Effort BA cascade (test) | Effort input | "
          "Latency CPU median / p95 | Rest FP raw / smoothed | Switches/s raw / smoothed |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['display']} | {r['window_ms']:.0f} ms | {_f(r['test_pose_macro_f1'])} ({_f(r['cv_pose_macro_f1'])}) | "
                 f"{_f(r['test_effort_ba_true_pose'])} ({_f(r['cv_effort_ba_true_pose'])}) | {_f(r['test_effort_ba_cascade'])} | "
                 f"{r['effort_input']} | {r['latency_median_ms']:.2f} / {r['latency_p95_ms']:.2f} ms | "
                 f"{_f(r['rest_fp_raw'])} / {_f(r['rest_fp_smoothed'])} | {_f(r['switches_per_s_raw'], 2)} / {_f(r['switches_per_s_smoothed'], 2)} |")
    L += ["", f"**{res['recommendation']['text']}**", "", f"Rule: {res['recommendation'].get('rule', '')}", "",
          "Every model predicts every 50 ms (20 Hz) from the most recent window: EMG 8 ch @ 500 Hz "
          "(200 ms = 100 samples, 300 ms = 150, 500 ms = 250) and IMU 6 axes @ 50 Hz (10 / 15 / 25 samples).", "",
          "Latency = window extraction + preprocessing + model on one window, CPU, single thread-ish (not end-to-end; "
          "add the stride wait (<= 50 ms), smoothing delay and device/Wi-Fi delay for end-to-end).",
          f"Smoothed = causal average of the last {SMOOTH_K} window probabilities (adds ~{(SMOOTH_K - 1) / 2 * 50:.0f} ms delay).", ""]
    for wk, w in res["windows"].items():
        L += [f"## Window {wk} ms - details", "",
              f"Pose classes {w['pose_classes']}; effort poses {w['effort_poses']}; windows dev {w['n_windows']['dev']} / test "
              f"{w['n_windows']['test']}; trials dev {w['trials']['dev']} / test {w['trials']['test']}; dropped {w['dropped_windows']}", "",
              f"Median gyro magnitude by label (dps): {w['gyro_mag_median_by_label']}", ""]
        for fam, fr in w["families"].items():
            if "error" in fr:
                L += [f"### {fam}: FAILED - {fr['error']}", ""]
                continue
            L += [f"### {fr['display']}", "", "Model input (per prediction):", "", "```",
                  fr["summary"].get("input_spec_text", ""), "```", "",
                  f"Pose test per-class F1: {fr['test_pose']['per_class_f1']}", "",
                  f"Effort input chosen: **{fr['effort_input']}** (CV mean BA per input: "
                  + ", ".join(f"{k} {_f(v)}" for k, v in fr["effort_input_cv_scores"]["cv_ba"].items()) + "; CV speed-confound rho: "
                  + ", ".join(f"{k} {v:+.2f}" for k, v in fr["effort_input_cv_scores"]["cv_speed_confound_rho"].items())
                  + f") - {fr['effort_input_cv_scores']['reason']}", "",
                  "| effort input | pose | test BA (true pose) | CV BA | test BA cascade |", "|---|---|---|---|---|"]
            for opt, eo in fr["effort_options"].items():
                for p in w["effort_poses"]:
                    t = eo["test_true_pose"].get(p)
                    cv = (eo["cv_true_pose"] or {}).get(p)
                    ca = eo["test_cascade"].get(p)
                    L.append(f"| {opt} | {p} | {_f(t['balanced_accuracy']) if t else 'n/a'} | "
                             f"{_f(cv['balanced_accuracy']) if cv else 'n/a'} | {_f(ca['balanced_accuracy']) if ca else 'n/a'} |")
            sc = fr["effort_options"][fr["effort_input"]]["speed_confound_test"]
            if sc:
                L += ["", "Speed confound check (test; Spearman rho of gyro magnitude vs P(High) within an effort level; "
                          "|rho| > 0.3 means the effort output partly follows movement speed):", ""]
                for k, v in sc.items():
                    L.append(f"- {k}: rho {v['spearman_rho']:+.2f} (n={v['n']}, median |w| {v['median_gyro_dps']:.0f} dps)")
            L += ["", f"Run-time parity (predictor on re-extracted raw windows vs batch evaluation): max |diff| "
                      f"{fr['runtime_parity']['max_abs_diff']:.2e} over {fr['runtime_parity']['n_checked']} windows", "",
                  f"![confusion](w{wk}/{fam}/confusion_test.png)", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=float, nargs="+", default=[C.WINDOW_MS])
    ap.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES, choices=list(FAMILIES))
    ap.add_argument("--split", default="trial", choices=["trial", "session", "participant"])
    ap.add_argument("--sessions", nargs="*", default=[])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--test-reps", type=int, default=2)
    ap.add_argument("--run-id", default="")
    a = ap.parse_args()
    C.ensure_dirs()
    run(TrainConfig(a.sessions, a.windows, a.families, a.split, a.test_reps, quick=a.quick, max_epochs=a.epochs, run_id=a.run_id))


if __name__ == "__main__":
    main()
