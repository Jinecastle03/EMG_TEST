"""Headless system tests (no MindRove hardware needed).

  python tests/test_system.py            (or: python -m pytest tests -q)

Writes only into a temporary EMGMON_ROOT copy, never into the real data / model folders, except that it reads the
imported legacy session and legacy model.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
TMP = Path(tempfile.mkdtemp(prefix="emgmon_test_"))
os.environ["EMGMON_ROOT"] = str(TMP)
(TMP / "models").mkdir()
shutil.copytree(REPO / "models" / "legacy_20261103", TMP / "models" / "legacy_20261103")
LEGACY_SESSION = next((REPO / "data" / "sessions" / "S01").glob("*_legacy"))
(TMP / "data" / "sessions" / "S01").mkdir(parents=True)
os.symlink(LEGACY_SESSION, TMP / "data" / "sessions" / "S01" / LEGACY_SESSION.name, target_is_directory=True) \
    if hasattr(os, "symlink") and os.name != "nt" else shutil.copytree(LEGACY_SESSION, TMP / "data" / "sessions" / "S01" / LEGACY_SESSION.name)

from emgmonitor import config as C  # noqa: E402
from emgmonitor.acquisition import MindRoveSource, ReplaySource  # noqa: E402
from emgmonitor.dsp import CausalFilter, SampleClock, WindowConfig, extract_window  # noqa: E402
from emgmonitor.engine import Engine  # noqa: E402
from emgmonitor.predictors import legacy_predictor  # noqa: E402
from emgmonitor.session import SessionWriter, load_session  # noqa: E402

C.ensure_dirs()
LEG = TMP / "data" / "sessions" / "S01" / LEGACY_SESSION.name


def wait(cond, timeout):
    t = time.time()
    while time.time() - t < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_filter_chunk_invariance():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 50, (5000, 8)) + 30000
    a = CausalFilter(500).process(x)
    f = CausalFilter(500)
    b = np.concatenate([f.process(x[i:i + k]) for i, k in zip(range(0, 5000, 37), [37] * 200) if i < 5000])
    assert np.abs(a - b).max() < 1e-9


def test_sample_clock():
    c = SampleClock()
    a = c.process(np.array([10, 11, 12, 15, 16]))
    b = c.process(np.array([17, 0, 1]))            # wrap / reset -> +1
    assert a.tolist() == [0, 1, 2, 5, 6] and b.tolist() == [7, 8, 9] and c.lost == 2 and c.discontinuities == 1


def test_window_alignment():
    cfg = WindowConfig(200, 50, 500)
    clock = np.arange(0, 1000)
    clock = np.delete(clock, [500, 501])           # 2 lost samples
    emg = np.tile(clock[:, None].astype(float), (1, 8))
    iclk = np.arange(0, 1000, 10)
    imu = np.tile(iclk[:, None].astype(float), (1, 6))
    w = extract_window(clock, emg, iclk, imu, 549, cfg)   # window 450..549, gap inside
    assert w.valid and w.emg_grid.shape == (8, 100) and np.allclose(w.emg_grid[0], np.arange(450, 550))   # interpolated gap
    assert w.imu_grid.shape == (6, 10) and np.allclose(w.imu_grid[0], np.arange(450, 550, 10))
    assert len(w.emg_actual) == 98 and len(w.imu_actual) == 10


def test_replay_engine_legacy_model_and_roundtrip():
    """Replay the legacy session through the real-time engine with the legacy RF/LR model; record the replayed
    stream; check (1) streaming filter == offline filter, (2) recorded raw == source raw, (3) accuracy on held-out
    trials of the legacy test split, (4) latency."""
    eng = Engine()
    pred = legacy_predictor(TMP / "models" / "legacy_20261103")
    eng.set_predictor(pred)
    eng.inference_on = True
    eng.smooth_k = 1
    src = ReplaySource(LEG, speed=8.0, chunk=10, start_s=0.0)
    eng.start(src)
    w = SessionWriter(C.SESSIONS_DIR, "TEST", "Replay_roundtrip", {"kind": "free", "data_source": "REPLAY", "fs": 500.0})
    eng.start_recording(w)
    seen = []
    t0 = time.time()
    while src.is_alive() and time.time() - t0 < 200:
        p = eng.prediction
        if p is not None and (not seen or seen[-1] is not p):
            seen.append(p)
        time.sleep(0.004)
    eng.stop()
    assert src.state == "finished", src.state
    # (1)+(2)
    off = load_session(LEG)
    rec = load_session(w.dir)
    # the recording starts a few chunks after the replay (start_recording is called after start): align by sample_index
    k = int(np.searchsorted(off.sample_index, rec.sample_index[0]))
    print(f"  recorded {len(rec.emg_raw)} of {len(off.emg_raw)} rows (first {k} rows streamed before recording started)")
    assert k < 200 and np.array_equal(rec.sample_index, off.sample_index[k:])
    assert np.abs(rec.emg_raw - off.emg_raw[k:]).max() < 1e-3
    assert np.array_equal(np.diff(rec.clock), np.diff(off.clock[k:]))
    ki = len(off.imu) - len(rec.imu)
    assert 0 <= ki < 20 and np.abs(rec.imu - off.imu[ki:]).max() < 1e-5
    n = eng.filt.n
    tail = eng.filt.last(n)
    assert np.abs(tail - off.emg_filt[-n:]).max() < 1e-6, "streaming filter differs from offline filter"
    # (3) held-out trials of the legacy split (last 2 per label)
    t = off.trials[off.trials.status == "completed"].copy()
    t["label"] = t.pose + "|" + t.effort
    t = t.sort_values("hold_start_host")
    test_trials = t.groupby("label").tail(2)
    hits, total, eff_hits, eff_total = 0, 0, 0, 0
    for p in seen:
        m = test_trials[(test_trials.hold_start_idx <= p.sidx - 100) & (p.sidx <= test_trials.hold_end_idx)]
        if len(m) == 0 or p.pose == "UNKNOWN":
            continue
        total += 1
        hits += p.pose == m.pose.iloc[0]
        if m.pose.iloc[0] != "Rest" and p.pose == m.pose.iloc[0]:
            eff_total += 1
            eff_hits += p.effort == m.effort.iloc[0]
    acc = hits / max(total, 1)
    lat = np.median([p.latency_ms for p in seen])
    inf = np.median([p.inference_ms for p in seen])
    print(f"  replay: {len(seen)} predictions, test-trial pose acc {acc:.3f} (n={total}), effort acc {eff_hits / max(eff_total, 1):.3f}, "
          f"inference {inf:.1f} ms, processing latency {lat:.1f} ms, skipped windows {eng.status.dropped_windows}")
    assert total > 300 and acc > 0.85


def test_unknown_on_fs_mismatch_and_sdk_synthetic():
    """SDK synthetic board (250 Hz): exercises the MindRove SDK path; the 500 Hz model must answer UNKNOWN."""
    eng = Engine()
    eng.set_predictor(legacy_predictor(TMP / "models" / "legacy_20261103"))
    eng.inference_on = True
    src = MindRoveSource("synthetic")
    eng.start(src)
    ok = wait(lambda: eng.status.emg_rate > 0 and eng.prediction is not None, 15)
    time.sleep(1.5)
    st, pr = eng.status, eng.prediction
    eng.stop()
    print(f"  synthetic: state {src.state}, fs {src.fs}, EMG {st.emg_rate:.0f} Hz, IMU {st.imu_rate:.1f} Hz, layout {src.layout.get('emg_rows')}, "
          f"prediction {pr.pose if pr else None} ({pr.reason if pr else ''}); errors: {src.error}")
    assert ok and src.error == ""
    assert pr.pose == "UNKNOWN" and "fs" in pr.reason


def test_gui_collection_and_training_offscreen():
    """Drive the real Collect-Data window (offscreen Qt) on the SDK synthetic stream with a short protocol including a
    NEW pose, then load the session and train quickly on it -> checks the new session format, trial phases,
    inter-trial Rest labeling, new-pose handling and the trainer on new data (synthetic -> accuracy meaningless)."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from emgmonitor.gui import theme
    from emgmonitor.gui.main_window import MainWindow
    from emgmonitor.gui.collection import CollectionWindow
    from emgmonitor.poses import Pose, add_pose
    from emgmonitor import trainer
    app = QApplication.instance() or QApplication([])
    theme.apply(app)
    add_pose(Pose("Tripod Grip", "엄지·검지·중지 끝으로 집기", "펜"))
    mw = MainWindow()
    mw.participant.setText("TEST")
    mw.engine.start(MindRoveSource("synthetic"))
    assert wait(lambda: mw.engine.status.emg_rate > 0, 15)
    cw = CollectionWindow(mw)
    for i in range(cw.pose_list.count()):
        it = cw.pose_list.item(i)
        it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in ("Power Grip", "Tripod Grip") else Qt.Unchecked)
    cw.reps.setValue(3)
    cw.rest_trials.setValue(2)
    cw.prep.setValue(1.0)
    cw.hold.setValue(1.5)
    cw.rest.setValue(2.0)
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    cw.start_session()
    sess_dir = cw.writer.dir
    t0 = time.time()
    while cw.phase != "DONE" and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.01)
        if cw.phase == "REST" and len([t for t in cw.trials if t["status"] == "completed"]) == 1 and not getattr(cw, "_redone", False):
            cw._redone = True
            cw.redo_last()                       # exercise re-collection
    cw.finish()
    for _ in range(20):
        app.processEvents()
    mw.engine.stop()
    s = load_session(sess_dir)
    tr = s.trials
    print(f"  collection: {len(tr)} trial rows, statuses {tr.status.value_counts().to_dict()}, "
          f"labels {sorted(set(tr[tr.status == 'completed'].condition))}")
    assert (tr.status == "completed").sum() == 3 * 4 + 2
    assert (tr.status == "rejected").sum() == 1
    c = tr[tr.status == "completed"]
    assert (c.prep_start_idx <= c.hold_start_idx).all() and (c.hold_start_idx < c.hold_end_idx).all() and (c.hold_end_idx < c.rest_end_idx).all()
    hold_s = (c.hold_end_idx - c.hold_start_idx) / s.fs
    assert hold_s.between(1.2, 1.9).all(), hold_s.tolist()
    res = trainer.run(trainer.TrainConfig(sessions=[str(sess_dir)], families=["A_feature", "D_twostream"], quick=True,
                                          test_reps=1, include_synthetic=True, max_epochs=5, run_id="test_synth"),
                      log=lambda m: None)
    w = res["results"]["windows"]["200"]
    print(f"  trained on synthetic session: classes {w['pose_classes']}, effort poses {w['effort_poses']}, windows {w['n_windows']}")
    assert "Tripod Grip" in w["pose_classes"] and "Rest" in w["pose_classes"]
    assert set(w["effort_poses"]) == {"Power Grip", "Tripod Grip"}
    mw.close()


def test_realtime_1x_latency():
    """1x real-time replay (40 s of held-out legacy trials) with the legacy RF and with a CNN bundle if present:
    processing latency (SDK read of newest sample -> prediction ready), inference time, skipped windows."""
    from emgmonitor.predictors import load_predictor
    models = [("legacy RF/LR", legacy_predictor(TMP / "models" / "legacy_20261103"))]
    cnn = sorted((REPO / "models" / "runs").glob("*/w200/D_twostream/model.pt"))
    if cnn:
        models.append(("two-stream CNN 200 ms", load_predictor(cnn[-1].parent)))
    off = load_session(LEG)
    t = off.trials[off.trials.status == "completed"].sort_values("hold_start_host")
    start_s = float((t.prep_start_idx.iloc[-14] - off.sample_index[0]) / off.fs)
    for name, pred in models:
        eng = Engine()
        eng.set_predictor(pred)
        eng.inference_on = True
        src = ReplaySource(LEG, speed=1.0, chunk=10, start_s=start_s)
        eng.start(src)
        seen = []
        t0 = time.time()
        while time.time() - t0 < 40:
            p = eng.prediction
            if p is not None and (not seen or seen[-1] is not p):
                seen.append(p)
            time.sleep(0.003)
        st = eng.status
        eng.stop()
        lat = np.array([p.latency_ms for p in seen])
        inf = np.array([p.inference_ms for p in seen])
        print(f"  1x {name}: {len(seen) / 40:.1f} pred/s, inference median {np.median(inf):.2f} ms p95 {np.percentile(inf, 95):.2f}, "
              f"processing latency median {np.median(lat):.1f} ms p95 {np.percentile(lat, 95):.1f}, skipped {st.dropped_windows}, "
              f"EMG {st.emg_rate:.0f} Hz IMU {st.imu_rate:.1f} Hz")
        assert len(seen) > 15 * 40 and np.median(lat) < 50


def test_calibration_wizard_rest_max_offscreen():
    """Rest 5 s + wrist flexion/extension max on the SDK synthetic stream: ~20 s, saves rest/max/norm + raw session."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    from emgmonitor.gui import theme
    from emgmonitor.gui.main_window import MainWindow
    from emgmonitor.gui.calibration_wizard import CalibrationWizard
    from emgmonitor.calibration import Calibration
    app = QApplication.instance() or QApplication([])
    theme.apply(app)
    mw = MainWindow()
    mw.participant.setText("CALTEST")
    mw.engine.start(MindRoveSource("synthetic"))
    assert wait(lambda: mw.engine.status.emg_rate > 0, 15)
    time.sleep(1.0)
    cw = CalibrationWizard(mw)
    cw.start()
    t0 = time.time()
    while cw._seq and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.01)
    dur = time.time() - t0
    cw.save()
    mw.engine.stop()
    files = list((C.CALIB_DIR / "CALTEST").glob("*.json"))
    cal = Calibration.load(files[0])
    print(f"  calibration: {dur:.1f} s, tasks {list(cal.max)}, has_norm {cal.has_norm}, "
          f"raw session {cal.session_id}, activation(max) {cal.activation(cal.max_rms):.2f}")
    assert dur < 30 and cal.has_rest and cal.has_norm and set(cal.max) == {"flexion", "extension"}
    assert (C.SESSIONS_DIR / "CALTEST" / cal.session_id / "emg.csv").exists()
    assert mw.calibration is not None and mw.calibration.calibration_id == cal.calibration_id
    mw.view.setCurrentIndex(2)          # normalized view must render with the calibration
    mw._plot()
    mw.close()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t:.1f} s)")
            except Exception as e:
                import traceback
                fails += 1
                print(f"FAIL {name}: {e}")
                traceback.print_exc()
    shutil.rmtree(TMP, ignore_errors=True)
    print("ALL PASSED" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
