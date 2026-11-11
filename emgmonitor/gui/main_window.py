from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QSpinBox,
                               QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QDoubleSpinBox,
                               QAbstractItemView, QPlainTextEdit)

from .. import config as C
from ..acquisition import MindRoveSource, ReplaySource
from ..calibration import Calibration, latest_calibration
from ..engine import Engine
from ..poses import ko_name
from ..predictors import input_spec_text, list_models, load_predictor
from ..session import SessionWriter, list_sessions, new_session_id
from . import theme as T


# ============================================================================== dialogs
class ConnectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect")
        f = QFormLayout(self)
        self.kind = QComboBox()
        self.kind.addItems(["MindRove Armband 2 (Wi-Fi)", "SDK Synthetic board (test, no hardware)"])
        self.ip = QLineEdit()
        self.ip.setPlaceholderText("empty = SDK default (192.168.4.1)")
        self.port = QSpinBox()
        self.port.setRange(0, 65535)
        self.port.setSpecialValueText("SDK default")
        f.addRow("Source", self.kind)
        f.addRow("IP (optional)", self.ip)
        f.addRow("Port (optional)", self.port)
        f.addRow(QLabel("1) 암밴드 전원 ON  2) PC Wi-Fi를 암밴드 네트워크에 연결  3) 방화벽에서 python.exe 허용"))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def source(self):
        return MindRoveSource("wifi" if self.kind.currentIndex() == 0 else "synthetic", self.ip.text().strip(), self.port.value())


class ReplayDialog(QDialog):
    def __init__(self, parent=None, model_classes=None):
        super().__init__(parent)
        self.setWindowTitle("Replay recorded session")
        self.resize(820, 420)
        v = QVBoxLayout(self)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["participant", "session", "kind", "source", "trials", "labels"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sessions = [s for s in list_sessions() if s["kind"] in ("training", "free", "calibration")]
        for s in self.sessions:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, val in enumerate([s["participant"], s["session_id"], s["kind"], s["data_source"], s["n_completed"],
                                     ", ".join(sorted(s["labels"]))]):
                self.table.setItem(r, c, QTableWidgetItem(str(val)))
        self.table.resizeColumnsToContents()
        if self.sessions:
            self.table.selectRow(0)
        v.addWidget(self.table)
        h = QHBoxLayout()
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.25, 10)
        self.speed.setValue(1.0)
        self.speed.setSuffix(" x")
        self.start = QDoubleSpinBox()
        self.start.setRange(0, 100000)
        self.start.setSuffix(" s")
        self.loop = QCheckBox("loop")
        h.addWidget(QLabel("speed"))
        h.addWidget(self.speed)
        h.addWidget(QLabel("start at"))
        h.addWidget(self.start)
        h.addWidget(self.loop)
        h.addStretch()
        v.addLayout(h)
        v.addWidget(QLabel("Replay plays the stored RAW data in real time through the same filter / window / model path. "
                           "Predictions are the model's real outputs; the ground truth comes from the trial table."))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def selected(self):
        r = self.table.currentRow()
        return None if r < 0 else Path(self.sessions[r]["path"])


# ============================================================================== main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        C.ensure_dirs()
        self.setWindowTitle("EMG Pose × Effort Monitor — MindRove Armband 2")
        self.resize(1600, 950)
        self.engine = Engine()
        self.calibration: Calibration | None = None
        self.collect_win = None
        self.train_win = None
        self.pose_colors = {}
        self._build()
        self.refresh_models()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)          # GUI refresh ~30 Hz, independent from inference (engine thread)

    # ------------------------------------------------------------------ layout
    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(8, 6, 8, 6)

        # --- top status
        top = QHBoxLayout()
        self.mode_lab = T.big_label("DISCONNECTED", 18)
        top.addWidget(self.mode_lab)
        self.stat_labs = {}
        for k in ("device", "emg", "imu", "battery", "calib", "rec"):
            lab = QLabel("-")
            lab.setStyleSheet(f"color:{T.MUTED}; padding: 0 10px;")
            self.stat_labs[k] = lab
            top.addWidget(lab)
        top.addStretch()
        top.addWidget(QLabel("Participant"))
        self.participant = QLineEdit("S02")
        self.participant.setMaximumWidth(90)
        self.participant.editingFinished.connect(self._participant_changed)
        top.addWidget(self.participant)
        self.session_lab = QLabel("Session: -")
        self.session_lab.setStyleSheet(f"color:{T.MUTED}; padding-left: 8px;")
        top.addWidget(self.session_lab)
        v.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        v.addWidget(split, 1)

        # --- left: EMG + IMU
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        tb = QHBoxLayout()
        self.view = QComboBox()
        self.view.addItems(["Filtered (µV)", "Raw (µV)", "Normalized (÷ calibrated max RMS)"])
        self.view.setToolTip("Filtered: HP 20 Hz + notch 60 Hz (= model input)\nRaw: SDK values incl. DC offset\n"
                             "Normalized: filtered ÷ each channel's max RMS from calibration (1.0 = max), same scale for all channels")
        self.autoscale = QCheckBox("Auto scale")
        self.autoscale.setChecked(True)
        self.secs = QComboBox()
        self.secs.addItems(["2 s", "3 s", "5 s"])
        self.secs.setCurrentIndex(1)
        self.ch_info = QLabel("")
        self.ch_info.setStyleSheet(f"color:{T.MUTED}")
        self.emg_title = QLabel("EMG 8 ch")
        for w in (self.emg_title, self.view, self.autoscale, self.secs, self.ch_info):
            tb.addWidget(w)
        tb.addStretch()
        lv.addLayout(tb)
        pg.setConfigOptions(antialias=False, background=T.BG, foreground=T.MUTED)
        self.glw = pg.GraphicsLayoutWidget()
        self.emg_plots, self.emg_curves, self.ch_labels = [], [], []
        self._lab_t = 0.0
        for c in range(C.N_EMG):
            p = self.glw.addPlot(row=c, col=0)
            p.setMenuEnabled(False)
            p.showGrid(x=True, y=False, alpha=0.15)
            p.setLabel("left", f"ch{c + 1}", color=T.CH_COLORS[c])
            p.getAxis("left").setWidth(52)
            p.setDownsampling(auto=True, mode="peak")
            p.setClipToView(True)
            if c < C.N_EMG - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            if self.emg_plots:
                p.setXLink(self.emg_plots[0])
            self.emg_curves.append(p.plot(pen=pg.mkPen(T.CH_COLORS[c], width=1)))
            self.emg_plots.append(p)
            lab = self.glw.addLabel("", row=c, col=1, justify="right")
            lab.setMinimumWidth(92)
            lab.setMaximumWidth(92)
            self.ch_labels.append(lab)
        lv.addWidget(self.glw, 5)
        self.imu_glw = pg.GraphicsLayoutWidget()
        self.acc_plot = self.imu_glw.addPlot(row=0, col=0, title="Accel (g)")
        self.gyro_plot = self.imu_glw.addPlot(row=0, col=1, title="Gyro (deg/s)")
        cols = ["#ef5350", "#66bb6a", "#42a5f5"]
        self.acc_curves = [self.acc_plot.plot(pen=pg.mkPen(cols[i], width=1.5), name=a) for i, a in enumerate("xyz")]
        self.gyro_curves = [self.gyro_plot.plot(pen=pg.mkPen(cols[i], width=1.5), name=a) for i, a in enumerate("xyz")]
        for p in (self.acc_plot, self.gyro_plot):
            p.setMenuEnabled(False)
            p.showGrid(x=True, y=True, alpha=0.15)
        lv.addWidget(self.imu_glw, 2)
        self.imu_vals = QLabel("IMU: -")
        self.imu_vals.setStyleSheet("font-family: Consolas; color: #b0bec5;")
        lv.addWidget(self.imu_vals)
        split.addWidget(left)

        # --- right: prediction
        right = QWidget()
        rv = QVBoxLayout(right)
        self.replay_banner = T.big_label("", 26, "#000")
        self.replay_banner.setAlignment(Qt.AlignCenter)
        self.replay_banner.hide()
        rv.addWidget(self.replay_banner)
        rv.addWidget(T.big_label("POSE", 20, T.MUTED))
        self.pose_lab = T.big_label("—", 60)
        self.pose_lab.setWordWrap(True)
        rv.addWidget(self.pose_lab)
        self.pose_ko = T.big_label("", 30, T.TEXT)
        rv.addWidget(self.pose_ko)
        self.pose_conf = T.bar()
        rv.addWidget(self.pose_conf)
        rv.addSpacing(10)
        rv.addWidget(T.big_label("EFFORT", 20, T.MUTED))
        self.eff_lab = T.big_label("—", 56)
        rv.addWidget(self.eff_lab)
        self.eff_conf = T.bar(T.HIGH)
        rv.addWidget(self.eff_conf)
        self.reason_lab = QLabel("")
        self.reason_lab.setWordWrap(True)
        self.reason_lab.setStyleSheet(f"color:{T.WARN}")
        rv.addWidget(self.reason_lab)
        self.truth_lab = T.big_label("", 20, T.MUTED)
        rv.addWidget(self.truth_lab)
        g = QGroupBox("Relative EMG activation (calibration-normalized; not force)")
        gl = QVBoxLayout(g)
        self.act_bar = T.bar(T.ACCENT, 150)
        self.act_bar.setFormat("%v %")
        gl.addWidget(self.act_bar)
        self.act_info = QLabel("")
        self.act_info.setStyleSheet(f"color:{T.MUTED}")
        gl.addWidget(self.act_info)
        rv.addWidget(g)
        g2 = QGroupBox("Class probabilities (smoothed)")
        self.prob_grid = QGridLayout(g2)
        self.prob_bars = {}
        rv.addWidget(g2)
        g3 = QGroupBox("Model — 무엇을, 얼마나 자주 읽는지")
        f3 = QFormLayout(g3)
        self.model_box = QComboBox()
        self.model_box.setMinimumWidth(300)
        self.model_box.currentIndexChanged.connect(self._model_changed)
        f3.addRow("Current model", self.model_box)
        self.smooth_box = QComboBox()
        self.smooth_box.addItems(["Raw (no smoothing)", "250 ms (5 windows)", "500 ms (10)", "1000 ms (20)"])
        self.smooth_box.setCurrentIndex(1)
        self.smooth_box.currentIndexChanged.connect(self._smooth_changed)
        f3.addRow("Smoothing", self.smooth_box)
        self.model_info = QLabel("")
        self.model_info.setWordWrap(True)
        self.model_info.setStyleSheet("color:#b0bec5; font-family: Consolas, 'Malgun Gothic'; font-size: 12px;")
        f3.addRow(self.model_info)
        self.lat_lab = QLabel("-")
        f3.addRow("Inference / latency", self.lat_lab)
        rv.addWidget(g3)
        rv.addStretch()
        split.addWidget(right)
        split.setSizes([950, 650])

        # --- bottom controls
        bot = QHBoxLayout()
        self.btn = {}
        for key, text in (("connect", "Connect"), ("calibrate", "Calibrate"), ("collect", "Collect Data"), ("train", "Train"),
                          ("live", "Live Inference"), ("replay", "Replay"), ("save", "Save (Record)")):
            b = QPushButton(text)
            b.setMinimumHeight(44)
            self.btn[key] = b
            bot.addWidget(b)
        self.btn["live"].setCheckable(True)
        self.btn["save"].setCheckable(True)
        self.btn["connect"].clicked.connect(self.on_connect)
        self.btn["calibrate"].clicked.connect(self.on_calibrate)
        self.btn["collect"].clicked.connect(self.on_collect)
        self.btn["train"].clicked.connect(self.on_train)
        self.btn["live"].toggled.connect(self.on_live)
        self.btn["replay"].clicked.connect(self.on_replay)
        self.btn["save"].toggled.connect(self.on_save)
        v.addLayout(bot)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(80)
        self.log_box.setStyleSheet("font-family: Consolas; font-size: 11px; color: #9aa4b1;")
        v.addWidget(self.log_box)
        self._participant_changed()

    def log(self, msg):
        self.log_box.appendPlainText(time.strftime("%H:%M:%S ") + msg)

    # ------------------------------------------------------------------ models
    def refresh_models(self, select_path: str | None = None):
        self.models = list_models()
        self.model_box.blockSignals(True)
        self.model_box.clear()
        self.model_box.addItem("(none)")
        idx = 0
        for i, m in enumerate(self.models):
            self.model_box.addItem(m["label"])
            if (select_path and m["path"] == select_path) or (not select_path and idx == 0 and m["recommended"]):
                idx = i + 1
        if idx == 0 and self.models and not select_path:
            idx = 1
        self.model_box.blockSignals(False)
        self.model_box.setCurrentIndex(idx)
        self._model_changed(idx)

    def _model_changed(self, i):
        if i <= 0:
            self.engine.set_predictor(None)
            self.model_info.setText("no model")
            self._build_prob_bars([])
            return
        m = self.models[i - 1]
        try:
            p = load_predictor(Path(m["path"]))
        except Exception as e:
            self.engine.set_predictor(None)
            self.model_info.setText(f"load failed: {e}")
            self.log(traceback.format_exc())
            return
        self.engine.set_predictor(p)
        meta = p.meta or {}
        txt = input_spec_text(p) + f"\n  클래스: {', '.join(p.pose_classes)} · Effort 대상: {', '.join(p.active_poses)}"
        if "test_pose_macro_f1" in meta:
            txt += (f"\n  test: pose macro-F1 {meta['test_pose_macro_f1']:.3f} · effort BA "
                    f"{meta.get('test_effort_ba_true_pose', float('nan')):.3f}")
        self.model_info.setText(txt)
        self._build_prob_bars(p.pose_classes)
        self.pose_colors = {c: T.POSE_COLORS[k % len(T.POSE_COLORS)] for k, c in enumerate(p.pose_classes)}
        self.log(f"model: {p.name} ({m['path']})")
        self._check_replay_classes()

    def _build_prob_bars(self, classes):
        while self.prob_grid.count():
            w = self.prob_grid.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.prob_bars = {}
        for r, c in enumerate(classes):
            self.prob_grid.addWidget(QLabel(c), r, 0)
            b = T.bar(T.POSE_COLORS[r % len(T.POSE_COLORS)])
            b.setMaximumHeight(16)
            self.prob_grid.addWidget(b, r, 1)
            self.prob_bars[c] = b

    def _smooth_changed(self, i):
        self.engine.smooth_k = [1, 5, 10, 20][i]

    # ------------------------------------------------------------------ actions
    def _participant_changed(self):
        pid = self.participant.text().strip() or "S00"
        p = latest_calibration(pid)
        if p is not None:
            try:
                self.set_calibration(Calibration.load(p))
                self.log(f"loaded calibration {p}")
                return
            except Exception as e:
                self.log(f"calibration load failed: {e}")
        if self.calibration is not None and self.calibration.participant_id != pid:
            self.set_calibration(None)

    def set_calibration(self, cal):
        self.calibration = cal
        self.engine.calibration = cal

    def is_replay(self):
        return self.engine.source is not None and self.engine.source.kind == "REPLAY"

    def is_connected(self):
        return self.engine.source is not None and self.engine.source.state in ("streaming", "stalled", "connecting")

    def on_connect(self):
        if self.engine.source is not None and not self.is_replay() and self.is_connected():
            if self.btn["save"].isChecked():
                self.btn["save"].setChecked(False)
            self.engine.stop()
            self.log("disconnected")
            return
        d = ConnectDialog(self)
        if d.exec() != QDialog.Accepted:
            return
        if self.btn["save"].isChecked():
            self.btn["save"].setChecked(False)
        self.engine.stop()
        src = d.source()
        self.engine.start(src)
        self.log(f"connecting {src.kind} ...")

    def on_replay(self):
        if self.is_replay():
            if self.btn["save"].isChecked():
                self.btn["save"].setChecked(False)
            self.engine.stop()
            self.log("replay stopped")
            return
        d = ReplayDialog(self)
        if d.exec() != QDialog.Accepted or d.selected() is None:
            return
        if self.btn["save"].isChecked():
            self.btn["save"].setChecked(False)
        self.engine.stop()
        src = ReplaySource(d.selected(), d.speed.value(), loop=d.loop.isChecked(), start_s=d.start.value())
        self.engine.start(src)
        self.log(f"REPLAY {d.selected()} (labels in data: {src.labels_in_data})")
        self._check_replay_classes()

    def _check_replay_classes(self):
        src, p = self.engine.source, self.engine.predictor
        if src is None or src.kind != "REPLAY" or p is None:
            return
        extra = [c for c in p.pose_classes if c not in src.poses_in_data]
        missing = [c for c in src.poses_in_data if c not in p.pose_classes]
        if extra or missing:
            self.log(f"WARNING replay/model mismatch - model classes not in this recording: {extra}; "
                     f"recorded poses unknown to the model: {missing}. Such predictions are shown as model outputs only.")

    def on_live(self, on):
        if on and self.engine.predictor is None:
            QMessageBox.information(self, "Live Inference", "Select a model first (Model ▸ Current model). Train one with 'Train'.")
            self.btn["live"].setChecked(False)
            return
        self.engine.inference_on = on
        self.engine.prediction = None
        self.log("live inference " + ("ON" if on else "OFF"))

    def on_save(self, on):
        if on:
            if self.engine.source is None:
                QMessageBox.information(self, "Save", "Connect a device (or start a replay) first.")
                self.btn["save"].setChecked(False)
                return
            if self.engine.writer is not None:
                QMessageBox.information(self, "Save", "A recording is already running (data collection / calibration).")
                self.btn["save"].setChecked(False)
                return
            src = self.engine.source
            pid = self.participant.text().strip() or "S00"
            sid = new_session_id("Free")
            meta = {"kind": "free", "data_source": src.kind, "fs": src.fs, "board_layout": src.layout, "filter": dict(C.FILTER),
                    "model": getattr(self.engine.predictor, "name", None),
                    "model_path": str(getattr(self.engine.predictor, "path", "")),
                    "calibration": self.calibration.__dict__ if self.calibration else None,
                    "note": "free recording (raw + live predictions); unlabeled - not used for training"}
            if src.kind == "REPLAY":
                meta["replay_of"] = str(src.dir)
            w = SessionWriter(C.SESSIONS_DIR, pid, sid, meta)
            self.engine.start_recording(w)
            self.btn["save"].setText("Stop && Save")
            self.log(f"recording raw + predictions -> {w.dir}")
        else:
            w = self.engine.stop_recording()
            self.btn["save"].setText("Save (Record)")
            if w is not None:
                self.log(f"saved {w.dir} ({w.emg_rows} EMG rows, {w.imu_rows} IMU rows)")
                if self.calibration is not None:
                    self.calibration.save(w.dir / "calibration.json")

    def on_calibrate(self):
        if not self.is_connected() or self.is_replay():
            QMessageBox.information(self, "Calibrate", "Connect the MindRove armband first (calibration is not possible in Replay).")
            return
        from .calibration_wizard import CalibrationWizard
        w = CalibrationWizard(self)
        w.exec()

    def on_collect(self):
        if not self.is_connected() or self.is_replay():
            QMessageBox.information(self, "Collect Data", "Connect the MindRove armband first (data collection is disabled in Replay).")
            return
        from .collection import CollectionWindow
        if self.collect_win is None or not self.collect_win.isVisible():
            self.collect_win = CollectionWindow(self)
        self.collect_win.show()
        self.collect_win.raise_()

    def on_train(self):
        from .training import TrainingWindow
        if self.train_win is None or not self.train_win.isVisible():
            self.train_win = TrainingWindow(self)
        self.train_win.show()
        self.train_win.raise_()

    # ------------------------------------------------------------------ refresh
    def tick(self):
        e = self.engine
        st = e.status
        src = e.source
        # mode banner
        if src is None:
            mode, col = "DISCONNECTED", T.MUTED
        elif src.kind == "REPLAY":
            mode, col = "REPLAY MODE", T.REPLAY
        elif src.kind == "SYNTHETIC":
            mode, col = "SYNTHETIC (SDK test)", T.WARN
        else:
            mode, col = {"streaming": ("LIVE", T.OK), "stalled": ("LIVE - STALLED", T.ERR), "connecting": ("CONNECTING…", T.WARN),
                         "error": ("ERROR", T.ERR)}.get(src.state, (src.state.upper(), T.MUTED))
        self.mode_lab.setText(mode)
        T.set_color(self.mode_lab, col)
        if src is not None and src.kind == "REPLAY":
            fr, ts = src.progress()
            self.replay_banner.setText(f"REPLAY MODE — {src.dir.name}  ({fr * 100:.0f} %, {src.state})")
            self.replay_banner.setStyleSheet(f"background:{T.REPLAY}; color:#000; border-radius:6px; padding:6px; font-size:26px; font-weight:700;")
            self.replay_banner.show()
        else:
            self.replay_banner.hide()
        self.btn["connect"].setText("Disconnect" if (src is not None and src.kind != "REPLAY" and self.is_connected()) else "Connect")
        self.btn["replay"].setText("Stop Replay" if self.is_replay() else "Replay")
        dev = "MindRove Disconnected" if src is None else f"{src.kind}: {src.state}"
        if src is not None and src.error:
            dev += f" ({src.error[:60]})"
        self.stat_labs["device"].setText(dev)
        T.set_color(self.stat_labs["device"], T.ERR if (src and src.state in ("error", "stalled")) else T.TEXT)
        self.stat_labs["emg"].setText(f"EMG {st.emg_rate:5.0f} Hz (nominal {e.fs:.0f}) lost {st.lost_samples}")
        self.stat_labs["imu"].setText(f"IMU {st.imu_rate:4.1f} Hz")
        self.stat_labs["battery"].setText(f"Battery {st.battery:.0f} %" if np.isfinite(st.battery) else "")
        self.stat_labs["calib"].setText("Calibration: " + (self.calibration.status_text() if self.calibration else "NOT calibrated"))
        T.set_color(self.stat_labs["calib"], T.OK if self.calibration and self.calibration.has_norm else T.WARN)
        rec = e.writer
        self.stat_labs["rec"].setText(f"● REC {rec.dir.name} ({rec.emg_rows} rows)" if rec else "")
        T.set_color(self.stat_labs["rec"], T.ERR)
        self.session_lab.setText(f"Session: {rec.session_id if rec else '-'}")
        if src is not None:
            msgs = src.messages
            if getattr(self, "_nmsg", 0) != len(msgs):
                for m in msgs[getattr(self, "_nmsg", 0):]:
                    self.log(m)
                self._nmsg = len(msgs)
        else:
            self._nmsg = 0
        self._plot()
        self._show_prediction()

    def _plot(self):
        e = self.engine
        if e.source is None and e.raw.n == 0:
            return
        secs = [2, 3, 5][self.secs.currentIndex()]
        mode = self.view.currentIndex()          # 0 filtered uV, 1 raw uV, 2 normalized
        t, y, it, iv = e.snapshot(secs, mode != 1)
        if len(t) < 2:
            return
        cal = self.calibration
        norm = mode == 2 and cal is not None and cal.has_norm
        if norm:
            y = y / cal.max_rms[None, :]
            m = max(float(np.percentile(np.abs(y), 99.5)) * 1.2, 0.2) if self.autoscale.isChecked() else 2.0
        for c in range(C.N_EMG):
            self.emg_curves[c].setData(t, y[:, c])
            if norm:
                self.emg_plots[c].setYRange(-m, m, padding=0)          # same scale for all channels
            elif self.autoscale.isChecked():
                yy = y[:, c]
                if mode == 0:
                    mm = max(np.percentile(np.abs(yy), 99.5) * 1.2, 20)
                    self.emg_plots[c].setYRange(-mm, mm, padding=0)
                else:
                    lo, hi = np.percentile(yy, [0.5, 99.5])
                    pad = max((hi - lo) * 0.2, 20)
                    self.emg_plots[c].setYRange(lo - pad, hi + pad, padding=0)
        self.emg_plots[0].setXRange(-secs, 0, padding=0)
        self.emg_title.setText(f"EMG 8 ch @ {e.fs:.0f} Hz")
        now = time.time()
        if now - self._lab_t > 0.2 and e.activity is not None:      # per-channel readout at 5 Hz
            self._lab_t = now
            rms = e.activity.rms
            pct = cal.channel_activation(rms) * 100 if (cal is not None and cal.has_norm) else None
            for c in range(C.N_EMG):
                txt = f"<span style='color:{T.CH_COLORS[c]}'>ch{c + 1}</span><br>{rms[c]:.0f} µV rms"
                if pct is not None:
                    txt += f"<br><b>{pct[c]:.0f} %</b> max"
                self.ch_labels[c].setText(txt, size="9pt")
        st = e.status
        msg = []
        if mode == 2 and not norm:
            msg.append("Normalized 보기는 캘리브레이션(Rest + Max)이 필요 — 지금은 Filtered µV 표시")
        elif norm:
            msg.append("단위: 채널별 최대 RMS의 배수 (1.0 = 캘리브레이션 최대), 8채널 같은 스케일")
        if st.flat_channels:
            msg.append(f"FLAT ch {st.flat_channels}")
        if st.noisy_channels:
            msg.append(f"very high amplitude ch {st.noisy_channels}")
        self.ch_info.setText("  ".join(msg))
        T.set_color(self.ch_info, T.ERR if (st.flat_channels or st.noisy_channels or (mode == 2 and not norm)) else T.MUTED)
        if len(it) >= 2:
            for k in range(3):
                self.acc_curves[k].setData(it, iv[:, k])
                self.gyro_curves[k].setData(it, iv[:, 3 + k])
            self.acc_plot.setXRange(-secs, 0, padding=0)
            self.gyro_plot.setXRange(-secs, 0, padding=0)
            last = iv[-1]
            sat = np.abs(iv[-int(min(len(iv), 50)):, 3:]).max() >= C.GYRO_SATURATION_DPS
            self.imu_vals.setText(f"IMU @ {st.imu_rate:.0f} Hz  " + "acc[g] x {:+.2f} y {:+.2f} z {:+.2f}   gyro[°/s] x {:+7.1f} y {:+7.1f} z {:+7.1f}".format(*last)
                                  + ("   GYRO SATURATED (±500 °/s)" if sat else ""))

    def _show_prediction(self):
        e = self.engine
        act = e.activation
        if np.isfinite(act):
            self.act_bar.setValue(int(min(150, act * 100)))
            self.act_info.setText(f"8채널 평균 {act * 100:.0f} % (0 % = 휴식, 100 % = 캘리브레이션 손목 최대)")
        else:
            self.act_bar.setValue(0)
            self.act_info.setText("calibrate to see relative activation" if e.source else "")
        p = e.predictor
        if not e.inference_on or e.source is None:
            self.pose_ko.setText("")
            self.pose_lab.setText("—" if p is not None else "NO MODEL")
            T.set_color(self.pose_lab, T.MUTED)
            self.eff_lab.setText("—")
            T.set_color(self.eff_lab, T.MUTED)
            self.reason_lab.setText("Live Inference off" if e.source else "")
            self.truth_lab.setText("")
            self.pose_conf.setValue(0)
            self.eff_conf.setValue(0)
            return
        pr = e.prediction
        if pr is None:
            self.pose_lab.setText("UNKNOWN")
            T.set_color(self.pose_lab, T.MUTED)
            self.reason_lab.setText("waiting for data")
            return
        stale = time.time() - pr.t > 0.5
        if stale:
            self.pose_lab.setText("UNKNOWN")
            T.set_color(self.pose_lab, T.MUTED)
            self.eff_lab.setText("—")
            T.set_color(self.eff_lab, T.MUTED)
            self.reason_lab.setText("no new prediction for > 0.5 s (stream stopped / stalled)")
            return
        name = pr.pose.upper()
        if self.is_replay() and pr.pose not in ("UNKNOWN",) and pr.pose not in getattr(e.source, "poses_in_data", []):
            name += "  (not in this recording)"
        self.pose_lab.setText(name)
        self.pose_ko.setText(ko_name(pr.pose) if pr.pose != "UNKNOWN" else "")
        T.set_color(self.pose_lab, self.pose_colors.get(pr.pose, T.MUTED) if pr.pose != "UNKNOWN" else T.MUTED)
        self.pose_conf.setValue(int(pr.pose_conf * 100))
        self.pose_conf.setFormat(f"pose confidence {pr.pose_conf * 100:.0f} %")
        if pr.pose == C.REST:
            self.eff_lab.setText("— (Rest)")
            T.set_color(self.eff_lab, T.MUTED)
            self.eff_conf.setValue(0)
            self.eff_conf.setFormat("")
        elif pr.effort is None:
            self.eff_lab.setText("—" if pr.pose == "UNKNOWN" else "n/a")
            T.set_color(self.eff_lab, T.MUTED)
            self.eff_conf.setValue(0)
            self.eff_conf.setFormat("")
        else:
            self.eff_lab.setText(pr.effort.upper())
            T.set_color(self.eff_lab, T.HIGH if pr.effort == "High" else T.LOW)
            self.eff_conf.setValue(int(pr.effort_conf * 100))
            self.eff_conf.setFormat(f"effort confidence {pr.effort_conf * 100:.0f} %")
        self.reason_lab.setText(pr.reason)
        for c, b in self.prob_bars.items():
            if c in pr.classes:
                b.setValue(int(pr.proba[pr.classes.index(c)] * 100))
        self.lat_lab.setText(f"{p.cfg.stride_ms:.0f} ms마다 예측 (목표 {1000 / p.cfg.stride_ms:.0f} Hz, 실측 {e.status.infer_rate:.1f} Hz) · "
                             f"모델 {pr.inference_ms:.1f} ms · 처리 지연 {pr.latency_ms:.1f} ms · skipped {e.status.dropped_windows}")
        if self.is_replay():
            if pr.truth is None:
                self.truth_lab.setText("TRUTH: (unlabeled / transition)")
                T.set_color(self.truth_lab, T.MUTED)
            else:
                tp, te, ph = pr.truth
                ok = pr.pose == tp and (tp == C.REST or pr.effort == te)
                self.truth_lab.setText(f"TRUTH: {tp}{(' / ' + te) if te else ''}  {'✔' if ok else '✘'}")
                T.set_color(self.truth_lab, T.OK if ok else T.ERR)
        else:
            self.truth_lab.setText("")

    def closeEvent(self, ev):
        for w in (self.collect_win, self.train_win):
            if w is not None:
                w.close()
        self.engine.stop()
        super().closeEvent(ev)
