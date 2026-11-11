"""Calibration (~20 s): REST 5 s -> wrist flexion MAX 3 s -> (wrist extension MAX 3 s) -> result table -> save.

The device check (EMG / IMU rate, flat / noisy channels) is computed from the REST recording.
Raw data of the calibration is stored as its own session (kind = "calibration"), never used as training data.
Calibration does not train a pose classifier.
"""
from __future__ import annotations

import threading
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from .. import config as C
from ..calibration import MAX_TASKS, Calibration, device_check, finalize, max_stats, rest_stats
from ..session import SessionWriter, new_session_id
from . import theme as T

REST_S, PREP_S, MAX_S, RELAX_S = 5.0, 3.0, 3.0, 3.0


class CalibrationWizard(QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main, self.engine = main, main.engine
        self.setWindowTitle("Calibration — Rest + Max")
        self.resize(900, 660)
        self._frames, self._lock, self._capture = [], threading.Lock(), False
        self.engine.add_listener(self._on_frame)
        self.writer, self.trials, self._tid = None, [], 0
        self._seq, self._i = [], -1

        v = QVBoxLayout(self)
        intro = QLabel("<b>2단계, 약 20초</b> — ① <b>가만히</b> 5초 (힘 완전히 빼기)  ② <b>손목 꺾기 최대</b> 3초 "
                       "(굽히기, 선택: 펴기)<br>"
                       "손목은 실제로 움직이지 않고 다른 손으로 막은 채 <i>버티기만</i> 합니다(등척성). "
                       "결과는 정규화 파형 보기와 '상대 활성도'(0 = 휴식, 100 % = 최대)에 쓰입니다. 모델 학습과는 별개입니다.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        h = QHBoxLayout()
        self.ext = QCheckBox("손목 펴기도 측정 (권장: 손등 쪽 채널의 최대값)")
        self.ext.setChecked(True)
        self.rec = QCheckBox("원신호를 세션으로 저장 (kind = calibration)")
        self.rec.setChecked(True)
        h.addWidget(self.ext)
        h.addWidget(self.rec)
        h.addStretch()
        v.addLayout(h)
        self.cue = T.big_label("Start 를 누르세요", 44)
        self.cue.setAlignment(Qt.AlignCenter)
        self.cue.setWordWrap(True)
        v.addWidget(self.cue)
        self.instr = QLabel("")
        self.instr.setWordWrap(True)
        self.instr.setAlignment(Qt.AlignCenter)
        self.instr.setStyleSheet("font-size: 18px; color: #cfd8dc;")
        v.addWidget(self.instr)
        self.prog = T.bar()
        self.prog.setMinimumHeight(26)
        v.addWidget(self.prog)
        self.table = QTableWidget(C.N_EMG, 5)
        self.table.setHorizontalHeaderLabels(["channel", "status", "rest RMS (µV)", "max RMS (µV)", "max / rest"])
        v.addWidget(self.table, 1)
        self.result = QLabel("")
        self.result.setWordWrap(True)
        v.addWidget(self.result)
        hb = QHBoxLayout()
        self.load_btn = QPushButton("Load existing…")
        self.load_btn.clicked.connect(self.load_existing)
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start)
        self.save_btn = QPushButton("Save calibration")
        self.save_btn.clicked.connect(self.save)
        self.save_btn.setEnabled(False)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        for b in (self.load_btn, self.start_btn):
            hb.addWidget(b)
        hb.addStretch()
        hb.addWidget(close)
        hb.addWidget(self.save_btn)
        v.addLayout(hb)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)

    # ------------------------------------------------------------------ capture plumbing
    def _on_frame(self, fr):              # engine thread
        if self._capture:
            with self._lock:
                self._frames.append(fr)

    def _new_cal(self):
        src = self.engine.source
        self.cal = Calibration(participant_id=self.main.participant.text().strip() or "S00",
                               calibration_id=time.strftime("Cal_%Y%m%d_%H%M%S"), created=time.strftime("%Y-%m-%d %H:%M:%S"),
                               device={"kind": src.kind, "layout": src.layout, "fs": src.fs, "sdk": getattr(src, "sdk_version", "")},
                               signal={"filter": dict(C.FILTER), "rms_window_ms": C.WINDOW_MS, "stride_ms": C.STRIDE_MS})

    def _ensure_writer(self):
        if self.writer is None and self.rec.isChecked() and self.engine.writer is None:
            src = self.engine.source
            self.writer = SessionWriter(C.SESSIONS_DIR, self.cal.participant_id, new_session_id("Calib"),
                                        {"kind": "calibration", "data_source": src.kind, "fs": src.fs, "board_layout": src.layout,
                                         "filter": dict(C.FILTER), "calibration_id": self.cal.calibration_id,
                                         "protocol": {"name": "calibration_rest_max", "inter_trial_rest": False, "onset_margin_s": 0.5}})
            self.engine.start_recording(self.writer)
            self.cal.session_id = self.writer.session_id

    def _mark(self, label, start):
        if self.writer is None:
            return
        if start:
            self._tid += 1
            self.engine.trial_id = self._tid
            self.trials.append({"trial_id": self._tid, "pose": label, "effort": "", "repetition": 1, "condition": label,
                                "status": "running", "prep_start_idx": self.engine.last_sidx, "hold_start_idx": self.engine.last_sidx,
                                "hold_start_host": time.time(), "quality": "", "note": "calibration"})
        else:
            self.trials[-1].update({"status": "completed", "hold_end_idx": self.engine.last_sidx,
                                    "rest_end_idx": self.engine.last_sidx, "hold_end_host": time.time()})
            self.engine.trial_id = -1
            self.writer.set_trials(self.trials)

    # ------------------------------------------------------------------ sequence
    def start(self):
        if self._seq:
            return
        self._new_cal()
        self._ensure_writer()
        if self.writer is not None:
            self.cal.session_id = self.writer.session_id
        seq = [("rest", "rest", REST_S, "가만히 — 힘을 완전히 빼세요", "팔 자세는 그대로, 손과 팔의 힘을 모두 뺍니다.")]
        tasks = ["flexion"] + (["extension"] if self.ext.isChecked() else [])
        for t in tasks:
            name, how = MAX_TASKS[t]
            seq += [("prep", t, PREP_S, f"준비: {name}", how),
                    ("max", t, MAX_S, f"{name} — 힘껏!", how),
                    ("relax", t, RELAX_S if t != tasks[-1] else 1.5, "힘 빼기", "")]
        self._seq, self._i = seq, -1
        self.start_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.result.setText("")
        self._advance()

    def _advance(self):
        if 0 <= self._i < len(self._seq):
            self._phase_done(self._seq[self._i])
        self._i += 1
        if self._i >= len(self._seq):
            self._seq = []
            self.start_btn.setEnabled(True)
            self.start_btn.setText("Redo")
            self.prog.setValue(0)
            self._show_result()
            return
        kind, task, dur, text, how = self._seq[self._i]
        self.cue.setText(text)
        T.set_color(self.cue, T.HIGH if kind == "max" else T.OK if kind == "rest" else T.TEXT)
        self.instr.setText(how)
        self._dur, self._end_t = dur, time.time() + dur
        if kind in ("rest", "max"):
            self._mark("Rest" if kind == "rest" else f"MAX {task}", True)
            with self._lock:
                self._frames = []
            self._capture = True
            if kind == "max":
                QApplication.beep()

    def _phase_done(self, ph):
        kind, task, dur, _, _ = ph
        if kind not in ("rest", "max"):
            return
        self._capture = False
        with self._lock:
            frames = list(self._frames)
        self._mark("", False)
        frames = [f for f in frames if f.host_ts >= frames[0].host_ts + 0.5] if frames else []   # drop the onset
        if len(frames) < 10:
            self.result.setText("⚠ not enough data — is the stream running?")
            return
        if kind == "rest":
            st = self.engine.status
            self.cal.rest = rest_stats(frames) | {"duration_s": dur}
            self.cal.device_check = device_check(frames, self.engine.recent_raw_std(2.0), st.emg_rate, st.imu_rate, self.engine.fs)
        else:
            self.cal.max[task] = max_stats(frames) | {"duration_s": dur}

    def _show_result(self):
        cal = finalize(self.cal)
        dc = cal.device_check
        if not cal.has_rest or not dc:
            return
        mx = np.asarray(cal.norm.get("ref_rms", [np.nan] * C.N_EMG))
        for c, ch in enumerate(dc["channels"]):
            r = ch["rest_rms_uv"]
            vals = [f"ch{c + 1}", ch["status"], f"{r:.1f}", f"{mx[c]:.1f}", f"{mx[c] / max(r, 1e-6):.1f} ×"]
            for k, val in enumerate(vals):
                it = QTableWidgetItem(val)
                if k == 1 and ch["status"] != "OK":
                    it.setForeground(Qt.red)
                self.table.setItem(c, k, it)
        warn = []
        if not dc["emg_rate_ok"]:
            warn.append(f"EMG rate {dc['emg_rate_hz']:.0f} Hz (expected {self.engine.fs:.0f})")
        if not dc["imu_rate_ok"]:
            warn.append(f"IMU rate {dc['imu_rate_hz']:.0f} Hz (expected ~50)")
        bad = [f"ch{c['channel']} {c['status']}" for c in dc["channels"] if c["status"] != "OK"]
        warn += bad
        if cal.norm.get("weak_channels"):
            warn.append(f"max < 2 × rest on ch {cal.norm['weak_channels']} (not activated by the wrist tasks or poor contact)")
        if cal.rest.get("gyro_mag_p95", 0) > 30:
            warn.append("arm moved during REST — consider Redo")
        self.result.setText(f"EMG {dc['emg_rate_hz']:.0f} Hz · IMU {dc['imu_rate_hz']:.1f} Hz · " +
                            ("✔ all OK" if not warn else "⚠ " + "; ".join(warn)))
        T.set_color(self.result, T.OK if not warn else T.WARN)
        self.cue.setText("완료 — 결과 확인 후 Save")
        T.set_color(self.cue, T.TEXT)
        self.instr.setText("")
        self.save_btn.setEnabled(cal.has_norm)

    def _tick(self):
        if self._seq:
            rem = self._end_t - time.time()
            self.prog.setValue(int(100 * (1 - max(rem, 0) / max(self._dur, 1e-6))))
            self.prog.setFormat(f"{max(rem, 0):.1f} s")
            if rem <= 0:
                self._advance()

    def save(self):
        path = self.cal.save()
        if self.writer is not None:
            self.cal.save(self.writer.dir / "calibration.json")
        self.main.set_calibration(self.cal)
        self._finish_writer()
        self.main.log(f"calibration saved: {path}")
        self.close()

    def load_existing(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load calibration", str(C.CALIB_DIR), "calibration (*.json)")
        if fn:
            cal = Calibration.load(fn)
            self.main.set_calibration(cal)
            QMessageBox.information(self, "Calibration", f"Loaded {cal.calibration_id} ({cal.participant_id})")
            self.close()

    def _finish_writer(self):
        if self.writer is not None and self.engine.writer is self.writer:
            self.engine.stop_recording("completed", {"calibration": self.cal.__dict__})
        self.writer = None

    def closeEvent(self, ev):
        self.timer.stop()
        self.engine.remove_listener(self._on_frame)
        if self.writer is not None:
            self._finish_writer()
        super().closeEvent(ev)
