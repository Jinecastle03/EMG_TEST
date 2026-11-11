"""Pose x Effort data collection.

Trial = PREPARE (cue + instruction) -> HOLD (beep, keep pose at the requested effort) -> REST (release, relax).
The sample index at every phase change is stored in trials.csv; the raw stream is recorded continuously.
Training later uses  HOLD (minus an onset margin)  as the pose/effort label and  REST (after 1 s settle) as Rest;
PREPARE and the release transition are never labeled.
"""
from __future__ import annotations

import random
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                               QTextEdit, QVBoxLayout, QWidget)

from .. import config as C
from ..poses import Pose, add_pose, load_poses
from ..session import SessionWriter, new_session_id
from . import theme as T


class AddPoseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add new pose")
        f = QFormLayout(self)
        self.name = QLineEdit()
        self.instr = QTextEdit()
        self.instr.setMaximumHeight(90)
        self.obj = QLineEdit()
        self.eff = QCheckBox("has Low / High effort")
        self.eff.setChecked(True)
        f.addRow("Name (e.g. Tripod Grip)", self.name)
        f.addRow("Instruction", self.instr)
        f.addRow("Object", self.obj)
        f.addRow(self.eff)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def pose(self):
        return Pose(self.name.text().strip(), self.instr.toPlainText().strip(), self.obj.text().strip(), self.eff.isChecked())


class CollectionWindow(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main, self.engine = main, main.engine
        self.setWindowTitle("Collect Data — Pose × Effort")
        self.resize(1300, 860)
        self.writer = None
        self.queue, self.trials = [], []
        self.cur = None
        self.phase, self.phase_end, self.paused = "IDLE", 0.0, False
        self._frames = []
        self.engine.add_listener(self._on_frame)
        self._build()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)

    # ------------------------------------------------------------------ UI
    def _build(self):
        v = QVBoxLayout(self)
        sp = QSplitter(Qt.Horizontal)
        v.addWidget(sp, 1)
        left = QWidget()
        lv = QVBoxLayout(left)
        g = QGroupBox("Protocol")
        f = QFormLayout(g)
        self.pid = QLineEdit(self.main.participant.text().strip() or "S00")
        f.addRow("Participant ID", self.pid)
        self.pose_list = QListWidget()
        self._fill_poses()
        f.addRow("Poses", self.pose_list)
        b = QPushButton("Add new pose…")
        b.clicked.connect(self._add_pose)
        f.addRow(b)
        self.reps = QSpinBox()
        self.reps.setRange(1, 50)
        self.reps.setValue(10)
        self.rest_trials = QSpinBox()
        self.rest_trials.setRange(0, 50)
        self.rest_trials.setValue(10)
        self.prep = QDoubleSpinBox()
        self.prep.setRange(1, 10)
        self.prep.setValue(2.0)
        self.hold = QDoubleSpinBox()
        self.hold.setRange(1, 10)
        self.hold.setValue(3.0)
        self.rest = QDoubleSpinBox()
        self.rest.setRange(2, 20)
        self.rest.setValue(4.0)
        self.onset = QDoubleSpinBox()
        self.onset.setRange(0, 2)
        self.onset.setValue(0.5)
        self.shuffle = QCheckBox("shuffle order (block-randomized: every condition once per block)")
        self.shuffle.setChecked(True)
        for lab, w in (("trials per pose × effort", self.reps), ("Rest trials", self.rest_trials), ("prepare (s)", self.prep),
                       ("hold (s)", self.hold), ("rest between trials (s)", self.rest), ("onset margin excluded from labels (s)", self.onset)):
            f.addRow(lab, w)
        f.addRow(self.shuffle)
        self.total_lab = QLabel("")
        f.addRow(self.total_lab)
        lv.addWidget(g)
        h = QHBoxLayout()
        self.start_btn = QPushButton("Start session")
        self.start_btn.clicked.connect(self.start_session)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._pause)
        self.finish_btn = QPushButton("Finish && Save")
        self.finish_btn.clicked.connect(self.finish)
        for w in (self.start_btn, self.pause_btn, self.finish_btn):
            h.addWidget(w)
        lv.addLayout(h)
        h2 = QHBoxLayout()
        self.redo_btn = QPushButton("Redo last trial")
        self.redo_btn.clicked.connect(self.redo_last)
        self.bad_btn = QPushButton("Reject selected → re-collect")
        self.bad_btn.clicked.connect(self.reject_selected)
        h2.addWidget(self.redo_btn)
        h2.addWidget(self.bad_btn)
        lv.addLayout(h2)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "pose", "effort", "status", "activation", "quality"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        lv.addWidget(self.table, 1)
        sp.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.progress_lab = T.big_label("", 18, T.MUTED)
        rv.addWidget(self.progress_lab)
        self.phase_lab = T.big_label("READY", 40, T.MUTED)
        self.phase_lab.setAlignment(Qt.AlignCenter)
        rv.addWidget(self.phase_lab)
        self.pose_lab = T.big_label("", 72)
        self.pose_lab.setAlignment(Qt.AlignCenter)
        rv.addWidget(self.pose_lab)
        self.eff_lab = T.big_label("", 60)
        self.eff_lab.setAlignment(Qt.AlignCenter)
        rv.addWidget(self.eff_lab)
        self.count = T.bar()
        self.count.setMinimumHeight(28)
        rv.addWidget(self.count)
        self.en_lab = T.big_label("", 22, T.MUTED)
        self.en_lab.setAlignment(Qt.AlignCenter)
        rv.addWidget(self.en_lab)
        self.key_lab = QLabel("")
        self.key_lab.setWordWrap(True)
        self.key_lab.setAlignment(Qt.AlignCenter)
        self.key_lab.setStyleSheet(f"font-size: 22px; font-weight: 700; color: {T.WARN};")
        rv.addWidget(self.key_lab)
        self.instr = QLabel("")
        self.instr.setWordWrap(True)
        self.instr.setStyleSheet("font-size: 20px; color: #cfd8dc;")
        rv.addWidget(self.instr)
        g2 = QGroupBox("Relative EMG activation (0 = rest, 100 % = calibrated wrist max)")
        gl = QVBoxLayout(g2)
        self.act = T.bar(T.ACCENT, 150)
        self.act.setMinimumHeight(30)
        self.act.setFormat("%v %")
        gl.addWidget(self.act)
        self.band_lab = QLabel("")
        self.band_lab.setStyleSheet("font-size: 16px;")
        gl.addWidget(self.band_lab)
        rv.addWidget(g2)
        self.warn_lab = QLabel("")
        self.warn_lab.setWordWrap(True)
        self.warn_lab.setStyleSheet(f"font-size: 16px; color: {T.WARN};")
        rv.addWidget(self.warn_lab)
        rv.addStretch()
        sp.addWidget(right)
        sp.setSizes([480, 820])
        for w in (self.reps, self.rest_trials, self.prep, self.hold, self.rest):
            w.valueChanged.connect(self._update_total)
        self.pose_list.itemChanged.connect(self._update_total)
        self._update_total()
        self._set_running(False)

    def _fill_poses(self):
        self.pose_list.clear()
        for p in load_poses():
            if p.name == C.REST:
                continue
            it = QListWidgetItem((f"{p.ko} — {p.name}" if p.ko else p.name) + ("" if p.has_effort else "  (no effort)"))
            it.setToolTip(f"{p.instruction}<br>구분: {p.key_point}<br>물체: {p.obj}")
            it.setData(Qt.UserRole, p.name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if p.builtin else Qt.Unchecked)
            self.pose_list.addItem(it)

    def _add_pose(self):
        d = AddPoseDialog(self)
        if d.exec() == QDialog.Accepted:
            try:
                add_pose(d.pose())
            except ValueError as e:
                QMessageBox.warning(self, "Add pose", str(e))
                return
            self._fill_poses()
            for i in range(self.pose_list.count()):
                if self.pose_list.item(i).data(Qt.UserRole) == d.pose().name:
                    self.pose_list.item(i).setCheckState(Qt.Checked)
            QMessageBox.information(self, "Add pose", f"'{d.pose().name}' added (config/poses.json). Record it here, then Train: "
                                    "models learn every pose present in the selected sessions.")

    def _selected_poses(self):
        names = [self.pose_list.item(i).data(Qt.UserRole) for i in range(self.pose_list.count())
                 if self.pose_list.item(i).checkState() == Qt.Checked]
        reg = {p.name: p for p in load_poses()}
        return [reg[n] for n in names if n in reg]

    def _update_total(self, *_):
        poses = self._selected_poses()
        n = sum(2 if p.has_effort else 1 for p in poses) * self.reps.value() + self.rest_trials.value()
        dur = n * (self.prep.value() + self.hold.value() + self.rest.value())
        self.total_lab.setText(f"{n} trials ≈ {dur / 60:.1f} min")

    def _set_running(self, on):
        for w in (self.pid, self.pose_list, self.reps, self.rest_trials, self.prep, self.hold, self.rest, self.onset, self.shuffle, self.start_btn):
            w.setEnabled(not on)
        for w in (self.pause_btn, self.finish_btn, self.redo_btn, self.bad_btn):
            w.setEnabled(on)

    # ------------------------------------------------------------------ session
    def start_session(self):
        if self.engine.source is None or self.engine.source.kind == "REPLAY":
            QMessageBox.warning(self, "Collect", "No live device stream.")
            return
        if self.engine.writer is not None:
            QMessageBox.warning(self, "Collect", "Another recording is running (stop 'Save' first).")
            return
        poses = self._selected_poses()
        if not poses and self.rest_trials.value() == 0:
            return
        cal = self.main.calibration
        if cal is None or not cal.has_rest:
            if QMessageBox.question(self, "Collect", "No calibration for this participant. Continue without calibration?") != QMessageBox.Yes:
                return
        conds = []
        for p in poses:
            conds += [(p.name, e) for e in (C.EFFORTS if p.has_effort else [""])]
        blocks = []
        for r in range(max(self.reps.value(), self.rest_trials.value())):
            b = [c for c in conds] if r < self.reps.value() else []
            if r < self.rest_trials.value():
                b.append((C.REST, ""))
            if self.shuffle.isChecked():
                random.shuffle(b)
            blocks += [(p, e, r + 1) for p, e in b]
        self.queue = blocks
        src = self.engine.source
        pid = self.pid.text().strip() or "S00"
        self.main.participant.setText(pid)
        reg = {p.name: p for p in load_poses()}
        meta = {"kind": "training", "data_source": src.kind, "fs": src.fs, "board_layout": src.layout,
                "sdk_version": getattr(src, "sdk_version", ""), "filter": dict(C.FILTER),
                "protocol": {"name": "pose_effort_v1", "inter_trial_rest": True, "onset_margin_s": self.onset.value(),
                             "prep_s": self.prep.value(), "hold_s": self.hold.value(), "rest_s": self.rest.value(),
                             "reps": self.reps.value(), "rest_trials": self.rest_trials.value(), "shuffle": self.shuffle.isChecked(),
                             "poses": [p.name for p in poses], "planned_order": [f"{p}|{e}" for p, e, _ in blocks]},
                "pose_definitions": {n: reg[n].__dict__ for n in [p.name for p in poses] + [C.REST] if n in reg},
                "calibration": cal.__dict__ if cal else None,
                "calibration_file": str(C.CALIB_DIR / cal.participant_id / f"{cal.calibration_id}.json") if cal else None,
                "effort_note": "Effort = instructed relative contraction level (Low/High), NOT force in N/kgf."}
        self.writer = SessionWriter(C.SESSIONS_DIR, pid, new_session_id(), meta)
        self.engine.start_recording(self.writer)
        self.trials, self.cur = [], None
        self.table.setRowCount(0)
        self._lost0 = self.engine.status.lost_samples
        self._set_running(True)
        self._next_trial(initial_rest=3.0)

    def _next_trial(self, initial_rest=0.0):
        if self.cur is None and initial_rest:
            self.phase, self.phase_end, self.phase_dur = "SETTLE", time.time() + initial_rest, initial_rest
            return
        if not self.queue:
            self.phase = "DONE"
            self.phase_lab.setText("ALL TRIALS DONE — press Finish & Save")
            T.set_color(self.phase_lab, T.OK)
            self.pose_lab.setText("")
            self.eff_lab.setText("")
            return
        pose, eff, rep = self.queue.pop(0)
        tid = (max(t["trial_id"] for t in self.trials) + 1) if self.trials else 1
        self.cur = {"trial_id": tid, "pose": pose, "effort": eff, "repetition": rep, "condition": f"{pose}|{eff}" if eff else pose,
                    "status": "running", "prep_start_idx": self.engine.last_sidx, "hold_start_idx": -1, "hold_end_idx": -1,
                    "rest_end_idx": -1, "hold_start_host": float("nan"), "hold_end_host": float("nan"), "quality": "", "note": ""}
        self.trials.append(self.cur)
        self.engine.trial_id = tid
        self.writer.event("prepare", self.cur["condition"])
        self._enter("PREPARE", self.prep.value())

    def _enter(self, phase, dur):
        self.phase, self.phase_dur, self.phase_end = phase, dur, time.time() + dur
        c = self.cur
        reg = {p.name: p for p in load_poses()}
        p = reg.get(c["pose"])
        if phase == "PREPARE":
            self.phase_lab.setText("다음 동작 준비 — PREPARE")
            T.set_color(self.phase_lab, T.WARN)
        elif phase == "HOLD":
            self.phase_lab.setText("유지 — HOLD")
            T.set_color(self.phase_lab, T.OK)
            c["hold_start_idx"], c["hold_start_host"] = self.engine.last_sidx, time.time()
            self._frames = []
            self.writer.event("hold_start", c["condition"])
            QApplication.beep()
        elif phase == "REST":
            self.phase_lab.setText("힘 빼고 휴식 — REST")
            T.set_color(self.phase_lab, T.LOW)
            c["hold_end_idx"], c["hold_end_host"] = self.engine.last_sidx, time.time()
            self.writer.event("hold_end", c["condition"])
            self._evaluate(c)
        self.pose_lab.setText((p.ko if p is not None and p.ko else c["pose"]))
        self.en_lab.setText(c["pose"])
        self.key_lab.setText(("구분 포인트: " + p.key_point) if (p is not None and p.key_point and phase != "REST") else "")
        T.set_color(self.pose_lab, T.TEXT if phase != "REST" else T.MUTED)
        e = c["effort"]
        self.eff_lab.setText(e.upper() if e else "")
        T.set_color(self.eff_lab, (T.HIGH if e == "High" else T.LOW) if phase != "REST" else T.MUTED)
        hint = ""
        if p is not None:
            hint = f"{p.instruction}\n물체: {p.obj}"
            if e:
                hint += f"\n강도: {p.high_hint if e == 'High' else p.low_hint}"
        if phase == "REST":
            hint = "동작을 풀고 손과 팔의 힘을 완전히 빼세요. 팔 자세는 그대로 유지합니다."
        self.instr.setText(hint)
        done = sum(t["status"] == "completed" for t in self.trials)
        self.progress_lab.setText(f"trial {c['trial_id']} | completed {done} | remaining {len(self.queue)} | "
                                  f"{c['condition']} rep {c['repetition']}")
        self.band_lab.setText(self._band_text(c["pose"]) if e else "")

    def _median_act(self, pose, effort, min_n=2):
        """Median activation of this session's completed trials of pose/effort (None if < min_n)."""
        a = [t["_act"] for t in self.trials if t["pose"] == pose and t["effort"] == effort and t["status"] == "completed"
             and np.isfinite(t.get("_act", np.nan))]
        return float(np.median(a)) if len(a) >= min_n else None

    def _band_text(self, pose):
        if self.main.calibration is None or not self.main.calibration.has_norm:
            return "캘리브레이션 없음 — 활성도 표시 불가 (강도는 안내 문구 기준)"
        lo, hi = self._median_act(pose, "Low"), self._median_act(pose, "High")
        f = lambda v: "—" if v is None else f"{v * 100:.0f} %"
        return f"이번 세션 {pose}: Low 중앙값 {f(lo)} · High 중앙값 {f(hi)}  (Low < High 가 유지되도록)"

    def _on_frame(self, fr):
        if self.phase == "HOLD":
            self._frames.append(fr)

    def _evaluate(self, c):
        """Quality flags of the hold that just ended (warnings only; the experimenter decides)."""
        frames = [f for f in self._frames if f.host_ts >= c["hold_start_host"] + self.onset.value()]
        flags = []
        st = self.engine.status
        if st.flat_channels:
            flags.append(f"flat ch {st.flat_channels}")
        lost = st.lost_samples - self._lost0
        self._lost0 = st.lost_samples
        if lost > 0.05 * self.engine.fs * (self.prep.value() + self.hold.value() + self.rest.value()):
            flags.append(f"packet loss {lost} samples")
        if len(frames) < 0.5 * self.hold.value() / 0.05:
            flags.append("too little data")
        act = float("nan")
        cal = self.main.calibration
        if frames:
            rms = np.array([f.rms for f in frames])
            gm = np.array([f.gyro_mag for f in frames])
            if cal is not None and cal.has_norm:
                act = float(np.median([cal.activation(r) for r in rms]))
                lo, hi = self._median_act(c["pose"], "Low"), self._median_act(c["pose"], "High")
                if c["effort"] == "Low" and hi is not None and act >= hi:
                    flags.append(f"Low as strong as this session's High (median {hi * 100:.0f} %)")
                if c["effort"] == "High" and lo is not None and act <= lo:
                    flags.append(f"High as weak as this session's Low (median {lo * 100:.0f} %)")
                if c["pose"] == C.REST and act > 0.15:
                    flags.append("not relaxed (EMG high)")
            gmed = float(np.nanmedian(gm)) if np.isfinite(gm).any() else 0.0
            if c["pose"] == C.REST and gmed > 30:
                flags.append(f"arm moved during Rest (|ω| {gmed:.0f} dps)")
            pose = next((p for p in load_poses() if p.name == c["pose"]), None)
            if pose is not None and "dynamic" not in pose.tags and c["pose"] != C.REST and gmed > 60:
                flags.append(f"arm moving (|ω| {gmed:.0f} dps)")
            if np.isfinite(gm).any() and np.nanmax(gm) >= C.GYRO_SATURATION_DPS:
                flags.append("gyro saturated")
        c["quality"] = "; ".join(flags)
        c["_act"] = act
        self.warn_lab.setText(("⚠ " + c["quality"]) if flags else "")

    def _tick(self):
        a = self.engine.activation
        self.act.setValue(int(min(150, a * 100)) if np.isfinite(a) else 0)
        if self.writer is None or self.paused or self.phase in ("IDLE", "DONE"):
            return
        rem = self.phase_end - time.time()
        self.count.setValue(int(100 * (1 - max(rem, 0) / max(self.phase_dur, 1e-6))))
        self.count.setFormat(f"{max(rem, 0):.1f} s")
        if rem > 0:
            return
        if self.phase == "SETTLE":
            self.cur = None
            self.phase = "IDLE"
            self._next_trial()
        elif self.phase == "PREPARE":
            self._enter("HOLD", self.hold.value())
        elif self.phase == "HOLD":
            self._enter("REST", self.rest.value())
        elif self.phase == "REST":
            c = self.cur
            c["rest_end_idx"] = self.engine.last_sidx
            c["status"] = "completed"
            self.engine.trial_id = -1
            self.writer.event("trial_completed", c["condition"])
            self._sync()
            self._next_trial()

    def _sync(self):
        rows = [{k: v for k, v in t.items() if not k.startswith("_")} for t in self.trials if t["status"] != "running"]
        self.writer.set_trials(rows)
        self.table.setRowCount(len(self.trials))
        for r, t in enumerate(self.trials):
            a = t.get("_act", float("nan"))
            for k, val in enumerate([t["trial_id"], t["pose"], t["effort"], t["status"],
                                     f"{a * 100:.0f} %" if np.isfinite(a) else "", t["quality"]]):
                it = QTableWidgetItem(str(val))
                if t["status"] == "rejected":
                    it.setForeground(Qt.gray)
                elif t["quality"] and k == 5:
                    it.setForeground(Qt.yellow)
                self.table.setItem(r, k, it)
        self.table.scrollToBottom()

    def _pause(self, on):
        self.paused = on
        if on:
            self._pause_rem = self.phase_end - time.time()
            self.phase_lab.setText("PAUSED")
            if self.phase in ("PREPARE", "HOLD") and self.cur is not None:
                # an interrupted trial is aborted and re-queued
                c = self.cur
                c["status"] = "aborted"
                c["note"] = "paused during " + self.phase
                self.queue.insert(0, (c["pose"], c["effort"], c["repetition"]))
                self.engine.trial_id = -1
                self.writer.event("trial_aborted", c["condition"])
                self._sync()
                self.phase = "SETTLE"
                self._pause_rem = 2.0
        else:
            self.phase_end = time.time() + max(self._pause_rem, 0.5)

    def redo_last(self):
        done = [t for t in self.trials if t["status"] == "completed"]
        if not done:
            return
        t = done[-1]
        t["status"] = "rejected"
        t["note"] = (t["note"] + " redo").strip()
        self.queue.insert(0, (t["pose"], t["effort"], t["repetition"]))
        self.writer.event("trial_rejected", f"{t['trial_id']} redo")
        self._sync()

    def reject_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        for r in rows:
            t = self.trials[r]
            if t["status"] == "completed":
                t["status"] = "rejected"
                t["note"] = (t["note"] + " rejected by experimenter").strip()
                self.queue.append((t["pose"], t["effort"], t["repetition"]))
                self.writer.event("trial_rejected", str(t["trial_id"]))
        self._sync()

    def finish(self):
        if self.writer is None:
            return
        if self.cur is not None and self.cur["status"] == "running":
            self.cur["status"] = "aborted"
            self.cur["note"] = "session finished"
        self.engine.trial_id = -1
        self._sync()
        w = self.writer
        self.writer = None
        self.engine.stop_recording("completed", {"remaining_queue": [f"{p}|{e}" for p, e, _ in self.queue]})
        if self.main.calibration is not None:
            self.main.calibration.save(w.dir / "calibration.json")
        n = sum(t["status"] == "completed" for t in self.trials)
        self.phase = "IDLE"
        self.phase_lab.setText("SAVED")
        self._set_running(False)
        QMessageBox.information(self, "Collect Data", f"Saved {n} completed trials to\n{w.dir}\n\nNext: Train.")

    def closeEvent(self, ev):
        if self.writer is not None:
            if QMessageBox.question(self, "Collect Data", "Finish and save the running session?") != QMessageBox.Yes:
                ev.ignore()
                return
            self.finish()
        self.timer.stop()
        self.engine.remove_listener(self._on_frame)
        super().closeEvent(ev)
