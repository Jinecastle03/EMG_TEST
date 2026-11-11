"""Training window. Training runs in a separate Python process (QProcess -> python -m emgmonitor.trainer) so it
can neither block nor slow down the real-time engine / GUI, and a crash cannot take the monitor down."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                               QPlainTextEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import config as C
from ..session import list_sessions
from ..trainer import DEFAULT_FAMILIES, FAMILIES
from . import theme as T


class TrainingWindow(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setWindowTitle("Train & compare models")
        self.resize(1250, 880)
        self.proc = None
        self.run_id = ""
        v = QVBoxLayout(self)
        g = QGroupBox("Training data (only labeled 'training' sessions; synthetic / free / calibration sessions are excluded by default)")
        gl = QVBoxLayout(g)
        self.sess = QTableWidget(0, 6)
        self.sess.setHorizontalHeaderLabels(["use", "participant", "session", "source", "trials", "labels"])
        self.sess.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sessions = [s for s in list_sessions() if s["kind"] == "training"]
        pid = main.participant.text().strip()
        for s in self.sessions:
            r = self.sess.rowCount()
            self.sess.insertRow(r)
            it = QTableWidgetItem()
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            # default: only the current participant's real sessions (never mix people / label sets silently)
            use = "SYNTHETIC" not in s["data_source"] and s["participant"] == pid and s["n_completed"] > 0
            it.setCheckState(Qt.Checked if use else Qt.Unchecked)
            self.sess.setItem(r, 0, it)
            for c, val in enumerate([s["participant"], s["session_id"], s["data_source"], s["n_completed"],
                                     ", ".join(f"{k}:{n}" for k, n in sorted(s["labels"].items()))], 1):
                self.sess.setItem(r, c, QTableWidgetItem(str(val)))
        self.sess.resizeColumnsToContents()
        self.sess.setMaximumHeight(180)
        gl.addWidget(self.sess)
        self.mix_lab = QLabel("")
        self.mix_lab.setWordWrap(True)
        gl.addWidget(self.mix_lab)
        v.addWidget(g)

        h = QHBoxLayout()
        g2 = QGroupBox("Models")
        l2 = QVBoxLayout(g2)
        self.fam = {}
        for k, spec in FAMILIES.items():
            cb = QCheckBox(f"{k} — {spec['display']}")
            cb.setChecked(k in DEFAULT_FAMILIES)
            self.fam[k] = cb
            l2.addWidget(cb)
        h.addWidget(g2)
        g3 = QGroupBox("Windows / evaluation")
        l3 = QVBoxLayout(g3)
        self.win = {}
        hw = QHBoxLayout()
        for w in C.WINDOW_CHOICES_MS:
            cb = QCheckBox(f"{w:.0f} ms")
            cb.setChecked(w == C.WINDOW_MS)
            self.win[w] = cb
            hw.addWidget(cb)
        l3.addLayout(hw)
        hs = QHBoxLayout()
        hs.addWidget(QLabel("Test split"))
        self.split = QComboBox()
        self.split.addItems(["trial", "session", "participant"])
        hs.addWidget(self.split)
        self.test_reps = QSpinBox()
        self.test_reps.setRange(1, 5)
        self.test_reps.setValue(2)
        hs.addWidget(QLabel("test trials / label"))
        hs.addWidget(self.test_reps)
        l3.addLayout(hs)
        self.split_info = QLabel("")
        self.split_info.setWordWrap(True)
        self.split_info.setStyleSheet(f"color:{T.MUTED}")
        l3.addWidget(self.split_info)
        self.quick = QCheckBox("quick (no cross-validation → no model selection; for testing only)")
        l3.addWidget(self.quick)
        he = QHBoxLayout()
        he.addWidget(QLabel("max epochs"))
        self.epochs = QSpinBox()
        self.epochs.setRange(5, 500)
        self.epochs.setValue(100)
        he.addWidget(self.epochs)
        he.addStretch()
        l3.addLayout(he)
        h.addWidget(g3)
        v.addLayout(h)
        self.sess.itemChanged.connect(self._split_info)
        self.split.currentIndexChanged.connect(self._split_info)
        self._split_info()

        hb = QHBoxLayout()
        self.start_btn = QPushButton("Start training")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        self.report_btn = QPushButton("Open report")
        self.report_btn.clicked.connect(self.open_report)
        self.use_btn = QPushButton("Use recommended model")
        self.use_btn.clicked.connect(self.use_recommended)
        for b in (self.start_btn, self.stop_btn, self.report_btn, self.use_btn):
            hb.addWidget(b)
        hb.addStretch()
        v.addLayout(hb)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family: Consolas; font-size: 11px;")
        v.addWidget(self.log, 1)
        self.rec_lab = T.big_label("", 16, T.OK, True)
        self.rec_lab.setWordWrap(True)
        v.addWidget(self.rec_lab)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Model", "Window", "Pose Macro-F1 test (CV)", "Effort BA test (CV)",
                                              "Effort BA cascade", "Effort input", "Latency ms", "Rest FP smoothed", "Switch/s smoothed"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        v.addWidget(self.table, 1)

    def _chosen_sessions(self):
        return [self.sessions[r]["path"] for r in range(self.sess.rowCount()) if self.sess.item(r, 0).checkState() == Qt.Checked]

    def _split_info(self, *_):
        ch = [self.sessions[r] for r in range(self.sess.rowCount()) if self.sess.item(r, 0) and self.sess.item(r, 0).checkState() == Qt.Checked]
        ns, npart = len(ch), len({s["participant"] for s in ch})
        poses = [frozenset(k.split("|")[0] for k in s["labels"]) for s in ch if s["labels"]]
        warn = []
        if npart > 1:
            warn.append(f"⚠ {npart}명의 참가자 세션이 섞여 있습니다 ({', '.join(sorted({s['participant'] for s in ch}))}). "
                        "사람별 모델이 목적이면 한 사람 것만 선택하세요.")
        if len(set(poses)) > 1 and any(not (a <= b or b <= a) for a in poses for b in poses):
            warn.append("⚠ 세션마다 동작 세트가 다릅니다 (예: 기존 Grip/Twist CCW + 새 5동작) → 모든 라벨이 한 모델의 클래스로 합쳐집니다.")
        small = [s["session_id"] for s in ch if s["n_completed"] < 10]
        if small:
            warn.append(f"참고: trial이 10개 미만인 세션 {small}")
        self.mix_lab.setText("<br>".join(warn))
        self.mix_lab.setStyleSheet(f"color:{T.WARN}")
        txt = {"trial": "Trial-independent: last N trials of every label (per session) are held out.",
               "session": f"Session-independent: newest session held out. Selected: {ns} session(s)" + ("" if ns >= 2 else " → NOT possible"),
               "participant": f"Participant-independent: one participant held out. Selected: {npart} participant(s)" +
                              ("" if npart >= 2 else " → NOT possible with a single participant")}[self.split.currentText()]
        self.split_info.setText(txt)

    def start(self):
        sess = self._chosen_sessions()
        fams = [k for k, cb in self.fam.items() if cb.isChecked()]
        wins = [str(int(w)) for w, cb in self.win.items() if cb.isChecked()]
        if not sess or not fams or not wins:
            QMessageBox.warning(self, "Train", "Select sessions, models and at least one window.")
            return
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        args = ["-u", "-W", "ignore", "-m", "emgmonitor.trainer", "--run-id", self.run_id, "--split", self.split.currentText(),
                "--test-reps", str(self.test_reps.value()), "--epochs", str(self.epochs.value()),
                "--windows", *wins, "--families", *fams, "--sessions", *sess]
        if self.quick.isChecked():
            args.append("--quick")
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(C.ROOT))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()      # full environment (HOME/USERPROFILE etc.)
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.log.clear()
        self.log.appendPlainText(f"python {' '.join(args)}")
        self.proc.start(sys.executable, args)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _read(self):
        self.log.appendPlainText(bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace").rstrip())

    def stop(self):
        if self.proc is not None:
            self.proc.kill()

    def _done(self, code, _status):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        run_dir = C.RUNS_DIR / self.run_id
        if code != 0 or not (run_dir / "comparison.csv").exists():
            self.log.appendPlainText(f"training process ended with code {code}")
            return
        df = pd.read_csv(run_dir / "comparison.csv")
        self.table.setRowCount(len(df))

        def f(x, nd=3):
            return "n/a" if pd.isna(x) else f"{x:.{nd}f}"
        for r, row in df.iterrows():
            vals = [row.display, f"{row.window_ms:.0f}", f"{f(row.test_pose_macro_f1)} ({f(row.cv_pose_macro_f1)})",
                    f"{f(row.test_effort_ba_true_pose)} ({f(row.cv_effort_ba_true_pose)})", f(row.test_effort_ba_cascade),
                    row.effort_input, f"{row.latency_median_ms:.2f}", f(row.rest_fp_smoothed), f(row.switches_per_s_smoothed, 2)]
            for c, val in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(str(val)))
        self.table.resizeColumnsToContents()
        res = C.load_json(run_dir / "results.json")
        self.rec_lab.setText(res["recommendation"]["text"])
        self.main.refresh_models()

    def open_report(self):
        p = C.RUNS_DIR / self.run_id / "report.md" if self.run_id else None
        if p is None or not p.exists():
            runs = sorted(C.RUNS_DIR.glob("*/report.md"))
            p = runs[-1] if runs else None
        if p:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def use_recommended(self):
        if not self.run_id:
            return
        for mp in (C.RUNS_DIR / self.run_id).glob("*/*/meta.json"):
            m = C.load_json(mp)
            if m.get("recommended"):
                self.main.refresh_models(str(mp.parent))
                QMessageBox.information(self, "Model", f"Selected {m['display']} {m['window_ms']:.0f} ms")
                return
