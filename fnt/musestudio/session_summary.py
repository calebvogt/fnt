"""Post-session summary — what you see the moment you open your eyes.

Answers three questions without making you go hunting: is the data good, did
anything go wrong, and did the measure actually move?
"""

import os
import subprocess
import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QVBoxLayout,
)

from fnt.musestudio import theme
from fnt.musestudio.review import compare_to_baseline, load_session, summarize_phases


class SessionSummaryDialog(QDialog):
    """Shown automatically when a recording finishes."""

    review_requested = None     # set by the host to a callable(path)

    def __init__(self, session_root, counts, frames, dropouts, parent=None):
        super().__init__(parent)
        self.session_root = session_root
        self.setWindowTitle("Session complete")
        self.setStyleSheet(theme.STYLESHEET)
        self.resize(720, 560)

        root = QVBoxLayout(self)

        title = QLabel("Session complete")
        f = QFont()
        f.setPointSize(16)
        f.setBold(True)
        title.setFont(f)
        root.addWidget(title)

        self.path_label = QLabel(os.path.basename(session_root or ""))
        self.path_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        root.addWidget(self.path_label)

        # --- data integrity ------------------------------------------------
        root.addWidget(_heading("Data captured"))
        self.data_table = _table(["Stream", "Samples"])
        rows = sorted(counts.items())
        if frames:
            rows.append(("webcam (frames)", frames))
        self.data_table.setRowCount(len(rows))
        for r, (name, n) in enumerate(rows):
            self.data_table.setItem(r, 0, QTableWidgetItem(str(name)))
            item = QTableWidgetItem(f"{n:,}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.data_table.setItem(r, 1, item)
        self.data_table.setMaximumHeight(140)
        root.addWidget(self.data_table)

        # --- dropouts ------------------------------------------------------
        if dropouts:
            lost = sum(d[2] for d in dropouts)
            warn = QLabel(
                f"⚠  {len(dropouts)} stream dropout(s), {lost:.1f}s of data lost. "
                "Treat gaps in this recording with caution.")
            warn.setWordWrap(True)
            warn.setStyleSheet(
                f"background:#3A2A10; color:{theme.WARN}; border:1px solid "
                f"{theme.WARN}; border-radius:6px; padding:8px;")
            root.addWidget(warn)
        else:
            ok = QLabel("✓  No stream dropouts detected.")
            ok.setStyleSheet(f"color: {theme.GOOD};")
            root.addWidget(ok)

        # --- did it move? --------------------------------------------------
        root.addWidget(_heading("Did the measure move?"))
        self.result_label = QLabel("Analysing…")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        root.addWidget(self.result_label)

        self.phase_table = _table(["Phase", "Duration", "PLV", "Δ vs baseline",
                                   "Contact"])
        root.addWidget(self.phase_table, stretch=1)

        buttons = QHBoxLayout()
        self.review_btn = QPushButton("Open in Review")
        self.review_btn.setProperty("accent", True)
        self.review_btn.setToolTip(
            "Load this recording into the Review tab for the full timelines.")
        buttons.addWidget(self.review_btn)
        folder_btn = QPushButton("Show folder")
        folder_btn.setToolTip("Reveal the recording folder in Finder.")
        folder_btn.clicked.connect(self._open_folder)
        buttons.addWidget(folder_btn)
        buttons.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        root.addLayout(buttons)

        self._analyse()

    # ------------------------------------------------------------------
    def _analyse(self):
        """Reuse the offline review analysis to fill the phase table."""
        if not self.session_root or not os.path.isdir(self.session_root):
            self.result_label.setText("Nothing was recorded.")
            return
        try:
            data = load_session(self.session_root)
            rows = summarize_phases(data)
        except Exception as exc:  # noqa: BLE001
            self.result_label.setText(f"Could not analyse this session: {exc}")
            return

        base, deltas = compare_to_baseline(rows)
        delta_by_phase = {d["phase"]: d for d in deltas}
        self.phase_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            is_base = base is not None and row is base
            d = delta_by_phase.get(row["phase"], {})
            dv = d.get("d_plv")
            cells = [
                row["phase"] + ("  (baseline)" if is_base else ""),
                f"{row['duration']:.0f} s",
                _fmt(row.get("plv")),
                "—" if is_base else _fmt(dv, signed=True),
                _fmt_pct(row.get("contact")),
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if c == 3 and dv is not None and not _nan(dv) and not is_base:
                    item.setForeground(
                        _brush(theme.GOOD if dv > 0 else theme.DANGER))
                self.phase_table.setItem(r, c, item)

        if base and deltas:
            best = max(deltas, key=lambda x: (x["d_plv"] if not _nan(x["d_plv"])
                                              else -9))
            if not _nan(best["d_plv"]):
                verb = "rose" if best["d_plv"] > 0 else "fell"
                self.result_label.setText(
                    f"Synchrony {verb} {abs(best['d_plv']):.3f} during "
                    f"{best['phase']} compared with baseline.")
                return
        if not rows:
            self.result_label.setText("No phases were recorded.")
        else:
            self.result_label.setText(
                "No baseline phase to compare against — open Review for the "
                "full timeline.")

    def _open_folder(self):
        if not self.session_root:
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", self.session_root])
            elif sys.platform.startswith("win"):
                os.startfile(self.session_root)   # noqa: S606
            else:
                subprocess.Popen(["xdg-open", self.session_root])
        except Exception:
            pass


def _heading(text):
    label = QLabel(text)
    label.setStyleSheet(
        f"color: {theme.TEXT_DIM}; font-weight: 600; margin-top: 6px;")
    return label


def _table(headers):
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.NoSelection)
    table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    for c in range(1, len(headers)):
        table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
    return table


def _nan(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def _fmt(v, signed=False):
    if _nan(v):
        return "—"
    return f"{v:+.3f}" if signed else f"{v:.3f}"


def _fmt_pct(v):
    return "—" if _nan(v) else f"{v * 100:.0f}%"


def _brush(colour):
    from PyQt5.QtGui import QBrush, QColor
    return QBrush(QColor(colour))
