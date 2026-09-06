"""Read-only viewer for a ``.mad`` (FNT-MAD HDF5) sidecar file.

The ``.mad`` is MAD's source of truth — labels, hard negatives, prediction
crops and the grid params they were computed on all live there, and nothing
else records them since the CSVs became an export step. Without a way to look
inside it, "why is this call still pending?" or "what did that run actually
store?" can only be answered by writing h5py by hand.

Two rules shape this module:

* **Never hold the file open.** Every read opens the file, takes what it
  needs, and closes. MAD writes to the same path whenever a call is
  confirmed, and an inspector holding a handle would either block those
  writes or read a half-written file. Snapshotting also makes "read-only"
  structural rather than a promise.
* **Never read pixels until asked.** A file can hold thousands of prediction
  crops; listing names, shapes and attributes is cheap, decompressing every
  mask is not. Groups fill in when expanded, arrays load when selected.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

try:
    import h5py
    HAS_H5PY = True
except Exception:                                   # pragma: no cover
    HAS_H5PY = False

#: Children listed per group before the rest are collapsed behind a summary
#: row. 4000 prediction crops in a tree is neither readable nor fast.
PAGE = 500

#: Root attributes, grouped and ordered for reading rather than for storage.
#: The file keeps them flat — restructuring an on-disk format that already
#: holds real labels would mean migrating every .mad for a presentation win —
#: so the grouping lives here, where it costs nothing and cannot corrupt
#: anything.
ATTR_SECTIONS = (
    ("Provenance", ("format", "format_version", "schema_version",
                    "fnt_version", "created", "updated", "source_wav")),
    ("Spectrogram grid", ("sample_rate", "nperseg", "noverlap", "nfft",
                          "n_freq_bins", "n_time_frames")),
    ("Last inference run", ("last_infer_at", "last_infer_model",
                            "last_infer_threshold", "last_infer_min_blob",
                            "last_infer_merge", "last_infer_n",
                            "n_pred_blobs")),
)


def humanize(key, val, attrs):
    """A readable gloss for an attribute, or '' when the raw value says it all.

    Frame counts and window sizes are the ones worth translating: 'nperseg
    512' means nothing until it is 2.05 ms, and 'n_time_frames 1171913' means
    nothing until it is 600 s.
    """
    try:
        sr = float(attrs.get("sample_rate") or 0)
        nperseg = float(attrs.get("nperseg") or 0)
        noverlap = float(attrs.get("noverlap") or 0)
        hop = nperseg - noverlap
        v = float(val)
    except Exception:
        return ""
    if key == "sample_rate":
        return f"{v / 1000:.0f} kHz  (Nyquist {v / 2000:.0f} kHz)"
    if key in ("nperseg", "noverlap") and sr:
        return f"{1000 * v / sr:.2f} ms"
    if key == "nfft" and sr:
        return f"{sr / v:.0f} Hz per bin"
    if key == "n_time_frames" and sr and hop > 0:
        return f"{v * hop / sr:.1f} s of audio  ({1000 * hop / sr:.3f} ms/frame)"
    if key == "n_freq_bins" and sr:
        return f"0 - {sr / 2000:.0f} kHz"
    if key == "last_infer_merge":
        return "on" if v else "off"
    if key in ("last_infer_n", "n_pred_blobs"):
        return "detections written by that run"
    return ""


def sort_key(name: str):
    """Numeric ids sort numerically.

    Prediction crops are keyed by integer, and lexical order puts 1000 between
    1 and 101 — which is what made a 2760-crop group unreadable.
    """
    return (0, int(name), "") if name.isdigit() else (1, 0, name.lower())


_ROLE_PATH = Qt.UserRole + 1
_ROLE_KIND = Qt.UserRole + 2        # 'group' | 'dataset' | 'more'
_ROLE_LOADED = Qt.UserRole + 3


def mad_path_for(wav_path: str) -> Optional[str]:
    """The .mad sibling for a recording, or None when there isn't one."""
    try:
        from fnt.usv.usv_detector.fnt_mask_store import masks_sibling_path
        p = masks_sibling_path(wav_path)
        return p if p and os.path.isfile(p) else None
    except Exception:
        return None


def _fmt_attr(v):
    if isinstance(v, bytes):
        try:
            v = v.decode()
        except Exception:
            return repr(v)
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, np.ndarray):
        return f"array{v.shape} {v.dtype}"
    if isinstance(v, float):
        # Show what was meant, not the binary expansion of it.
        return f"{v:g}"
    return v


class MADInspectorDialog(QDialog):
    """Tree view of a .mad file's structure, attributes and arrays."""

    def __init__(self, parent, path: str, parented: bool = True):
        # Un-parented by default from MAD: a parented widget is an owned
        # native window and Windows keeps an owned window above its owner.
        super().__init__(parent if parented else None)
        self._path = path
        self.setWindowTitle(f"Inspect .mad — {os.path.basename(path)}")
        self.setModal(False)
        # Independent top-level window so it can be sent behind the main
        # window while cross-referencing (see MADTrainGraphDialog).
        self.setWindowFlags(Qt.Window
                            | Qt.WindowMinimizeButtonHint
                            | Qt.WindowMaximizeButtonHint
                            | Qt.WindowCloseButtonHint)
        self.resize(1040, 660)

        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)

        hdr = QLabel(
            f"<b>{os.path.basename(path)}</b> — "
            f"{os.path.getsize(path) / 1e6:.1f} MB · read-only")
        hdr.setStyleSheet("font-size:11px;")
        v.addWidget(hdr)

        note = QLabel(
            "The file is opened, read and closed for each request, so nothing "
            "here can change it and MAD can keep writing while this is open.")
        note.setStyleSheet("color:#9fb8c8; font-size:10px;")
        note.setWordWrap(True)
        v.addWidget(note)

        self._filter = QLineEdit()
        self._filter.setPlaceholderText(
            "Filter the visible rows by name (e.g. a call id, or 'neg')")
        self._filter.textChanged.connect(self._apply_filter)
        v.addWidget(self._filter)

        split = QSplitter(Qt.Horizontal)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Item", "Type", "Shape / count"])
        self._tree.setColumnWidth(0, 340)
        self._tree.header().setSectionResizeMode(0, QHeaderView.Interactive)
        self._tree.itemExpanded.connect(self._on_expand)
        self._tree.currentItemChanged.connect(self._on_select)
        split.addWidget(self._tree)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(QLabel("Attributes"))
        self._attrs = QTableWidget(0, 3)
        self._attrs.setHorizontalHeaderLabels(["key", "value", "meaning"])
        self._attrs.horizontalHeader().setStretchLastSection(True)
        self._attrs.verticalHeader().setVisible(False)
        self._attrs.setMaximumHeight(220)
        rv.addWidget(self._attrs)
        rv.addWidget(QLabel("Value"))
        self._preview = QLabel("Select an item.")
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumHeight(150)
        self._preview.setStyleSheet("background:#1b1b1b;")
        rv.addWidget(self._preview, 1)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumHeight(180)
        rv.addWidget(self._text)
        split.addWidget(right)
        split.setSizes([460, 580])
        v.addWidget(split, 1)

        row = QHBoxLayout()
        row.addStretch()
        b = QPushButton("Close")
        b.clicked.connect(self.close)
        row.addWidget(b)
        v.addLayout(row)

        self._build_root()

    # -- structure ----------------------------------------------------
    def _open(self):
        if not HAS_H5PY:
            raise RuntimeError("h5py is not installed")
        return h5py.File(self._path, "r")

    def _build_root(self):
        self._tree.clear()
        try:
            with self._open() as f:
                root = QTreeWidgetItem(["/", "file", ""])
                root.setData(0, _ROLE_PATH, "/")
                root.setData(0, _ROLE_KIND, "group")
                root.setData(0, _ROLE_LOADED, True)
                self._tree.addTopLevelItem(root)
                self._add_children(root, f, "/")
                root.setExpanded(True)
                self._show_attrs(dict(f.attrs))
                self._text.setPlainText(
                    "Root attributes describe the spectrogram grid every mask "
                    "in this file was computed on, plus the last inference "
                    "run.")
        except Exception as e:
            self._text.setPlainText(f"Could not open the file:\n\n{e}")

    def _add_children(self, parent_item, h5group, path):
        keys = sorted(h5group.keys(), key=sort_key)
        for k in keys[:PAGE]:
            obj = h5group[k]
            if isinstance(obj, h5py.Group):
                it = QTreeWidgetItem([k, "group", f"{len(obj)} item(s)"])
                it.setData(0, _ROLE_KIND, "group")
                it.setData(0, _ROLE_LOADED, False)
                # A placeholder gives the row an expand arrow without reading
                # the group; the real children arrive on expand.
                it.addChild(QTreeWidgetItem(["…", "", ""]))
            else:
                it = QTreeWidgetItem(
                    [k, str(obj.dtype), " x ".join(str(d) for d in obj.shape)])
                it.setData(0, _ROLE_KIND, "dataset")
                it.setData(0, _ROLE_LOADED, True)
            child_path = (path.rstrip("/") + "/" + k) if path != "/" else "/" + k
            it.setData(0, _ROLE_PATH, child_path)
            parent_item.addChild(it)
        if len(keys) > PAGE:
            more = QTreeWidgetItem(
                [f"… {len(keys) - PAGE:,} more not listed", "", ""])
            more.setData(0, _ROLE_KIND, "more")
            more.setForeground(0, QColor(150, 150, 150))
            parent_item.addChild(more)

    def _on_expand(self, item):
        if item.data(0, _ROLE_LOADED):
            return
        item.setData(0, _ROLE_LOADED, True)
        item.takeChildren()                      # drop the placeholder
        path = item.data(0, _ROLE_PATH)
        try:
            with self._open() as f:
                self._add_children(item, f[path], path)
        except Exception as e:
            item.addChild(QTreeWidgetItem([f"error: {e}", "", ""]))

    # -- detail -------------------------------------------------------
    def _show_attrs(self, attrs: dict):
        """Fill the table, grouped into sections with a plain-English gloss."""
        rows = []
        seen = set()
        for title, keys in ATTR_SECTIONS:
            present = [k for k in keys if k in attrs]
            if not present:
                continue
            rows.append((title, None, None))            # section header
            for k in present:
                rows.append((k, _fmt_attr(attrs[k]), humanize(k, attrs[k], attrs)))
                seen.add(k)
        rest = sorted(k for k in attrs if k not in seen)
        if rest:
            rows.append(("Other", None, None))
            for k in rest:
                rows.append((k, _fmt_attr(attrs[k]), humanize(k, attrs[k], attrs)))

        self._attrs.setRowCount(len(rows))
        for r, (k, v, why) in enumerate(rows):
            if v is None:                               # section header row
                it = QTableWidgetItem(k)
                f = it.font(); f.setBold(True); it.setFont(f)
                it.setForeground(QColor(150, 190, 215))
                self._attrs.setItem(r, 0, it)
                self._attrs.setSpan(r, 0, 1, 3)
                continue
            self._attrs.setItem(r, 0, QTableWidgetItem("   " + str(k)))
            self._attrs.setItem(r, 1, QTableWidgetItem(str(v)))
            item = QTableWidgetItem(str(why))
            item.setForeground(QColor(150, 150, 150))
            self._attrs.setItem(r, 2, item)
        self._attrs.resizeColumnsToContents()

    def _on_select(self, item, _prev=None):
        if item is None or item.data(0, _ROLE_KIND) == "more":
            return
        path = item.data(0, _ROLE_PATH)
        if not path:
            return
        self._preview.setPixmap(QPixmap())
        self._preview.setText("")
        try:
            with self._open() as f:
                obj = f[path]
                attrs = dict(obj.attrs)
                self._show_attrs(attrs)
                if isinstance(obj, h5py.Group):
                    self._preview.setText(
                        f"{path}\n\n{len(obj)} item(s)")
                    self._text.setPlainText(self._describe_meta(attrs))
                    return
                arr = obj[()]                    # only now is data read
        except Exception as e:
            self._text.setPlainText(f"Could not read {path}:\n\n{e}")
            return
        self._show_array(path, arr, attrs)

    def _describe_meta(self, attrs: dict) -> str:
        raw = attrs.get("meta_json")
        if raw is None:
            return ""
        if isinstance(raw, bytes):
            raw = raw.decode(errors="replace")
        try:
            return json.dumps(json.loads(raw), indent=2, sort_keys=True)
        except Exception:
            return str(raw)

    def _show_array(self, path, arr, attrs):
        info = [f"{path}", f"shape {getattr(arr, 'shape', ())}  "
                           f"dtype {getattr(arr, 'dtype', type(arr).__name__)}"]
        a = np.asarray(arr)
        if a.ndim == 2 and a.size:
            nz = int((a > 0).sum())
            info.append(f"non-zero {nz:,} of {a.size:,} "
                        f"({100.0 * nz / a.size:.2f}%)")
            info.append(f"min {a.min()}  max {a.max()}")
            self._preview.setPixmap(self._as_pixmap(a))
        else:
            self._preview.setText(np.array2string(a, threshold=200))
        meta = self._describe_meta(attrs)
        self._text.setPlainText("\n".join(info) + (("\n\n" + meta) if meta else ""))

    def _as_pixmap(self, a: np.ndarray) -> QPixmap:
        """Render a 2D array as an image, scaled into the preview pane.

        Flipped vertically to match how the spectrogram is drawn, so a mask
        here has the same orientation as the call it came from.
        """
        f = a.astype(np.float32)
        lo, hi = float(f.min()), float(f.max())
        norm = (f - lo) / (hi - lo) if hi > lo else np.zeros_like(f)
        u8 = np.ascontiguousarray(np.flipud((norm * 255).astype(np.uint8)))
        h, w = u8.shape
        img = QImage(u8.data, w, h, w, QImage.Format_Grayscale8).copy()
        return QPixmap.fromImage(img).scaled(
            max(1, self._preview.width()), max(1, self._preview.height()),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)

    # -- filter -------------------------------------------------------
    def _apply_filter(self, text: str):
        text = (text or "").strip().lower()

        def walk(item) -> bool:
            hit = text in item.text(0).lower() if text else True
            shown = False
            for i in range(item.childCount()):
                shown = walk(item.child(i)) or shown
            visible = hit or shown
            item.setHidden(not visible)
            return visible

        for i in range(self._tree.topLevelItemCount()):
            walk(self._tree.topLevelItem(i))
