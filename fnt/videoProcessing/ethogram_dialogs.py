"""Ethogram design dialogs for the Behavior Scoring Studio.

The scoring model lives in ethogram.py; this is the UI for building one.

Modelled on BORIS's ethogram editor, which is the part of that tool worth
copying: a behavior owns several independent MODIFIER SETS, each with its own
selection type and its own values, and each value may carry a shortcut key so
a modifier can be chosen without leaving the keyboard.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QAbstractItemView, QColorDialog, QComboBox,
                             QDialog, QDialogButtonBox, QGroupBox,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QMessageBox,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QTabWidget, QVBoxLayout, QWidget)

from fnt.theme import BLUE_BUTTON_STYLE
from fnt.videoProcessing.ethogram import (
    MODIFIER_TYPES, MULTIPLE, NUMERIC, POINT, SINGLE, STATE, TEXT,
    BehaviorDefinition, ModifierSet, ModifierValue, Subject,
)

TYPE_LABELS = {
    SINGLE: "Single selection (pick one)",
    MULTIPLE: "Multiple selection (pick any)",
    NUMERIC: "Numeric entry",
    TEXT: "Free text",
}


class ModifierSetDialog(QDialog):
    """Define one modifier set: its name, how it is answered, and its values."""

    def __init__(self, parent=None, modifier_set=None):
        super().__init__(parent)
        self.setWindowTitle("Modifier Set")
        self.resize(520, 460)

        layout = QVBoxLayout(self)

        blurb = QLabel(
            "A modifier set is one question asked when the behavior is scored, "
            "e.g. 'Partner' or 'Body region'. A behavior can have several, and "
            "they are answered independently.")
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color:#bbbbbb;")
        layout.addWidget(blurb)

        layout.addWidget(QLabel("Set name:"))
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("e.g. Partner")
        self.edit_name.setToolTip(
            "Becomes a column in the exported CSV, named modifier_<set>.")
        layout.addWidget(self.edit_name)

        layout.addWidget(QLabel("Answered by:"))
        self.combo_type = QComboBox()
        for t in MODIFIER_TYPES:
            self.combo_type.addItem(TYPE_LABELS[t], t)
        self.combo_type.setToolTip(
            "Single selection: exactly one value.\n"
            "Multiple selection: any number, stored comma-separated.\n"
            "Numeric / Free text: typed at scoring time, so no value list.")
        self.combo_type.currentIndexChanged.connect(self._on_type_changed)
        layout.addWidget(self.combo_type)

        self.values_group = QGroupBox("Values")
        vg = QVBoxLayout()
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Value", "Key"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setToolTip(
            "Each value may carry a single-character key, so the modifier can "
            "be answered from the keyboard while scoring. Keys are optional.")
        vg.addWidget(self.table)

        row = QHBoxLayout()
        for label, slot, tip in (
                ("Add", self._add_value, "Add a value to this set."),
                ("Remove", self._remove_value, "Remove the selected value."),
                ("Move Up", lambda: self._move(-1), "Values export in this order."),
                ("Move Down", lambda: self._move(1), "Values export in this order.")):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        vg.addLayout(row)
        self.values_group.setLayout(vg)
        layout.addWidget(self.values_group, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setStyleSheet(BLUE_BUTTON_STYLE)
        layout.addWidget(buttons)

        if modifier_set:
            self.edit_name.setText(modifier_set.name)
            idx = self.combo_type.findData(modifier_set.type)
            self.combo_type.setCurrentIndex(max(0, idx))
            for v in modifier_set.values:
                self._append_row(v.value, v.key)
        self._on_type_changed()

    def _on_type_changed(self):
        # Numeric and free-text sets are typed at scoring time, so a value
        # list would be meaningless.
        self.values_group.setEnabled(
            self.combo_type.currentData() in (SINGLE, MULTIPLE))

    def _append_row(self, value="", key=""):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(value))
        self.table.setItem(r, 1, QTableWidgetItem(key))

    def _add_value(self):
        self._append_row()
        self.table.editItem(self.table.item(self.table.rowCount() - 1, 0))

    def _remove_value(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)

    def _move(self, delta):
        r = self.table.currentRow()
        t = r + delta
        if r < 0 or not (0 <= t < self.table.rowCount()):
            return
        for col in (0, 1):
            a = self.table.item(r, col).text() if self.table.item(r, col) else ""
            b = self.table.item(t, col).text() if self.table.item(t, col) else ""
            self.table.setItem(r, col, QTableWidgetItem(b))
            self.table.setItem(t, col, QTableWidgetItem(a))
        self.table.setCurrentCell(t, 0)

    def _collect(self):
        out = []
        for r in range(self.table.rowCount()):
            value = self.table.item(r, 0).text().strip() if self.table.item(r, 0) else ""
            key = self.table.item(r, 1).text().strip() if self.table.item(r, 1) else ""
            if value:
                out.append(ModifierValue(value, key[:1]))
        return out

    def _validate_and_accept(self):
        if not self.edit_name.text().strip():
            QMessageBox.warning(self, "Name required",
                                "Give the modifier set a name.")
            return
        values = self._collect()
        kind = self.combo_type.currentData()
        if kind in (SINGLE, MULTIPLE) and not values:
            QMessageBox.warning(self, "No values",
                                "A selection set needs at least one value.")
            return
        names = [v.value for v in values]
        if len(names) != len(set(names)):
            QMessageBox.warning(self, "Duplicate values",
                                "Each value must appear only once.")
            return
        keys = [v.key.lower() for v in values if v.key]
        if len(keys) != len(set(keys)):
            QMessageBox.warning(
                self, "Duplicate keys",
                "Two values share a key, so one could never be chosen.")
            return
        self.accept()

    def get_modifier_set(self):
        return ModifierSet(self.edit_name.text().strip(),
                           self.combo_type.currentData(), self._collect())


class BehaviorEditDialog(QDialog):
    """Define one behavior: identity, type, key, category, modifiers, exclusions."""

    def __init__(self, parent=None, behavior=None, existing_keys=None,
                 all_behaviors=None, categories=None):
        super().__init__(parent)
        self.setWindowTitle("Behavior")
        self.resize(560, 620)
        self.existing_keys = {k.lower() for k in (existing_keys or [])}
        self._color = ""
        self._modifier_sets = []

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._build_basics(categories or []), "Behavior")
        tabs.addTab(self._build_modifiers(), "Modifiers")
        tabs.addTab(self._build_exclusions(all_behaviors or [], behavior),
                    "Exclusions")
        layout.addWidget(tabs, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setStyleSheet(BLUE_BUTTON_STYLE)
        layout.addWidget(buttons)

        if behavior:
            self._load(behavior)
        else:
            self._color = BehaviorDefinition().color
        self._update_color_preview()

    # -- tabs --------------------------------------------------------------

    def _build_basics(self, categories):
        page = QWidget()
        v = QVBoxLayout(page)

        v.addWidget(QLabel("Name:"))
        self.edit_name = QLineEdit()
        self.edit_name.setToolTip("Appears in the timeline and the exported CSV.")
        v.addWidget(self.edit_name)

        v.addWidget(QLabel("Description:"))
        self.edit_description = QLineEdit()
        self.edit_description.setPlaceholderText("optional")
        self.edit_description.setToolTip(
            "Free text, so a second scorer knows what counts as this behavior.")
        v.addWidget(self.edit_description)

        v.addWidget(QLabel("Type:"))
        self.combo_type = QComboBox()
        self.combo_type.addItem("Point event (an instant)", POINT)
        self.combo_type.addItem("State event (has a start and a stop)", STATE)
        self.combo_type.setToolTip(
            "Point events mark a moment. State events are toggled on and off "
            "and carry a duration, so only these appear in a time budget.")
        v.addWidget(self.combo_type)

        v.addWidget(QLabel("Key:"))
        self.edit_key = QLineEdit()
        self.edit_key.setMaxLength(1)
        self.edit_key.setToolTip(
            "Single key that scores this behavior. Must not collide with "
            "another behavior OR a subject key -- they share one keyboard.")
        v.addWidget(self.edit_key)

        v.addWidget(QLabel("Category:"))
        self.combo_category = QComboBox()
        self.combo_category.setEditable(True)
        self.combo_category.addItem("")
        for c in categories:
            self.combo_category.addItem(c)
        self.combo_category.setToolTip(
            "Optional grouping such as Social or Locomotion. Time budgets can "
            "be aggregated by category.")
        v.addWidget(self.combo_category)

        row = QHBoxLayout()
        row.addWidget(QLabel("Colour:"))
        self.lbl_color = QLabel()
        self.lbl_color.setFixedSize(60, 22)
        row.addWidget(self.lbl_color)
        pick = QPushButton("Choose...")
        pick.setToolTip("Colour used for this behavior in the timeline.")
        pick.clicked.connect(self._pick_color)
        row.addWidget(pick)
        row.addStretch()
        v.addLayout(row)
        v.addStretch()
        return page

    def _build_modifiers(self):
        page = QWidget()
        v = QVBoxLayout(page)
        blurb = QLabel(
            "Questions asked when this behavior is scored. Each set is "
            "answered separately and exports to its own column, so 'Partner' "
            "and 'Body region' stay distinct rather than merging into one "
            "string.")
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color:#bbbbbb;")
        v.addWidget(blurb)

        self.list_sets = QListWidget()
        self.list_sets.setToolTip("Modifier sets belonging to this behavior.")
        self.list_sets.itemDoubleClicked.connect(lambda _: self._edit_set())
        v.addWidget(self.list_sets, 1)

        row = QHBoxLayout()
        for label, slot, tip in (
                ("Add Set", self._add_set, "Define a new modifier set."),
                ("Edit Set", self._edit_set, "Edit the selected set."),
                ("Remove Set", self._remove_set, "Delete the selected set.")):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        v.addLayout(row)
        return page

    def _build_exclusions(self, all_behaviors, behavior):
        page = QWidget()
        v = QVBoxLayout(page)
        blurb = QLabel(
            "State behaviors that this one ends. Starting this behavior "
            "automatically stops anything ticked here for the same subject.\n\n"
            "Exclusion is mutual, so it only needs stating once. This is what "
            "keeps mutually exclusive states from being left open by mistake.")
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color:#bbbbbb;")
        v.addWidget(blurb)

        self.list_exclusions = QListWidget()
        self.list_exclusions.setToolTip(
            "Tick the behaviors that cannot run at the same time as this one.")
        current = behavior.name if behavior else ""
        for b in all_behaviors:
            if b.name == current or b.event_type != STATE:
                continue          # a point event has nothing to stop
            item = QListWidgetItem(b.name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list_exclusions.addItem(item)
        v.addWidget(self.list_exclusions, 1)
        return page

    # -- modifier sets -----------------------------------------------------

    def _refresh_sets(self):
        self.list_sets.clear()
        for ms in self._modifier_sets:
            detail = (", ".join(ms.value_names[:6]) if ms.needs_values
                      else TYPE_LABELS[ms.type])
            if ms.needs_values and len(ms.values) > 6:
                detail += ", ..."
            self.list_sets.addItem(f"{ms.name}  [{ms.type}]  {detail}")

    def _add_set(self):
        dlg = ModifierSetDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            ms = dlg.get_modifier_set()
            if any(s.name == ms.name for s in self._modifier_sets):
                QMessageBox.warning(self, "Duplicate set",
                                    f"This behavior already has a set called "
                                    f"'{ms.name}'.")
                return
            self._modifier_sets.append(ms)
            self._refresh_sets()

    def _edit_set(self):
        r = self.list_sets.currentRow()
        if r < 0:
            return
        dlg = ModifierSetDialog(self, self._modifier_sets[r])
        if dlg.exec_() == QDialog.Accepted:
            self._modifier_sets[r] = dlg.get_modifier_set()
            self._refresh_sets()

    def _remove_set(self):
        r = self.list_sets.currentRow()
        if r >= 0:
            del self._modifier_sets[r]
            self._refresh_sets()

    # -- colour ------------------------------------------------------------

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(self._color or "#ff6b35"), self)
        if c.isValid():
            self._color = c.name()
            self._update_color_preview()

    def _update_color_preview(self):
        self.lbl_color.setStyleSheet(
            f"background-color:{self._color}; border:1px solid #555;")

    # -- load / save -------------------------------------------------------

    def _load(self, behavior):
        self.edit_name.setText(behavior.name)
        self.edit_description.setText(behavior.description)
        self.edit_key.setText(behavior.key)
        self.combo_type.setCurrentIndex(
            max(0, self.combo_type.findData(behavior.event_type)))
        self.combo_category.setCurrentText(behavior.category)
        self._color = behavior.color
        self._modifier_sets = [ModifierSet.from_dict(s.to_dict())
                               for s in behavior.modifier_sets]
        self._refresh_sets()
        for i in range(self.list_exclusions.count()):
            item = self.list_exclusions.item(i)
            if item.text() in behavior.exclusions:
                item.setCheckState(Qt.Checked)

    def _validate_and_accept(self):
        name = self.edit_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Name required", "Give the behavior a name.")
            return
        key = self.edit_key.text().strip()
        if key and key.lower() in self.existing_keys:
            QMessageBox.warning(
                self, "Key already used",
                f"'{key}' is already bound to another behavior or subject.")
            return
        self.accept()

    def get_behavior(self):
        exclusions = [self.list_exclusions.item(i).text()
                      for i in range(self.list_exclusions.count())
                      if self.list_exclusions.item(i).checkState() == Qt.Checked]
        return BehaviorDefinition(
            name=self.edit_name.text().strip(),
            key=self.edit_key.text().strip(),
            event_type=self.combo_type.currentData(),
            color=self._color,
            modifier_sets=self._modifier_sets,
            category=self.combo_category.currentText().strip(),
            description=self.edit_description.text().strip(),
            exclusions=exclusions,
        )


class SubjectEditDialog(QDialog):
    """Define one subject."""

    def __init__(self, parent=None, subject=None, existing_keys=None):
        super().__init__(parent)
        self.setWindowTitle("Subject")
        self.resize(400, 220)
        self.existing_keys = {k.lower() for k in (existing_keys or [])}

        v = QVBoxLayout(self)
        v.addWidget(QLabel("Name:"))
        self.edit_name = QLineEdit()
        self.edit_name.setToolTip(
            "Identifier for the animal, e.g. M1. Recorded on every event "
            "scored while this subject is in focus.")
        v.addWidget(self.edit_name)

        v.addWidget(QLabel("Key:"))
        self.edit_key = QLineEdit()
        self.edit_key.setMaxLength(1)
        self.edit_key.setToolTip(
            "Single key that switches focus to this subject while scoring. "
            "Must not collide with a behavior key.")
        v.addWidget(self.edit_key)

        v.addWidget(QLabel("Description:"))
        self.edit_description = QLineEdit()
        self.edit_description.setPlaceholderText("optional")
        v.addWidget(self.edit_description)
        v.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setStyleSheet(BLUE_BUTTON_STYLE)
        v.addWidget(buttons)

        if subject:
            self.edit_name.setText(subject.name)
            self.edit_key.setText(subject.key)
            self.edit_description.setText(subject.description)

    def _validate_and_accept(self):
        if not self.edit_name.text().strip():
            QMessageBox.warning(self, "Name required", "Give the subject a name.")
            return
        key = self.edit_key.text().strip()
        if key and key.lower() in self.existing_keys:
            QMessageBox.warning(
                self, "Key already used",
                f"'{key}' is already bound to another subject or behavior.")
            return
        self.accept()

    def get_subject(self):
        return Subject(self.edit_name.text().strip(),
                       self.edit_key.text().strip(),
                       self.edit_description.text().strip())


class ModifierPromptDialog(QDialog):
    """Ask a behavior's modifier sets at scoring time.

    Keyboard first: a value with a key is chosen by pressing it, and when every
    set is answered the dialog closes itself. Scoring should not require the
    mouse.
    """

    def __init__(self, parent=None, behavior=None):
        super().__init__(parent)
        self.behavior = behavior
        self.setWindowTitle(f"Modifiers - {behavior.name}")
        self.resize(430, 130 + 120 * len(behavior.modifier_sets))
        self._widgets = {}

        v = QVBoxLayout(self)
        for ms in behavior.modifier_sets:
            box = QGroupBox(f"{ms.name}  ({TYPE_LABELS[ms.type]})")
            bv = QVBoxLayout()
            if ms.needs_values:
                lst = QListWidget()
                lst.setSelectionMode(
                    QAbstractItemView.MultiSelection if ms.type == MULTIPLE
                    else QAbstractItemView.SingleSelection)
                for val in ms.values:
                    lst.addItem(f"{val.value}   [{val.key}]" if val.key else val.value)
                bv.addWidget(lst)
                self._widgets[ms.name] = ("list", lst, ms)
            else:
                edit = QLineEdit()
                edit.setPlaceholderText(
                    "number" if ms.type == NUMERIC else "text")
                bv.addWidget(edit)
                self._widgets[ms.name] = ("edit", edit, ms)
            box.setLayout(bv)
            v.addWidget(box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setStyleSheet(BLUE_BUTTON_STYLE)
        v.addWidget(buttons)

    def keyPressEvent(self, event):
        """Select by value key, and close once every set has an answer."""
        text = event.text()
        if text:
            for kind, widget, ms in self._widgets.values():
                if kind != "list":
                    continue
                value = ms.value_for_key(text)
                if value is None:
                    continue
                for i in range(widget.count()):
                    if widget.item(i).text().split("   [")[0] == value:
                        if ms.type == MULTIPLE:
                            widget.item(i).setSelected(
                                not widget.item(i).isSelected())
                        else:
                            widget.setCurrentRow(i)
                        break
                if self._all_answered():
                    self.accept()
                return
        super().keyPressEvent(event)

    def _all_answered(self):
        for kind, widget, ms in self._widgets.values():
            if kind == "list" and not widget.selectedItems():
                return False
        return True

    def get_modifiers(self):
        out = {}
        for name, (kind, widget, ms) in self._widgets.items():
            if kind == "list":
                chosen = [i.text().split("   [")[0] for i in widget.selectedItems()]
                out[name] = ms.normalise(chosen)
            else:
                out[name] = ms.normalise(widget.text().strip())
        return out
