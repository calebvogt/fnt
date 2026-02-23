"""
GitHub CSV Transfer Tool
Transfer CSV/TXT/JSON files to a GitHub repository with automatic splitting for large files.

This tool:
- Recursively copies selected file types from source to destination
- Preserves folder structure
- Automatically splits files larger than the configured threshold
- Uses CSV-aware splitting to preserve headers in each chunk
- Optional file curation via tree-view explorer with checkboxes
"""

import os
import csv
import shutil
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QGroupBox, QFileDialog, QSpinBox, QCheckBox, QProgressBar, QTextEdit,
    QMessageBox, QDialog, QSizePolicy, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QStyle
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# File extensions considered for each type
FILE_TYPE_MAP = {
    'CSV': ['.csv', '.tsv'],
    'TXT': ['.txt'],
    'JSON': ['.json'],
}


class SourceCurationDialog(QDialog):
    """File explorer-style dialog for curating which files/folders to transfer.

    Displays the source directory tree with checkboxes. Users can check/uncheck
    folders and files. Unchecking a parent cascades to all children. File type
    checkboxes filter which files are visible in real time.
    """

    def __init__(self, source_dir, preselected_paths=None, parent=None):
        super().__init__(parent)
        self.source_dir = source_dir
        self.preselected_paths = preselected_paths  # None = all checked, list = only these checked
        self.setWindowTitle("Curate Source Files")
        self.setModal(True)
        self.setMinimumSize(700, 500)
        self.resize(800, 600)
        self._updating = False  # Guard against recursive signal loops

        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #cccccc;
                font-family: Arial;
            }
            QTreeWidget {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                color: #cccccc;
                font-family: Consolas, monospace;
                font-size: 10pt;
            }
            QTreeWidget::item:selected {
                background-color: #264f78;
                color: white;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1e90ff;
            }
            QLabel {
                color: #cccccc;
            }
            QCheckBox {
                color: #cccccc;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #3c3c3c;
                padding: 4px;
                font-weight: bold;
            }
        """)

        layout = QVBoxLayout()

        # Title
        title = QLabel(f"Select files to transfer from: {os.path.basename(source_dir)}")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet("color: #0078d4;")
        title.setWordWrap(True)
        layout.addWidget(title)

        # File type filters
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("File types:"))

        self.csv_checkbox = QCheckBox("CSV")
        self.csv_checkbox.setChecked(True)
        self.csv_checkbox.stateChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.csv_checkbox)

        self.txt_checkbox = QCheckBox("TXT")
        self.txt_checkbox.setChecked(True)
        self.txt_checkbox.stateChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.txt_checkbox)

        self.json_checkbox = QCheckBox("JSON")
        self.json_checkbox.setChecked(True)
        self.json_checkbox.stateChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.json_checkbox)

        filter_layout.addStretch()

        # Select All / Deselect All buttons
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        filter_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all)
        filter_layout.addWidget(deselect_all_btn)

        layout.addLayout(filter_layout)

        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Size"])
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.setAnimated(True)
        layout.addWidget(self.tree)

        # Status label
        self.status_label = QLabel("Loading...")
        self.status_label.setStyleSheet("color: #999999; font-size: 10pt;")
        layout.addWidget(self.status_label)

        # Confirm button
        confirm_btn = QPushButton("Confirm Selection")
        confirm_btn.setMinimumHeight(40)
        confirm_btn.clicked.connect(self.accept)
        layout.addWidget(confirm_btn)

        self.setLayout(layout)

        # Populate tree
        self._populate_tree()

        # Connect item changed signal for cascade logic
        self.tree.itemChanged.connect(self._on_item_changed)

        # Expand first level and update counts
        self.tree.expandToDepth(0)
        self._update_status()

    def _get_visible_extensions(self):
        """Get set of file extensions currently enabled by the filter checkboxes."""
        exts = set()
        if self.csv_checkbox.isChecked():
            exts.update(FILE_TYPE_MAP['CSV'])
        if self.txt_checkbox.isChecked():
            exts.update(FILE_TYPE_MAP['TXT'])
        if self.json_checkbox.isChecked():
            exts.update(FILE_TYPE_MAP['JSON'])
        return exts

    def _on_filter_changed(self):
        """Rebuild the tree when file type filters change."""
        self._populate_tree()
        self.tree.expandToDepth(0)
        self._update_status()

    def _populate_tree(self):
        """Build the tree from the source directory, showing only matching file types."""
        self._updating = True
        self.tree.clear()

        folder_icon = self.style().standardIcon(QStyle.SP_DirIcon)
        file_icon = self.style().standardIcon(QStyle.SP_FileIcon)

        visible_exts = self._get_visible_extensions()

        # Build a set of preselected paths for fast lookup
        preselected_set = None
        if self.preselected_paths is not None:
            preselected_set = set(self.preselected_paths)

        # Map directory paths to tree items
        dir_items = {}

        # Root item is the source folder itself
        root_item = QTreeWidgetItem(self.tree)
        root_item.setText(0, os.path.basename(self.source_dir))
        root_item.setIcon(0, folder_icon)
        root_item.setFlags(root_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
        root_item.setCheckState(0, Qt.Checked)
        root_item.setData(0, Qt.UserRole, "")  # rel_path = "" for root
        dir_items[self.source_dir] = root_item

        for dirpath, dirnames, filenames in os.walk(self.source_dir):
            # Sort for consistent display
            dirnames.sort(key=str.lower)
            filenames.sort(key=str.lower)

            # Get parent item
            parent_item = dir_items.get(dirpath, root_item)

            # Add subdirectories
            for dirname in dirnames:
                full_path = os.path.join(dirpath, dirname)
                rel_path = os.path.relpath(full_path, self.source_dir)

                dir_item = QTreeWidgetItem(parent_item)
                dir_item.setText(0, dirname)
                dir_item.setIcon(0, folder_icon)
                dir_item.setFlags(dir_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
                dir_item.setCheckState(0, Qt.Checked)
                dir_item.setData(0, Qt.UserRole, rel_path)
                dir_items[full_path] = dir_item

            # Add files (only those matching visible extensions)
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in visible_exts:
                    continue

                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, self.source_dir)
                file_size = os.path.getsize(full_path)

                file_item = QTreeWidgetItem(parent_item)
                file_item.setText(0, filename)
                file_item.setText(1, self._format_size(file_size))
                file_item.setIcon(0, file_icon)
                file_item.setFlags(file_item.flags() | Qt.ItemIsUserCheckable)
                file_item.setData(0, Qt.UserRole, rel_path)
                file_item.setData(0, Qt.UserRole + 1, 'file')

                # Determine check state
                if preselected_set is not None:
                    file_item.setCheckState(0, Qt.Checked if rel_path in preselected_set else Qt.Unchecked)
                else:
                    file_item.setCheckState(0, Qt.Checked)

        # Remove empty folders (folders that have no visible file descendants)
        self._prune_empty_folders(root_item)

        # If preselected paths were provided, update parent states bottom-up
        if preselected_set is not None:
            self._update_parent_states(root_item)

        self._updating = False

    def _prune_empty_folders(self, item):
        """Recursively remove folder items that contain no visible files."""
        i = item.childCount() - 1
        while i >= 0:
            child = item.child(i)
            is_file = child.data(0, Qt.UserRole + 1) == 'file'
            if not is_file:
                # It's a folder — prune its children first, then check if empty
                self._prune_empty_folders(child)
                if child.childCount() == 0:
                    item.removeChild(child)
            i -= 1

    def _update_parent_states(self, item):
        """Recursively update parent check states based on children (bottom-up)."""
        for i in range(item.childCount()):
            child = item.child(i)
            if child.childCount() > 0:
                self._update_parent_states(child)

        if item.childCount() > 0:
            checked = 0
            unchecked = 0
            partial = 0
            for i in range(item.childCount()):
                state = item.child(i).checkState(0)
                if state == Qt.Checked:
                    checked += 1
                elif state == Qt.Unchecked:
                    unchecked += 1
                else:
                    partial += 1

            if partial > 0 or (checked > 0 and unchecked > 0):
                item.setCheckState(0, Qt.PartiallyChecked)
            elif checked > 0:
                item.setCheckState(0, Qt.Checked)
            else:
                item.setCheckState(0, Qt.Unchecked)

    def _on_item_changed(self, item, column):
        """Handle checkbox state changes with cascade logic."""
        if self._updating or column != 0:
            return

        self._updating = True

        state = item.checkState(0)

        # Cascade to children (only for fully checked/unchecked, not partial)
        if state != Qt.PartiallyChecked:
            self._set_children_state(item, state)

        # Update parent states
        self._update_ancestors(item)

        self._updating = False
        self._update_status()

    def _set_children_state(self, item, state):
        """Recursively set all children to the given check state."""
        for i in range(item.childCount()):
            child = item.child(i)
            child.setCheckState(0, state)
            self._set_children_state(child, state)

    def _update_ancestors(self, item):
        """Update ancestor check states based on sibling states."""
        parent = item.parent()
        if parent is None:
            return

        checked = 0
        unchecked = 0
        partial = 0
        for i in range(parent.childCount()):
            state = parent.child(i).checkState(0)
            if state == Qt.Checked:
                checked += 1
            elif state == Qt.Unchecked:
                unchecked += 1
            else:
                partial += 1

        if partial > 0 or (checked > 0 and unchecked > 0):
            parent.setCheckState(0, Qt.PartiallyChecked)
        elif checked > 0:
            parent.setCheckState(0, Qt.Checked)
        else:
            parent.setCheckState(0, Qt.Unchecked)

        # Continue up the tree
        self._update_ancestors(parent)

    def _update_status(self):
        """Update the status label with counts of selected folders and files."""
        self._folder_count = 0
        self._file_count = 0
        self._count_items(self.tree.invisibleRootItem())
        self.status_label.setText(f"{self._folder_count} folder(s) and {self._file_count} file(s) selected")

    def _count_items(self, item):
        """Walk tree and count checked folders and files."""
        for i in range(item.childCount()):
            child = item.child(i)
            is_file = child.data(0, Qt.UserRole + 1) == 'file'
            if is_file:
                if child.checkState(0) == Qt.Checked:
                    self._file_count += 1
            else:
                if child.checkState(0) in (Qt.Checked, Qt.PartiallyChecked):
                    self._folder_count += 1
            self._count_items(child)

    def select_all(self):
        """Check all items."""
        self._updating = True
        root = self.tree.topLevelItem(0)
        if root:
            root.setCheckState(0, Qt.Checked)
            self._set_children_state(root, Qt.Checked)
        self._updating = False
        self._update_status()

    def deselect_all(self):
        """Uncheck all items."""
        self._updating = True
        root = self.tree.topLevelItem(0)
        if root:
            root.setCheckState(0, Qt.Unchecked)
            self._set_children_state(root, Qt.Unchecked)
        self._updating = False
        self._update_status()

    def get_selected_paths(self):
        """Return list of relative file paths that are checked."""
        selected = []
        self._collect_selected(self.tree.invisibleRootItem(), selected)
        return selected

    def get_selected_file_types(self):
        """Return list of file extensions currently enabled in the filter."""
        return list(self._get_visible_extensions())

    def _collect_selected(self, item, selected):
        """Recursively collect checked file paths."""
        for i in range(item.childCount()):
            child = item.child(i)
            is_file = child.data(0, Qt.UserRole + 1) == 'file'
            if is_file and child.checkState(0) == Qt.Checked:
                rel_path = child.data(0, Qt.UserRole)
                selected.append(rel_path)
            # Recurse into folders regardless of state (partial folders contain checked files)
            self._collect_selected(child, selected)

    @staticmethod
    def _format_size(size_bytes):
        """Format file size as human-readable string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / 1024 / 1024:.1f} MB"
        else:
            return f"{size_bytes / 1024 / 1024 / 1024:.1f} GB"


class ConflictDialog(QDialog):
    """Dialog for handling file conflicts with Skip/Overwrite options."""

    # Constants for return values
    SKIP = 0
    OVERWRITE = 1
    SKIP_ALL = 2
    OVERWRITE_ALL = 3
    CANCEL = 4

    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("File Conflict")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # Message
        message = QLabel(f"The following file already exists in the destination:\n\n{file_path}\n\nWhat would you like to do?")
        message.setWordWrap(True)
        layout.addWidget(message)

        # Buttons
        button_layout = QHBoxLayout()

        skip_btn = QPushButton("Skip")
        skip_btn.clicked.connect(lambda: self.done(self.SKIP))
        button_layout.addWidget(skip_btn)

        overwrite_btn = QPushButton("Overwrite")
        overwrite_btn.clicked.connect(lambda: self.done(self.OVERWRITE))
        button_layout.addWidget(overwrite_btn)

        skip_all_btn = QPushButton("Skip All")
        skip_all_btn.clicked.connect(lambda: self.done(self.SKIP_ALL))
        button_layout.addWidget(skip_all_btn)

        overwrite_all_btn = QPushButton("Overwrite All")
        overwrite_all_btn.clicked.connect(lambda: self.done(self.OVERWRITE_ALL))
        button_layout.addWidget(overwrite_all_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(lambda: self.done(self.CANCEL))
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)


class TransferWorker(QThread):
    """Worker thread for file transfer operations."""

    progress = pyqtSignal(int, int, str)  # current, total, message
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)  # summary stats
    conflict = pyqtSignal(str, object)  # file_path, callback

    def __init__(self, source_dir, dest_dir, file_types, max_size_mb, curated_paths=None):
        super().__init__()
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.file_types = file_types  # list of extensions like ['.csv', '.txt', '.json']
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.curated_paths = curated_paths  # None = all files, list = only these relative paths
        self.conflict_resolution = None  # None, 'skip_all', or 'overwrite_all'
        self.pending_conflict_response = None
        self.cancelled = False

    def set_conflict_response(self, response):
        """Set the response for a pending conflict."""
        self.pending_conflict_response = response

    def cancel(self):
        """Cancel the transfer operation."""
        self.cancelled = True

    def run(self):
        """Execute the transfer operation."""
        stats = {
            'files_copied': 0,
            'files_split': 0,
            'files_skipped': 0,
            'total_size': 0,
            'split_files_list': [],
            'skipped_files_list': []
        }

        # Get source folder name to include in destination
        source_folder_name = os.path.basename(self.source_dir)

        # Build curated set for fast lookup
        curated_set = set(self.curated_paths) if self.curated_paths is not None else None

        # Collect all files to process
        files_to_process = []
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                source_path = os.path.join(root, file)
                rel_path = os.path.relpath(source_path, self.source_dir)

                # Check against curated paths if set
                if curated_set is not None and rel_path not in curated_set:
                    continue

                # Check file type filter
                ext = os.path.splitext(file)[1].lower()
                if ext in self.file_types:
                    # Include source folder name in destination path
                    dest_path = os.path.join(self.dest_dir, source_folder_name, rel_path)
                    display_path = os.path.join(source_folder_name, rel_path)
                    files_to_process.append((source_path, dest_path, display_path))

        total_files = len(files_to_process)
        self.log.emit(f"Found {total_files} files to transfer")

        for i, (source_path, dest_path, rel_path) in enumerate(files_to_process):
            if self.cancelled:
                self.log.emit("Transfer cancelled by user")
                break

            self.progress.emit(i + 1, total_files, f"Processing: {rel_path}")

            try:
                # Create destination directory if needed
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)

                # Check for conflicts
                if os.path.exists(dest_path):
                    if self.conflict_resolution == 'skip_all':
                        self.log.emit(f"⏭️ Skipped (exists): {rel_path}")
                        stats['files_skipped'] += 1
                        stats['skipped_files_list'].append(rel_path)
                        continue
                    elif self.conflict_resolution == 'overwrite_all':
                        pass  # Continue to overwrite
                    else:
                        # Need to ask user
                        self.pending_conflict_response = None
                        self.conflict.emit(dest_path, self)

                        # Wait for response
                        while self.pending_conflict_response is None and not self.cancelled:
                            self.msleep(100)

                        if self.cancelled:
                            break

                        response = self.pending_conflict_response

                        if response == ConflictDialog.CANCEL:
                            self.cancelled = True
                            break
                        elif response == ConflictDialog.SKIP:
                            self.log.emit(f"⏭️ Skipped: {rel_path}")
                            stats['files_skipped'] += 1
                            stats['skipped_files_list'].append(rel_path)
                            continue
                        elif response == ConflictDialog.SKIP_ALL:
                            self.conflict_resolution = 'skip_all'
                            self.log.emit(f"⏭️ Skipped: {rel_path}")
                            stats['files_skipped'] += 1
                            stats['skipped_files_list'].append(rel_path)
                            continue
                        elif response == ConflictDialog.OVERWRITE_ALL:
                            self.conflict_resolution = 'overwrite_all'
                        # OVERWRITE falls through to copy

                # Check file size
                file_size = os.path.getsize(source_path)
                stats['total_size'] += file_size

                if file_size >= self.max_size_bytes:
                    # Need to split
                    self.log.emit(f"✂️ Splitting: {rel_path} ({file_size / 1024 / 1024:.1f}MB)")
                    chunks = self.split_and_copy_file(source_path, dest_path)
                    stats['files_split'] += 1
                    stats['split_files_list'].append(f"{rel_path} ({chunks} parts)")
                else:
                    # Direct copy
                    shutil.copy2(source_path, dest_path)
                    self.log.emit(f"✅ Copied: {rel_path}")
                    stats['files_copied'] += 1

            except Exception as e:
                self.log.emit(f"❌ Error processing {rel_path}: {str(e)}")
                stats['files_skipped'] += 1
                stats['skipped_files_list'].append(f"{rel_path} (error)")

        self.finished.emit(stats)

    def split_and_copy_file(self, source_path, dest_path):
        """Split a file and copy chunks to destination."""
        ext = os.path.splitext(source_path)[1].lower()

        if ext in ['.csv', '.tsv']:
            return self.split_csv_file(source_path, dest_path)
        else:
            return self.split_binary_file(source_path, dest_path)

    def split_csv_file(self, source_path, dest_path):
        """Split CSV file by rows to preserve data structure."""
        dest_dir = os.path.dirname(dest_path)
        file_name = os.path.basename(dest_path)
        base_name, ext = os.path.splitext(file_name)

        chunk_number = 1
        current_size = 0
        current_rows = []
        header_row = None

        # Detect CSV dialect and read file
        with open(source_path, 'r', newline='', encoding='utf-8') as input_file:
            sample = input_file.read(8192)
            input_file.seek(0)

            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel

            reader = csv.reader(input_file, dialect)

            try:
                header_row = next(reader)
                header_size = len(','.join(header_row).encode('utf-8'))
            except StopIteration:
                return 0

            row_count = 0
            for row in reader:
                row_count += 1
                row_text = ','.join(str(cell) for cell in row)
                row_size = len(row_text.encode('utf-8'))

                if (current_size + row_size + header_size) > self.max_size_bytes and current_rows:
                    self.write_csv_chunk(dest_dir, base_name, ext, chunk_number, header_row, current_rows)
                    chunk_number += 1
                    current_rows = []
                    current_size = header_size

                current_rows.append(row)
                current_size += row_size

        if current_rows:
            self.write_csv_chunk(dest_dir, base_name, ext, chunk_number, header_row, current_rows)

        # Create info file
        self.create_csv_info_file(dest_dir, base_name, ext, source_path, chunk_number, row_count)

        return chunk_number

    def write_csv_chunk(self, dest_dir, base_name, ext, chunk_number, header_row, data_rows):
        """Write a single CSV chunk with header."""
        chunk_filename = f"{base_name}.part{chunk_number:03d}{ext}"
        chunk_path = os.path.join(dest_dir, chunk_filename)

        with open(chunk_path, 'w', newline='', encoding='utf-8') as chunk_file:
            writer = csv.writer(chunk_file)
            writer.writerow(header_row)
            writer.writerows(data_rows)

        chunk_size = os.path.getsize(chunk_path)
        self.log.emit(f"   Created: {chunk_filename} ({chunk_size / 1024 / 1024:.1f}MB, {len(data_rows):,} rows)")

    def create_csv_info_file(self, dest_dir, base_name, ext, source_path, total_chunks, total_rows):
        """Create info file for CSV splits."""
        info_filename = f"{base_name}.split_info.txt"
        info_path = os.path.join(dest_dir, info_filename)

        original_size = os.path.getsize(source_path)

        with open(info_path, 'w') as info_file:
            info_file.write(f"File Type: CSV (Smart Split)\n")
            info_file.write(f"Original file: {os.path.basename(source_path)}\n")
            info_file.write(f"Original size: {original_size} bytes ({original_size / 1024 / 1024:.1f}MB)\n")
            info_file.write(f"Total chunks: {total_chunks}\n")
            info_file.write(f"Total rows: {total_rows:,}\n")
            info_file.write(f"Split method: Row-based (preserves data structure)\n")
            info_file.write(f"Headers: Included in each chunk\n")
            info_file.write(f"\nNote: Each chunk contains headers and can be processed independently.\n")
            info_file.write(f"\nChunk files:\n")
            for i in range(1, total_chunks + 1):
                chunk_name = f"{base_name}.part{i:03d}{ext}"
                info_file.write(f"  {chunk_name}\n")
            info_file.write(f"\nTo rejoin:\n")
            info_file.write(f"- Python: pd.concat([pd.read_csv(f) for f in chunk_files])\n")
            info_file.write(f"- R: rbind(read.csv('part001'), read.csv('part002'), ...)\n")

    def split_binary_file(self, source_path, dest_path):
        """Split non-CSV files using byte-level splitting."""
        dest_dir = os.path.dirname(dest_path)
        file_name = os.path.basename(dest_path)
        base_name, ext = os.path.splitext(file_name)

        chunk_number = 1

        with open(source_path, 'rb') as input_file:
            while True:
                chunk_data = input_file.read(self.max_size_bytes)
                if not chunk_data:
                    break

                chunk_filename = f"{base_name}.part{chunk_number:03d}{ext}"
                chunk_path = os.path.join(dest_dir, chunk_filename)

                with open(chunk_path, 'wb') as chunk_file:
                    chunk_file.write(chunk_data)

                self.log.emit(f"   Created: {chunk_filename} ({len(chunk_data) / 1024 / 1024:.1f}MB)")
                chunk_number += 1

        # Create info file
        total_chunks = chunk_number - 1
        info_filename = f"{base_name}.split_info.txt"
        info_path = os.path.join(dest_dir, info_filename)

        original_size = os.path.getsize(source_path)

        with open(info_path, 'w') as info_file:
            info_file.write(f"File Type: Binary\n")
            info_file.write(f"Original file: {os.path.basename(source_path)}\n")
            info_file.write(f"Original size: {original_size} bytes ({original_size / 1024 / 1024:.1f}MB)\n")
            info_file.write(f"Total chunks: {total_chunks}\n")
            info_file.write(f"Chunk size: {self.max_size_bytes} bytes\n")
            info_file.write(f"\nTo rejoin: concatenate all .partXXX files in order\n")

        return total_chunks


class GitHubCSVTransferWindow(QWidget):
    """Main window for GitHub CSV Transfer tool."""

    def __init__(self):
        super().__init__()
        self.worker = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("GitHub CSV Transfer - FNT")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(600, 500)

        # Apply dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                font-family: Arial;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3c3c3c;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #0078d4;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1e90ff;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666666;
            }
            QLineEdit, QSpinBox {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                padding: 5px;
                color: #cccccc;
            }
            QCheckBox {
                color: #cccccc;
            }
            QProgressBar {
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
            QTextEdit {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                color: #cccccc;
                font-family: Consolas, monospace;
            }
        """)

        layout = QVBoxLayout()

        # Title
        title = QLabel("GitHub CSV Transfer")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #0078d4;")
        layout.addWidget(title)

        # Description
        desc = QLabel("Transfer CSV, TXT, and JSON files to a GitHub repository with automatic splitting for large files.")
        desc.setFont(QFont("Arial", 10))
        desc.setStyleSheet("color: #999999; font-style: italic; margin-bottom: 10px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Folder Selection Group
        folder_group = QGroupBox("Folder Selection")
        folder_layout = QGridLayout()

        # Source folder
        folder_layout.addWidget(QLabel("Source Folder:"), 0, 0)
        self.source_label = QLabel("No folder selected")
        self.source_label.setStyleSheet("color: #999999;")
        folder_layout.addWidget(self.source_label, 0, 1)

        source_btn_layout = QHBoxLayout()
        self.source_btn = QPushButton("Browse...")
        self.source_btn.clicked.connect(self.select_source_folder)
        source_btn_layout.addWidget(self.source_btn)

        self.curate_btn = QPushButton("Curate...")
        self.curate_btn.clicked.connect(self.open_curation_dialog)
        self.curate_btn.setEnabled(False)
        self.curate_btn.setToolTip("Select which files and folders to include in the transfer")
        source_btn_layout.addWidget(self.curate_btn)
        folder_layout.addLayout(source_btn_layout, 0, 2)

        # Destination folder
        folder_layout.addWidget(QLabel("Destination (GitHub Repo):"), 1, 0)
        self.dest_label = QLabel("No folder selected")
        self.dest_label.setStyleSheet("color: #999999;")
        folder_layout.addWidget(self.dest_label, 1, 1)
        self.dest_btn = QPushButton("Browse...")
        self.dest_btn.clicked.connect(self.select_dest_folder)
        folder_layout.addWidget(self.dest_btn, 1, 2)

        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)

        # Configuration Group (split threshold only — file types are in curation dialog)
        config_group = QGroupBox("Transfer Configuration")
        config_layout = QHBoxLayout()

        config_layout.addWidget(QLabel("Split files larger than:"))
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setRange(1, 100)
        self.size_spinbox.setValue(45)
        self.size_spinbox.setSuffix(" MB")
        self.size_spinbox.setToolTip("Files larger than this will be automatically split (GitHub limit is 100MB, recommended 45MB)")
        config_layout.addWidget(self.size_spinbox)
        config_layout.addStretch()

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Progress Group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to transfer")
        self.status_label.setStyleSheet("color: #999999;")
        progress_layout.addWidget(self.status_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Log Group
        log_group = QGroupBox("Transfer Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.transfer_btn = QPushButton("Start Transfer")
        self.transfer_btn.clicked.connect(self.start_transfer)
        self.transfer_btn.setMinimumHeight(40)
        button_layout.addWidget(self.transfer_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_transfer)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setMinimumHeight(40)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Store folder paths and curation state
        self.source_folder = None
        self.dest_folder = None
        self.curated_paths = None  # None = transfer everything, list = only these relative paths
        self.curated_file_types = None  # File types selected in curation dialog

    def select_source_folder(self):
        """Select source folder for transfer."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Source Folder",
            "",
            QFileDialog.ShowDirsOnly
        )
        if folder:
            self.source_folder = folder
            self.source_label.setText(folder)
            self.source_label.setStyleSheet("color: #4ec9b0;")
            self.curated_paths = None  # Reset curation on new folder
            self.curated_file_types = None
            self.curate_btn.setEnabled(True)
            self.log_text.append(f"Source folder: {folder}")

            # Ask if user wants to curate
            reply = QMessageBox.question(
                self,
                "Curate Source Files",
                "Would you like to curate which files and folders are transferred?\n\n"
                "If No, all CSV, TXT, and JSON files will be transferred.",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.open_curation_dialog()

    def open_curation_dialog(self):
        """Open the file curation dialog."""
        if not self.source_folder:
            return

        dialog = SourceCurationDialog(self.source_folder, self.curated_paths, self)
        if dialog.exec_() == QDialog.Accepted:
            self.curated_paths = dialog.get_selected_paths()
            self.curated_file_types = dialog.get_selected_file_types()
            count = len(self.curated_paths)
            self.log_text.append(f"Curated selection: {count} file(s) selected for transfer")
            self.source_label.setText(f"{self.source_folder}  ({count} files curated)")
            self.source_label.setStyleSheet("color: #4ec9b0;")

    def select_dest_folder(self):
        """Select destination folder (GitHub repo)."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Destination Folder (GitHub Repository)",
            "",
            QFileDialog.ShowDirsOnly
        )
        if folder:
            self.dest_folder = folder
            self.dest_label.setText(folder)
            self.dest_label.setStyleSheet("color: #4ec9b0;")
            self.log_text.append(f"Destination folder: {folder}")

    def _get_file_types_for_transfer(self):
        """Get list of file extensions to transfer.

        If the user curated, use the file types from the curation dialog.
        Otherwise, default to all three types (CSV/TSV, TXT, JSON).
        """
        if self.curated_file_types is not None:
            return self.curated_file_types
        # Default: all types
        all_types = []
        for exts in FILE_TYPE_MAP.values():
            all_types.extend(exts)
        return all_types

    def start_transfer(self):
        """Start the file transfer operation."""
        # Validate inputs
        if not self.source_folder:
            QMessageBox.warning(self, "Missing Input", "Please select a source folder.")
            return

        if not self.dest_folder:
            QMessageBox.warning(self, "Missing Input", "Please select a destination folder.")
            return

        file_types = self._get_file_types_for_transfer()

        # Build confirmation message
        curation_note = ""
        if self.curated_paths is not None:
            curation_note = f"\nCurated: {len(self.curated_paths)} file(s) selected\n"

        # Confirm transfer
        reply = QMessageBox.question(
            self,
            "Confirm Transfer",
            f"Transfer files from:\n{self.source_folder}\n\nTo:\n{self.dest_folder}\n\n"
            f"File types: {', '.join(file_types)}\n"
            f"Split threshold: {self.size_spinbox.value()} MB\n"
            f"{curation_note}\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Disable UI
        self.transfer_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.source_btn.setEnabled(False)
        self.curate_btn.setEnabled(False)
        self.dest_btn.setEnabled(False)

        # Clear log
        self.log_text.clear()
        self.log_text.append("Starting transfer...")

        # Create and start worker
        self.worker = TransferWorker(
            self.source_folder,
            self.dest_folder,
            file_types,
            self.size_spinbox.value(),
            self.curated_paths
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.log.connect(self.log_message)
        self.worker.finished.connect(self.transfer_finished)
        self.worker.conflict.connect(self.handle_conflict)
        self.worker.start()

    def cancel_transfer(self):
        """Cancel the transfer operation."""
        if self.worker:
            self.worker.cancel()
            self.log_text.append("Cancelling transfer...")

    def update_progress(self, current, total, message):
        """Update progress bar and status."""
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))
        self.status_label.setText(message)

    def log_message(self, message):
        """Add message to log."""
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def handle_conflict(self, file_path, worker):
        """Handle file conflict by showing dialog."""
        dialog = ConflictDialog(file_path, self)
        result = dialog.exec_()
        worker.set_conflict_response(result)

    def transfer_finished(self, stats):
        """Handle transfer completion."""
        # Re-enable UI
        self.transfer_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.source_btn.setEnabled(True)
        self.curate_btn.setEnabled(self.source_folder is not None)
        self.dest_btn.setEnabled(True)
        self.progress_bar.setValue(100)

        # Show summary
        self.log_text.append("\n" + "=" * 50)
        self.log_text.append("TRANSFER COMPLETE")
        self.log_text.append("=" * 50)
        self.log_text.append(f"Files copied: {stats['files_copied']}")
        self.log_text.append(f"Files split: {stats['files_split']}")
        self.log_text.append(f"Files skipped: {stats['files_skipped']}")
        self.log_text.append(f"Total size processed: {stats['total_size'] / 1024 / 1024:.1f} MB")

        if stats['split_files_list']:
            self.log_text.append("\nSplit files:")
            for f in stats['split_files_list']:
                self.log_text.append(f"  - {f}")

        if stats['skipped_files_list']:
            self.log_text.append("\nSkipped files:")
            for f in stats['skipped_files_list']:
                self.log_text.append(f"  - {f}")

        # Show summary dialog
        QMessageBox.information(
            self,
            "Transfer Complete",
            f"Transfer completed!\n\n"
            f"Files copied: {stats['files_copied']}\n"
            f"Files split: {stats['files_split']}\n"
            f"Files skipped: {stats['files_skipped']}\n"
            f"Total size: {stats['total_size'] / 1024 / 1024:.1f} MB"
        )

        self.status_label.setText("Transfer complete")
        self.worker = None


def main():
    """Standalone entry point."""
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = GitHubCSVTransferWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
