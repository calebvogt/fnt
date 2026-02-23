import os
import sys
import sqlite3
import struct
import numpy as np
import pandas as pd
import pytz
import gc
import json
import shutil
import xml.etree.ElementTree as ET
import cv2
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy.signal import savgol_filter
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox,
                             QGroupBox, QCheckBox, QScrollArea, QComboBox,
                             QSpinBox, QDoubleSpinBox, QSplitter, QFrame, QSlider, QLineEdit,
                             QDialog, QDialogButtonBox, QFormLayout, QTableWidget,
                             QTableWidgetItem, QHeaderView, QTextEdit, QProgressBar,
                             QDateTimeEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QDateTime
from PyQt5.QtGui import QFont


class ExportConflictDialog(QDialog):
    """Dialog shown when export would overwrite existing files."""

    SKIP = 0
    OVERWRITE = 1
    NEW_FOLDER = 2

    def __init__(self, total_new, num_conflicts, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Conflict")
        self.setModal(True)
        self.setMinimumWidth(450)

        layout = QVBoxLayout()

        message = QLabel(
            f"{total_new} file(s) will be produced, but {num_conflicts} existing "
            f"file(s) have the same name and would be overwritten.\n\n"
            "What would you like to do?"
        )
        message.setWordWrap(True)
        layout.addWidget(message)

        btn_layout = QHBoxLayout()

        skip_btn = QPushButton("Skip Existing")
        skip_btn.setToolTip("Only write files that don't already exist")
        skip_btn.clicked.connect(lambda: self.done(self.SKIP))
        btn_layout.addWidget(skip_btn)

        overwrite_btn = QPushButton("Overwrite")
        overwrite_btn.setToolTip("Replace all conflicting files")
        overwrite_btn.clicked.connect(lambda: self.done(self.OVERWRITE))
        btn_layout.addWidget(overwrite_btn)

        new_folder_btn = QPushButton("New Folder")
        new_folder_btn.setToolTip("Create a new timestamped analysis folder instead")
        new_folder_btn.clicked.connect(lambda: self.done(self.NEW_FOLDER))
        btn_layout.addWidget(new_folder_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)


class IdentityAssignmentDialog(QDialog):
    """Dialog for assigning sex, custom identities, and active time windows to tags"""
    def __init__(self, available_tags, existing_identities=None, tag_time_ranges=None, parent=None):
        super().__init__(parent)
        self.available_tags = available_tags
        self.identities = existing_identities if existing_identities else {}
        self.tag_time_ranges = tag_time_ranges if tag_time_ranges else {}
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Assign Tag Identities")
        self.setMinimumWidth(700)

        layout = QVBoxLayout()

        # Instructions
        instructions = QLabel(
            "Assign sex (M/F), IDs, and active time windows. "
            "To merge tags (e.g., lost tag replaced), assign the same ID to both tags."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Form for each tag
        form_layout = QFormLayout()
        self.sex_combos = {}
        self.identity_edits = {}
        self.start_edits = {}
        self.stop_edits = {}

        for tag in sorted(self.available_tags):
            # Sex selection
            sex_combo = QComboBox()
            sex_combo.addItems(["M", "F"])

            # Identity text input
            identity_edit = QLineEdit()
            identity_edit.setPlaceholderText(f"e.g., {tag}")

            # Start/Stop time pickers
            start_edit = QDateTimeEdit()
            start_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            start_edit.setCalendarPopup(True)
            stop_edit = QDateTimeEdit()
            stop_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            stop_edit.setCalendarPopup(True)

            # Set defaults from tag time ranges
            if tag in self.tag_time_ranges:
                tr = self.tag_time_ranges[tag]
                start_edit.setDateTime(QDateTime.fromString(tr['start'], "yyyy-MM-dd HH:mm:ss"))
                stop_edit.setDateTime(QDateTime.fromString(tr['end'], "yyyy-MM-dd HH:mm:ss"))
                # Set min/max to observed range
                start_edit.setMinimumDateTime(QDateTime.fromString(tr['start'], "yyyy-MM-dd HH:mm:ss"))
                stop_edit.setMaximumDateTime(QDateTime.fromString(tr['end'], "yyyy-MM-dd HH:mm:ss"))

            # Load existing identity values if available
            if tag in self.identities:
                sex_idx = 0 if self.identities[tag].get('sex', 'M') == 'M' else 1
                sex_combo.setCurrentIndex(sex_idx)
                identity_edit.setText(self.identities[tag].get('identity', ''))
                # Restore saved start/stop times
                if 'start_time' in self.identities[tag]:
                    dt = QDateTime.fromString(self.identities[tag]['start_time'], "yyyy-MM-dd HH:mm:ss")
                    if dt.isValid():
                        start_edit.setDateTime(dt)
                if 'stop_time' in self.identities[tag]:
                    dt = QDateTime.fromString(self.identities[tag]['stop_time'], "yyyy-MM-dd HH:mm:ss")
                    if dt.isValid():
                        stop_edit.setDateTime(dt)
            else:
                sex_combo.setCurrentIndex(-1)  # No default selection
                identity_edit.setText("")  # Blank until user configures

            # Layout: Sex + ID on first row, Start/Stop on second row
            tag_widget = QWidget()
            tag_vlayout = QVBoxLayout()
            tag_vlayout.setContentsMargins(0, 0, 0, 0)

            row1 = QHBoxLayout()
            row1.addWidget(QLabel("Sex:"))
            row1.addWidget(sex_combo)
            row1.addWidget(QLabel("ID:"))
            row1.addWidget(identity_edit)
            tag_vlayout.addLayout(row1)

            row2 = QHBoxLayout()
            row2.addWidget(QLabel("Start:"))
            row2.addWidget(start_edit)
            row2.addWidget(QLabel("Stop:"))
            row2.addWidget(stop_edit)
            tag_vlayout.addLayout(row2)

            tag_widget.setLayout(tag_vlayout)

            # Convert DEC to HEX for display
            hex_id = hex(tag).upper().replace('0X', '')
            form_layout.addRow(f"HexID {hex_id}:", tag_widget)

            self.sex_combos[tag] = sex_combo
            self.identity_edits[tag] = identity_edit
            self.start_edits[tag] = start_edit
            self.stop_edits[tag] = stop_edit

        layout.addLayout(form_layout)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_identities(self):
        """Return the configured identities with start/stop times"""
        result = {}
        for tag in self.available_tags:
            sex = self.sex_combos[tag].currentText()
            identity = self.identity_edits[tag].text().strip()
            if not identity:
                identity = str(tag)
            result[tag] = {
                'sex': sex,
                'identity': identity,
                'start_time': self.start_edits[tag].dateTime().toString("yyyy-MM-dd HH:mm:ss"),
                'stop_time': self.stop_edits[tag].dateTime().toString("yyyy-MM-dd HH:mm:ss"),
            }
        return result


class PlotSaverWorker(QThread):
    """Worker thread for saving plots without blocking the UI"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, db_path, table_name, selected_tags, downsample, smoothing_method,
                 plot_types=None, skip_existing=False, rolling_window=10, timezone='US/Mountain',
                 tag_identities=None, use_identities=False, background_image=None,
                 bg_width_meters=None, bg_height_meters=None, csv_path=None, save_svg=False,
                 output_dir=None, plots_dir=None):
        super().__init__()
        self.db_path = db_path
        self.table_name = table_name
        self.csv_path = csv_path
        self.selected_tags = selected_tags
        self.downsample = downsample
        self.smoothing_method = smoothing_method
        self.plot_types = plot_types if plot_types is not None else {
            'daily_paths': True,
            'trajectory_overview': True,
            'battery_levels': True
        }
        self.skip_existing = skip_existing
        self.output_dir = output_dir
        self.plots_dir = plots_dir  # Subfolder for plot output (PNGs/SVGs)
        self.rolling_window = rolling_window
        self.timezone = timezone
        self.tag_identities = tag_identities if tag_identities else {}
        self.use_identities = use_identities
        self.background_image = background_image
        self.bg_width_meters = bg_width_meters
        self.bg_height_meters = bg_height_meters
        self.save_svg = save_svg

    def save_figure(self, fig, output_path):
        """Save figure as PNG, and optionally also as SVG"""
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        if self.save_svg:
            svg_path = os.path.splitext(output_path)[0] + '.svg'
            fig.savefig(svg_path, format='svg', bbox_inches='tight')
            self.progress.emit(f"Saved SVG: {os.path.basename(svg_path)}")

    def run(self):
        try:
            # Load from CSV if available (much faster and ensures consistency)
            if self.csv_path and os.path.exists(self.csv_path):
                self.progress.emit("Loading data from CSV...")
                data = pd.read_csv(self.csv_path, low_memory=False)
                
                # Parse Timestamp column (let pandas infer the format automatically)
                # This handles timezone-aware timestamps like "2025-10-13 18:09:10-06:00"
                data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='mixed')
                
                self.progress.emit(f"Loaded {len(data)} records from CSV")
            else:
                # Fallback: Load from database (old behavior)
                self.progress.emit("Loading data from database...")
                
                # Connect to database
                conn = sqlite3.connect(self.db_path)
                query = f"SELECT * FROM {self.table_name}"
                data = pd.read_sql_query(query, conn)
                conn.close()
                
                self.progress.emit(f"Loaded {len(data)} records")
                
                # Process data
                data['Timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', origin='unix', utc=True)
                tz = pytz.timezone(self.timezone)
                data['Timestamp'] = data['Timestamp'].dt.tz_convert(tz)
                
                # Convert location to meters
                data['location_x'] *= 0.0254
                data['location_y'] *= 0.0254
                
                data = data.sort_values(by=['shortid', 'Timestamp'])
                
                # Filter to selected tags
                if self.selected_tags:
                    data = data[data['shortid'].isin(self.selected_tags)]
                
                # Apply custom sex and identities if configured
                if self.use_identities and self.tag_identities:
                    data['sex'] = data['shortid'].map(lambda x: self.tag_identities.get(x, {}).get('sex', 'M'))
                    data['identity'] = data['shortid'].map(lambda x: self.tag_identities.get(x, {}).get('identity', f'Tag{x}'))
                else:
                    data['sex'] = 'M'
                    data['identity'] = data['shortid'].apply(lambda x: f'Tag{x}')
                
                # Apply smoothing FIRST (on full resolution data)
                if self.smoothing_method != "None":
                    self.progress.emit("Applying smoothing to full resolution data...")
                    data = self.apply_smoothing(data, self.smoothing_method)
                
                # Downsample AFTER smoothing (if requested)
                if self.downsample:
                    self.progress.emit("Downsampling to 1Hz...")
                    data = self.apply_downsampling(data)
            
            # Get output directory
            if self.output_dir:
                output_dir = self.output_dir
            else:
                db_dir = os.path.dirname(self.db_path)
                db_filename = os.path.basename(self.db_path)
                db_name = os.path.splitext(db_filename)[0]
                output_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis")

            # Use plots subfolder for all plot output
            plots_dir = self.plots_dir if self.plots_dir else os.path.join(output_dir, 'plots')

            db_name = os.path.splitext(os.path.basename(self.db_path))[0]
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(plots_dir, exist_ok=True)
            
            # Generate and save plots based on selection
            generated_count = 0
            skipped_count = 0
            
            if self.plot_types.get('daily_paths', False):
                result = self.save_daily_paths_per_tag(data, plots_dir, db_name)
                if result:
                    generated_count += result

            if self.plot_types.get('trajectory_overview', False):
                result = self.save_trajectory_overview(data, plots_dir, db_name)
                if result:
                    generated_count += 1
                else:
                    skipped_count += 1

            if self.plot_types.get('battery_levels', False):
                result = self.save_battery_levels(data, plots_dir, db_name)
                if result:
                    generated_count += 1
                else:
                    skipped_count += 1

            if self.plot_types.get('3d_occupancy', False):
                result = self.save_3d_occupancy(data, plots_dir, db_name)
                if result:
                    generated_count += result

            if self.plot_types.get('activity_timeline', False):
                result = self.save_activity_timeline(data, plots_dir, db_name)
                if result:
                    generated_count += 1
                else:
                    skipped_count += 1

            if self.plot_types.get('velocity_distribution', False):
                result = self.save_velocity_distribution(data, plots_dir, db_name)
                if result:
                    generated_count += 1
                else:
                    skipped_count += 1
            
            if self.plot_types.get('cumulative_distance', False):
                result = self.save_cumulative_distance(data, plots_dir, db_name)
                if result:
                    generated_count += 1
                else:
                    skipped_count += 1

            if self.plot_types.get('velocity_timeline', False):
                result = self.save_velocity_timeline(data, plots_dir, db_name)
                if result:
                    generated_count += result

            if self.plot_types.get('actogram', False):
                result = self.save_actogram(data, plots_dir, db_name)
                if result:
                    generated_count += result

            if self.plot_types.get('data_quality', False):
                result = self.save_data_quality(data, plots_dir, db_name)
                if result:
                    generated_count += 1
                else:
                    skipped_count += 1
            
            msg = f"Generated {generated_count} plot(s)"
            if skipped_count > 0:
                msg += f", skipped {skipped_count} existing file(s)"
            msg += f" in {output_dir}"
            self.finished.emit(True, msg)
            
        except Exception as e:
            self.finished.emit(False, f"Error generating plots: {str(e)}")
    
    def apply_downsampling(self, data):
        """Downsample data to 1Hz"""
        data = data.copy()
        data['time_sec'] = (data['Timestamp'].astype(np.int64) // 1_000_000_000).astype(int)
        data = data.groupby(['shortid', 'time_sec']).first().reset_index()
        return data
    
    def apply_smoothing(self, data, method):
        """Apply smoothing to trajectory data"""
        def apply_savgol_filter(group):
            window_length = min(31, len(group))
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(2, window_length - 1)
            if len(group) > polyorder:
                return savgol_filter(group, window_length=window_length, polyorder=polyorder)
            return group
        
        if method == "Savitzky-Golay":
            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(apply_savgol_filter)
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(apply_savgol_filter)
        elif method == "Rolling Average":
            # Use rolling window from worker parameter
            window_size = max(3, self.rolling_window)  # Minimum window of 3
            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(
                lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(
                lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())
        elif method == "Rolling Median":
            window_size = max(3, self.rolling_window)
            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(
                lambda x: x.rolling(window=window_size, center=True, min_periods=1).median())
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(
                lambda x: x.rolling(window=window_size, center=True, min_periods=1).median())

        return data
    
    def save_daily_paths_per_tag(self, data, output_dir, db_name):
        """Save daily paths - one PNG per tag with all days
        Returns: number of plots generated"""
        self.progress.emit("Generating daily paths per tag...")
        
        data = data.copy()
        if 'Date' not in data.columns:
            data['Date'] = data['Timestamp'].dt.date
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Get global min/max coordinates across ALL data
        x_min, x_max = data[x_col].min(), data[x_col].max()
        y_min, y_max = data[y_col].min(), data[y_col].max()
        
        # If background image exists, adjust limits to include it (using meters)
        if self.background_image is not None and self.bg_width_meters is not None:
            x_min = min(x_min, 0)
            x_max = max(x_max, self.bg_width_meters)
            y_min = min(y_min, 0)
            y_max = max(y_max, self.bg_height_meters)
        
        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_pad = x_range * 0.05 if x_range > 0 else 1
        y_pad = y_range * 0.05 if y_range > 0 else 1
        
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad
        
        unique_dates = sorted(data['Date'].unique())
        unique_tags = sorted(data['shortid'].unique())
        
        generated = 0
        
        # Create one plot per tag
        for tag in unique_tags:
            # Generate filename with HexID or sex-identity
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                file_suffix = f"{sex}-{identity}"
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                file_suffix = f"HexID{hex_id}"
            
            output_path = os.path.join(output_dir, f'{db_name}_DailyPaths_{file_suffix}.png')
            
            # Check if file exists and overwrite is False
            if self.skip_existing and os.path.exists(output_path):
                self.progress.emit(f"Skipped (exists): {db_name}_DailyPaths_{file_suffix}.png")
                continue
            
            tag_data = data[data['shortid'] == tag]
            num_days = len(unique_dates)
            
            fig = Figure(figsize=(min(4 * num_days, 20), 4))
            
            for day_idx, date in enumerate(unique_dates):
                day_data = tag_data[tag_data['Date'] == date]
                
                ax = fig.add_subplot(1, num_days, day_idx + 1)
                
                if not day_data.empty:
                    ax.plot(day_data[x_col], day_data[y_col], 
                           linewidth=1.5, alpha=0.8, color='blue')
                    ax.scatter(day_data[x_col].iloc[0], day_data[y_col].iloc[0], 
                              c='black', s=50, marker='o', zorder=5)
                    ax.scatter(day_data[x_col].iloc[-1], day_data[y_col].iloc[-1], 
                              c='black', s=50, marker='s', zorder=5)
                
                # Apply global axis limits
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                
                ax.set_xlabel('X (m)', fontsize=9)
                ax.set_ylabel('Y (m)', fontsize=9)
                ax.set_title(f'{date}', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
            
            fig.suptitle(f'Daily Paths - {file_suffix}', fontsize=14, fontweight='bold')
            fig.tight_layout()
            
            self.save_figure(fig, output_path)
            plt.close(fig)
            generated += 1
            
            self.progress.emit(f"Saved: {db_name}_DailyPaths_{file_suffix}.png")
        
        return generated
    
    def save_trajectory_overview(self, data, output_dir, db_name):
        """Save trajectory overview
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating trajectory overview...")
        
        output_path = os.path.join(output_dir, f'{db_name}_TrajectoryOverview.png')
        
        # Check if file exists and overwrite is False
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_TrajectoryOverview.png")
            return False
        
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Display background image if available (with 0,0 at lower-left corner)
        if self.background_image is not None:
            if self.bg_width_meters is not None and self.bg_height_meters is not None:
                # Use scaled dimensions in meters
                ax.imshow(self.background_image, 
                         extent=[0, self.bg_width_meters, 0, self.bg_height_meters],
                         origin='lower',
                         aspect='auto',
                         alpha=0.6,
                         zorder=0)
            else:
                # Fallback to pixel dimensions
                img_height, img_width = self.background_image.shape[:2]
                ax.imshow(self.background_image, 
                         extent=[0, img_width, 0, img_height],
                         origin='lower',
                         aspect='auto',
                         alpha=0.6,
                         zorder=0)
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Plot each tag with sex-based coloring
        for tag in data['shortid'].unique():
            tag_data = data[data['shortid'] == tag]
            
            # Determine label and color based on identity configuration
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                label = f"{sex}-{identity}"
                color = 'blue' if sex == 'M' else 'red'
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                label = f"HexID {hex_id}"
                color = 'blue'  # Default to blue
            
            ax.plot(tag_data[x_col], tag_data[y_col], 
                   linewidth=1, alpha=0.7, color=color, label=label)
        
        ax.set_xlabel('X Position (m)', fontsize=10)
        ax.set_ylabel('Y Position (m)', fontsize=10)
        ax.set_title('Trajectory Overview - All Tags', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        fig.tight_layout()
        
        self.save_figure(fig, output_path)
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_TrajectoryOverview.png")
        return True
    
    def save_battery_levels(self, data, output_dir, db_name):
        """Save battery levels plot
        Returns: True if generated, False if skipped or no battery data"""
        self.progress.emit("Generating battery levels...")
        
        output_path = os.path.join(output_dir, f'{db_name}_BatteryLevels.png')
        
        # Check if file exists and overwrite is False
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_BatteryLevels.png")
            return False
        
        # Search for battery column
        battery_col = None
        possible_names = ['battery_voltage', 'vbat', 'battery', 'bat', 'voltage']
        for col_name in possible_names:
            if col_name in data.columns:
                battery_col = col_name
                break
        
        if battery_col is None:
            self.progress.emit("No battery column found, skipping battery plot")
            return False
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        for tag in data['shortid'].unique():
            tag_data = data[data['shortid'] == tag]
            ax.plot(tag_data['Timestamp'], tag_data[battery_col], 
                   label=f'Tag {tag}', linewidth=1.5, marker='o', markersize=2)
        
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel(f'{battery_col} (V)', fontsize=10)
        ax.set_title('Battery Levels Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        
        self.save_figure(fig, output_path)
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_BatteryLevels.png")
        return True
    
    def save_3d_occupancy(self, data, output_dir, db_name):
        """Save 3D occupancy heatmap - one file per tag faceted by day
        Returns: number of plots generated"""
        self.progress.emit("Generating 3D occupancy heatmaps...")
        
        from mpl_toolkits.mplot3d import Axes3D
        
        data = data.copy()
        if 'Date' not in data.columns:
            data['Date'] = data['Timestamp'].dt.date
        
        unique_dates = sorted(data['Date'].unique())
        date_to_day = {date: i+1 for i, date in enumerate(unique_dates)}
        data['Day'] = data['Date'].map(date_to_day)
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        unique_tags = sorted(data['shortid'].unique())
        num_days = len(unique_dates)
        generated = 0
        
        # Create one plot per tag with all days
        for tag in unique_tags:
            # Generate filename with HexID or sex-identity
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                file_suffix = f"{sex}-{identity}"
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                file_suffix = f"HexID{hex_id}"
            
            output_path = os.path.join(output_dir, f'{db_name}_3D_Occupancy_{file_suffix}.png')
            
            if self.skip_existing and os.path.exists(output_path):
                self.progress.emit(f"Skipped (exists): {db_name}_3D_Occupancy_{file_suffix}.png")
                continue
            
            tag_data = data[data['shortid'] == tag]
            
            # Create subplots for each day
            cols = min(3, num_days)
            rows = (num_days + cols - 1) // cols
            fig = Figure(figsize=(6 * cols, 5 * rows))
            
            for day_idx, day in enumerate(sorted(tag_data['Day'].unique())):
                day_data = tag_data[tag_data['Day'] == day]
                
                ax = fig.add_subplot(rows, cols, day_idx + 1, projection='3d')
                
                # Create 3D histogram
                hist, xedges, yedges = np.histogram2d(
                    day_data[x_col], day_data[y_col], bins=20
                )
                
                xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
                xpos = xpos.ravel()
                ypos = ypos.ravel()
                zpos = 0
                
                dx = dy = (xedges[1] - xedges[0]) * np.ones_like(zpos)
                dz = hist.ravel()
                
                # Color by sex if available
                if 'sex' in day_data.columns:
                    sex = day_data['sex'].iloc[0]
                    color = 'blue' if sex == 'M' else 'red'
                else:
                    color = 'steelblue'
                
                ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', alpha=0.8, color=color)
                ax.set_xlabel('X (m)', fontsize=8)
                ax.set_ylabel('Y (m)', fontsize=8)
                ax.set_zlabel('Count', fontsize=8)
                ax.set_title(f'Day {day}', fontsize=10)
            
            fig.suptitle(f'3D Occupancy - {file_suffix}', fontsize=14, fontweight='bold')
            fig.tight_layout()
            self.save_figure(fig, output_path)
            plt.close(fig)
            generated += 1
            
            self.progress.emit(f"Saved: {db_name}_3D_Occupancy_{file_suffix}.png")
        
        return generated
    
    def save_activity_timeline(self, data, output_dir, db_name):
        """Save activity timeline
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating activity timeline...")
        
        output_path = os.path.join(output_dir, f'{db_name}_ActivityTimeline.png')
        
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_ActivityTimeline.png")
            return False
        
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        for tag in data['shortid'].unique():
            tag_data = data[data['shortid'] == tag]
            hourly_counts = tag_data.set_index('Timestamp').resample('h').size()
            ax.plot(hourly_counts.index, hourly_counts.values, label=f'Tag {tag}', linewidth=1.5)
        
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Data Points per Hour', fontsize=10)
        ax.set_title('Activity Timeline - Data Points Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        self.save_figure(fig, output_path)
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_ActivityTimeline.png")
        return True
    
    def save_velocity_distribution(self, data, output_dir, db_name):
        """Save velocity distribution
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating velocity distribution...")
        
        output_path = os.path.join(output_dir, f'{db_name}_VelocityDistribution.png')
        
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_VelocityDistribution.png")
            return False
        
        data = data.copy()
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Calculate velocity
        data['time_diff'] = data.groupby('shortid')['Timestamp'].diff().dt.total_seconds()
        data['distance'] = np.sqrt(
            (data[x_col] - data.groupby('shortid')[x_col].shift())**2 +
            (data[y_col] - data.groupby('shortid')[y_col].shift())**2
        )
        data['velocity'] = data['distance'] / data['time_diff']
        
        # Filter out unrealistic velocities
        data = data[(data['velocity'] <= 2) | (data['velocity'].isna())]
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        for tag in data['shortid'].unique():
            tag_data = data[data['shortid'] == tag]['velocity'].dropna()
            if len(tag_data) > 0:
                # Generate label with HexID or sex-identity
                if self.use_identities and tag in self.tag_identities:
                    info = self.tag_identities[tag]
                    sex = info.get('sex', 'M')
                    identity = info.get('identity', str(tag))
                    label = f"{sex}-{identity}"
                else:
                    hex_id = hex(tag).upper().replace('0X', '')
                    label = f"HexID {hex_id}"
                
                ax.hist(tag_data, bins=50, alpha=0.5, label=label, density=True)
        
        ax.set_xlabel('Velocity (m/s)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title('Velocity Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.save_figure(fig, output_path)
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_VelocityDistribution.png")
        return True
    
    def save_cumulative_distance(self, data, output_dir, db_name):
        """Save cumulative distance plots (reset daily)
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating cumulative distance plots...")
        
        output_path = os.path.join(output_dir, f'{db_name}_CumulativeDistance.png')
        
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_CumulativeDistance.png")
            return False
        
        data = data.copy()
        if 'Date' not in data.columns:
            data['Date'] = data['Timestamp'].dt.date
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Calculate distance between consecutive points
        data = data.sort_values(['shortid', 'Timestamp'])
        data['distance_step'] = data.groupby('shortid', group_keys=False).apply(
            lambda group: np.sqrt(
                group[x_col].diff()**2 + group[y_col].diff()**2
            ).fillna(0)
        ).reset_index(level=0, drop=True)
        
        # Cumulative distance per day (reset each day)
        data['cumulative_distance'] = data.groupby(['shortid', 'Date'])['distance_step'].cumsum()
        data['time_of_day'] = (data['Timestamp'] - data['Timestamp'].dt.normalize()).dt.total_seconds() / 3600
        
        unique_days = sorted(data['Date'].unique())
        num_days = len(unique_days)
        num_cols = min(4, num_days)
        num_rows = (num_days + num_cols - 1) // num_cols
        
        fig = Figure(figsize=(5 * num_cols, 4 * num_rows))
        
        for i, day in enumerate(unique_days):
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            day_data = data[data['Date'] == day]
            
            for tag in day_data['shortid'].unique():
                tag_data = day_data[day_data['shortid'] == tag]
                
                # Generate label
                if self.use_identities and tag in self.tag_identities:
                    info = self.tag_identities[tag]
                    sex = info.get('sex', 'M')
                    identity = info.get('identity', str(tag))
                    label = f"{sex}-{identity}"
                else:
                    hex_id = hex(tag).upper().replace('0X', '')
                    label = f"HexID {hex_id}"
                
                ax.plot(tag_data['time_of_day'], tag_data['cumulative_distance'], label=label, alpha=0.7)
            
            ax.set_xlabel('Hour of Day', fontsize=9)
            ax.set_ylabel('Distance (m)', fontsize=9)
            ax.set_title(f'Day {i+1}: {day}', fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Cumulative Distance Traveled by Day', fontsize=12, fontweight='bold')
        fig.tight_layout()
        self.save_figure(fig, output_path)
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_CumulativeDistance.png")
        return True
    
    def save_velocity_timeline(self, data, output_dir, db_name):
        """Save velocity timeline plots
        Returns: number of plots generated"""
        self.progress.emit("Generating velocity timeline plots...")
        
        data = data.copy()
        if 'Date' not in data.columns:
            data['Date'] = data['Timestamp'].dt.date
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Calculate velocity
        data['time_diff'] = data.groupby('shortid')['Timestamp'].diff().dt.total_seconds()
        data['distance'] = np.sqrt(
            (data[x_col] - data.groupby('shortid')[x_col].shift())**2 +
            (data[y_col] - data.groupby('shortid')[y_col].shift())**2
        )
        data['velocity'] = data['distance'] / data['time_diff']
        data = data[(data['velocity'] <= 2) | (data['velocity'].isna())]
        
        generated = 0
        for tag in data['shortid'].unique():
            # Generate filename
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                filename = f'{db_name}_VelocityTimeline_{sex}-{identity}.png'
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                filename = f'{db_name}_VelocityTimeline_HexID{hex_id}.png'
            
            output_path = os.path.join(output_dir, filename)
            
            if self.skip_existing and os.path.exists(output_path):
                self.progress.emit(f"Skipped (exists): {filename}")
                continue
            
            tag_data = data[data['shortid'] == tag].copy()
            
            fig = Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            
            ax.plot(tag_data['Timestamp'], tag_data['velocity'], alpha=0.6, linewidth=0.5, color='blue')
            ax.axhline(y=0.1, color='red', linestyle='--', label='Activity threshold (0.1 m/s)')
            
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Velocity (m/s)', fontsize=10)
            
            # Generate title
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                ax.set_title(f'Velocity Timeline: {sex}-{identity}', fontsize=12, fontweight='bold')
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                ax.set_title(f'Velocity Timeline: HexID {hex_id}', fontsize=12, fontweight='bold')
            
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            fig.tight_layout()
            self.save_figure(fig, output_path)
            plt.close(fig)
            
            self.progress.emit(f"Saved: {filename}")
            generated += 1
        
        return generated
    
    def save_actogram(self, data, output_dir, db_name):
        """Save circadian actogram plots
        Returns: number of plots generated"""
        self.progress.emit("Generating actogram plots...")
        
        data = data.copy()
        if 'Date' not in data.columns:
            data['Date'] = data['Timestamp'].dt.date
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Calculate velocity for activity
        data['time_diff'] = data.groupby('shortid')['Timestamp'].diff().dt.total_seconds()
        data['distance'] = np.sqrt(
            (data[x_col] - data.groupby('shortid')[x_col].shift())**2 +
            (data[y_col] - data.groupby('shortid')[y_col].shift())**2
        )
        data['velocity'] = data['distance'] / data['time_diff']
        data = data[(data['velocity'] <= 2) | (data['velocity'].isna())]
        
        # Add hour and day columns
        data['hour'] = data['Timestamp'].dt.hour + data['Timestamp'].dt.minute / 60
        unique_dates = sorted(data['Date'].unique())
        date_to_day = {date: i+1 for i, date in enumerate(unique_dates)}
        data['Day'] = data['Date'].map(date_to_day)
        
        generated = 0
        for tag in data['shortid'].unique():
            # Generate filename
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                filename = f'{db_name}_Actogram_{sex}-{identity}.png'
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                filename = f'{db_name}_Actogram_HexID{hex_id}.png'
            
            output_path = os.path.join(output_dir, filename)
            
            if self.skip_existing and os.path.exists(output_path):
                self.progress.emit(f"Skipped (exists): {filename}")
                continue
            
            tag_data = data[data['shortid'] == tag].copy()
            
            # Bin activity by hour and day
            tag_data['active'] = (tag_data['velocity'] > 0.1).astype(int)
            activity_grid = tag_data.groupby(['Day', 'hour'])['active'].sum().unstack(fill_value=0)
            
            fig = Figure(figsize=(12, max(6, len(unique_dates) * 0.3)))
            ax = fig.add_subplot(111)
            
            im = ax.imshow(activity_grid.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax.set_xlabel('Hour of Day', fontsize=10)
            ax.set_ylabel('Day', fontsize=10)
            ax.set_xticks(range(0, 24, 2))
            ax.set_xticklabels(range(0, 24, 2))
            
            # Generate title
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                ax.set_title(f'Circadian Actogram: {sex}-{identity}', fontsize=12, fontweight='bold')
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                ax.set_title(f'Circadian Actogram: HexID {hex_id}', fontsize=12, fontweight='bold')
            
            fig.colorbar(im, ax=ax, label='Activity Count')
            fig.tight_layout()
            self.save_figure(fig, output_path)
            plt.close(fig)
            
            self.progress.emit(f"Saved: {filename}")
            generated += 1
        
        return generated
    
    def save_data_quality(self, data, output_dir, db_name):
        """Save data quality metrics table
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating data quality metrics...")
        
        output_path = os.path.join(output_dir, f'{db_name}_DataQuality.png')
        
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_DataQuality.png")
            return False
        
        quality_data = []
        for tag in sorted(data['shortid'].unique()):
            tag_data = data[data['shortid'] == tag].sort_values('Timestamp')
            gaps = tag_data['Timestamp'].diff().dt.total_seconds()
            
            # Generate label
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                label = f"{sex}-{identity}"
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                label = f"HexID {hex_id}"
            
            median_gap = gaps.median()
            max_gap = gaps.max()
            large_gaps = (gaps > 60).sum()
            
            quality_data.append([
                label,
                f"{median_gap:.2f}s" if pd.notna(median_gap) else "N/A",
                f"{max_gap:.2f}s" if pd.notna(max_gap) else "N/A",
                str(large_gaps)
            ])
        
        fig = Figure(figsize=(10, max(4, len(quality_data) * 0.5)))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        table = ax.table(cellText=quality_data,
                        colLabels=['Tag', 'Median Gap', 'Max Gap', 'Gaps >60s'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.25])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Data Quality Metrics', fontsize=12, fontweight='bold', pad=20)
        fig.tight_layout()
        self.save_figure(fig, output_path)
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_DataQuality.png")
        return True


class UWBQuickVisualizationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.db_path = None
        self.table_name = None
        self.available_tags = []
        self.data = None
        self.worker = None
        self.show_trail = False
        self.trail_length = 30  # Changed to seconds

        # Export control flags
        self.export_cancelled = False
        self.exporting = False
        
        # Tag identity and sex mapping
        self.tag_identities = {}  # {tag_id: {'sex': 'M', 'identity': 'Animal1'}}
        
        # XML configuration and background image
        self.xml_config_path = None
        self.background_image_path = None
        self.background_image = None  # Loaded matplotlib image
        self.xml_scale = None  # Scale from XML in inches/pixel
        self.bg_width_meters = None  # Background image width in meters
        self.bg_height_meters = None  # Background image height in meters
        self.arena_zones = None  # DataFrame with zone coordinates from XML
        self.anchor_positions = []  # List of dicts: {'shortid': int, 'x': float, 'y': float, 'z': float}

        # Preview state (separate from export data)
        self.preview_data = None
        self.unique_timestamps = []
        self.preview_loaded = False
        self.is_playing = False
        self.playback_speed = 1
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.advance_playback)

        # Zoom state tracking — used to preserve user zoom across redraws
        self._default_xlim = None
        self._default_ylim = None

        self.initUI()
        
    def log_message(self, message):
        """Add a message to the messages window"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.txt_messages.append(f"[{timestamp}] {message}")
        # Auto-scroll to bottom
        self.txt_messages.verticalScrollBar().setValue(
            self.txt_messages.verticalScrollBar().maximum()
        )
        # Also update legacy status label for compatibility
        self.lbl_status.setText(message)
    
    def save_message_log(self, output_dir):
        """Save the message log to a text file"""
        try:
            db_name = os.path.splitext(os.path.basename(self.db_path))[0]
            log_path = os.path.join(output_dir, f"{db_name}_messageLog.txt")
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(self.txt_messages.toPlainText())
            
            self.log_message(f"✓ Message log saved: {os.path.basename(log_path)}")
        except Exception as e:
            self.log_message(f"Warning: Could not save message log: {str(e)}")
    
    def save_run_summary(self, output_dir):
        """Save run summary with filtering statistics to CSV"""
        try:
            from datetime import datetime
            db_name = os.path.splitext(os.path.basename(self.db_path))[0]
            summary_path = os.path.join(output_dir, f"{db_name}_runSummary.csv")
            
            # Collect run parameters
            summary_data = {
                'Parameter': [],
                'Value': []
            }
            
            # General info
            summary_data['Parameter'].append('Run Date')
            summary_data['Value'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            summary_data['Parameter'].append('Database')
            summary_data['Value'].append(os.path.basename(self.db_path))
            
            summary_data['Parameter'].append('Table')
            summary_data['Value'].append(self.table_name)
            
            summary_data['Parameter'].append('Timezone')
            summary_data['Value'].append(self.combo_timezone.currentText())
            
            # Selected tags
            selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]
            summary_data['Parameter'].append('Selected Tags')
            summary_data['Value'].append(', '.join([str(t) for t in selected_tags]))
            
            # Filtering settings
            summary_data['Parameter'].append('Velocity Filter Enabled')
            summary_data['Value'].append('Yes' if self.chk_velocity_filter.isChecked() else 'No')
            
            if self.chk_velocity_filter.isChecked():
                summary_data['Parameter'].append('Velocity Threshold (m/s)')
                summary_data['Value'].append(self.spin_velocity_threshold.value())
            
            summary_data['Parameter'].append('Jump Filter Enabled')
            summary_data['Value'].append('Yes' if self.chk_jump_filter.isChecked() else 'No')
            
            if self.chk_jump_filter.isChecked():
                summary_data['Parameter'].append('Jump Threshold (m)')
                summary_data['Value'].append(self.spin_jump_threshold.value())
            
            summary_data['Parameter'].append('Time Gap Threshold (s)')
            summary_data['Value'].append(self.spin_time_gap.value())
            
            # Smoothing settings
            summary_data['Parameter'].append('Smoothing')
            summary_data['Value'].append(self.combo_smoothing.currentText())
            
            # Downsampling
            summary_data['Parameter'].append('Downsample Hz')
            summary_data['Value'].append(f"{self.spin_downsample_hz.value()} Hz" if self.chk_export_downsampled_csv.isChecked() else 'N/A')
            
            # Filtering statistics (if available)
            if hasattr(self, 'filter_stats') and self.filter_stats:
                summary_data['Parameter'].append('')
                summary_data['Value'].append('')
                
                summary_data['Parameter'].append('--- Filtering Statistics ---')
                summary_data['Value'].append('')
                
                summary_data['Parameter'].append('Initial Data Points')
                summary_data['Value'].append(self.filter_stats.get('initial_count', 'N/A'))
                
                summary_data['Parameter'].append('Points Removed (Velocity)')
                summary_data['Value'].append(self.filter_stats.get('removed_velocity', 0))
                
                summary_data['Parameter'].append('Points Removed (Jump)')
                summary_data['Value'].append(self.filter_stats.get('removed_jump', 0))
                
                summary_data['Parameter'].append('Final Data Points')
                summary_data['Value'].append(self.filter_stats.get('final_count', 'N/A'))
                
                summary_data['Parameter'].append('Percent Filtered')
                summary_data['Value'].append(f"{self.filter_stats.get('percent_filtered', 0):.2f}%")
            
            # Export options
            summary_data['Parameter'].append('')
            summary_data['Value'].append('')
            
            summary_data['Parameter'].append('--- Export Options ---')
            summary_data['Value'].append('')
            
            summary_data['Parameter'].append('Raw CSV Exported')
            summary_data['Value'].append('Yes' if self.chk_export_raw_csv.isChecked() else 'No')

            summary_data['Parameter'].append('Smoothed CSV Exported')
            summary_data['Value'].append('Yes' if self.chk_export_smoothed_csv.isChecked() else 'No')

            summary_data['Parameter'].append('Downsampled CSV Exported')
            if self.chk_export_downsampled_csv.isChecked():
                summary_data['Value'].append(f'Yes ({self.spin_downsample_hz.value()} Hz)')
            else:
                summary_data['Value'].append('No')

            summary_data['Parameter'].append('Plots Generated')
            summary_data['Value'].append('Yes' if self.chk_save_plots.isChecked() else 'No')
            
            summary_data['Parameter'].append('Animation Generated')
            summary_data['Value'].append('Yes' if self.chk_save_animation.isChecked() else 'No')
            
            summary_data['Parameter'].append('Behaviors Detected')
            summary_data['Value'].append('Yes' if self.chk_detect_behaviors.isChecked() else 'No')
            
            # Create DataFrame and save
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_path, index=False)
            
            self.log_message(f"✓ Run summary saved: {os.path.basename(summary_path)}")
        except Exception as e:
            self.log_message(f"Warning: Could not save run summary: {str(e)}")
    
    def initUI(self):
        self.setWindowTitle("UWB PreProcessing Tool")
        self.setGeometry(50, 50, 1400, 900)
        
        # Set dark theme style
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #cccccc;
                font-family: Arial;
            }
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #666666;
            }
            QGroupBox {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 8px;
                background-color: #2b2b2b;
                color: #0078d4;
            }
            QCheckBox {
                color: #cccccc;
                spacing: 8px;
            }
            QComboBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 100px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3f3f3f;
                height: 6px;
                background: #1e1e1e;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #e08030;
                border: 1px solid #3f3f3f;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #f09040;
            }
        """)

        # Main layout - splitter with settings (left) and preview (right)
        main_layout = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)

        left_panel = self.create_settings_panel()
        left_panel.setMinimumWidth(350)
        splitter.addWidget(left_panel)

        right_panel = self.create_visualization_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 1000])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
    def create_settings_panel(self):
        """Create the left settings panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Header frame (matching Video PreProcessing Tool style)
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        header_frame.setStyleSheet("background-color: #1e1e1e; padding: 15px; border: 1px solid #3f3f3f;")
        header_layout = QVBoxLayout()
        header_frame.setLayout(header_layout)

        title = QLabel("UWB PreProcessing Tool")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #0078d4; background-color: transparent;")
        header_layout.addWidget(title)

        desc = QLabel("Preprocess and export UWB tracking data")
        desc.setAlignment(Qt.AlignCenter)
        desc.setFont(QFont("Arial", 10))
        desc.setStyleSheet("color: #999999; font-style: italic; background-color: transparent;")
        header_layout.addWidget(desc)

        layout.addWidget(header_frame)
        
        # Database selection
        db_group = QGroupBox("Database Selection")
        db_layout = QVBoxLayout()
        
        btn_select = QPushButton("Select SQLite Database")
        btn_select.clicked.connect(self.select_database)
        db_layout.addWidget(btn_select)
        
        self.lbl_db = QLabel("No database selected")
        self.lbl_db.setStyleSheet("color: #666666; font-style: italic;")
        self.lbl_db.setWordWrap(True)
        db_layout.addWidget(self.lbl_db)
        
        # Table selection
        table_layout = QHBoxLayout()
        table_layout.addWidget(QLabel("Table:"))
        self.combo_table = QComboBox()
        self.combo_table.setEnabled(False)
        self.combo_table.currentTextChanged.connect(self.on_table_selected)
        table_layout.addWidget(self.combo_table)
        db_layout.addLayout(table_layout)
        
        # Preview table button
        self.btn_preview_table = QPushButton("Preview Table")
        self.btn_preview_table.clicked.connect(self.preview_table)
        self.btn_preview_table.setEnabled(False)
        db_layout.addWidget(self.btn_preview_table)
        
        db_group.setLayout(db_layout)
        layout.addWidget(db_group)

        # Timezone
        tz_group = QGroupBox("Timezone")
        tz_group_layout = QVBoxLayout()
        tz_layout = QHBoxLayout()
        tz_layout.addWidget(QLabel("Timezone:"))
        self.combo_timezone = QComboBox()
        common_timezones = [
            "US/Mountain", "US/Pacific", "US/Central", "US/Eastern",
            "UTC", "Europe/London", "Europe/Paris", "Asia/Tokyo"
        ]
        self.combo_timezone.addItems(common_timezones)
        self.combo_timezone.setCurrentText("US/Mountain")
        self.combo_timezone.currentTextChanged.connect(self.mark_options_changed)
        tz_layout.addWidget(self.combo_timezone)
        tz_group_layout.addLayout(tz_layout)
        tz_group.setLayout(tz_group_layout)
        layout.addWidget(tz_group)

        # Tag selection
        self.tag_group = QGroupBox("Tag Selection")
        self.tag_layout = QVBoxLayout()
        self.tag_checkboxes = {}
        
        self.lbl_no_tags = QLabel("Load a database to see available tags")
        self.lbl_no_tags.setStyleSheet("color: #666666; font-style: italic;")
        self.tag_layout.addWidget(self.lbl_no_tags)
        
        # Note: Select All/None and Configure Identities buttons will be added
        # dynamically below the tag checkboxes in load_tags_from_table()
        
        self.tag_group.setLayout(self.tag_layout)
        layout.addWidget(self.tag_group)

        # Smoothing & Filtering Options
        options_group = QGroupBox("Smoothing & Filtering Options")
        options_layout = QVBoxLayout()

        # --- Step 1: Filtering ---

        # Velocity Filtering
        velocity_filter_layout = QHBoxLayout()
        self.chk_velocity_filter = QCheckBox("Apply velocity filtering (remove points >")
        self.chk_velocity_filter.setChecked(True)
        self.chk_velocity_filter.setToolTip("Remove data points with unrealistic velocities")
        self.chk_velocity_filter.stateChanged.connect(self.mark_options_changed)
        velocity_filter_layout.addWidget(self.chk_velocity_filter)

        self.spin_velocity_threshold = QDoubleSpinBox()
        self.spin_velocity_threshold.setRange(0.1, 10.0)
        self.spin_velocity_threshold.setValue(2.0)
        self.spin_velocity_threshold.setSuffix(" m/s)")
        self.spin_velocity_threshold.setDecimals(1)
        self.spin_velocity_threshold.setSingleStep(0.1)
        self.spin_velocity_threshold.setToolTip("Maximum allowed velocity in meters per second")
        self.spin_velocity_threshold.valueChanged.connect(self.mark_options_changed)
        velocity_filter_layout.addWidget(self.spin_velocity_threshold)
        velocity_filter_layout.addStretch()
        options_layout.addLayout(velocity_filter_layout)

        # Jump Filtering (distance threshold)
        jump_filter_layout = QHBoxLayout()
        self.chk_jump_filter = QCheckBox("Apply jump filtering (remove jumps >")
        self.chk_jump_filter.setChecked(True)
        self.chk_jump_filter.setToolTip("Remove data points with unrealistic spatial jumps")
        self.chk_jump_filter.stateChanged.connect(self.mark_options_changed)
        jump_filter_layout.addWidget(self.chk_jump_filter)

        self.spin_jump_threshold = QDoubleSpinBox()
        self.spin_jump_threshold.setRange(0.1, 10.0)
        self.spin_jump_threshold.setValue(2.0)
        self.spin_jump_threshold.setSuffix(" m)")
        self.spin_jump_threshold.setDecimals(1)
        self.spin_jump_threshold.setSingleStep(0.1)
        self.spin_jump_threshold.setToolTip("Maximum allowed distance jump in meters")
        self.spin_jump_threshold.valueChanged.connect(self.mark_options_changed)
        jump_filter_layout.addWidget(self.spin_jump_threshold)
        jump_filter_layout.addStretch()
        options_layout.addLayout(jump_filter_layout)

        # Time Window Grouping
        time_window_layout = QHBoxLayout()
        time_window_layout.addWidget(QLabel("Time gap grouping:"))
        self.spin_time_gap = QSpinBox()
        self.spin_time_gap.setRange(5, 300)
        self.spin_time_gap.setValue(30)
        self.spin_time_gap.setSuffix(" sec")
        self.spin_time_gap.setToolTip("Group data by time gaps larger than this (prevents filtering across battery restarts)")
        self.spin_time_gap.valueChanged.connect(self.mark_options_changed)
        time_window_layout.addWidget(self.spin_time_gap)
        time_window_layout.addStretch()
        options_layout.addLayout(time_window_layout)

        # Trail options (visual setting, placed after filtering options)
        self.chk_show_trail = QCheckBox("Show trail")
        self.chk_show_trail.setChecked(False)
        self.chk_show_trail.stateChanged.connect(self.on_trail_toggled)
        options_layout.addWidget(self.chk_show_trail)

        trail_length_layout = QHBoxLayout()
        trail_length_layout.addWidget(QLabel("Trail length (seconds):"))
        self.spin_trail_length = QSpinBox()
        self.spin_trail_length.setRange(1, 300)
        self.spin_trail_length.setValue(30)
        self.spin_trail_length.setEnabled(False)
        trail_length_layout.addWidget(self.spin_trail_length)
        options_layout.addLayout(trail_length_layout)

        # --- Step 2: Smoothing ---

        options_layout.addWidget(QLabel("Smoothing method:"))
        self.combo_smoothing = QComboBox()
        self.combo_smoothing.addItems(["None", "Rolling Average (default)", "Rolling Median", "Savitzky-Golay"])
        self.combo_smoothing.setCurrentIndex(1)  # Default to Rolling Average
        self.combo_smoothing.currentTextChanged.connect(self.on_smoothing_changed)
        options_layout.addWidget(self.combo_smoothing)

        # Rolling window (shown for Rolling Average and Rolling Median)
        self.rolling_window_layout = QHBoxLayout()
        self.rolling_window_layout.addWidget(QLabel("Window (seconds):"))
        self.spin_rolling_window = QSpinBox()
        self.spin_rolling_window.setRange(1, 60)
        self.spin_rolling_window.setValue(30)
        self.spin_rolling_window.setEnabled(False)
        self.rolling_window_layout.addWidget(self.spin_rolling_window)
        options_layout.addLayout(self.rolling_window_layout)
        self.spin_rolling_window.hide()
        self.rolling_window_layout.itemAt(0).widget().hide()

        # --- Preview Color By ---
        preview_color_layout = QHBoxLayout()
        preview_color_layout.addWidget(QLabel("Color by:"))
        self.combo_preview_color_by = QComboBox()
        self.combo_preview_color_by.addItems(["ID", "Sex"])
        self.combo_preview_color_by.setToolTip("Color trajectories in preview by unique ID or by sex (M=blue, F=red)")
        self.combo_preview_color_by.currentTextChanged.connect(self.on_preview_color_changed)
        preview_color_layout.addWidget(self.combo_preview_color_by)
        options_layout.addLayout(preview_color_layout)

        # --- Background Image ---

        bg_buttons_layout = QHBoxLayout()

        self.btn_load_background = QPushButton("Load Background")
        self.btn_load_background.clicked.connect(self.select_background_image)
        self.btn_load_background.setEnabled(False)
        self.btn_load_background.setToolTip("Load a background map/floorplan image to overlay on visualizations")
        self.btn_load_background.setStyleSheet("padding: 8px; font-size: 11px;")
        bg_buttons_layout.addWidget(self.btn_load_background)

        self.btn_remove_background = QPushButton("Remove Background")
        self.btn_remove_background.clicked.connect(self.remove_background)
        self.btn_remove_background.setEnabled(False)
        self.btn_remove_background.setToolTip("Remove the background image from visualizations")
        self.btn_remove_background.setStyleSheet("padding: 8px; font-size: 11px;")
        bg_buttons_layout.addWidget(self.btn_remove_background)

        options_layout.addLayout(bg_buttons_layout)

        self.lbl_background_status = QLabel("No background image loaded")
        self.lbl_background_status.setStyleSheet("color: #666666; font-style: italic; font-size: 9px;")
        self.lbl_background_status.setWordWrap(True)
        options_layout.addWidget(self.lbl_background_status)

        # Show anchors toggle
        self.chk_show_anchors = QCheckBox("Show anchor positions")
        self.chk_show_anchors.setChecked(True)
        self.chk_show_anchors.setEnabled(False)  # Enabled when anchors are parsed from XML
        self.chk_show_anchors.setToolTip("Toggle visibility of UWB anchor/antenna positions (triangles)")
        self.chk_show_anchors.stateChanged.connect(self.on_show_anchors_toggled)
        options_layout.addWidget(self.chk_show_anchors)

        # Refresh Tracking Preview button
        self.btn_refresh_preview = QPushButton("Refresh Tracking Preview")
        self.btn_refresh_preview.clicked.connect(self.load_preview)
        self.btn_refresh_preview.setEnabled(False)
        self.btn_refresh_preview.setToolTip("Reload preview with current options (tag selection, smoothing, filtering, identities)")
        self.btn_refresh_preview.setStyleSheet("padding: 8px; font-size: 11px;")
        options_layout.addWidget(self.btn_refresh_preview)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Export Options
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout()
        
        # Export Raw CSV checkbox
        self.chk_export_raw_csv = QCheckBox("Export Raw CSV")
        self.chk_export_raw_csv.setChecked(True)
        self.chk_export_raw_csv.setToolTip("Export raw database contents as CSV (no processing)")
        export_layout.addWidget(self.chk_export_raw_csv)

        # Export Smoothed CSV checkbox
        self.chk_export_smoothed_csv = QCheckBox("Export Smoothed CSV")
        self.chk_export_smoothed_csv.setChecked(True)
        self.chk_export_smoothed_csv.setToolTip("Export filtered + smoothed data at full resolution (no downsampling)")
        export_layout.addWidget(self.chk_export_smoothed_csv)

        # Export Smoothed Downsampled CSV checkbox + Hz spinner
        downsample_row = QWidget()
        downsample_layout = QHBoxLayout()
        downsample_layout.setContentsMargins(0, 0, 0, 0)
        self.chk_export_downsampled_csv = QCheckBox("Export Smoothed Downsampled CSV")
        self.chk_export_downsampled_csv.setChecked(True)
        self.chk_export_downsampled_csv.setToolTip("Export filtered + smoothed + downsampled data at specified sample rate")
        downsample_layout.addWidget(self.chk_export_downsampled_csv)
        self.spin_downsample_hz = QSpinBox()
        self.spin_downsample_hz.setRange(1, 5)
        self.spin_downsample_hz.setValue(1)
        self.spin_downsample_hz.setSuffix(" Hz")
        self.spin_downsample_hz.setToolTip("Target sample rate for downsampled CSV (1-5 Hz)")
        self.spin_downsample_hz.setFixedWidth(70)
        downsample_layout.addWidget(self.spin_downsample_hz)
        downsample_layout.addStretch()
        downsample_row.setLayout(downsample_layout)
        export_layout.addWidget(downsample_row)

        # Save Plots checkbox (master)
        self.chk_save_plots = QCheckBox("Save Plots")
        self.chk_save_plots.setChecked(True)  # Default checked
        self.chk_save_plots.stateChanged.connect(self.on_save_plots_toggled)
        self.chk_save_plots.setToolTip("Generate and save visualization plots (PNG always included)")
        export_layout.addWidget(self.chk_save_plots)

        # Indented plot type options
        self.plot_types_widget = QWidget()
        plot_types_layout = QVBoxLayout()
        plot_types_layout.setContentsMargins(30, 0, 0, 0)  # Indent

        self.plot_type_checkboxes = {}
        plot_types = [
            ("daily_paths", "Daily Paths per Tag", "One PNG per tag with all days"),
            ("trajectory_overview", "Trajectory Overview", "All tags overlaid"),
            ("battery_levels", "Battery Levels", "Battery voltage over time"),
            ("3d_occupancy", "3D Occupancy Heatmap", "3D visualization of occupancy over time"),
            ("activity_timeline", "Activity Timeline", "Data points per hour over time"),
            ("velocity_distribution", "Velocity Distribution", "Velocity distribution for each tag"),
            ("cumulative_distance", "Cumulative Distance", "Distance traveled over time (reset daily)"),
            ("velocity_timeline", "Velocity Timeline", "Velocity over time with activity threshold"),
            ("actogram", "Circadian Actogram", "24-hour activity patterns across days"),
            ("data_quality", "Data Quality Metrics", "Table showing data gaps and quality statistics")
        ]

        for key, plot_name, plot_desc in plot_types:
            cb = QCheckBox(plot_name)
            cb.setChecked(True)
            cb.setToolTip(plot_desc)
            self.plot_type_checkboxes[key] = cb
            plot_types_layout.addWidget(cb)

        self.plot_types_widget.setLayout(plot_types_layout)
        self.plot_types_widget.setVisible(True)  # Visible by default since Save Plots is checked
        export_layout.addWidget(self.plot_types_widget)

        # SVG option (indented under Save Plots, after plot types)
        self.svg_option_widget = QWidget()
        svg_option_layout = QHBoxLayout()
        svg_option_layout.setContentsMargins(30, 0, 0, 0)  # Indent
        self.chk_save_svg = QCheckBox("Also save as SVG")
        self.chk_save_svg.setChecked(False)
        self.chk_save_svg.setToolTip("Additionally save plots in SVG format (vector graphics)")
        svg_option_layout.addWidget(self.chk_save_svg)
        svg_option_layout.addStretch()
        self.svg_option_widget.setLayout(svg_option_layout)
        self.svg_option_widget.setVisible(True)  # Visible by default since Save Plots is checked
        export_layout.addWidget(self.svg_option_widget)

        # Detect Behaviors checkbox
        self.chk_detect_behaviors = QCheckBox("Detect Behaviors (beta)")
        self.chk_detect_behaviors.setChecked(False)
        self.chk_detect_behaviors.setToolTip("Analyze and export behavioral patterns and social interactions")
        export_layout.addWidget(self.chk_detect_behaviors)

        # Save Animation checkbox (master)
        self.chk_save_animation = QCheckBox("Save Animation")
        self.chk_save_animation.setChecked(False)  # Default unchecked
        self.chk_save_animation.stateChanged.connect(self.on_save_animation_toggled)
        self.chk_save_animation.setToolTip("Generate animated video of tracking data")
        export_layout.addWidget(self.chk_save_animation)
        
        # Indented animation options
        self.animation_options_widget = QWidget()
        animation_options_layout = QVBoxLayout()
        animation_options_layout.setContentsMargins(30, 0, 0, 0)  # Indent
        
        # Animation trail length
        trail_layout = QHBoxLayout()
        trail_layout.addWidget(QLabel("Trail length (seconds):"))
        self.spin_animation_trail = QSpinBox()
        self.spin_animation_trail.setRange(1, 1000)
        self.spin_animation_trail.setValue(500)
        self.spin_animation_trail.setToolTip("How much trailing data to show in animation")
        trail_layout.addWidget(self.spin_animation_trail)
        animation_options_layout.addLayout(trail_layout)
        
        # Animation Speed
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Animation Speed:"))
        self.combo_animation_speed = QComboBox()
        self.combo_animation_speed.addItems(["1x", "5x", "10x", "20x", "40x", "80x", "100x", "120x", "150x", "160x", "200x", "400x", "500x", "1000x"])
        self.combo_animation_speed.setCurrentText("80x")
        self.combo_animation_speed.setToolTip("Playback speed multiplier (e.g., 10x = 10 seconds of real time per second of video)")
        speed_layout.addWidget(self.combo_animation_speed)
        animation_options_layout.addLayout(speed_layout)
        
        # Animation FPS
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.combo_animation_fps = QComboBox()
        self.combo_animation_fps.addItems(["1", "5", "10", "20", "30"])
        self.combo_animation_fps.setCurrentText("30")
        self.combo_animation_fps.setToolTip("Frames per second for output video (affects smoothness)")
        fps_layout.addWidget(self.combo_animation_fps)
        animation_options_layout.addLayout(fps_layout)
        
        # Time window per frame
        time_window_layout = QHBoxLayout()
        time_window_layout.addWidget(QLabel("Time window (seconds):"))
        self.spin_time_window = QSpinBox()
        self.spin_time_window.setRange(1, 300)
        self.spin_time_window.setValue(30)
        self.spin_time_window.setToolTip("Time window in seconds for each frame")
        time_window_layout.addWidget(self.spin_time_window)
        animation_options_layout.addLayout(time_window_layout)
        
        # Color by option
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color by:"))
        self.combo_color_by = QComboBox()
        self.combo_color_by.addItems(["ID", "sex"])
        self.combo_color_by.setToolTip("Color trajectories by ID or sex")
        color_layout.addWidget(self.combo_color_by)
        animation_options_layout.addLayout(color_layout)
        
        # Video quality option
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Video Quality:"))
        self.combo_video_quality = QComboBox()
        self.combo_video_quality.addItems(["Draft (Fast)", "Standard", "High Quality"])
        self.combo_video_quality.setCurrentText("High Quality")
        self.combo_video_quality.setToolTip("Draft=75dpi (4x faster), Standard=100dpi, High=150dpi")
        quality_layout.addWidget(self.combo_video_quality)
        animation_options_layout.addLayout(quality_layout)
        
        # Estimated frame count label
        self.lbl_estimated_frames = QLabel("Estimated frames: -- (load data first)")
        self.lbl_estimated_frames.setStyleSheet("color: #888; font-size: 10pt; font-style: italic;")
        animation_options_layout.addWidget(self.lbl_estimated_frames)
        
        # Connect animation parameter changes to update estimate
        self.combo_animation_speed.currentTextChanged.connect(self.update_frame_estimate)
        self.combo_animation_fps.currentTextChanged.connect(self.update_frame_estimate)
        
        # Daily animations checkbox
        self.chk_daily_animations = QCheckBox("Generate daily animations (one per day)")
        self.chk_daily_animations.setChecked(False)
        self.chk_daily_animations.stateChanged.connect(self.on_daily_animations_toggled)
        self.chk_daily_animations.setToolTip("Create separate animation for each day (midnight to midnight)")
        animation_options_layout.addWidget(self.chk_daily_animations)
        
        # Container for daily animation day selection (hidden by default)
        self.daily_animation_days_widget = QWidget()
        daily_days_layout = QVBoxLayout()
        daily_days_layout.setContentsMargins(20, 5, 0, 5)
        self.daily_animation_day_checkboxes = {}
        self.daily_days_layout_inner = QVBoxLayout()
        daily_days_layout.addLayout(self.daily_days_layout_inner)
        self.daily_animation_days_widget.setLayout(daily_days_layout)
        self.daily_animation_days_widget.setVisible(False)
        animation_options_layout.addWidget(self.daily_animation_days_widget)
        
        self.animation_options_widget.setLayout(animation_options_layout)
        self.animation_options_widget.setVisible(True)  # Visible by default
        export_layout.addWidget(self.animation_options_widget)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Progress bar (hidden by default)
        self.progress_widget = QWidget()
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(0, 5, 0, 5)
        
        self.lbl_export_progress = QLabel("")
        self.lbl_export_progress.setStyleSheet("color: #00aa00; font-weight: bold;")
        progress_layout.addWidget(self.lbl_export_progress)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
                background-color: #1e1e1e;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_widget.setLayout(progress_layout)
        self.progress_widget.setVisible(False)
        layout.addWidget(self.progress_widget)
        
        # Export buttons layout
        export_buttons_layout = QHBoxLayout()

        # Export button
        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self.export_data)
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet("padding: 8px; font-size: 11px; font-weight: bold;")
        export_buttons_layout.addWidget(self.btn_export)

        # Stop export button (hidden by default)
        self.btn_stop_export = QPushButton("Stop Export")
        self.btn_stop_export.clicked.connect(self.stop_export)
        self.btn_stop_export.setStyleSheet("padding: 8px; font-size: 11px; font-weight: bold; background-color: #d41100;")
        self.btn_stop_export.setVisible(False)
        export_buttons_layout.addWidget(self.btn_stop_export)

        layout.addLayout(export_buttons_layout)
        
        # Messages window
        messages_label = QLabel("Messages:")
        messages_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(messages_label)
        
        self.txt_messages = QTextEdit()
        self.txt_messages.setReadOnly(True)
        self.txt_messages.setMaximumHeight(150)
        self.txt_messages.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #555555;
                padding: 5px;
                font-family: Consolas, monospace;
                font-size: 9px;
            }
        """)
        layout.addWidget(self.txt_messages)
        
        # Status (legacy, kept for compatibility but hidden)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #666666; font-style: italic; font-size: 10px;")
        self.lbl_status.setVisible(False)  # Hidden, using messages window instead
        layout.addWidget(self.lbl_status)
        
        layout.addStretch()
        panel.setLayout(layout)
        
        # Make scrollable
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        
        return scroll

    def create_visualization_panel(self):
        """Create the right visualization/preview panel"""
        panel = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Tracking Preview")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        # Matplotlib canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Initial empty plot
        self.ax.set_xlabel('X Position (m)', fontsize=10)
        self.ax.set_ylabel('Y Position (m)', fontsize=10)
        self.ax.set_title('Load data to visualize tracking', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        layout.addWidget(self.canvas)

        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, panel)
        layout.addWidget(self.toolbar)

        # Playback controls
        playback_layout = QHBoxLayout()

        self.btn_rewind = QPushButton("\u23ee Rewind")
        self.btn_rewind.clicked.connect(self.rewind_playback)
        self.btn_rewind.setEnabled(False)
        self.btn_rewind.setMaximumWidth(100)
        playback_layout.addWidget(self.btn_rewind)

        self.btn_play_pause = QPushButton("\u25b6 Play")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.setMaximumWidth(100)
        playback_layout.addWidget(self.btn_play_pause)

        self.btn_fast_forward = QPushButton("Fast Forward \u23ed")
        self.btn_fast_forward.clicked.connect(self.fast_forward_playback)
        self.btn_fast_forward.setEnabled(False)
        self.btn_fast_forward.setMaximumWidth(120)
        playback_layout.addWidget(self.btn_fast_forward)

        playback_layout.addSpacing(20)

        playback_layout.addWidget(QLabel("Speed:"))
        self.combo_playback_speed = QComboBox()
        self.combo_playback_speed.addItems(["1x", "2x", "4x", "8x"])
        self.combo_playback_speed.setCurrentText("1x")
        self.combo_playback_speed.currentTextChanged.connect(self.on_speed_changed)
        self.combo_playback_speed.setMaximumWidth(80)
        playback_layout.addWidget(self.combo_playback_speed)

        playback_layout.addStretch()

        # Save Preview Image button
        self.btn_save_preview = QPushButton("Save Preview Image")
        self.btn_save_preview.clicked.connect(self.save_preview_image)
        self.btn_save_preview.setEnabled(False)
        self.btn_save_preview.setToolTip("Save the current preview view as PNG or SVG")
        playback_layout.addWidget(self.btn_save_preview)

        layout.addLayout(playback_layout)

        # Time slider
        slider_layout = QHBoxLayout()
        self.lbl_time = QLabel("--:--:--")
        self.lbl_time.setMinimumWidth(160)
        slider_layout.addWidget(self.lbl_time)

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(100)
        self.time_slider.setValue(0)
        self.time_slider.setEnabled(False)
        self.time_slider.valueChanged.connect(self.update_visualization)
        slider_layout.addWidget(self.time_slider)

        layout.addLayout(slider_layout)

        panel.setLayout(layout)
        return panel

    def select_database(self):
        """Select SQLite database"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select SQLite Database", "",
            "SQLite Files (*.sqlite *.db *.sql);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if not tables:
                QMessageBox.warning(self, "No Tables", "No tables found in database")
                return
            
            self.db_path = file_path
            self.lbl_db.setText(f"Selected: {os.path.basename(file_path)}")

            # Load config BEFORE populating combo_table, so that pending_tag_selection
            # is set before on_table_selected() triggers tag checkbox creation
            self.load_config_if_exists()

            self.combo_table.clear()
            self.combo_table.addItems(tables)
            self.combo_table.setEnabled(True)
            self.btn_preview_table.setEnabled(True)
            self.btn_load_background.setEnabled(True)  # Enable background image loading

            # Apply saved table name from config (if any), otherwise default to first
            if hasattr(self, 'pending_table_name') and self.pending_table_name:
                index = self.combo_table.findText(self.pending_table_name)
                if index >= 0:
                    self.combo_table.setCurrentIndex(index)
                delattr(self, 'pending_table_name')
            elif len(tables) == 1:
                self.combo_table.setCurrentIndex(0)

            # Check for XML configuration file in the database directory
            self.load_xml_config()

            # Auto-load background image if PNG exists alongside database
            self.auto_load_background()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open database: {str(e)}")
    
    def load_xml_config(self):
        """Look for XML configuration file in the database directory"""
        if not self.db_path:
            return
        
        db_dir = os.path.dirname(self.db_path)
        
        # Look for .xml files in the same directory
        xml_files = [f for f in os.listdir(db_dir) if f.endswith('.xml')]
        
        if not xml_files:
            self.log_message("No XML configuration file found in database directory")
            QMessageBox.warning(
                self, "XML Configuration Not Found",
                "No XML configuration file found in the database folder.\n\n"
                "Anchor positions and floorplan scale will not be available."
            )
            return
        
        # If multiple XML files, use the first one or one matching config/Config pattern
        xml_file = None
        for f in xml_files:
            if 'config' in f.lower():
                xml_file = f
                break
        if not xml_file:
            xml_file = xml_files[0]
        
        self.xml_config_path = os.path.join(db_dir, xml_file)
        self.log_message(f"Found XML config: {xml_file}")
        
        try:
            self.parse_xml_config()
        except Exception as e:
            self.log_message(f"Warning: Could not parse XML config: {str(e)}")
    
    def parse_xml_config(self):
        """Parse XML configuration file and check for background image"""
        if not self.xml_config_path or not os.path.exists(self.xml_config_path):
            return
        
        try:
            tree = ET.parse(self.xml_config_path)
            root = tree.getroot()
            
            # Extract scale attribute (inches/pixel)
            for elem in root.iter():
                if 'scale' in elem.attrib:
                    try:
                        self.xml_scale = float(elem.attrib['scale'])
                        self.log_message(f"Found XML scale: {self.xml_scale} inches/pixel")
                        break
                    except:
                        pass
            
            # Parse zone coordinates from Zones section
            zones_element = root.find('Zones')
            if zones_element is not None:
                zone_data = []
                for zone in zones_element.findall('Zone'):
                    zone_name = zone.get('name')
                    if zone_name is None:
                        continue
                    
                    shape = zone.find('Shape')
                    if shape is None:
                        continue
                    
                    for point in shape.findall('Point'):
                        x_str = point.get('x')
                        y_str = point.get('y')
                        
                        if x_str is not None and y_str is not None:
                            try:
                                # Convert coordinates to meters (inches to meters)
                                x_meters = float(x_str) * 0.0254
                                y_meters = float(y_str) * 0.0254
                                zone_data.append({
                                    'zone': zone_name,
                                    'x': x_meters,
                                    'y': y_meters
                                })
                            except ValueError:
                                continue
                
                if zone_data:
                    self.arena_zones = pd.DataFrame(zone_data)
                    num_zones = len(self.arena_zones['zone'].unique())
                    self.log_message(f"Parsed {num_zones} zones with {len(zone_data)} coordinate points from XML")
                else:
                    self.log_message("No valid zone coordinates found in XML")
            else:
                self.log_message("No Zones section found in XML")

            # Parse anchor positions
            self.anchor_positions = []
            for anchor in root.iter('Anchor'):
                try:
                    shortid = int(anchor.get('shortid', '0'))
                    x_hex = anchor.get('x', '0x0')
                    y_hex = anchor.get('y', '0x0')
                    z_hex = anchor.get('z', '0x0')

                    # Decode IEEE 754 hex-encoded doubles
                    x_inches = struct.unpack('d', struct.pack('Q', int(x_hex, 16)))[0]
                    y_inches = struct.unpack('d', struct.pack('Q', int(y_hex, 16)))[0]
                    z_inches = struct.unpack('d', struct.pack('Q', int(z_hex, 16)))[0]

                    # Convert inches to meters
                    self.anchor_positions.append({
                        'shortid': shortid,
                        'x': x_inches * 0.0254,
                        'y': y_inches * 0.0254,
                        'z': z_inches * 0.0254
                    })
                except (ValueError, struct.error):
                    continue

            if self.anchor_positions:
                self.log_message(f"Parsed {len(self.anchor_positions)} anchor positions from XML")
                self.chk_show_anchors.setEnabled(True)
            else:
                self.log_message("No anchor positions found in XML")

            # Check if XML contains an embedded background image (base64)
            has_embedded_image = False
            for elem in root.iter():
                if elem.tag in ('Map', 'BackgroundImage', 'Image'):
                    # Check for base64 image data in text content
                    if elem.text and len(elem.text.strip()) > 200:
                        has_embedded_image = True
                        break

            if has_embedded_image:
                self.log_message("XML contains an embedded background image")
                reply = QMessageBox.question(
                    self,
                    "Background Image Detected",
                    "The XML config contains an embedded background image.\n\n"
                    "Would you like to select the corresponding PNG file to use as background?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.select_background_image()
            else:
                self.log_message("No embedded background image found in XML")
                
        except Exception as e:
            self.log_message(f"Error parsing XML: {str(e)}")
    
    def auto_load_background(self):
        """Auto-load background image if a PNG exists in the database directory"""
        if not self.db_path or self.background_image is not None:
            return  # Already loaded (e.g., from config) or no database

        db_dir = os.path.dirname(self.db_path)
        db_name = os.path.splitext(os.path.basename(self.db_path))[0]

        # Find PNG files in the database directory
        png_files = [f for f in os.listdir(db_dir) if f.lower().endswith('.png')]

        if not png_files:
            return

        # Pick the best match: prefer one matching db name or xml config name, else use only if exactly one
        selected_png = None
        for f in png_files:
            fname = os.path.splitext(f)[0].lower()
            if db_name.lower() in fname or fname in db_name.lower():
                selected_png = f
                break
        if not selected_png and hasattr(self, 'xml_config_path') and self.xml_config_path:
            xml_name = os.path.splitext(os.path.basename(self.xml_config_path))[0].lower()
            for f in png_files:
                fname = os.path.splitext(f)[0].lower()
                if xml_name in fname or fname in xml_name:
                    selected_png = f
                    break
        if not selected_png and len(png_files) == 1:
            selected_png = png_files[0]

        if not selected_png:
            self.log_message(f"Multiple PNG files found in directory — skipping auto-load (use Load Background to select)")
            return

        file_path = os.path.join(db_dir, selected_png)
        try:
            self.background_image = plt.imread(file_path)
            self.background_image_path = file_path
            img_height_px, img_width_px = self.background_image.shape[:2]

            if self.xml_scale:
                self.bg_width_meters = img_width_px * self.xml_scale * 0.0254
                self.bg_height_meters = img_height_px * self.xml_scale * 0.0254
                self.log_message(f"✓ Background auto-loaded: {selected_png} ({self.bg_width_meters:.2f}m x {self.bg_height_meters:.2f}m)")
            else:
                self.bg_width_meters = img_width_px * 0.0254
                self.bg_height_meters = img_height_px * 0.0254
                self.log_message(f"✓ Background auto-loaded: {selected_png} (no XML scale — dimensions may be inaccurate)")

            self.lbl_background_status.setText(f"\u2713 Background: {selected_png}")
            self.lbl_background_status.setStyleSheet("color: #00aa00; font-style: normal; font-size: 9px;")
            self.btn_remove_background.setEnabled(True)

            # Reset stored axis defaults so preview recomputes limits to include background
            self._default_xlim = None
            self._default_ylim = None

            # Refresh preview if already loaded
            if self.preview_loaded:
                self.update_visualization(self.time_slider.value())

        except Exception as e:
            self.log_message(f"Warning: Could not auto-load background {selected_png}: {str(e)}")

    def select_background_image(self):
        """Allow user to select a background image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Background Image",
            os.path.dirname(self.db_path) if self.db_path else "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*.*)"
        )
        
        if file_path and os.path.exists(file_path):
            self.background_image_path = file_path
            self.log_message(f"Background image loaded: {os.path.basename(file_path)}")
            
            # Load the image for matplotlib
            try:
                self.background_image = plt.imread(file_path)
                self.log_message(f"Background image size: {self.background_image.shape}")
                
                # Calculate dimensions in meters using XML scale if available
                img_height_px, img_width_px = self.background_image.shape[:2]
                if self.xml_scale:
                    # Convert: pixels * inches/pixel * meters/inch = meters
                    # 1 inch = 0.0254 meters
                    self.bg_width_meters = img_width_px * self.xml_scale * 0.0254
                    self.bg_height_meters = img_height_px * self.xml_scale * 0.0254
                    self.log_message(f"Background dimensions: {self.bg_width_meters:.2f}m x {self.bg_height_meters:.2f}m")
                else:
                    # No XML scale — warn user that dimensions will be incorrect
                    self.bg_width_meters = img_width_px * 0.0254  # Assume 1 inch/pixel as rough fallback
                    self.bg_height_meters = img_height_px * 0.0254
                    self.log_message("WARNING: No XML scale found — background dimensions may be incorrect. Load an XML configuration for accurate scaling.")
                    QMessageBox.warning(
                        self, "No Scale Available",
                        "No XML scale has been loaded. The background image dimensions "
                        "may be incorrect.\n\nLoad an XML configuration file first for "
                        "accurate floorplan scaling."
                    )
                
                # Update status label and enable remove button
                self.lbl_background_status.setText(f"\u2713 Background: {os.path.basename(file_path)}")
                self.lbl_background_status.setStyleSheet("color: #00aa00; font-style: normal; font-size: 9px;")
                self.btn_remove_background.setEnabled(True)

                # Reset stored axis defaults so preview recomputes limits to include background
                self._default_xlim = None
                self._default_ylim = None

                # Refresh preview if loaded
                if self.preview_loaded:
                    self.update_visualization(self.time_slider.value())

            except Exception as e:
                self.log_message(f"Error loading background image: {str(e)}")
                self.background_image = None
                self.background_image_path = None
                self.bg_width_meters = None
                self.bg_height_meters = None
                self.lbl_background_status.setText("Error loading background image")
                self.lbl_background_status.setStyleSheet("color: #aa0000; font-style: italic; font-size: 9px;")

    def remove_background(self):
        """Remove the background image from visualizations"""
        self.background_image = None
        self.background_image_path = None
        self.bg_width_meters = None
        self.bg_height_meters = None
        self.lbl_background_status.setText("No background image loaded")
        self.lbl_background_status.setStyleSheet("color: #666666; font-style: italic; font-size: 9px;")
        self.btn_remove_background.setEnabled(False)
        self.log_message("Background image removed")

        # Reset stored axis defaults so preview recomputes limits without background
        self._default_xlim = None
        self._default_ylim = None

        # Refresh preview if loaded
        if self.preview_loaded:
            self.update_visualization(self.time_slider.value())

    def on_table_selected(self, table_name):
        """Handle table selection"""
        if table_name:
            self.table_name = table_name
            self.load_tags_from_table()
    
    def preview_table(self):
        """Preview table data in a dialog"""
        if not self.db_path or not self.table_name:
            return
        
        try:
            # Load first 100 rows for preview
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT * FROM {self.table_name} LIMIT 100"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Create preview dialog
            from PyQt5.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QHeaderView
            
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Preview: {self.table_name}")
            dialog.setGeometry(100, 100, 1000, 600)
            
            layout = QVBoxLayout()
            
            # Info label
            info_label = QLabel(f"Showing first 100 rows of {len(df)} columns")
            info_label.setStyleSheet("color: #cccccc; font-weight: bold; padding: 10px;")
            layout.addWidget(info_label)
            
            # Create table widget
            table = QTableWidget()
            table.setRowCount(len(df))
            table.setColumnCount(len(df.columns))
            table.setHorizontalHeaderLabels(df.columns.tolist())
            
            # Populate table
            for i in range(len(df)):
                for j in range(len(df.columns)):
                    item = QTableWidgetItem(str(df.iloc[i, j]))
                    table.setItem(i, j, item)
            
            # Auto-resize columns
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            
            # Style table
            table.setStyleSheet("""
                QTableWidget {
                    background-color: #1e1e1e;
                    color: #cccccc;
                    gridline-color: #3f3f3f;
                    border: 1px solid #3f3f3f;
                }
                QHeaderView::section {
                    background-color: #2b2b2b;
                    color: #0078d4;
                    font-weight: bold;
                    padding: 4px;
                    border: 1px solid #3f3f3f;
                }
            """)
            
            layout.addWidget(table)
            
            # Close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Failed to preview table: {str(e)}")
    
    def on_smoothing_changed(self, method):
        """Handle smoothing method change"""
        clean_method = method.replace(" (default)", "")
        is_rolling = clean_method in ("Rolling Average", "Rolling Median")
        self.spin_rolling_window.setEnabled(is_rolling)
        self.spin_rolling_window.setVisible(is_rolling)
        self.rolling_window_layout.itemAt(0).widget().setVisible(is_rolling)
        self.mark_options_changed()
    
    def on_preview_color_changed(self):
        """Handle preview color-by dropdown change - refresh preview if loaded"""
        if self.preview_loaded:
            self.update_visualization(self.time_slider.value())

    def on_save_plots_toggled(self):
        """Handle save plots checkbox toggle"""
        enabled = self.chk_save_plots.isChecked()
        self.plot_types_widget.setVisible(enabled)
        self.svg_option_widget.setVisible(enabled)
    
    def on_save_animation_toggled(self):
        """Handle save animation checkbox toggle"""
        enabled = self.chk_save_animation.isChecked()
        self.animation_options_widget.setVisible(enabled)
    
    def update_frame_estimate(self):
        """Update estimated frame count based on animation settings and loaded data"""
        source_data = self.preview_data if self.preview_data is not None else self.data
        if source_data is None or 'Timestamp' not in source_data.columns:
            self.lbl_estimated_frames.setText("Estimated frames: -- (load data first)")
            return

        try:
            # Get animation parameters
            fps = int(self.combo_animation_fps.currentText())
            speed_text = self.combo_animation_speed.currentText()
            speed_multiplier = int(speed_text.replace('x', ''))

            # Calculate frame interval (real seconds per frame)
            frame_interval = speed_multiplier / fps

            # Get total time span of data
            time_span = (source_data['Timestamp'].max() - source_data['Timestamp'].min()).total_seconds()
            
            # Estimate number of frames
            estimated_frames = int(time_span / frame_interval)
            
            # Format with commas for readability
            frames_formatted = f"{estimated_frames:,}"
            
            # Calculate estimated video duration
            video_duration = estimated_frames / fps
            
            if video_duration >= 60:
                duration_str = f"{video_duration/60:.1f} min"
            else:
                duration_str = f"{video_duration:.1f} sec"
            
            self.lbl_estimated_frames.setText(
                f"Estimated frames: {frames_formatted} (~{duration_str} video @ {fps} FPS)"
            )
        except Exception as e:
            self.lbl_estimated_frames.setText(f"Estimated frames: Error calculating ({str(e)})")
    
    def on_daily_animations_toggled(self):
        """Handle daily animations checkbox toggle"""
        enabled = self.chk_daily_animations.isChecked()
        self.daily_animation_days_widget.setVisible(enabled)
    
    def populate_animation_days_from_list(self, date_strings):
        """Populate day checkboxes from a list of date strings (works without loading full data)"""
        # Clear existing checkboxes
        for cb in self.daily_animation_day_checkboxes.values():
            cb.deleteLater()
        self.daily_animation_day_checkboxes.clear()
        
        if not date_strings:
            return
        
        # Create checkboxes for each day
        for i, date_str in enumerate(date_strings):
            cb = QCheckBox(f"Day {i+1}: {date_str}")
            cb.setChecked(True)  # Default to all days selected
            self.daily_animation_day_checkboxes[date_str] = cb
            self.daily_days_layout_inner.addWidget(cb)
    
    def populate_animation_days(self):
        """Populate day checkboxes based on loaded data (called after preview loads)"""
        # Clear existing checkboxes
        for cb in self.daily_animation_day_checkboxes.values():
            cb.deleteLater()
        self.daily_animation_day_checkboxes.clear()

        source_data = self.preview_data if self.preview_data is not None else self.data
        if source_data is None or 'Timestamp' not in source_data.columns:
            return

        # Get unique dates
        dates = pd.to_datetime(source_data['Timestamp']).dt.date.unique()
        dates = sorted(dates)
        date_strings = [date.strftime('%Y-%m-%d') for date in dates]
        
        # Use the list-based function
        self.populate_animation_days_from_list(date_strings)
        
        self.log_message(f"Found {len(dates)} unique days in dataset")
    
    def open_identity_dialog(self):
        """Open dialog to assign identities to tags"""
        if not self.available_tags:
            QMessageBox.warning(self, "No Tags", "Please load a database and table first")
            return

        # Get only selected tags
        selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]
        if not selected_tags:
            QMessageBox.warning(self, "No Tags Selected", "Please select at least one tag")
            return

        # Query per-tag time ranges from database
        tag_time_ranges = {}
        try:
            tz = pytz.timezone(self.combo_timezone.currentText())
            conn = sqlite3.connect(self.db_path)
            placeholders = ','.join(['?'] * len(selected_tags))
            query = f"SELECT shortid, MIN(timestamp) as first_ts, MAX(timestamp) as last_ts FROM {self.table_name} WHERE shortid IN ({placeholders}) GROUP BY shortid"
            cursor = conn.execute(query, selected_tags)
            for row in cursor:
                tag_id, first_ts, last_ts = row
                # Convert ms timestamps to timezone-aware datetimes
                first_dt = pd.Timestamp(first_ts, unit='ms', tz='UTC').tz_convert(tz)
                last_dt = pd.Timestamp(last_ts, unit='ms', tz='UTC').tz_convert(tz)
                tag_time_ranges[tag_id] = {
                    'start': first_dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'end': last_dt.strftime('%Y-%m-%d %H:%M:%S'),
                }
            conn.close()
        except Exception as e:
            self.log_message(f"Warning: Could not query tag time ranges: {str(e)}")

        dialog = IdentityAssignmentDialog(selected_tags, self.tag_identities, tag_time_ranges, self)
        if dialog.exec_() == QDialog.Accepted:
            self.tag_identities = dialog.get_identities()
            self.log_message(f"Updated identities for {len(self.tag_identities)} tags")
            # Update tag checkbox labels to reflect new identities
            self.update_tag_labels()
            self.lbl_status.setText(f"Identity assignments saved for {len(self.tag_identities)} tags")
            
            # Auto-save configuration to JSON
            if self.db_path:
                db_dir = os.path.dirname(self.db_path)
                db_filename = os.path.basename(self.db_path)
                db_name = os.path.splitext(db_filename)[0]
                output_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis")
                os.makedirs(output_dir, exist_ok=True)
                self.save_config(output_dir)
                self.log_message("✓ Configuration auto-saved to JSON")
    
    def stop_export(self):
        """Cancel ongoing export operations"""
        self.export_cancelled = True
        self.log_message("⚠ Export cancellation requested...")
        
        # Stop worker thread if running
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.log_message("✗ Plot export cancelled")
        
        # Reset UI
        self.exporting = False
        self.btn_export.setEnabled(True)
        self.btn_stop_export.setVisible(False)
        self.progress_widget.setVisible(False)
        self.progress_bar.setValue(0)
        self.lbl_export_progress.setText("")
    
    def mark_options_changed(self):
        """Mark that options have changed - prompt user to refresh preview"""
        if self.preview_loaded:
            self.btn_refresh_preview.setEnabled(True)
    
    def load_tags_from_table(self):
        """Load tags from selected table"""
        if not self.db_path or not self.table_name:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT DISTINCT shortid FROM {self.table_name} ORDER BY shortid"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            self.available_tags = df['shortid'].tolist()
            self.update_tag_selection()

            # Enable buttons
            self.btn_refresh_preview.setEnabled(True)
            self.btn_export.setEnabled(True)

            # Load unique days for daily animation options
            self.load_unique_days_from_database()

            # Auto-load preview with default settings
            self.load_preview()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load tags: {str(e)}")
    
    def load_unique_days_from_database(self):
        """Load unique days from database without loading full data"""
        if not self.db_path or not self.table_name:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            # Query for distinct dates
            query = f"""
                SELECT DISTINCT date(datetime(timestamp/1000, 'unixepoch'), 'localtime') as date
                FROM {self.table_name}
                ORDER BY date
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) > 0:
                self.populate_animation_days_from_list(df['date'].tolist())
                self.log_message(f"Found {len(df)} unique days in database")
            
        except Exception as e:
            self.log_message(f"Could not load unique days: {str(e)}")
    
    def update_tag_selection(self):
        """Update tag checkboxes"""
        for cb in self.tag_checkboxes.values():
            cb.deleteLater()
        self.tag_checkboxes.clear()
        
        if self.lbl_no_tags:
            self.lbl_no_tags.deleteLater()
            self.lbl_no_tags = None
        
        # Remove existing buttons if they exist
        if hasattr(self, 'tag_buttons_layout'):
            while self.tag_buttons_layout.count():
                item = self.tag_buttons_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.tag_layout.removeItem(self.tag_buttons_layout)
        
        if hasattr(self, 'btn_assign_identities'):
            self.btn_assign_identities.deleteLater()
        
        for tag in self.available_tags:
            hex_id = hex(tag).upper().replace('0X', '')
            # Show HexID with identity info only if user has configured it
            if tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', '')
                identity = info.get('identity', '')
                if sex and identity:
                    cb = QCheckBox(f"HexID {hex_id} ({sex}, {identity})")
                else:
                    cb = QCheckBox(f"HexID {hex_id}")
            else:
                cb = QCheckBox(f"HexID {hex_id}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.mark_options_changed)  # Track changes
            cb.stateChanged.connect(self.update_identity_button_state)  # Enable/disable identity button
            self.tag_checkboxes[tag] = cb
            self.tag_layout.addWidget(cb)
        
        # Add Select All/None buttons below checkboxes
        self.tag_buttons_layout = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_select_all.setStyleSheet("padding: 6px; font-size: 10px;")
        btn_select_all.clicked.connect(self.select_all_tags)
        btn_select_none = QPushButton("Select None")
        btn_select_none.setStyleSheet("padding: 6px; font-size: 10px;")
        btn_select_none.clicked.connect(self.select_none_tags)
        self.tag_buttons_layout.addWidget(btn_select_all)
        self.tag_buttons_layout.addWidget(btn_select_none)
        self.tag_layout.addLayout(self.tag_buttons_layout)
        
        # Add Configure Identities button below Select All/None
        self.btn_assign_identities = QPushButton("Configure Identities...")
        self.btn_assign_identities.clicked.connect(self.open_identity_dialog)
        self.btn_assign_identities.setEnabled(False)
        self.btn_assign_identities.setToolTip("Assign custom sex (M/F) and alphanumeric IDs to selected tags")
        self.btn_assign_identities.setStyleSheet("padding: 6px; font-size: 10px;")
        self.tag_layout.addWidget(self.btn_assign_identities)
        
        # Apply pending tag selection from loaded config
        self.apply_pending_tag_selection()
        
        # Update identity button state
        self.update_identity_button_state()
    
    def update_identity_button_state(self):
        """Enable Configure Identities button if any tag is selected"""
        any_selected = any(cb.isChecked() for cb in self.tag_checkboxes.values())
        self.btn_assign_identities.setEnabled(any_selected)
    
    def update_tag_labels(self):
        """Update tag checkbox labels to reflect sex and ID information"""
        for tag, cb in self.tag_checkboxes.items():
            hex_id = hex(tag).upper().replace('0X', '')
            if tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', '')
                identity = info.get('identity', '')
                if sex and identity:
                    cb.setText(f"HexID {hex_id} ({sex}, {identity})")
                else:
                    cb.setText(f"HexID {hex_id}")
            else:
                cb.setText(f"HexID {hex_id}")
    
    def select_all_tags(self):
        """Select all tags"""
        for cb in self.tag_checkboxes.values():
            cb.setChecked(True)
        self.mark_options_changed()
    
    def select_none_tags(self):
        """Deselect all tags"""
        for cb in self.tag_checkboxes.values():
            cb.setChecked(False)
        self.mark_options_changed()
    
    def on_trail_toggled(self):
        """Handle trail checkbox toggle"""
        enabled = self.chk_show_trail.isChecked()
        self.spin_trail_length.setEnabled(enabled)
        # Refresh preview if loaded (trail is a visual-only setting, no reload needed)
        if self.preview_loaded:
            self.update_visualization(self.time_slider.value())

    def on_show_anchors_toggled(self):
        """Handle show anchors checkbox toggle"""
        # Visual-only setting, no data reload needed
        if self.preview_loaded:
            self.update_visualization(self.time_slider.value())

    def get_smoothing_method(self):
        """Get the current smoothing method name (stripped of UI hints like '(default)')"""
        text = self.combo_smoothing.currentText()
        return text.replace(" (default)", "")

    def apply_smoothing(self):
        """Apply smoothing to self.data"""
        self.data = self.apply_smoothing_to_data(self.data, self.get_smoothing_method())
    
    def apply_smoothing_to_data(self, data, method):
        """Apply smoothing to a dataframe (works on any dataframe, not just self.data)"""
        def apply_savgol(group):
            window_length = min(31, len(group))
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(2, window_length - 1)
            if len(group) > polyorder:
                return savgol_filter(group, window_length=window_length, polyorder=polyorder)
            return group
        
        if method == "Savitzky-Golay":
            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(apply_savgol)
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(apply_savgol)
        elif method == "Rolling Average":
            # Get window size in seconds from spinbox
            window_seconds = self.spin_rolling_window.value()

            # Calculate window size in number of samples (assuming 1Hz after downsampling)
            window_size = max(3, window_seconds)  # Minimum window of 3

            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(
                lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(
                lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())
        elif method == "Rolling Median":
            window_seconds = self.spin_rolling_window.value()
            window_size = max(3, window_seconds)

            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(
                lambda x: x.rolling(window=window_size, center=True, min_periods=1).median())
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(
                lambda x: x.rolling(window=window_size, center=True, min_periods=1).median())

        return data
    
    def apply_filters(self, data):
        """Apply velocity and jump filtering with time window grouping"""
        initial_count = len(data)
        
        # Calculate time differences and group by time gaps (prevents filtering across battery restarts)
        time_gap_threshold = self.spin_time_gap.value()
        data['time_diff'] = data.groupby('shortid')['Timestamp'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
        data['time_diff_s'] = np.ceil(data['time_diff']).astype(int)
        data['tw_group'] = data.groupby('shortid')['time_diff_s'].apply(
            lambda x: (x > time_gap_threshold).cumsum()
        ).reset_index(level=0, drop=True)
        
        # Calculate distance and velocity within time window groups
        data['distance'] = np.sqrt(
            (data['location_x'] - data.groupby(['shortid', 'tw_group'])['location_x'].shift())**2 +
            (data['location_y'] - data.groupby(['shortid', 'tw_group'])['location_y'].shift())**2
        )
        data['velocity'] = data['distance'] / data['time_diff']
        
        # Apply velocity filtering if enabled
        if self.chk_velocity_filter.isChecked():
            velocity_threshold = self.spin_velocity_threshold.value()
            before_velocity = len(data)
            data = data[(data['velocity'] <= velocity_threshold) | (data['velocity'].isna())]
            removed_velocity = before_velocity - len(data)
            if removed_velocity > 0:
                self.log_message(f"  Removed {removed_velocity} points with velocity > {velocity_threshold} m/s")
        
        # Apply jump filtering if enabled
        if self.chk_jump_filter.isChecked():
            jump_threshold = self.spin_jump_threshold.value()
            before_jump = len(data)
            data['is_jump'] = (data['distance'] > jump_threshold)
            data = data[~data['is_jump']]
            removed_jump = before_jump - len(data)
            if removed_jump > 0:
                self.log_message(f"  Removed {removed_jump} points with distance jump > {jump_threshold} m")
        
        # Clean up temporary columns
        data = data.drop(columns=['time_diff', 'time_diff_s', 'tw_group', 'distance', 'velocity'], errors='ignore')
        if 'is_jump' in data.columns:
            data = data.drop(columns=['is_jump'])
        
        final_count = len(data)
        if initial_count != final_count:
            self.log_message(f"  Total filtered: {initial_count - final_count} points ({100*(initial_count-final_count)/initial_count:.1f}%)")
        
        return data
    
    def apply_filters_to_data(self, data):
        """Apply velocity and jump filtering with time window grouping to any dataframe"""
        initial_count = len(data)
        removed_velocity = 0
        removed_jump = 0
        
        # Make explicit copy at start to avoid any SettingWithCopyWarning
        data = data.copy()
        
        # Calculate time differences and group by time gaps (prevents filtering across battery restarts)
        time_gap_threshold = self.spin_time_gap.value()
        data['time_diff'] = data.groupby('shortid')['Timestamp'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
        data['time_diff_s'] = np.ceil(data['time_diff']).astype(int)
        data['tw_group'] = data.groupby('shortid')['time_diff_s'].apply(
            lambda x: (x > time_gap_threshold).cumsum()
        ).reset_index(level=0, drop=True)
        
        # Calculate distance and velocity within time window groups
        data['distance'] = np.sqrt(
            (data['location_x'] - data.groupby(['shortid', 'tw_group'])['location_x'].shift())**2 +
            (data['location_y'] - data.groupby(['shortid', 'tw_group'])['location_y'].shift())**2
        )
        data['velocity'] = data['distance'] / data['time_diff']
        
        # Apply velocity filtering if enabled
        if self.chk_velocity_filter.isChecked():
            velocity_threshold = self.spin_velocity_threshold.value()
            before_velocity = len(data)
            data = data[(data['velocity'] <= velocity_threshold) | (data['velocity'].isna())].copy()
            removed_velocity = before_velocity - len(data)
            if removed_velocity > 0:
                self.log_message(f"  Removed {removed_velocity} points with velocity > {velocity_threshold} m/s")
        else:
            removed_velocity = 0
        
        # Apply jump filtering if enabled
        if self.chk_jump_filter.isChecked():
            jump_threshold = self.spin_jump_threshold.value()
            before_jump = len(data)
            data['is_jump'] = (data['distance'] > jump_threshold)
            data = data[~data['is_jump']].copy()
            removed_jump = before_jump - len(data)
            if removed_jump > 0:
                self.log_message(f"  Removed {removed_jump} points with distance jump > {jump_threshold} m")
        else:
            removed_jump = 0
        
        # Clean up temporary columns
        data = data.drop(columns=['time_diff', 'time_diff_s', 'tw_group', 'distance', 'velocity'], errors='ignore')
        if 'is_jump' in data.columns:
            data = data.drop(columns=['is_jump'])
        
        final_count = len(data)
        if initial_count != final_count:
            self.log_message(f"  Total filtered: {initial_count - final_count} points ({100*(initial_count-final_count)/initial_count:.1f}%)")
        
        # Store stats for summary report
        if not hasattr(self, 'filter_stats'):
            self.filter_stats = {}
        self.filter_stats = {
            'initial_count': initial_count,
            'removed_velocity': removed_velocity,
            'removed_jump': removed_jump,
            'final_count': final_count,
            'percent_filtered': 100 * (initial_count - final_count) / initial_count if initial_count > 0 else 0
        }
        
        return data

    # ---- Preview / Playback Methods ----

    def load_preview(self):
        """Load data and setup interactive preview (always downsampled to 1Hz)"""
        if not self.db_path or not self.table_name:
            return

        selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]
        if not selected_tags:
            QMessageBox.warning(self, "No Tags", "Please select at least one tag")
            return

        try:
            self.btn_refresh_preview.setText("Refreshing...")
            self.btn_refresh_preview.setEnabled(False)
            QApplication.processEvents()

            self.log_message("Loading data from database for preview...")

            # Load data
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT * FROM {self.table_name}"
            preview_data = pd.read_sql_query(query, conn)
            conn.close()

            self.log_message(f"Loaded {len(preview_data)} records from database")

            # Process data
            self.log_message("Processing timestamps and converting units...")
            preview_data['Timestamp'] = pd.to_datetime(preview_data['timestamp'], unit='ms', origin='unix', utc=True)
            tz = pytz.timezone(self.combo_timezone.currentText())
            preview_data['Timestamp'] = preview_data['Timestamp'].dt.tz_convert(tz)

            preview_data['location_x'] *= 0.0254
            preview_data['location_y'] *= 0.0254

            preview_data = preview_data.sort_values(by=['shortid', 'Timestamp'])

            # Filter tags
            preview_data = preview_data[preview_data['shortid'].isin(selected_tags)]

            # Apply custom sex and identities
            if self.tag_identities:
                self.log_message("Applying custom identities...")
                preview_data['sex'] = preview_data['shortid'].map(lambda x: self.tag_identities.get(x, {}).get('sex', 'M'))
                preview_data['identity'] = preview_data['shortid'].map(lambda x: self.tag_identities.get(x, {}).get('identity', f'Tag{x}'))
            else:
                preview_data['sex'] = 'M'
                preview_data['identity'] = preview_data['shortid'].apply(lambda x: f'Tag{x}')

            # Apply per-tag time trimming (before filtering/smoothing)
            if self.tag_identities:
                tz = pytz.timezone(self.combo_timezone.currentText())
                for tag, info in self.tag_identities.items():
                    if 'start_time' in info and 'stop_time' in info:
                        start = pd.Timestamp(info['start_time']).tz_localize(tz)
                        stop = pd.Timestamp(info['stop_time']).tz_localize(tz)
                        mask = (preview_data['shortid'] == tag) & (
                            (preview_data['Timestamp'] < start) | (preview_data['Timestamp'] > stop))
                        trimmed = mask.sum()
                        if trimmed > 0:
                            preview_data = preview_data[~mask]
                            self.log_message(f"  Trimmed {trimmed} points outside time window for tag {tag}")

            # Apply filtering if enabled
            if self.chk_velocity_filter.isChecked() or self.chk_jump_filter.isChecked():
                self.log_message("Applying velocity/jump filtering...")
                preview_data = self.apply_filters_to_data(preview_data)

            # Apply smoothing if enabled (on full resolution data BEFORE downsampling)
            if self.get_smoothing_method() != "None":
                self.log_message("Applying smoothing to full resolution data...")
                preview_data = self.apply_smoothing_to_data(preview_data, self.get_smoothing_method())

            # Always downsample preview to 1Hz for performance
            self.log_message("Downsampling preview to 1Hz...")
            preview_data['time_sec'] = (preview_data['Timestamp'].astype(np.int64) // 1_000_000_000).astype(int)
            preview_data = preview_data.groupby(['shortid', 'time_sec']).first().reset_index()

            # Store preview data
            self.preview_data = preview_data

            # Setup slider based on unique timestamps
            self.log_message("Setting up visualization controls...")
            unique_times = sorted(self.preview_data['Timestamp'].unique())
            self.unique_timestamps = unique_times
            self.time_slider.setMaximum(len(unique_times) - 1)
            self.time_slider.setValue(0)
            self.time_slider.setEnabled(True)

            # Initial visualization
            self.update_visualization(0)

            # Enable playback controls
            self.btn_rewind.setEnabled(True)
            self.btn_play_pause.setEnabled(True)
            self.btn_fast_forward.setEnabled(True)
            self.btn_save_preview.setEnabled(True)

            # Mark preview as loaded
            self.preview_loaded = True
            self.btn_refresh_preview.setText("Refresh Tracking Preview")
            self.btn_refresh_preview.setEnabled(True)
            self.btn_export.setEnabled(True)

            # Populate animation days and update frame estimate
            self.populate_animation_days()
            self.update_frame_estimate()

            self.log_message(f"\u2713 Preview loaded: {len(self.preview_data)} data points across {len(unique_times)} unique timestamps")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load preview: {str(e)}")
            self.log_message(f"\u2717 Error loading data: {str(e)}")
            self.btn_refresh_preview.setText("Refresh Tracking Preview")
            self.btn_refresh_preview.setEnabled(True)

    def update_visualization(self, slider_value):
        """Update visualization based on slider position"""
        if self.preview_data is None or len(self.preview_data) == 0 or not self.unique_timestamps:
            return

        # Save current axis limits before clearing (to preserve user zoom)
        old_xlim = self.ax.get_xlim()
        old_ylim = self.ax.get_ylim()

        self.ax.clear()

        x_col = 'smoothed_x' if 'smoothed_x' in self.preview_data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in self.preview_data.columns else 'location_y'

        # Get current timestamp from slider
        current_timestamp = self.unique_timestamps[slider_value]
        self.lbl_time.setText(current_timestamp.strftime('%Y-%m-%d %H:%M:%S'))

        # Display background image if available
        if self.background_image is not None and self.bg_width_meters is not None:
            try:
                self.ax.imshow(self.background_image,
                              extent=[0, self.bg_width_meters, 0, self.bg_height_meters],
                              origin='lower',
                              aspect='auto',
                              alpha=0.6,
                              zorder=0)
            except Exception as e:
                self.log_message(f"Error drawing background: {str(e)}")

        # Draw arena zones if available
        if self.arena_zones is not None and not self.arena_zones.empty:
            try:
                from matplotlib.patches import Polygon
                for zone_name in self.arena_zones['zone'].unique():
                    zone_points = self.arena_zones[self.arena_zones['zone'] == zone_name]
                    coords = zone_points[['x', 'y']].values
                    if len(coords) >= 3:
                        poly = Polygon(coords, fill=False, edgecolor='black', linewidth=1.5, linestyle='--', zorder=1)
                        self.ax.add_patch(poly)
                        centroid_x = coords[:, 0].mean()
                        centroid_y = coords[:, 1].mean()
                        self.ax.text(centroid_x, centroid_y, zone_name,
                                   fontsize=8, ha='center', va='center',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))
            except Exception as e:
                self.log_message(f"Error drawing zones: {str(e)}")

        # Plot anchor positions if available and toggled on
        if self.anchor_positions and self.chk_show_anchors.isChecked():
            try:
                for anchor in self.anchor_positions:
                    self.ax.scatter(anchor['x'], anchor['y'],
                                  marker='^', s=80, c='black', edgecolors='white',
                                  linewidths=1, zorder=4)
                    self.ax.text(anchor['x'], anchor['y'] + 0.05,
                               str(anchor['shortid']),
                               fontsize=7, ha='center', va='bottom', color='black')
            except Exception as e:
                self.log_message(f"Error drawing anchors: {str(e)}")

        # Get all data up to and including current timestamp
        current_data = self.preview_data[self.preview_data['Timestamp'] <= current_timestamp]

        # Determine color mode from preview dropdown
        preview_color_mode = self.combo_preview_color_by.currentText()  # "ID" or "Sex"

        # Build a color palette for ID-based coloring
        sorted_tags = sorted(self.preview_data['shortid'].unique())
        id_color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        tag_id_colors = {}
        for i, t in enumerate(sorted_tags):
            tag_id_colors[t] = id_color_palette[i % len(id_color_palette)]

        # Plot each tag
        for tag in sorted_tags:
            tag_all_data = current_data[current_data['shortid'] == tag]

            if len(tag_all_data) == 0:
                continue

            # Get the most recent position for this tag
            current_pos = tag_all_data.iloc[-1]

            # Determine label and color based on preview color mode
            if tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                label = f"{sex}-{identity}"
                if preview_color_mode == "Sex":
                    color = 'blue' if sex == 'M' else 'red'
                else:
                    color = tag_id_colors[tag]
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                label = f"HexID {hex_id}"
                color = tag_id_colors[tag]

            # Plot trail if enabled
            if self.chk_show_trail.isChecked() and len(tag_all_data) > 1:
                trail_seconds = self.spin_trail_length.value()
                trail_start_time = current_timestamp - pd.Timedelta(seconds=trail_seconds)
                trail_data = tag_all_data[tag_all_data['Timestamp'] >= trail_start_time]

                if len(trail_data) > 1:
                    self.ax.plot(trail_data[x_col], trail_data[y_col],
                               color=color, linewidth=2, alpha=0.6)

            # Plot current position with label
            self.ax.scatter(current_pos[x_col], current_pos[y_col],
                          c=color, s=100, marker='o', edgecolors='black', linewidths=2, zorder=5)
            self.ax.text(current_pos[x_col], current_pos[y_col], f'  {label}',
                        fontsize=10, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color))

        # Compute default (full-extent) limits
        x_min, x_max = self.preview_data[x_col].min(), self.preview_data[x_col].max()
        y_min, y_max = self.preview_data[y_col].min(), self.preview_data[y_col].max()

        # Include background image bounds
        if self.background_image is not None and self.bg_width_meters is not None:
            x_min = min(x_min, 0)
            x_max = max(x_max, self.bg_width_meters)
            y_min = min(y_min, 0)
            y_max = max(y_max, self.bg_height_meters)

        x_range = x_max - x_min
        y_range = y_max - y_min
        x_pad = x_range * 0.05 if x_range > 0 else 1
        y_pad = y_range * 0.05 if y_range > 0 else 1

        default_xlim = (x_min - x_pad, x_max + x_pad)
        default_ylim = (y_min - y_pad, y_max + y_pad)

        # Preserve user zoom: if old limits differ from stored defaults, user had zoomed
        if (self._default_xlim is not None
                and (old_xlim != self._default_xlim or old_ylim != self._default_ylim)):
            self.ax.set_xlim(old_xlim)
            self.ax.set_ylim(old_ylim)
        else:
            self.ax.set_xlim(default_xlim)
            self.ax.set_ylim(default_ylim)

        # Store current defaults so we can detect user zoom on next redraw
        self._default_xlim = default_xlim
        self._default_ylim = default_ylim

        self.ax.set_xlabel('X Position (m)', fontsize=10)
        self.ax.set_ylabel('Y Position (m)', fontsize=10)
        self.ax.set_title('UWB Tracking Preview', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        self.canvas.draw_idle()

    def rewind_playback(self):
        """Rewind to the beginning"""
        self.time_slider.setValue(0)
        if self.is_playing:
            self.toggle_play_pause()

    def toggle_play_pause(self):
        """Toggle between play and pause"""
        if self.is_playing:
            self.is_playing = False
            self.playback_timer.stop()
            self.btn_play_pause.setText("\u25b6 Play")
        else:
            self.is_playing = True
            base_interval = 1000
            interval = int(base_interval / self.playback_speed)
            self.playback_timer.start(interval)
            self.btn_play_pause.setText("\u23f8 Pause")

    def fast_forward_playback(self):
        """Fast forward to the end"""
        self.time_slider.setValue(self.time_slider.maximum())
        if self.is_playing:
            self.toggle_play_pause()

    def on_speed_changed(self, speed_text):
        """Handle playback speed change"""
        self.playback_speed = int(speed_text.replace('x', ''))
        if self.is_playing:
            base_interval = 1000
            interval = int(base_interval / self.playback_speed)
            self.playback_timer.start(interval)

    def advance_playback(self):
        """Advance playback by one frame"""
        current_value = self.time_slider.value()
        max_value = self.time_slider.maximum()
        if current_value < max_value:
            self.time_slider.setValue(current_value + 1)
        else:
            self.toggle_play_pause()

    def save_preview_image(self):
        """Save the current preview view as PNG or SVG"""
        if self.preview_data is None:
            QMessageBox.warning(self, "No Preview", "Please load a preview first")
            return

        # Default filename based on current timestamp
        current_idx = self.time_slider.value()
        if current_idx < len(self.unique_timestamps):
            ts = self.unique_timestamps[current_idx]
            default_name = f"preview_{ts.strftime('%Y%m%d_%H%M%S')}"
        else:
            default_name = "preview_image"

        # Determine default directory (same as database file)
        default_dir = os.path.dirname(self.db_path) if self.db_path else ""
        default_path = os.path.join(default_dir, default_name)

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Preview Image", default_path,
            "PNG Image (*.png);;SVG Image (*.svg);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            # Determine format from extension
            fmt = 'svg' if file_path.lower().endswith('.svg') else 'png'
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight', format=fmt)
            self.log_message(f"\u2713 Preview image saved: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")
            self.log_message(f"\u2717 Error saving preview image: {str(e)}")

    def get_config_dict(self):
        """Get current configuration as dictionary"""
        config = {
            'table_name': self.table_name,
            'selected_tags': [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()],
            'timezone': self.combo_timezone.currentText(),
            'smoothing_method': self.combo_smoothing.currentText(),
            'rolling_window': self.spin_rolling_window.value(),
            'velocity_filter': self.chk_velocity_filter.isChecked(),
            'velocity_threshold': self.spin_velocity_threshold.value(),
            'jump_filter': self.chk_jump_filter.isChecked(),
            'jump_threshold': self.spin_jump_threshold.value(),
            'time_gap': self.spin_time_gap.value(),
            'show_trail': self.chk_show_trail.isChecked(),
            'trail_length': self.spin_trail_length.value(),
            'export_raw_csv': self.chk_export_raw_csv.isChecked(),
            'export_smoothed_csv': self.chk_export_smoothed_csv.isChecked(),
            'export_downsampled_csv': self.chk_export_downsampled_csv.isChecked(),
            'downsample_hz': self.spin_downsample_hz.value(),
            'detect_behaviors': self.chk_detect_behaviors.isChecked(),
            'save_plots': self.chk_save_plots.isChecked(),
            'save_svg': self.chk_save_svg.isChecked(),
            'plot_types': {
                'daily_paths': self.plot_type_checkboxes['daily_paths'].isChecked(),
                'trajectory_overview': self.plot_type_checkboxes['trajectory_overview'].isChecked(),
                'battery_levels': self.plot_type_checkboxes['battery_levels'].isChecked(),
                '3d_occupancy': self.plot_type_checkboxes['3d_occupancy'].isChecked(),
                'activity_timeline': self.plot_type_checkboxes['activity_timeline'].isChecked(),
                'velocity_distribution': self.plot_type_checkboxes['velocity_distribution'].isChecked(),
                'cumulative_distance': self.plot_type_checkboxes['cumulative_distance'].isChecked(),
                'velocity_timeline': self.plot_type_checkboxes['velocity_timeline'].isChecked(),
                'actogram': self.plot_type_checkboxes['actogram'].isChecked(),
                'data_quality': self.plot_type_checkboxes['data_quality'].isChecked()
            },
            'save_animation': self.chk_save_animation.isChecked(),
            'animation_trail': self.spin_animation_trail.value(),
            'animation_speed': self.combo_animation_speed.currentText(),
            'animation_fps': self.combo_animation_fps.currentText(),
            'time_window': self.spin_time_window.value(),
            'color_by': self.combo_color_by.currentText(),
            'preview_color_by': self.combo_preview_color_by.currentText(),
            'tag_identities': self.tag_identities,
            'background_image_path': self.background_image_path,  # Save background image path
            'arena_zones': self.arena_zones.to_dict('records') if self.arena_zones is not None else None  # Save zone data
        }
        return config
    
    def save_config(self, output_dir):
        """Save current configuration to JSON file"""
        config = self.get_config_dict()
        config_path = os.path.join(output_dir, 'fnt_config.json')
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            self.lbl_status.setText(f"Config saved to {config_path}")
        except Exception as e:
            print(f"Warning: Could not save config: {str(e)}")
    
    def load_config_if_exists(self):
        """Check for existing config file and load it"""
        if not self.db_path:
            return
        
        db_dir = os.path.dirname(self.db_path)
        db_filename = os.path.basename(self.db_path)
        db_name = os.path.splitext(db_filename)[0]
        analysis_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis")
        config_path = os.path.join(analysis_dir, 'fnt_config.json')
        
        if not os.path.exists(config_path):
            return
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load configuration into GUI
            if 'table_name' in config and config['table_name']:
                # Store for deferred application after combo_table is populated
                self.pending_table_name = config['table_name']
                # Also try to set directly if combo is already populated
                index = self.combo_table.findText(config['table_name'])
                if index >= 0:
                    self.combo_table.setCurrentIndex(index)
            
            if 'timezone' in config:
                index = self.combo_timezone.findText(config['timezone'])
                if index >= 0:
                    self.combo_timezone.setCurrentIndex(index)
            
            if 'smoothing_method' in config:
                index = self.combo_smoothing.findText(config['smoothing_method'])
                if index >= 0:
                    self.combo_smoothing.setCurrentIndex(index)
            
            if 'rolling_window' in config:
                self.spin_rolling_window.setValue(config['rolling_window'])
            
            if 'velocity_filter' in config:
                self.chk_velocity_filter.setChecked(config['velocity_filter'])
            
            if 'velocity_threshold' in config:
                self.spin_velocity_threshold.setValue(config['velocity_threshold'])
            
            if 'jump_filter' in config:
                self.chk_jump_filter.setChecked(config['jump_filter'])
            
            if 'jump_threshold' in config:
                self.spin_jump_threshold.setValue(config['jump_threshold'])
            
            if 'time_gap' in config:
                self.spin_time_gap.setValue(config['time_gap'])
            
            if 'show_trail' in config:
                self.chk_show_trail.setChecked(config['show_trail'])
            
            if 'trail_length' in config:
                self.spin_trail_length.setValue(config['trail_length'])
            
            if 'export_raw_csv' in config:
                self.chk_export_raw_csv.setChecked(config['export_raw_csv'])

            if 'export_smoothed_csv' in config:
                self.chk_export_smoothed_csv.setChecked(config['export_smoothed_csv'])

            if 'export_downsampled_csv' in config:
                self.chk_export_downsampled_csv.setChecked(config['export_downsampled_csv'])

            if 'downsample_hz' in config:
                self.spin_downsample_hz.setValue(config['downsample_hz'])

            if 'detect_behaviors' in config:
                self.chk_detect_behaviors.setChecked(config['detect_behaviors'])
            
            if 'save_plots' in config:
                self.chk_save_plots.setChecked(config['save_plots'])

            if 'save_svg' in config:
                self.chk_save_svg.setChecked(config['save_svg'])

            if 'plot_types' in config:
                for key, value in config['plot_types'].items():
                    if key in self.plot_type_checkboxes:
                        self.plot_type_checkboxes[key].setChecked(value)
            
            if 'save_animation' in config:
                self.chk_save_animation.setChecked(config['save_animation'])
            
            if 'animation_trail' in config:
                self.spin_animation_trail.setValue(config['animation_trail'])
            
            if 'animation_speed' in config:
                index = self.combo_animation_speed.findText(config['animation_speed'])
                if index >= 0:
                    self.combo_animation_speed.setCurrentIndex(index)
            
            if 'animation_fps' in config:
                index = self.combo_animation_fps.findText(str(config['animation_fps']))
                if index >= 0:
                    self.combo_animation_fps.setCurrentIndex(index)
            
            if 'time_window' in config:
                self.spin_time_window.setValue(config['time_window'])
            
            if 'color_by' in config:
                index = self.combo_color_by.findText(config['color_by'])
                if index >= 0:
                    self.combo_color_by.setCurrentIndex(index)

            if 'preview_color_by' in config:
                index = self.combo_preview_color_by.findText(config['preview_color_by'])
                if index >= 0:
                    self.combo_preview_color_by.setCurrentIndex(index)

            if 'tag_identities' in config:
                # Convert string keys back to integers if needed
                self.tag_identities = {}
                for key, value in config['tag_identities'].items():
                    tag_key = int(key) if isinstance(key, str) and key.isdigit() else key
                    self.tag_identities[tag_key] = value
            
            # Load background image if path is saved and file exists
            if 'background_image_path' in config and config['background_image_path']:
                bg_path = config['background_image_path']
                # Ensure XML config is parsed first to get scale
                if not self.xml_scale and self.db_path:
                    db_dir = os.path.dirname(self.db_path)
                    xml_files = [f for f in os.listdir(db_dir) if f.lower().endswith('.xml')]
                    if xml_files:
                        xml_file = next((f for f in xml_files if 'config' in f.lower()), xml_files[0])
                        self.xml_config_path = os.path.join(db_dir, xml_file)
                        try:
                            self.parse_xml_config()
                        except:
                            pass
                
                # Try absolute path first
                if os.path.exists(bg_path):
                    self.background_image_path = bg_path
                    try:
                        self.background_image = plt.imread(bg_path)
                        img_height_px, img_width_px = self.background_image.shape[:2]
                        if self.xml_scale:
                            self.bg_width_meters = img_width_px * self.xml_scale * 0.0254
                            self.bg_height_meters = img_height_px * self.xml_scale * 0.0254
                            self.log_message(f"Background dimensions: {self.bg_width_meters:.2f}m x {self.bg_height_meters:.2f}m (scale: {self.xml_scale} in/px)")
                        else:
                            self.bg_width_meters = img_width_px * 0.0254
                            self.bg_height_meters = img_height_px * 0.0254
                            self.log_message(f"WARNING: No XML scale — background dimensions may be incorrect")
                        self.log_message(f"Background image loaded: {os.path.basename(bg_path)}")
                        self.lbl_background_status.setText(f"✓ Background: {os.path.basename(bg_path)}")
                        self.lbl_background_status.setStyleSheet("color: #00aa00; font-style: normal; font-size: 9px;")
                        self.btn_remove_background.setEnabled(True)
                        self._default_xlim = None
                        self._default_ylim = None
                    except Exception as e:
                        self.log_message(f"Warning: Could not load background image: {str(e)}")
                # Try relative to database directory
                elif os.path.exists(os.path.join(db_dir, os.path.basename(bg_path))):
                    bg_path = os.path.join(db_dir, os.path.basename(bg_path))
                    self.background_image_path = bg_path
                    try:
                        self.background_image = plt.imread(bg_path)
                        img_height_px, img_width_px = self.background_image.shape[:2]
                        if self.xml_scale:
                            self.bg_width_meters = img_width_px * self.xml_scale * 0.0254
                            self.bg_height_meters = img_height_px * self.xml_scale * 0.0254
                            self.log_message(f"Background dimensions: {self.bg_width_meters:.2f}m x {self.bg_height_meters:.2f}m (scale: {self.xml_scale} in/px)")
                        else:
                            self.bg_width_meters = img_width_px * 0.0254
                            self.bg_height_meters = img_height_px * 0.0254
                            self.log_message(f"WARNING: No XML scale — background dimensions may be incorrect")
                        self.log_message(f"Background image loaded: {os.path.basename(bg_path)}")
                        self.lbl_background_status.setText(f"✓ Background: {os.path.basename(bg_path)}")
                        self.lbl_background_status.setStyleSheet("color: #00aa00; font-style: normal; font-size: 9px;")
                        self.btn_remove_background.setEnabled(True)
                        self._default_xlim = None
                        self._default_ylim = None
                    except Exception as e:
                        self.log_message(f"Warning: Could not load background image: {str(e)}")
                else:
                    self.log_message(f"Warning: Saved background image not found: {bg_path}")
            
            # Note: selected_tags will be loaded after tags are populated from table
            if 'selected_tags' in config:
                self.pending_tag_selection = config['selected_tags']
            
            # Load zone data if present
            if 'arena_zones' in config and config['arena_zones'] is not None:
                self.arena_zones = pd.DataFrame(config['arena_zones'])
                if not self.arena_zones.empty:
                    num_zones = self.arena_zones['zone'].nunique()
                    num_points = len(self.arena_zones)
                    self.log_message(f"Loaded {num_zones} zones with {num_points} coordinate points from config")
            
            self.log_message(f"Loaded previous configuration from {config_path}")

            # Update tag labels if identities were loaded
            if self.tag_identities and self.tag_checkboxes:
                self.update_tag_labels()

            # --- Migration prompt: move loose files into subfolders ---
            plots_subdir = os.path.join(analysis_dir, 'plots')
            animations_subdir = os.path.join(analysis_dir, 'animations')

            loose_plots = []
            loose_animations = []

            if not os.path.exists(plots_subdir):
                loose_plots = [f for f in os.listdir(analysis_dir)
                               if os.path.isfile(os.path.join(analysis_dir, f))
                               and f.lower().endswith(('.png', '.svg'))]

            if not os.path.exists(animations_subdir):
                loose_animations = [f for f in os.listdir(analysis_dir)
                                    if os.path.isfile(os.path.join(analysis_dir, f))
                                    and f.lower().endswith('.mp4')]

            if loose_plots or loose_animations:
                parts = []
                if loose_plots:
                    parts.append(f"{len(loose_plots)} plot file(s)")
                if loose_animations:
                    parts.append(f"{len(loose_animations)} animation file(s)")
                file_desc = " and ".join(parts)

                reply = QMessageBox.question(
                    self, "Update Folder Structure",
                    f"Your analysis folder has {file_desc} in the root directory. "
                    f"Would you like to organize them into subfolders (plots/ and animations/) "
                    f"for better organization?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    moved = 0
                    if loose_plots:
                        os.makedirs(plots_subdir, exist_ok=True)
                        for plot_file in loose_plots:
                            src = os.path.join(analysis_dir, plot_file)
                            dst = os.path.join(plots_subdir, plot_file)
                            try:
                                shutil.move(src, dst)
                                moved += 1
                            except Exception as move_err:
                                self.log_message(f"Warning: Could not move {plot_file}: {move_err}")
                    if loose_animations:
                        os.makedirs(animations_subdir, exist_ok=True)
                        for anim_file in loose_animations:
                            src = os.path.join(analysis_dir, anim_file)
                            dst = os.path.join(animations_subdir, anim_file)
                            try:
                                shutil.move(src, dst)
                                moved += 1
                            except Exception as move_err:
                                self.log_message(f"Warning: Could not move {anim_file}: {move_err}")
                    self.log_message(f"Migrated {moved} file(s) into subfolders")

        except Exception as e:
            print(f"Warning: Could not load config: {str(e)}")
    
    def apply_pending_tag_selection(self):
        """Apply tag selection from loaded config after tags are populated"""
        if hasattr(self, 'pending_tag_selection') and self.pending_tag_selection:
            for tag, cb in self.tag_checkboxes.items():
                cb.setChecked(tag in self.pending_tag_selection)
            delattr(self, 'pending_tag_selection')
    
    def generate_animation(self, output_dir, total_export_steps=1, current_export_step=1, csv_path=None, animations_dir=None):
        """Generate animation video from tracking data"""
        try:
            if self.export_cancelled:
                return
            
            self.log_message("Preparing animation data...")

            # Resolve animations output directory
            if animations_dir is None:
                animations_dir = os.path.join(output_dir, 'animations')
            os.makedirs(animations_dir, exist_ok=True)

            # Use temp folder on C: drive (SSD) for faster frame writing
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            temp_frames_dir = os.path.join(desktop_path, "temp_animation_frames")
            os.makedirs(temp_frames_dir, exist_ok=True)
            self.log_message(f"Saving animation frames to: {temp_frames_dir}")
            self.log_message(f"Temp folder path: {os.path.abspath(temp_frames_dir)}")
            
            # Load data from CSV if provided (for consistency with plots)
            if csv_path and os.path.exists(csv_path):
                self.log_message(f"Loading animation data from CSV...")
                anim_data = pd.read_csv(csv_path, low_memory=False)
                # Parse Timestamp column (with mixed format to handle timezone-aware timestamps)
                anim_data['Timestamp'] = pd.to_datetime(anim_data['Timestamp'], format='mixed')
                self.log_message(f"Loaded {len(anim_data)} records from CSV")
            else:
                # Fallback to using self.data
                self.log_message("Using in-memory data for animation...")
                anim_data = self.data.copy()
            
            # Map column names to match animate_path expectations
            if 'shortid' in anim_data.columns:
                anim_data['ID'] = anim_data['shortid']
            
            # Add required columns if they don't exist
            if 'sex' not in anim_data.columns:
                anim_data['sex'] = 'M'  # Default to male if sex not available
            
            # Apply custom sex and identities if configured
            if self.tag_identities:
                for tag_id, info in self.tag_identities.items():
                    mask = anim_data['ID'] == tag_id
                    anim_data.loc[mask, 'sex'] = info['sex']
                    anim_data.loc[mask, 'custom_identity'] = info['identity']
            
            if 'Day' not in anim_data.columns:
                anim_data['Day'] = 1  # Default to day 1
            if 'trial' not in anim_data.columns:
                anim_data['trial'] = self.table_name
            
            # Use smoothed coordinates if available, otherwise use location
            x_col = 'smoothed_x' if 'smoothed_x' in anim_data.columns else 'location_x'
            y_col = 'smoothed_y' if 'smoothed_y' in anim_data.columns else 'location_y'
            
            if x_col != 'smoothed_x':
                anim_data['smoothed_x'] = anim_data[x_col]
            if y_col != 'smoothed_y':
                anim_data['smoothed_y'] = anim_data[y_col]
            
            # Calculate heading and velocity
            anim_data = anim_data.sort_values(by=['ID', 'Timestamp'])
            
            # Calculate heading (direction of movement)
            def calc_heading(group):
                return np.arctan2(group['smoothed_y'].diff(1), group['smoothed_x'].diff(1))
            
            anim_data['heading'] = anim_data.groupby('ID', group_keys=False).apply(calc_heading, include_groups=False).reset_index(level=0, drop=True)
            
            # Smooth heading with rolling average
            anim_data['heading'] = anim_data.groupby('ID', group_keys=False)['heading'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            
            # Calculate velocity
            def calc_velocity(group):
                return np.sqrt(group['smoothed_x'].diff()**2 + group['smoothed_y'].diff()**2) / group['Timestamp'].diff().dt.total_seconds()
            
            anim_data['velocity'] = anim_data.groupby('ID', group_keys=False).apply(calc_velocity, include_groups=False).reset_index(level=0, drop=True)
            
            # Get animation parameters
            time_window = self.spin_time_window.value()
            trailing_window = self.spin_animation_trail.value()
            fps = int(self.combo_animation_fps.currentText())
            speed_text = self.combo_animation_speed.currentText()
            speed_multiplier = int(speed_text.replace('x', ''))
            color_by = self.combo_color_by.currentText()
            
            # Calculate frame interval: how many real seconds each frame represents
            # speed_multiplier seconds of real time per second of video / fps frames per second
            frame_interval = speed_multiplier / fps
            self.log_message(f"Animation: {speed_text} speed at {fps} FPS (each frame = {frame_interval:.2f}s of real time)")
            
            self.log_message("Setting up animation frames...")
            
            # Check if daily animations are requested
            generate_daily = self.chk_daily_animations.isChecked()
            
            if generate_daily:
                # Get selected days
                selected_days = [date_str for date_str, cb in self.daily_animation_day_checkboxes.items() if cb.isChecked()]
                if not selected_days:
                    self.log_message("⚠ No days selected for daily animations")
                    return
                
                self.log_message(f"Generating {len(selected_days)} daily animations...")
                
                # Generate one animation per selected day
                for day_idx, date_str in enumerate(selected_days):
                    if self.export_cancelled:
                        return
                    
                    self.log_message(f"Processing Day {day_idx + 1}/{len(selected_days)}: {date_str}")
                    
                    # Filter data for this specific day (midnight to midnight)
                    # Get timezone from the data to ensure compatible comparison
                    data_tz = anim_data['Timestamp'].dt.tz
                    date_obj = pd.to_datetime(date_str).date()
                    
                    if data_tz is not None:
                        # Create timezone-aware timestamps matching the data's timezone
                        day_start = pd.Timestamp(date_obj, tz=data_tz)
                        day_end = day_start + pd.Timedelta(days=1)
                    else:
                        # Data is timezone-naive
                        day_start = pd.Timestamp(date_obj)
                        day_end = day_start + pd.Timedelta(days=1)
                    
                    day_data = anim_data[(anim_data['Timestamp'] >= day_start) & (anim_data['Timestamp'] < day_end)].copy()
                    
                    if len(day_data) == 0:
                        self.log_message(f"⚠ No data for {date_str}, skipping")
                        continue
                    
                    # Get speed text for display
                    speed_text = self.combo_animation_speed.currentText()
                    
                    # Generate animation for this day
                    video_path = self.create_animation_frames(
                        day_data, temp_frames_dir, frame_interval, trailing_window,
                        fps, color_by, bool(self.tag_identities),
                        total_export_steps, current_export_step,
                        day_suffix=f"_Day{day_idx + 1}_{date_str}",
                        speed_text=speed_text
                    )
                    
                    if video_path and not self.export_cancelled:
                        # Move video to final location with FPS and speed in filename
                        db_filename = os.path.basename(self.db_path)
                        db_name = os.path.splitext(db_filename)[0]
                        speed_text = self.combo_animation_speed.currentText()
                        final_video_path = os.path.join(animations_dir, f"{db_name}_Animation_Day{day_idx + 1}_{date_str}_{fps}fps_{speed_text}.mp4")
                        
                        # Check if file exists and skip if not overwriting
                        if os.path.exists(final_video_path):
                            self.log_message(f"⚠ Day {day_idx + 1} animation already exists, skipping: {os.path.basename(final_video_path)}")
                        elif os.path.exists(video_path):
                            shutil.move(video_path, final_video_path)
                            self.log_message(f"✓ Day {day_idx + 1} animation saved: {final_video_path}")
                
                # Clean up temp frames
                try:
                    shutil.rmtree(temp_frames_dir)
                    self.log_message("✓ Temp frames cleaned up")
                except Exception as e:
                    self.log_message(f"Warning: Could not clean temp frames: {str(e)}")
            else:
                # Generate single animation for all data
                # Clean existing temp frames
                if os.path.exists(temp_frames_dir):
                    for filename in os.listdir(temp_frames_dir):
                        file_path = os.path.join(temp_frames_dir, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                
                # Create animation
                self.log_message("Generating animation frames (this may take a while)...")
                speed_text = self.combo_animation_speed.currentText()
                video_path = self.create_animation_frames(anim_data, temp_frames_dir, frame_interval, trailing_window, 
                                            fps, color_by, bool(self.tag_identities),
                                            total_export_steps, current_export_step,
                                            speed_text=speed_text)
                
                if video_path and not self.export_cancelled:
                    # Move video to final location with FPS and speed in filename
                    db_filename = os.path.basename(self.db_path)
                    db_name = os.path.splitext(db_filename)[0]
                    speed_text = self.combo_animation_speed.currentText()
                    final_video_path = os.path.join(animations_dir, f"{db_name}_Animation_{fps}fps_{speed_text}.mp4")
                    
                    # Check if file exists and skip if not overwriting
                    if os.path.exists(final_video_path):
                        self.log_message(f"⚠ Animation already exists, skipping: {os.path.basename(final_video_path)}")
                    elif os.path.exists(video_path):
                        shutil.move(video_path, final_video_path)
                        self.log_message(f"✓ Animation saved: {final_video_path}")
                    
                    # Clean up temp frames
                    try:
                        shutil.rmtree(temp_frames_dir)
                        self.log_message("✓ Temp frames cleaned up")
                    except Exception as e:
                        self.log_message(f"Warning: Could not clean temp frames: {str(e)}")
            
            self.log_message("✓ Animation generation complete!")
            
            # Reset UI after animation completes
            self.exporting = False
            self.btn_export.setEnabled(True)
            self.btn_stop_export.setVisible(False)
            self.progress_bar.setValue(100)
            self.lbl_export_progress.setText("All exports complete!")
            QMessageBox.information(self, "Success", "All exports completed successfully!")
            QTimer.singleShot(3000, lambda: self.progress_widget.setVisible(False))
            
        except Exception as e:
            QMessageBox.critical(self, "Animation Error", f"Failed to generate animation: {str(e)}")
            self.log_message(f"✗ Animation generation failed: {str(e)}")
    
    def create_animation_frames(self, data, output_dir, frame_interval, trailing_window, 
                               fps, color_by, use_custom_identities=False,
                               total_export_steps=1, current_export_step=1, day_suffix="", speed_text=""):
        """Create animation frames and compile video with optimization strategies:
        1. In-memory rendering (no temp PNG files)
        2. Configurable DPI for speed vs quality
        3. Matplotlib blitting for faster redraw
        4. Parallel frame generation
        """
        
        # Get DPI from quality setting
        quality_map = {"Draft (Fast)": 75, "Standard": 100, "High Quality": 150}
        dpi = quality_map.get(self.combo_video_quality.currentText(), 100)
        self.log_message(f"Using {dpi} DPI for video generation")
        
        # Get global min/max for consistent axis limits
        x_min, x_max = data['smoothed_x'].min(), data['smoothed_x'].max()
        y_min, y_max = data['smoothed_y'].min(), data['smoothed_y'].max()
        
        # If background image exists, adjust limits to include it (using meters)
        if self.background_image is not None and self.bg_width_meters is not None:
            x_min = min(x_min, 0)
            x_max = max(x_max, self.bg_width_meters)
            y_min = min(y_min, 0)
            y_max = max(y_max, self.bg_height_meters)
        
        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_pad = x_range * 0.05 if x_range > 0 else 1
        y_pad = y_range * 0.05 if y_range > 0 else 1
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad
        
        # Calculate time range based on frame_interval
        start = data['Timestamp'].min()
        end = data['Timestamp'].max()
        
        # Generate time points manually to avoid floating point precision issues with pd.date_range
        total_seconds = (end - start).total_seconds()
        num_frames = int(total_seconds / frame_interval) + 1
        time_starts = [start + pd.Timedelta(seconds=i * frame_interval) for i in range(num_frames)]
        time_starts = [t for t in time_starts if t <= end]  # Filter to ensure we don't exceed end time
        
        total_frames = len(time_starts)
        self.log_message(f"Creating {total_frames} animation frames (optimized pipeline)...")
        
        # Pre-compute all data for each tag to avoid repeated filtering
        self.log_message("Pre-computing trajectories...")
        tag_data_dict = {}
        for tag in data['shortid'].unique():
            tag_subset = data[data['shortid'] == tag][['Timestamp', 'smoothed_x', 'smoothed_y']].copy()
            tag_subset = tag_subset.sort_values('Timestamp')
            tag_data_dict[tag] = tag_subset
            
            # Pre-compute label and color
            if use_custom_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                label_text = f"{sex}-{identity}"
                color = 'blue' if sex == 'M' else 'red'
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                label_text = f"HexID {hex_id}"
                color = 'blue'
            tag_data_dict[tag] = {
                'data': tag_subset,
                'label': label_text,
                'color': color
            }
        
        #===========================================
        # OPTIMIZATION 1 & 3: In-memory rendering with background blitting
        #===========================================
        
        # Create figure once and reuse (blitting optimization)
        fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
        ax.grid(False)
        
        # Draw static background once
        bg_artist = None
        if self.background_image is not None and self.bg_width_meters is not None:
            bg_artist = ax.imshow(self.background_image, 
                     extent=[0, self.bg_width_meters, 0, self.bg_height_meters],
                     origin='lower',
                     aspect='auto',
                     alpha=0.6,
                     zorder=0)
        
        # Draw arena zones if available (static, drawn once)
        zone_artists = []
        if self.arena_zones is not None and not self.arena_zones.empty:
            try:
                from matplotlib.patches import Polygon
                for zone_name in self.arena_zones['zone'].unique():
                    zone_points = self.arena_zones[self.arena_zones['zone'] == zone_name]
                    coords = zone_points[['x', 'y']].values
                    if len(coords) >= 3:  # Need at least 3 points for a polygon
                        poly = Polygon(coords, fill=False, edgecolor='black', linewidth=1.5, linestyle='--', zorder=1)
                        ax.add_patch(poly)
                        zone_artists.append(poly)
                        # Add zone label at centroid
                        centroid_x = coords[:, 0].mean()
                        centroid_y = coords[:, 1].mean()
                        text = ax.text(centroid_x, centroid_y, zone_name, 
                                     fontsize=8, ha='center', va='center', 
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5),
                                     zorder=1)
                        zone_artists.append(text)
            except Exception as e:
                self.log_message(f"Error drawing zones in animation: {str(e)}")
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_xlabel("X Position (m)", fontsize=12)
        ax.set_ylabel("Y Position (m)", fontsize=12)
        
        # Render figure to get dimensions
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        
        # Initialize video writer
        video_filename = f'animation_temp{day_suffix}.mp4' if day_suffix else 'animation_temp.mp4'
        video_output_path = os.path.join(output_dir, video_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            self.log_message("✗ Could not open VideoWriter")
            plt.close(fig)
            return None
        
        self.log_message(f"Video dimensions: {width}x{height}")
        
        #===========================================
        # Generate and write frames directly to video (in-memory)
        #===========================================
        
        for i, frame_start in enumerate(time_starts):
            if self.export_cancelled:
                video_writer.release()
                plt.close(fig)
                self.log_message("✗ Animation cancelled")
                return None
            
            # Update progress
            if i % 10 == 0 or i == total_frames - 1:
                progress_pct = int(((current_export_step - 1) + (i + 1) / total_frames) / total_export_steps * 100)
                self.progress_bar.setValue(progress_pct)
                self.lbl_export_progress.setText(f"Step {current_export_step}/{total_export_steps}: Rendering frame {i+1}/{total_frames}...")
                if i % 50 == 0:
                    self.log_message(f"Rendering frame {i+1}/{total_frames}...")
                QApplication.processEvents()
            
            frame_end = frame_start + pd.Timedelta(seconds=trailing_window)
            
            # Clear previous frame's dynamic content
            ax.clear()
            
            # Redraw static background
            if bg_artist is not None:
                ax.imshow(self.background_image, 
                         extent=[0, self.bg_width_meters, 0, self.bg_height_meters],
                         origin='lower',
                         aspect='auto',
                         alpha=0.6,
                         zorder=0)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')
            ax.set_xlabel("X Position (m)", fontsize=12)
            ax.set_ylabel("Y Position (m)", fontsize=12)
            
            # Build title with speed if provided
            title_text = "UWB Tracking Animation"
            if speed_text:
                title_text += f" - Speed: {speed_text}"
            title_text += f"\nTime: {frame_start.strftime('%Y-%m-%d %H:%M:%S')}"
            ax.set_title(title_text, fontsize=14, fontweight='bold')
            
            # Plot each tag's trajectory
            for tag, tag_info in tag_data_dict.items():
                tag_df = tag_info['data']
                trailing_data = tag_df[(tag_df['Timestamp'] >= frame_start) & 
                                       (tag_df['Timestamp'] <= frame_end)]
                
                if trailing_data.empty:
                    continue
                
                label = tag_info['label']
                color = tag_info['color']
                
                # Plot trailing line
                ax.plot(trailing_data['smoothed_x'], trailing_data['smoothed_y'], 
                       color=color, alpha=0.5, linewidth=1)
                
                # Plot current position
                current_x = trailing_data['smoothed_x'].iloc[-1]
                current_y = trailing_data['smoothed_y'].iloc[-1]
                ax.plot(current_x, current_y, 'o', color=color, markersize=10)
                
                # Add label
                ax.text(current_x, current_y + (y_range * 0.02), label, 
                       fontsize=10, ha='center', color=color, fontweight='bold')
            
            # Render to numpy array (IN-MEMORY - no disk I/O!)
            fig.canvas.draw()
            # Use buffer_rgba for Qt backend compatibility
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            
            # Convert RGBA to BGR for OpenCV (drop alpha channel)
            frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
            
            # Write directly to video
            video_writer.write(frame_bgr)
        
        # Clean up
        video_writer.release()
        plt.close(fig)
        gc.collect()
        
        # Check if video was created successfully
        if os.path.exists(video_output_path) and os.path.getsize(video_output_path) > 0:
            self.log_message(f"✓ Video compilation complete: {os.path.getsize(video_output_path):,} bytes")
            return video_output_path
        else:
            self.log_message("✗ Video file was not created or is empty")
            return None
    
    def stop_export(self):
        """Cancel ongoing export operations"""
        self.export_cancelled = True
        self.log_message("⚠ Export cancellation requested...")
        
        # Stop worker thread if running
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.log_message("✗ Plot export cancelled")
        
        # Reset UI
        self.exporting = False
        self.btn_export.setEnabled(True)
        self.btn_stop_export.setVisible(False)
        self.progress_widget.setVisible(False)
        self.progress_bar.setValue(0)
        self.lbl_export_progress.setText("")
    
    def export_data(self):
        """Export data and/or plots based on selected options"""
        if not self.db_path:
            QMessageBox.warning(self, "No Database", "Please select a database first")
            return
        
        # Note: self.data can be None if this is a fresh export
        # This is OK - plots and animations will load data directly from CSV/database
        
        # Initialize export state
        self.export_cancelled = False
        self.exporting = True
        self.btn_export.setEnabled(False)
        self.btn_stop_export.setVisible(True)
        self.progress_widget.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Gather export settings
        db_dir = os.path.dirname(self.db_path)
        db_filename = os.path.basename(self.db_path)
        db_name = os.path.splitext(db_filename)[0]  # Remove extension

        export_raw_csv = self.chk_export_raw_csv.isChecked()
        export_smoothed_csv = self.chk_export_smoothed_csv.isChecked()
        export_downsampled_csv = self.chk_export_downsampled_csv.isChecked()
        save_plots = self.chk_save_plots.isChecked()
        save_animation = self.chk_save_animation.isChecked()
        downsample_hz = self.spin_downsample_hz.value()

        detect_behaviors = self.chk_detect_behaviors.isChecked()

        if not export_raw_csv and not export_smoothed_csv and not export_downsampled_csv and not save_plots and not save_animation and not detect_behaviors:
            QMessageBox.warning(self, "No Export Selected", "Please select at least one export option (CSV, Plots, or Animation)")
            return

        # --- Conflict detection ---
        base_output_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis")
        plots_subdir = os.path.join(base_output_dir, 'plots')
        animations_subdir = os.path.join(base_output_dir, 'animations')

        # Predict which files will be produced (root-level CSVs and behavior files)
        predicted_files = []
        if export_raw_csv:
            predicted_files.append(f'{db_name}_raw.csv')
        if export_smoothed_csv:
            predicted_files.append(f'{db_name}_smoothed.csv')
        if export_downsampled_csv:
            predicted_files.append(f'{db_name}_downsampled_{downsample_hz}Hz.csv')
        if detect_behaviors:
            predicted_files.append(f'{db_name}_behavior_timeline.csv')
            predicted_files.append(f'{db_name}_behavior_summary.csv')
            predicted_files.append(f'{db_name}_social_interactions.csv')
            predicted_files.append(f'{db_name}_social_summary.csv')

        # Predict animation files (in animations/ subfolder)
        predicted_animation_files = []
        if save_animation:
            fps = int(self.combo_animation_fps.currentText())
            speed_text = self.combo_animation_speed.currentText()
            if self.chk_daily_animations.isChecked():
                selected_days = [d for d, cb in self.daily_animation_day_checkboxes.items() if cb.isChecked()]
                for day_idx, date_str in enumerate(selected_days):
                    predicted_animation_files.append(f'{db_name}_Animation_Day{day_idx + 1}_{date_str}_{fps}fps_{speed_text}.mp4')
            else:
                predicted_animation_files.append(f'{db_name}_Animation_{fps}fps_{speed_text}.mp4')

        # Predict plot files (in plots/ subfolder)
        predicted_plot_files = []
        if save_plots:
            selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]
            plot_types = {k: self.plot_type_checkboxes[k].isChecked() for k in self.plot_type_checkboxes}

            if plot_types.get('daily_paths', False):
                for tag in selected_tags:
                    hex_id = hex(tag).upper().replace('0X', '')
                    predicted_plot_files.append(f'{db_name}_daily_paths_HexID_{hex_id}.png')
                    if self.chk_save_svg.isChecked():
                        predicted_plot_files.append(f'{db_name}_daily_paths_HexID_{hex_id}.svg')
            if plot_types.get('trajectory_overview', False):
                predicted_plot_files.append(f'{db_name}_trajectory_overview.png')
                if self.chk_save_svg.isChecked():
                    predicted_plot_files.append(f'{db_name}_trajectory_overview.svg')
            if plot_types.get('battery_levels', False):
                predicted_plot_files.append(f'{db_name}_battery_levels.png')
                if self.chk_save_svg.isChecked():
                    predicted_plot_files.append(f'{db_name}_battery_levels.svg')
            if plot_types.get('3d_occupancy', False):
                for tag in selected_tags:
                    hex_id = hex(tag).upper().replace('0X', '')
                    predicted_plot_files.append(f'{db_name}_3d_occupancy_HexID_{hex_id}.png')
                    if self.chk_save_svg.isChecked():
                        predicted_plot_files.append(f'{db_name}_3d_occupancy_HexID_{hex_id}.svg')
            if plot_types.get('activity_timeline', False):
                predicted_plot_files.append(f'{db_name}_activity_timeline.png')
                if self.chk_save_svg.isChecked():
                    predicted_plot_files.append(f'{db_name}_activity_timeline.svg')
            if plot_types.get('velocity_distribution', False):
                predicted_plot_files.append(f'{db_name}_velocity_distribution.png')
                if self.chk_save_svg.isChecked():
                    predicted_plot_files.append(f'{db_name}_velocity_distribution.svg')
            if plot_types.get('cumulative_distance', False):
                predicted_plot_files.append(f'{db_name}_cumulative_distance.png')
                if self.chk_save_svg.isChecked():
                    predicted_plot_files.append(f'{db_name}_cumulative_distance.svg')
            if plot_types.get('velocity_timeline', False):
                predicted_plot_files.append(f'{db_name}_velocity_timeline.png')
                if self.chk_save_svg.isChecked():
                    predicted_plot_files.append(f'{db_name}_velocity_timeline.svg')
            if plot_types.get('actogram', False):
                for tag in selected_tags:
                    hex_id = hex(tag).upper().replace('0X', '')
                    predicted_plot_files.append(f'{db_name}_actogram_HexID_{hex_id}.png')
                    if self.chk_save_svg.isChecked():
                        predicted_plot_files.append(f'{db_name}_actogram_HexID_{hex_id}.svg')
            if plot_types.get('data_quality', False):
                predicted_plot_files.append(f'{db_name}_data_quality.png')
                if self.chk_save_svg.isChecked():
                    predicted_plot_files.append(f'{db_name}_data_quality.svg')

        # Check for conflicts against existing files
        conflicting_root = [f for f in predicted_files if os.path.exists(os.path.join(base_output_dir, f))]
        conflicting_plots = [f for f in predicted_plot_files if os.path.exists(os.path.join(plots_subdir, f))]
        conflicting_animations = [f for f in predicted_animation_files if os.path.exists(os.path.join(animations_subdir, f))]
        num_conflicts = len(conflicting_root) + len(conflicting_plots) + len(conflicting_animations)
        total_new = len(predicted_files) + len(predicted_plot_files) + len(predicted_animation_files)

        skip_existing = False
        output_dir = base_output_dir

        if num_conflicts > 0:
            dialog = ExportConflictDialog(total_new, num_conflicts, parent=self)
            result = dialog.exec_()

            if result == ExportConflictDialog.SKIP:
                skip_existing = True
                output_dir = base_output_dir
            elif result == ExportConflictDialog.OVERWRITE:
                skip_existing = False
                output_dir = base_output_dir
                # Delete conflicting files so they get cleanly rewritten
                for f in conflicting_root:
                    try:
                        os.remove(os.path.join(base_output_dir, f))
                    except Exception:
                        pass
                for f in conflicting_plots:
                    try:
                        os.remove(os.path.join(plots_subdir, f))
                    except Exception:
                        pass
                for f in conflicting_animations:
                    try:
                        os.remove(os.path.join(animations_subdir, f))
                    except Exception:
                        pass
            elif result == ExportConflictDialog.NEW_FOLDER:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis_{timestamp}")
                skip_existing = False
            else:
                # User cancelled
                self.exporting = False
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setVisible(False)
                self.progress_widget.setVisible(False)
                return

        # Create output directory and subfolders
        os.makedirs(output_dir, exist_ok=True)
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        animations_dir = os.path.join(output_dir, 'animations')
        os.makedirs(animations_dir, exist_ok=True)

        try:
            self.log_message(f"Starting export to {output_dir}")
            self.lbl_export_progress.setText("Initializing export...")

            # Calculate total steps for progress
            total_steps = 0
            if export_raw_csv:
                total_steps += 1
            if export_smoothed_csv:
                total_steps += 1
            if export_downsampled_csv:
                total_steps += 1
            if save_plots:
                total_steps += 1
            if save_animation:
                total_steps += 1

            current_step = 0

            # Note: Conflict resolution (skip/overwrite/new folder) was handled above

            # Export raw CSV (unprocessed database dump)
            if export_raw_csv:
                if self.export_cancelled:
                    self.stop_export()
                    return

                current_step += 1
                self.lbl_export_progress.setText(f"Step {current_step}/{total_steps}: Exporting raw CSV...")
                self.progress_bar.setValue(int(current_step / total_steps * 100))
                QApplication.processEvents()

                raw_csv_filename = f'{db_name}_raw.csv'
                raw_csv_path = os.path.join(output_dir, raw_csv_filename)
                self.log_message("Exporting raw database to CSV...")
                conn = sqlite3.connect(self.db_path)
                raw_data = pd.read_sql_query(f"SELECT * FROM {self.table_name}", conn)
                conn.close()
                raw_data.to_csv(raw_csv_path, index=False)
                self.log_message(f"✓ Raw CSV exported: {raw_csv_filename}")

            # Prepare processed data (needed for smoothed CSV, downsampled CSV, plots, animation, behaviors)
            needs_processed_data = export_smoothed_csv or export_downsampled_csv or save_plots or save_animation or detect_behaviors
            smoothed_data = None
            csv_path = None  # Path to the CSV that plots/animation will use

            if needs_processed_data:
                if self.export_cancelled:
                    self.stop_export()
                    return

                self.log_message("Preparing processed data...")

                # Load data if not already in memory
                if self.data is None:
                    self.log_message("Loading data from database...")
                    conn = sqlite3.connect(self.db_path)
                    query = f"SELECT * FROM {self.table_name}"
                    csv_data = pd.read_sql_query(query, conn)
                    conn.close()

                    csv_data['Timestamp'] = pd.to_datetime(csv_data['timestamp'], unit='ms', origin='unix', utc=True)
                    tz = pytz.timezone(self.combo_timezone.currentText())
                    csv_data['Timestamp'] = csv_data['Timestamp'].dt.tz_convert(tz)
                    csv_data['location_x'] *= 0.0254
                    csv_data['location_y'] *= 0.0254
                    csv_data = csv_data.sort_values(by=['shortid', 'Timestamp'])

                    selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]
                    if selected_tags:
                        csv_data = csv_data[csv_data['shortid'].isin(selected_tags)]

                    # Apply per-tag time trimming (before filtering/smoothing)
                    if self.tag_identities:
                        for tag, info in self.tag_identities.items():
                            if 'start_time' in info and 'stop_time' in info:
                                start = pd.Timestamp(info['start_time']).tz_localize(tz)
                                stop = pd.Timestamp(info['stop_time']).tz_localize(tz)
                                mask = (csv_data['shortid'] == tag) & (
                                    (csv_data['Timestamp'] < start) | (csv_data['Timestamp'] > stop))
                                trimmed = mask.sum()
                                if trimmed > 0:
                                    csv_data = csv_data[~mask]
                                    self.log_message(f"  Trimmed {trimmed} points outside time window for tag {tag}")

                    # Apply filtering (before smoothing)
                    if self.chk_velocity_filter.isChecked() or self.chk_jump_filter.isChecked():
                        self.log_message("Applying velocity/jump filtering...")
                        csv_data = self.apply_filters_to_data(csv_data)

                    # Apply smoothing (after filtering)
                    if self.get_smoothing_method() != "None":
                        self.log_message("Applying smoothing...")
                        csv_data = self.apply_smoothing_to_data(csv_data, self.get_smoothing_method())
                else:
                    csv_data = self.data.copy()

                # Ensure sex and identity columns
                if 'sex' not in csv_data.columns or 'identity' not in csv_data.columns:
                    if self.tag_identities:
                        csv_data['sex'] = csv_data['shortid'].map(lambda x: self.tag_identities.get(x, {}).get('sex', 'M'))
                        csv_data['identity'] = csv_data['shortid'].map(lambda x: self.tag_identities.get(x, {}).get('identity', f'Tag{x}'))
                    else:
                        csv_data['sex'] = 'M'
                        csv_data['identity'] = csv_data['shortid'].apply(lambda x: f'Tag{x}')

                # This is the smoothed (full resolution) data
                smoothed_data = csv_data

                # Export smoothed CSV
                if export_smoothed_csv:
                    current_step += 1
                    self.lbl_export_progress.setText(f"Step {current_step}/{total_steps}: Exporting smoothed CSV...")
                    self.progress_bar.setValue(int(current_step / total_steps * 100))
                    QApplication.processEvents()

                    smoothed_csv_filename = f'{db_name}_smoothed.csv'
                    smoothed_csv_path = os.path.join(output_dir, smoothed_csv_filename)
                    smoothed_data.to_csv(smoothed_csv_path, index=False)
                    self.log_message(f"✓ Smoothed CSV exported: {smoothed_csv_filename}")
                    csv_path = smoothed_csv_path  # Use smoothed for plots/animation

                # Export downsampled CSV
                if export_downsampled_csv:
                    current_step += 1
                    self.lbl_export_progress.setText(f"Step {current_step}/{total_steps}: Exporting downsampled CSV ({downsample_hz}Hz)...")
                    self.progress_bar.setValue(int(current_step / total_steps * 100))
                    QApplication.processEvents()

                    downsampled_data = smoothed_data.copy()
                    # Downsample by grouping into time bins of 1/Hz seconds
                    bin_ns = int(1_000_000_000 / downsample_hz)
                    downsampled_data['time_bin'] = (downsampled_data['Timestamp'].astype(np.int64) // bin_ns).astype(int)
                    downsampled_data = downsampled_data.groupby(['shortid', 'time_bin']).first().reset_index()
                    if 'time_bin' in downsampled_data.columns:
                        downsampled_data = downsampled_data.drop(columns=['time_bin'])

                    ds_csv_filename = f'{db_name}_smoothed_{downsample_hz}Hz.csv'
                    ds_csv_path = os.path.join(output_dir, ds_csv_filename)
                    downsampled_data.to_csv(ds_csv_path, index=False)
                    self.log_message(f"✓ Downsampled CSV exported: {ds_csv_filename}")
                    csv_path = ds_csv_path  # Prefer downsampled for plots/animation

                # If neither CSV was exported but plots/animation need data, create a temp CSV
                if csv_path is None and (save_plots or save_animation):
                    csv_filename = f'{db_name}_smoothed_{downsample_hz}Hz.csv'
                    csv_path = os.path.join(output_dir, csv_filename)
                    # Downsample for plots/animation
                    temp_data = smoothed_data.copy()
                    bin_ns = int(1_000_000_000 / downsample_hz)
                    temp_data['time_bin'] = (temp_data['Timestamp'].astype(np.int64) // bin_ns).astype(int)
                    temp_data = temp_data.groupby(['shortid', 'time_bin']).first().reset_index()
                    if 'time_bin' in temp_data.columns:
                        temp_data = temp_data.drop(columns=['time_bin'])
                    temp_data.to_csv(csv_path, index=False)
                    self.log_message(f"✓ Temporary CSV created for plots/animation")
            
            # Save configuration file
            self.save_config(output_dir)
            
            # Save message log and run summary
            self.save_message_log(output_dir)
            self.save_run_summary(output_dir)
            
            # Detect behaviors if requested
            if detect_behaviors:
                if self.export_cancelled:
                    self.stop_export()
                    return
                
                self.log_message("=" * 50)
                self.log_message("Starting behavioral analysis...")
                self.lbl_export_progress.setText("Analyzing behaviors and social interactions...")
                QApplication.processEvents()
                
                try:
                    from fnt.uwb.behavior_detection import BehaviorDetector
                    
                    # Load CSV data if not already loaded
                    if not os.path.exists(csv_path):
                        self.log_message("Error: CSV file not found for behavior analysis")
                    else:
                        behavior_data = pd.read_csv(csv_path, low_memory=False)
                        behavior_data['Timestamp'] = pd.to_datetime(behavior_data['Timestamp'], format='ISO8601')
                        
                        # Initialize behavior detector with default thresholds
                        detector = BehaviorDetector(
                            speed_threshold_rest=0.05,      # m/s
                            speed_threshold_active=0.3,     # m/s
                            distance_threshold_social=0.5,  # m
                            window_seconds=10.0,
                            overlap_seconds=5.0
                        )
                        
                        # Run behavior detection
                        behavior_timeline, social_interactions = detector.analyze_behaviors(
                            behavior_data, 
                            log_callback=self.log_message
                        )
                        
                        # Generate summary reports
                        if not behavior_timeline.empty:
                            # Behavior summary per animal
                            behavior_summary = detector.generate_behavior_summary(
                                behavior_timeline, 
                                social_interactions,
                                tag_identities=self.tag_identities
                            )
                            
                            # Social interaction summary
                            social_summary = detector.generate_social_interaction_summary(
                                social_interactions,
                                tag_identities=self.tag_identities
                            )
                            
                            # Export behavior timeline (detailed)
                            behavior_timeline_path = os.path.join(output_dir, f'{db_name}_behavior_timeline.csv')
                            behavior_timeline.to_csv(behavior_timeline_path, index=False)
                            self.log_message(f"✓ Exported behavior timeline: {os.path.basename(behavior_timeline_path)}")
                            
                            # Export behavior summary
                            behavior_summary_path = os.path.join(output_dir, f'{db_name}_behavior_summary.csv')
                            behavior_summary.to_csv(behavior_summary_path, index=False)
                            self.log_message(f"✓ Exported behavior summary: {os.path.basename(behavior_summary_path)}")
                            
                            # Export social interactions (detailed)
                            if not social_interactions.empty:
                                social_interactions_path = os.path.join(output_dir, f'{db_name}_social_interactions.csv')
                                social_interactions.to_csv(social_interactions_path, index=False)
                                self.log_message(f"✓ Exported social interactions: {os.path.basename(social_interactions_path)}")
                            
                            # Export social interaction summary
                            if not social_summary.empty:
                                social_summary_path = os.path.join(output_dir, f'{db_name}_social_summary.csv')
                                social_summary.to_csv(social_summary_path, index=False)
                                self.log_message(f"✓ Exported social interaction summary: {os.path.basename(social_summary_path)}")
                            else:
                                self.log_message("No social interactions detected")
                            
                            self.log_message("✓ Behavioral analysis complete!")
                        else:
                            self.log_message("Warning: No behaviors detected in data")
                        
                except ImportError as e:
                    self.log_message(f"Error: Could not import behavior detection module: {e}")
                except Exception as e:
                    self.log_message(f"Error during behavioral analysis: {e}")
                    import traceback
                    traceback.print_exc()
                
                self.log_message("=" * 50)
            
            # Export plots if requested (BEFORE animation, which takes longer)
            if save_plots:
                if self.export_cancelled:
                    self.stop_export()
                    return
                
                current_step += 1
                self.lbl_export_progress.setText(f"Step {current_step}/{total_steps}: Generating plots...")
                self.progress_bar.setValue(int(current_step / total_steps * 100))
                QApplication.processEvents()
                
                selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]

                # Get selected plot types - include ALL plot types
                plot_types = {
                    'daily_paths': self.plot_type_checkboxes['daily_paths'].isChecked(),
                    'trajectory_overview': self.plot_type_checkboxes['trajectory_overview'].isChecked(),
                    'battery_levels': self.plot_type_checkboxes['battery_levels'].isChecked(),
                    '3d_occupancy': self.plot_type_checkboxes['3d_occupancy'].isChecked(),
                    'activity_timeline': self.plot_type_checkboxes['activity_timeline'].isChecked(),
                    'velocity_distribution': self.plot_type_checkboxes['velocity_distribution'].isChecked(),
                    'cumulative_distance': self.plot_type_checkboxes['cumulative_distance'].isChecked(),
                    'velocity_timeline': self.plot_type_checkboxes['velocity_timeline'].isChecked(),
                    'actogram': self.plot_type_checkboxes['actogram'].isChecked(),
                    'data_quality': self.plot_type_checkboxes['data_quality'].isChecked()
                }

                # Get rolling window value
                rolling_window = self.spin_rolling_window.value()

                self.btn_export.setEnabled(False)
                self.log_message("Starting plot generation in background...")

                self.worker = PlotSaverWorker(
                    self.db_path,
                    self.table_name,
                    selected_tags,
                    False,  # downsample handled in CSV creation
                    self.get_smoothing_method(),
                    plot_types,
                    skip_existing,
                    rolling_window,
                    self.combo_timezone.currentText(),
                    self.tag_identities,
                    bool(self.tag_identities),  # Use identities if any are configured
                    self.background_image,  # Pass background image
                    self.bg_width_meters,  # Pass scaled width
                    self.bg_height_meters,  # Pass scaled height
                    csv_path,  # Pass CSV path for reuse
                    self.chk_save_svg.isChecked(),  # Save SVG copies
                    output_dir,  # Pass output directory
                    plots_dir  # Pass plots subfolder
                )
                self.worker.progress.connect(self.update_status)
                self.worker.finished.connect(lambda success, msg: self.export_finished(success, msg, save_animation, output_dir, total_steps, current_step, csv_path, animations_dir))
                self.worker.start()
            
            # Animation will be started from export_finished() after plots complete
            elif save_animation:
                # If no plots, start animation directly
                if self.export_cancelled:
                    self.stop_export()
                    return
                
                current_step += 1
                self.lbl_export_progress.setText(f"Step {current_step}/{total_steps}: Generating animation...")
                self.progress_bar.setValue(int((current_step - 1) / total_steps * 100))
                QApplication.processEvents()
                
                self.generate_animation(output_dir, total_steps, current_step, csv_path, animations_dir)

            # If no plots or animation, show success message now
            any_csv = export_raw_csv or export_smoothed_csv or export_downsampled_csv
            if (any_csv or detect_behaviors) and not save_plots and not save_animation:
                self.log_message("✓ Export completed successfully")
                msg = f"Export completed to:\n{output_dir}"
                QMessageBox.information(self, "Success", msg)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
            self.log_message(f"✗ Export failed: {str(e)}")
    
    def update_status(self, message):
        """Update status label and messages window"""
        self.log_message(message)
    
    def export_finished(self, success, message, start_animation=False, output_dir=None, total_steps=1, current_step=1, csv_path=None, animations_dir=None):
        """Handle export completion"""
        if not self.export_cancelled:
            if success:
                self.log_message("✓ Plot export completed successfully")

                # Start animation if requested (plots are now complete)
                if start_animation and output_dir:
                    self.log_message("Starting animation generation...")
                    self.lbl_export_progress.setText(f"Step {current_step + 1}/{total_steps}: Generating animation...")
                    self.progress_bar.setValue(int(current_step / total_steps * 100))
                    QApplication.processEvents()
                    self.generate_animation(output_dir, total_steps, current_step + 1, csv_path, animations_dir)
                    return  # Don't reset UI yet, animation will do that
                else:
                    self.progress_bar.setValue(100)
                    self.lbl_export_progress.setText("Export complete!")
                    QMessageBox.information(self, "Success", "Export completed successfully!")
            else:
                self.log_message(f"✗ Plot export failed: {message}")
                QMessageBox.critical(self, "Error", message)
        
        # Reset UI state
        self.exporting = False
        self.btn_export.setEnabled(True)
        self.btn_stop_export.setVisible(False)
        
        # Hide progress after a delay
        QTimer.singleShot(3000, lambda: self.progress_widget.setVisible(False))
    
    def closeEvent(self, event):
        """Handle window close event - stop any ongoing exports and playback"""
        if self.is_playing:
            self.playback_timer.stop()
        if self.exporting:
            reply = QMessageBox.question(self, 'Export in Progress', 
                                        'An export is in progress. Do you want to cancel it and close?',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.stop_export()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
        
        self.log_message("Ready for next operation")


def main():
    app = QApplication(sys.argv)
    window = UWBQuickVisualizationWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
