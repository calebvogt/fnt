import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import pytz
import subprocess
import gc
import json
import shutil
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.signal import savgol_filter
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QMessageBox, 
                             QGroupBox, QCheckBox, QScrollArea, QComboBox,
                             QSpinBox, QSplitter, QFrame, QSlider, QLineEdit,
                             QDialog, QDialogButtonBox, QFormLayout, QTableWidget,
                             QTableWidgetItem, QHeaderView, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont


class IdentityAssignmentDialog(QDialog):
    """Dialog for assigning sex and custom identities to tags"""
    def __init__(self, available_tags, existing_identities=None, parent=None):
        super().__init__(parent)
        self.available_tags = available_tags
        self.identities = existing_identities if existing_identities else {}
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Assign Tag Identities")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("Assign sex (M/F) and custom alphanumeric IDs to each tag:")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Form for each tag
        form_layout = QFormLayout()
        self.sex_combos = {}
        self.identity_edits = {}
        
        for tag in sorted(self.available_tags):
            # Sex selection
            sex_combo = QComboBox()
            sex_combo.addItems(["M", "F"])
            
            # Identity text input
            identity_edit = QLineEdit()
            identity_edit.setPlaceholderText(f"e.g., Animal{tag}")
            
            # Load existing values if available
            if tag in self.identities:
                sex_idx = 0 if self.identities[tag].get('sex', 'M') == 'M' else 1
                sex_combo.setCurrentIndex(sex_idx)
                identity_edit.setText(self.identities[tag].get('identity', ''))
            else:
                # Default values
                sex_combo.setCurrentIndex(0)  # Default to M
                identity_edit.setText(f"Tag{tag}")
            
            # Horizontal layout for sex and identity
            tag_layout = QHBoxLayout()
            tag_layout.addWidget(QLabel("Sex:"))
            tag_layout.addWidget(sex_combo)
            tag_layout.addWidget(QLabel("ID:"))
            tag_layout.addWidget(identity_edit)
            
            form_layout.addRow(f"Tag {tag}:", tag_layout)
            
            self.sex_combos[tag] = sex_combo
            self.identity_edits[tag] = identity_edit
        
        layout.addLayout(form_layout)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def get_identities(self):
        """Return the configured identities"""
        result = {}
        for tag in self.available_tags:
            sex = self.sex_combos[tag].currentText()
            identity = self.identity_edits[tag].text().strip()
            if not identity:
                identity = f"Tag{tag}"
            result[tag] = {'sex': sex, 'identity': identity}
        return result


class PlotSaverWorker(QThread):
    """Worker thread for saving plots without blocking the UI"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, db_path, table_name, selected_tags, downsample, smoothing_method, 
                 plot_types=None, overwrite=True, rolling_window=10, timezone='US/Mountain',
                 tag_identities=None, use_identities=False):
        super().__init__()
        self.db_path = db_path
        self.table_name = table_name
        self.selected_tags = selected_tags
        self.downsample = downsample
        self.smoothing_method = smoothing_method
        self.plot_types = plot_types if plot_types is not None else {
            'daily_paths': True,
            'trajectory_overview': True,
            'battery_levels': True
        }
        self.overwrite = overwrite
        self.rolling_window = rolling_window
        self.timezone = timezone
        self.tag_identities = tag_identities if tag_identities else {}
        self.use_identities = use_identities
        
    def run(self):
        try:
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
            
            # Flip Y-axis so 0,0 is at bottom-left
            data['location_y'] = -data['location_y']
            
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
            
            # Get output directory - use the analysis folder
            db_dir = os.path.dirname(self.db_path)
            db_filename = os.path.basename(self.db_path)
            db_name = os.path.splitext(db_filename)[0]
            output_dir = os.path.join(db_dir, f"{db_name}_fntUwbAnalysis")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate and save plots based on selection
            generated_count = 0
            skipped_count = 0
            
            if self.plot_types.get('daily_paths', False):
                result = self.save_daily_paths_per_tag(data, output_dir, db_name)
                if result:
                    generated_count += result
            
            if self.plot_types.get('trajectory_overview', False):
                result = self.save_trajectory_overview(data, output_dir, db_name)
                if result:
                    generated_count += 1
                else:
                    skipped_count += 1
            
            if self.plot_types.get('battery_levels', False):
                result = self.save_battery_levels(data, output_dir, db_name)
                if result:
                    generated_count += 1
                else:
                    skipped_count += 1
            
            if self.plot_types.get('3d_occupancy', False):
                result = self.save_3d_occupancy(data, output_dir, db_name)
                if result:
                    generated_count += result
            
            if self.plot_types.get('activity_timeline', False):
                result = self.save_activity_timeline(data, output_dir, db_name)
                if result:
                    generated_count += 1
                else:
                    skipped_count += 1
            
            if self.plot_types.get('velocity_distribution', False):
                result = self.save_velocity_distribution(data, output_dir, db_name)
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
            output_path = os.path.join(output_dir, f'{db_name}_DailyPaths_Tag{tag}.png')
            
            # Check if file exists and overwrite is False
            if not self.overwrite and os.path.exists(output_path):
                self.progress.emit(f"Skipped (exists): {db_name}_DailyPaths_Tag{tag}.png")
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
            
            fig.suptitle(f'Daily Paths - Tag {tag}', fontsize=14, fontweight='bold')
            fig.tight_layout()
            
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            generated += 1
            
            self.progress.emit(f"Saved: {db_name}_DailyPaths_Tag{tag}.png")
        
        return generated
    
    def save_trajectory_overview(self, data, output_dir, db_name):
        """Save trajectory overview
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating trajectory overview...")
        
        output_path = os.path.join(output_dir, f'{db_name}_TrajectoryOverview.png')
        
        # Check if file exists and overwrite is False
        if not self.overwrite and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_TrajectoryOverview.png")
            return False
        
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(data['shortid'].unique())))
        
        for i, tag in enumerate(data['shortid'].unique()):
            tag_data = data[data['shortid'] == tag]
            ax.plot(tag_data[x_col], tag_data[y_col], 
                   linewidth=1, alpha=0.7, color=colors[i], label=f'Tag {tag}')
        
        ax.set_xlabel('X Position (m)', fontsize=10)
        ax.set_ylabel('Y Position (m)', fontsize=10)
        ax.set_title('Trajectory Overview - All Tags', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        fig.tight_layout()
        
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_TrajectoryOverview.png")
        return True
    
    def save_battery_levels(self, data, output_dir, db_name):
        """Save battery levels plot
        Returns: True if generated, False if skipped or no battery data"""
        self.progress.emit("Generating battery levels...")
        
        output_path = os.path.join(output_dir, f'{db_name}_BatteryLevels.png')
        
        # Check if file exists and overwrite is False
        if not self.overwrite and os.path.exists(output_path):
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
        
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
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
            # Get identity if available
            if 'identity' in data.columns:
                tag_identity = data[data['shortid'] == tag]['identity'].iloc[0]
            else:
                tag_identity = f'Tag{tag}'
            
            output_path = os.path.join(output_dir, f'{db_name}_3D_Occupancy_{tag_identity}.png')
            
            if not self.overwrite and os.path.exists(output_path):
                self.progress.emit(f"Skipped (exists): {db_name}_3D_Occupancy_{tag_identity}.png")
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
            
            fig.suptitle(f'3D Occupancy - {tag_identity}', fontsize=14, fontweight='bold')
            fig.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            generated += 1
            
            self.progress.emit(f"Saved: {db_name}_3D_Occupancy_{tag_identity}.png")
        
        return generated
    
    def save_activity_timeline(self, data, output_dir, db_name):
        """Save activity timeline
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating activity timeline...")
        
        output_path = os.path.join(output_dir, f'{db_name}_ActivityTimeline.png')
        
        if not self.overwrite and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_ActivityTimeline.png")
            return False
        
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        for tag in data['shortid'].unique():
            tag_data = data[data['shortid'] == tag]
            hourly_counts = tag_data.set_index('Timestamp').resample('H').size()
            ax.plot(hourly_counts.index, hourly_counts.values, label=f'Tag {tag}', linewidth=1.5)
        
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Data Points per Hour', fontsize=10)
        ax.set_title('Activity Timeline - Data Points Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_ActivityTimeline.png")
        return True
    
    def save_velocity_distribution(self, data, output_dir, db_name):
        """Save velocity distribution
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating velocity distribution...")
        
        output_path = os.path.join(output_dir, f'{db_name}_VelocityDistribution.png')
        
        if not self.overwrite and os.path.exists(output_path):
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
                ax.hist(tag_data, bins=50, alpha=0.5, label=f'Tag {tag}', density=True)
        
        ax.set_xlabel('Velocity (m/s)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title('Velocity Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_VelocityDistribution.png")
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
        self.preview_loaded = False
        
        # Playback control variables
        self.is_playing = False
        self.playback_speed = 1  # 1x, 2x, 4x, etc.
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.advance_playback)
        
        # Tag identity and sex mapping
        self.tag_identities = {}  # {tag_id: {'sex': 'M', 'identity': 'Animal1'}}
        
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
    
    def initUI(self):
        self.setWindowTitle("UWB Quick Visualization")
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
                height: 8px;
                background: #1e1e1e;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 1px solid #0078d4;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #106ebe;
            }
        """)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Settings
        left_panel = self.create_settings_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Visualization
        right_panel = self.create_visualization_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([350, 1050])
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
    def create_settings_panel(self):
        """Create the left settings panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("UWB Quick Visualization")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Interactive visualization of UWB tracking data with time slider")
        desc.setFont(QFont("Arial", 9))
        desc.setStyleSheet("color: #666666; margin-bottom: 10px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
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
        
        # Tag selection
        self.tag_group = QGroupBox("Tag Selection")
        self.tag_layout = QVBoxLayout()
        self.tag_checkboxes = {}
        
        self.lbl_no_tags = QLabel("Load a database to see available tags")
        self.lbl_no_tags.setStyleSheet("color: #666666; font-style: italic;")
        self.tag_layout.addWidget(self.lbl_no_tags)
        
        # Select All/None buttons
        tag_btn_layout = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_select_all.clicked.connect(self.select_all_tags)
        btn_select_none = QPushButton("Select None")
        btn_select_none.clicked.connect(self.select_none_tags)
        tag_btn_layout.addWidget(btn_select_all)
        tag_btn_layout.addWidget(btn_select_none)
        self.tag_layout.addLayout(tag_btn_layout)
        
        # Button to open identity assignment dialog (no checkbox, always available)
        self.btn_assign_identities = QPushButton("Configure Identities...")
        self.btn_assign_identities.clicked.connect(self.open_identity_dialog)
        self.btn_assign_identities.setEnabled(False)
        self.btn_assign_identities.setToolTip("Assign custom sex (M/F) and alphanumeric IDs to selected tags")
        self.tag_layout.addWidget(self.btn_assign_identities)
        
        self.tag_group.setLayout(self.tag_layout)
        layout.addWidget(self.tag_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        
        # Timezone
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
        options_layout.addLayout(tz_layout)
        
        # Smoothing (moved up, before downsample)
        options_layout.addWidget(QLabel("Smoothing method:"))
        self.combo_smoothing = QComboBox()
        self.combo_smoothing.addItems(["None", "Savitzky-Golay", "Rolling Average"])
        self.combo_smoothing.currentTextChanged.connect(self.on_smoothing_changed)
        options_layout.addWidget(self.combo_smoothing)
        
        # Rolling average window (only shown when Rolling Average is selected)
        self.rolling_window_layout = QHBoxLayout()
        self.rolling_window_layout.addWidget(QLabel("Window (seconds):"))
        self.spin_rolling_window = QSpinBox()
        self.spin_rolling_window.setRange(1, 60)
        self.spin_rolling_window.setValue(30)  # Changed default to 30s
        self.spin_rolling_window.setEnabled(False)
        self.rolling_window_layout.addWidget(self.spin_rolling_window)
        options_layout.addLayout(self.rolling_window_layout)
        self.spin_rolling_window.hide()
        self.rolling_window_layout.itemAt(0).widget().hide()  # Hide label too
        
        # Trail options
        self.chk_show_trail = QCheckBox("Show trail")
        self.chk_show_trail.setChecked(False)
        self.chk_show_trail.stateChanged.connect(self.on_trail_toggled)
        options_layout.addWidget(self.chk_show_trail)
        
        trail_length_layout = QHBoxLayout()
        trail_length_layout.addWidget(QLabel("Trail length (seconds):"))  # Changed to seconds
        self.spin_trail_length = QSpinBox()
        self.spin_trail_length.setRange(1, 300)  # 1-300 seconds
        self.spin_trail_length.setValue(30)  # Changed default to 30s
        self.spin_trail_length.setEnabled(False)
        trail_length_layout.addWidget(self.spin_trail_length)
        options_layout.addLayout(trail_length_layout)
        
        # Downsample (moved to below trail options, above Load Preview button)
        self.chk_downsample = QCheckBox("Downsample to 1Hz")
        self.chk_downsample.setChecked(True)
        self.chk_downsample.setToolTip("Downsample output to 1Hz (smoothing is applied to full resolution data first)")
        self.chk_downsample.stateChanged.connect(self.mark_options_changed)
        options_layout.addWidget(self.chk_downsample)
        
        # Load Preview button (moved here)
        self.btn_load_preview = QPushButton("Load Preview")
        self.btn_load_preview.clicked.connect(self.load_preview)
        self.btn_load_preview.setEnabled(False)
        self.btn_load_preview.setStyleSheet("padding: 10px; font-size: 12px; font-weight: bold;")
        options_layout.addWidget(self.btn_load_preview)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Export Options
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout()
        
        # Export CSV checkbox
        self.chk_export_csv = QCheckBox("Export CSV")
        self.chk_export_csv.setChecked(True)
        self.chk_export_csv.setToolTip("Export processed data as CSV with current settings applied")
        export_layout.addWidget(self.chk_export_csv)
        
        # Save Plots checkbox (master)
        self.chk_save_plots = QCheckBox("Save Plots")
        self.chk_save_plots.setChecked(False)  # Default unchecked
        self.chk_save_plots.stateChanged.connect(self.on_save_plots_toggled)
        self.chk_save_plots.setToolTip("Generate and save visualization plots")
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
            ("velocity_distribution", "Velocity Distribution", "Velocity distribution for each tag")
        ]
        
        for key, plot_name, plot_desc in plot_types:
            cb = QCheckBox(plot_name)
            cb.setChecked(True)
            cb.setToolTip(plot_desc)
            self.plot_type_checkboxes[key] = cb
            plot_types_layout.addWidget(cb)
        
        self.plot_types_widget.setLayout(plot_types_layout)
        self.plot_types_widget.setVisible(False)  # Hidden by default
        export_layout.addWidget(self.plot_types_widget)
        
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
        self.spin_animation_trail.setValue(1000)
        self.spin_animation_trail.setToolTip("How much trailing data to show in animation")
        trail_layout.addWidget(self.spin_animation_trail)
        animation_options_layout.addLayout(trail_layout)
        
        # Animation FPS
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.spin_animation_fps = QSpinBox()
        self.spin_animation_fps.setRange(1, 60)
        self.spin_animation_fps.setValue(20)
        self.spin_animation_fps.setToolTip("Frames per second for output video")
        fps_layout.addWidget(self.spin_animation_fps)
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
        
        # Viewing angle
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Viewing angle (degrees):"))
        self.spin_viewing_angle = QSpinBox()
        self.spin_viewing_angle.setRange(15, 180)
        self.spin_viewing_angle.setValue(45)
        self.spin_viewing_angle.setToolTip("Viewing angle for direction indicator")
        angle_layout.addWidget(self.spin_viewing_angle)
        animation_options_layout.addLayout(angle_layout)
        
        self.animation_options_widget.setLayout(animation_options_layout)
        self.animation_options_widget.setVisible(False)  # Hidden by default
        export_layout.addWidget(self.animation_options_widget)
        
        # Overwrite checkbox (moved to bottom, applies to all exports)
        self.chk_overwrite = QCheckBox("Overwrite existing files")
        self.chk_overwrite.setChecked(False)
        self.chk_overwrite.setToolTip("If unchecked, will skip files that already exist (applies to all export types)")
        export_layout.addWidget(self.chk_overwrite)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Export button
        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self.export_data)
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet("padding: 10px; font-size: 12px; font-weight: bold;")
        layout.addWidget(self.btn_export)
        
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
        scroll.setMinimumWidth(350)
        
        return scroll
        
    def create_visualization_panel(self):
        """Create the right visualization panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Tracking Preview")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Canvas
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
        
        # Playback controls
        playback_layout = QHBoxLayout()
        
        # Rewind button
        self.btn_rewind = QPushButton("⏮ Rewind")
        self.btn_rewind.clicked.connect(self.rewind_playback)
        self.btn_rewind.setEnabled(False)
        self.btn_rewind.setMaximumWidth(100)
        playback_layout.addWidget(self.btn_rewind)
        
        # Play/Pause button
        self.btn_play_pause = QPushButton("▶ Play")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.setMaximumWidth(100)
        playback_layout.addWidget(self.btn_play_pause)
        
        # Fast forward button
        self.btn_fast_forward = QPushButton("Fast Forward ⏭")
        self.btn_fast_forward.clicked.connect(self.fast_forward_playback)
        self.btn_fast_forward.setEnabled(False)
        self.btn_fast_forward.setMaximumWidth(120)
        playback_layout.addWidget(self.btn_fast_forward)
        
        # Spacer
        playback_layout.addSpacing(20)
        
        # Playback speed label
        playback_layout.addWidget(QLabel("Speed:"))
        
        # Speed selector
        self.combo_playback_speed = QComboBox()
        self.combo_playback_speed.addItems(["1x", "2x", "4x", "8x"])
        self.combo_playback_speed.setCurrentText("1x")
        self.combo_playback_speed.currentTextChanged.connect(self.on_speed_changed)
        self.combo_playback_speed.setMaximumWidth(80)
        playback_layout.addWidget(self.combo_playback_speed)
        
        playback_layout.addStretch()
        
        layout.addLayout(playback_layout)
        
        # Time slider
        slider_layout = QVBoxLayout()
        
        slider_label_layout = QHBoxLayout()
        slider_label_layout.addWidget(QLabel("Time:"))
        self.lbl_time = QLabel("--:--:--")
        self.lbl_time.setStyleSheet("font-weight: bold;")
        slider_label_layout.addWidget(self.lbl_time)
        slider_label_layout.addStretch()
        slider_layout.addLayout(slider_label_layout)
        
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
            
            self.combo_table.clear()
            self.combo_table.addItems(tables)
            self.combo_table.setEnabled(True)
            self.btn_preview_table.setEnabled(True)
            
            if len(tables) == 1:
                self.combo_table.setCurrentIndex(0)
            
            # Check for existing config file and load it
            self.load_config_if_exists()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open database: {str(e)}")
    
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
        is_rolling = method == "Rolling Average"
        self.spin_rolling_window.setEnabled(is_rolling)
        self.spin_rolling_window.setVisible(is_rolling)
        self.rolling_window_layout.itemAt(0).widget().setVisible(is_rolling)
        self.mark_options_changed()
    
    def on_save_plots_toggled(self):
        """Handle save plots checkbox toggle"""
        enabled = self.chk_save_plots.isChecked()
        self.plot_types_widget.setVisible(enabled)
    
    def on_save_animation_toggled(self):
        """Handle save animation checkbox toggle"""
        enabled = self.chk_save_animation.isChecked()
        self.animation_options_widget.setVisible(enabled)
    
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
        
        dialog = IdentityAssignmentDialog(selected_tags, self.tag_identities, self)
        if dialog.exec_() == QDialog.Accepted:
            self.tag_identities = dialog.get_identities()
            self.log_message(f"Updated identities for {len(self.tag_identities)} tags")
            # Update tag checkbox labels to reflect new identities
            self.update_tag_labels()
            self.lbl_status.setText(f"Identity assignments saved for {len(self.tag_identities)} tags")
    
    def mark_options_changed(self):
        """Mark that options have changed, requiring reload"""
        if self.preview_loaded:
            self.btn_load_preview.setText("Reload Preview")
            self.btn_load_preview.setEnabled(True)
    
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
            self.btn_load_preview.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load tags: {str(e)}")
    
    def update_tag_selection(self):
        """Update tag checkboxes"""
        for cb in self.tag_checkboxes.values():
            cb.deleteLater()
        self.tag_checkboxes.clear()
        
        if self.lbl_no_tags:
            self.lbl_no_tags.deleteLater()
            self.lbl_no_tags = None
        
        for tag in self.available_tags:
            cb = QCheckBox(f"Tag {tag}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.mark_options_changed)  # Track changes
            cb.stateChanged.connect(self.update_identity_button_state)  # Enable/disable identity button
            self.tag_checkboxes[tag] = cb
            self.tag_layout.insertWidget(self.tag_layout.count() - 1, cb)
        
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
            if tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', f'Tag{tag}')
                cb.setText(f"Tag {tag} ({sex}, {identity})")
            else:
                cb.setText(f"Tag {tag}")
    
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
        if self.data is not None:
            self.update_visualization(self.time_slider.value())
    
    def load_preview(self):
        """Load data and setup preview"""
        if not self.db_path or not self.table_name:
            return
        
        selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]
        if not selected_tags:
            QMessageBox.warning(self, "No Tags", "Please select at least one tag")
            return
        
        try:
            self.log_message("Loading data from database...")
            
            # Load data
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT * FROM {self.table_name}"
            self.data = pd.read_sql_query(query, conn)
            conn.close()
            
            self.log_message(f"Loaded {len(self.data)} records from database")
            
            # Process data
            self.log_message("Processing timestamps and converting units...")
            self.data['Timestamp'] = pd.to_datetime(self.data['timestamp'], unit='ms', origin='unix', utc=True)
            tz = pytz.timezone(self.combo_timezone.currentText())
            self.data['Timestamp'] = self.data['Timestamp'].dt.tz_convert(tz)
            
            self.data['location_x'] *= 0.0254
            self.data['location_y'] *= 0.0254
            
            # Flip Y-axis so 0,0 is at bottom-left
            self.data['location_y'] = -self.data['location_y']
            
            self.data = self.data.sort_values(by=['shortid', 'Timestamp'])
            
            # Filter tags
            self.data = self.data[self.data['shortid'].isin(selected_tags)]
            
            # Apply custom sex and identities if configured
            if self.tag_identities:
                self.log_message("Applying custom identities...")
                self.data['sex'] = self.data['shortid'].map(lambda x: self.tag_identities.get(x, {}).get('sex', 'M'))
                self.data['identity'] = self.data['shortid'].map(lambda x: self.tag_identities.get(x, {}).get('identity', f'Tag{x}'))
            else:
                # Default values
                self.data['sex'] = 'M'
                self.data['identity'] = self.data['shortid'].apply(lambda x: f'Tag{x}')
            
            # Apply smoothing FIRST (on full resolution data)
            if self.combo_smoothing.currentText() != "None":
                self.log_message("Applying smoothing to full resolution data...")
                self.apply_smoothing()
            
            # Downsample AFTER smoothing (if requested)
            if self.chk_downsample.isChecked():
                self.log_message("Downsampling to 1Hz...")
                self.data['time_sec'] = (self.data['Timestamp'].astype(np.int64) // 1_000_000_000).astype(int)
                self.data = self.data.groupby(['shortid', 'time_sec']).first().reset_index()
            
            # Setup slider based on unique timestamps
            self.log_message("Setting up visualization controls...")
            unique_times = sorted(self.data['Timestamp'].unique())
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
            
            # Mark preview as loaded and disable button
            self.preview_loaded = True
            self.btn_load_preview.setEnabled(False)
            self.btn_export.setEnabled(True)
            self.log_message(f"✓ Preview loaded: {len(self.data)} data points across {len(unique_times)} unique timestamps")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load preview: {str(e)}")
            self.log_message(f"✗ Error loading data: {str(e)}")
    
    def apply_smoothing(self):
        """Apply smoothing to data"""
        method = self.combo_smoothing.currentText()
        
        def apply_savgol(group):
            window_length = min(31, len(group))
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(2, window_length - 1)
            if len(group) > polyorder:
                return savgol_filter(group, window_length=window_length, polyorder=polyorder)
            return group
        
        if method == "Savitzky-Golay":
            self.data['smoothed_x'] = self.data.groupby('shortid')['location_x'].transform(apply_savgol)
            self.data['smoothed_y'] = self.data.groupby('shortid')['location_y'].transform(apply_savgol)
        elif method == "Rolling Average":
            # Get window size in seconds from spinbox
            window_seconds = self.spin_rolling_window.value()
            
            # Calculate window size in number of samples (assuming 1Hz after downsampling)
            window_size = max(3, window_seconds)  # Minimum window of 3
            
            self.data['smoothed_x'] = self.data.groupby('shortid')['location_x'].transform(
                lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())
            self.data['smoothed_y'] = self.data.groupby('shortid')['location_y'].transform(
                lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())
    
    def update_visualization(self, slider_value):
        """Update visualization based on slider position"""
        if self.data is None or len(self.data) == 0 or not hasattr(self, 'unique_timestamps'):
            return
        
        self.ax.clear()
        
        x_col = 'smoothed_x' if 'smoothed_x' in self.data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in self.data.columns else 'location_y'
        
        # Get current timestamp from slider
        current_timestamp = self.unique_timestamps[slider_value]
        self.lbl_time.setText(current_timestamp.strftime('%Y-%m-%d %H:%M:%S'))
        
        # Get all data up to and including current timestamp
        current_data = self.data[self.data['Timestamp'] <= current_timestamp]
        
        # Plot each tag
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.data['shortid'].unique())))
        
        for i, tag in enumerate(sorted(self.data['shortid'].unique())):
            tag_all_data = current_data[current_data['shortid'] == tag]
            
            if len(tag_all_data) == 0:
                continue
            
            # Get the most recent position for this tag at current time
            current_pos = tag_all_data.iloc[-1]
            
            # Plot trail if enabled
            if self.chk_show_trail.isChecked() and len(tag_all_data) > 1:
                # Calculate time window for trail (in seconds)
                trail_seconds = self.spin_trail_length.value()
                trail_start_time = current_timestamp - pd.Timedelta(seconds=trail_seconds)
                trail_data = tag_all_data[tag_all_data['Timestamp'] >= trail_start_time]
                
                if len(trail_data) > 1:
                    self.ax.plot(trail_data[x_col], trail_data[y_col], 
                               color=colors[i], linewidth=2, alpha=0.6)
            
            # Plot current position with label
            self.ax.scatter(current_pos[x_col], current_pos[y_col], 
                          c=[colors[i]], s=100, marker='o', edgecolors='black', linewidths=2, zorder=5)
            self.ax.text(current_pos[x_col], current_pos[y_col], f'  Tag {tag}', 
                        fontsize=10, fontweight='bold', color=colors[i],
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=colors[i]))
        
        # Set limits based on all data
        x_min, x_max = self.data[x_col].min(), self.data[x_col].max()
        y_min, y_max = self.data[y_col].min(), self.data[y_col].max()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_pad = x_range * 0.05 if x_range > 0 else 1
        y_pad = y_range * 0.05 if y_range > 0 else 1
        
        self.ax.set_xlim(x_min - x_pad, x_max + x_pad)
        self.ax.set_ylim(y_min - y_pad, y_max + y_pad)
        
        self.ax.set_xlabel('X Position (m)', fontsize=10)
        self.ax.set_ylabel('Y Position (m)', fontsize=10)
        self.ax.set_title('UWB Tracking Visualization', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        self.canvas.draw()
    
    def rewind_playback(self):
        """Rewind to the beginning"""
        self.time_slider.setValue(0)
        if self.is_playing:
            self.toggle_play_pause()  # Pause if playing
    
    def toggle_play_pause(self):
        """Toggle between play and pause"""
        if self.is_playing:
            # Pause
            self.is_playing = False
            self.playback_timer.stop()
            self.btn_play_pause.setText("▶ Play")
        else:
            # Play
            self.is_playing = True
            # Calculate interval based on playback speed
            # Base interval is 1000ms (1 second) for 1x speed
            base_interval = 1000
            interval = int(base_interval / self.playback_speed)
            self.playback_timer.start(interval)
            self.btn_play_pause.setText("⏸ Pause")
    
    def fast_forward_playback(self):
        """Fast forward to the end"""
        self.time_slider.setValue(self.time_slider.maximum())
        if self.is_playing:
            self.toggle_play_pause()  # Pause if playing
    
    def on_speed_changed(self, speed_text):
        """Handle playback speed change"""
        self.playback_speed = int(speed_text.replace('x', ''))
        
        # If currently playing, restart timer with new interval
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
            # Reached the end, pause playback
            self.toggle_play_pause()
    
    def get_config_dict(self):
        """Get current configuration as dictionary"""
        config = {
            'table_name': self.table_name,
            'selected_tags': [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()],
            'timezone': self.combo_timezone.currentText(),
            'downsample': self.chk_downsample.isChecked(),
            'smoothing_method': self.combo_smoothing.currentText(),
            'rolling_window': self.spin_rolling_window.value(),
            'show_trail': self.chk_show_trail.isChecked(),
            'trail_length': self.spin_trail_length.value(),
            'export_csv': self.chk_export_csv.isChecked(),
            'save_plots': self.chk_save_plots.isChecked(),
            'plot_types': {
                'daily_paths': self.plot_type_checkboxes['daily_paths'].isChecked(),
                'trajectory_overview': self.plot_type_checkboxes['trajectory_overview'].isChecked(),
                'battery_levels': self.plot_type_checkboxes['battery_levels'].isChecked(),
                '3d_occupancy': self.plot_type_checkboxes['3d_occupancy'].isChecked(),
                'activity_timeline': self.plot_type_checkboxes['activity_timeline'].isChecked(),
                'velocity_distribution': self.plot_type_checkboxes['velocity_distribution'].isChecked()
            },
            'save_animation': self.chk_save_animation.isChecked(),
            'animation_trail': self.spin_animation_trail.value(),
            'animation_fps': self.spin_animation_fps.value(),
            'time_window': self.spin_time_window.value(),
            'color_by': self.combo_color_by.currentText(),
            'viewing_angle': self.spin_viewing_angle.value(),
            'tag_identities': self.tag_identities,
            'overwrite': self.chk_overwrite.isChecked()
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
        analysis_dir = os.path.join(db_dir, f"{db_name}_fntUwbAnalysis")
        config_path = os.path.join(analysis_dir, 'fnt_config.json')
        
        if not os.path.exists(config_path):
            return
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load configuration into GUI
            if 'table_name' in config and config['table_name']:
                # Set table if it exists in combo
                index = self.combo_table.findText(config['table_name'])
                if index >= 0:
                    self.combo_table.setCurrentIndex(index)
            
            if 'timezone' in config:
                index = self.combo_timezone.findText(config['timezone'])
                if index >= 0:
                    self.combo_timezone.setCurrentIndex(index)
            
            if 'downsample' in config:
                self.chk_downsample.setChecked(config['downsample'])
            
            if 'smoothing_method' in config:
                index = self.combo_smoothing.findText(config['smoothing_method'])
                if index >= 0:
                    self.combo_smoothing.setCurrentIndex(index)
            
            if 'rolling_window' in config:
                self.spin_rolling_window.setValue(config['rolling_window'])
            
            if 'show_trail' in config:
                self.chk_show_trail.setChecked(config['show_trail'])
            
            if 'trail_length' in config:
                self.spin_trail_length.setValue(config['trail_length'])
            
            if 'export_csv' in config:
                self.chk_export_csv.setChecked(config['export_csv'])
            
            if 'save_plots' in config:
                self.chk_save_plots.setChecked(config['save_plots'])
            
            if 'plot_types' in config:
                for key, value in config['plot_types'].items():
                    if key in self.plot_type_checkboxes:
                        self.plot_type_checkboxes[key].setChecked(value)
            
            if 'save_animation' in config:
                self.chk_save_animation.setChecked(config['save_animation'])
            
            if 'animation_trail' in config:
                self.spin_animation_trail.setValue(config['animation_trail'])
            
            if 'animation_fps' in config:
                self.spin_animation_fps.setValue(config['animation_fps'])
            
            if 'time_window' in config:
                self.spin_time_window.setValue(config['time_window'])
            
            if 'color_by' in config:
                index = self.combo_color_by.findText(config['color_by'])
                if index >= 0:
                    self.combo_color_by.setCurrentIndex(index)
            
            if 'viewing_angle' in config:
                self.spin_viewing_angle.setValue(config['viewing_angle'])
            
            if 'tag_identities' in config:
                # Convert string keys back to integers if needed
                self.tag_identities = {}
                for key, value in config['tag_identities'].items():
                    tag_key = int(key) if isinstance(key, str) and key.isdigit() else key
                    self.tag_identities[tag_key] = value
            
            if 'overwrite' in config:
                self.chk_overwrite.setChecked(config['overwrite'])
            
            # Note: selected_tags will be loaded after tags are populated from table
            if 'selected_tags' in config:
                self.pending_tag_selection = config['selected_tags']
            
            self.log_message(f"Loaded previous configuration from {config_path}")
            
            # Update tag labels if identities were loaded
            if self.tag_identities and self.tag_checkboxes:
                self.update_tag_labels()
            
        except Exception as e:
            print(f"Warning: Could not load config: {str(e)}")
    
    def apply_pending_tag_selection(self):
        """Apply tag selection from loaded config after tags are populated"""
        if hasattr(self, 'pending_tag_selection') and self.pending_tag_selection:
            for tag, cb in self.tag_checkboxes.items():
                cb.setChecked(tag in self.pending_tag_selection)
            delattr(self, 'pending_tag_selection')
    
    def generate_animation(self, output_dir):
        """Generate animation video from tracking data"""
        try:
            self.log_message("Preparing animation data...")
            
            # Prepare data for animation
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
            
            anim_data['heading'] = anim_data.groupby('ID', group_keys=False).apply(
                lambda group: np.arctan2(group['smoothed_y'].diff(1), group['smoothed_x'].diff(1))
            ).reset_index(level=0, drop=True)
            
            anim_data['heading'] = anim_data.groupby('ID', group_keys=False)['heading'].apply(
                lambda group: group.rolling(window=5, min_periods=1).mean()
            ).reset_index(level=0, drop=True)
            
            anim_data['velocity'] = anim_data.groupby('ID', group_keys=False).apply(
                lambda group: np.sqrt(group['smoothed_x'].diff()**2 + group['smoothed_y'].diff()**2) / 
                              group['Timestamp'].diff().dt.total_seconds()
            ).reset_index(level=0, drop=True)
            
            # Get animation parameters
            time_window = self.spin_time_window.value()
            trailing_window = self.spin_animation_trail.value()
            fps = self.spin_animation_fps.value()
            color_by = self.combo_color_by.currentText()
            viewing_angle = self.spin_viewing_angle.value()
            
            self.log_message("Setting up animation frames...")
            # Create frames directory
            frames_dir = os.path.join(output_dir, "animation_frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Clean existing frames
            for filename in os.listdir(frames_dir):
                file_path = os.path.join(frames_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # Create animation
            self.log_message("Generating animation frames (this may take a while)...")
            self.create_animation_frames(anim_data, frames_dir, time_window, trailing_window, 
                                        fps, color_by, viewing_angle, bool(self.tag_identities))
            
            self.log_message("✓ Animation generation complete!")
            
        except Exception as e:
            QMessageBox.critical(self, "Animation Error", f"Failed to generate animation: {str(e)}")
            self.log_message(f"✗ Animation generation failed: {str(e)}")
    
    def create_animation_frames(self, data, output_dir, time_window, trailing_window, 
                               fps, color_by, viewing_angle, use_custom_identities=False):
        """Create animation frames and compile video"""
        # Get global min/max for consistent axis limits
        x_min, x_max = data['smoothed_x'].min(), data['smoothed_x'].max()
        y_min, y_max = data['smoothed_y'].min(), data['smoothed_y'].max()
        
        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_pad = x_range * 0.05 if x_range > 0 else 1
        y_pad = y_range * 0.05 if y_range > 0 else 1
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad
        
        # Calculate time range
        start = data['Timestamp'].min()
        end = data['Timestamp'].max()
        time_starts = pd.date_range(start=start, end=end, freq=f'{time_window}s')
        
        # Define color palette
        if color_by == "ID" and not use_custom_identities:
            unique_ids = data['ID'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_ids)))
            id_color_map = {ID: colors(i) for i, ID in enumerate(unique_ids)}
        
        # Create frames
        self.lbl_status.setText(f"Creating {len(time_starts)} animation frames...")
        
        for i, frame_start in enumerate(time_starts):
            if i % 10 == 0:
                self.lbl_status.setText(f"Creating frame {i+1}/{len(time_starts)}...")
                QApplication.processEvents()  # Keep UI responsive
            
            frame_end = frame_start + pd.Timedelta(seconds=trailing_window)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.grid(False)
            
            # Plot each ID
            for ID in data['ID'].unique():
                trailing_data = data[(data['ID'] == ID) & 
                                    (data['Timestamp'] >= frame_start) & 
                                    (data['Timestamp'] <= frame_end)]
                
                if trailing_data.empty:
                    continue
                
                # Determine color
                if use_custom_identities or color_by == "sex":
                    # Use sex-based coloring: blue for M, red for F
                    color = 'blue' if trailing_data['sex'].values[0] == 'M' else 'red'
                elif color_by == "ID":
                    color = id_color_map[ID]
                
                # Plot trailing line
                ax.plot(trailing_data['smoothed_x'], trailing_data['smoothed_y'], 
                       color=color, alpha=0.5, linewidth=1)
                
                # Plot current position
                current_x = trailing_data['smoothed_x'].values[-1]
                current_y = trailing_data['smoothed_y'].values[-1]
                ax.plot(current_x, current_y, 'o', color=color, markersize=10)
                
                # Determine label text
                if use_custom_identities and 'custom_identity' in trailing_data.columns:
                    label_text = trailing_data['custom_identity'].values[-1]
                else:
                    label_text = str(ID)
                
                # Add label
                ax.text(current_x, current_y + (y_range * 0.02), label_text, 
                       fontsize=10, ha='center', color=color, fontweight='bold')
                
                # Plot heading if moving
                if trailing_data['velocity'].values[-1] > 0.01:
                    heading = trailing_data['heading'].values[-1]
                    from matplotlib.patches import Wedge
                    wedge = Wedge((current_x, current_y), x_range * 0.03, 
                                 np.degrees(heading) - viewing_angle / 2, 
                                 np.degrees(heading) + viewing_angle / 2, 
                                 color=color, alpha=0.3)
                    ax.add_patch(wedge)
            
            ax.set_title(f"UWB Tracking Animation\nTime: {frame_start.strftime('%Y-%m-%d %H:%M:%S')}", 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("X Position (m)", fontsize=12)
            ax.set_ylabel("Y Position (m)", fontsize=12)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')
            
            filename = os.path.join(output_dir, f"frame_{i:04d}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()
        
        # Compile video with ffmpeg
        self.lbl_status.setText("Compiling video with ffmpeg...")
        
        # Get database name for file prefix
        db_filename = os.path.basename(self.db_path)
        db_name = os.path.splitext(db_filename)[0]
        video_output = os.path.join(os.path.dirname(output_dir), 
                                   f'{db_name}_Animation.mp4')
        
        try:
            subprocess.call([
                'ffmpeg', '-y', '-framerate', str(fps), 
                '-i', os.path.join(output_dir, 'frame_%04d.png'),
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18', 
                '-pix_fmt', 'yuv420p', video_output
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Clean up frames
            for filename in os.listdir(output_dir):
                if filename.endswith(".png"):
                    os.remove(os.path.join(output_dir, filename))
            
            QMessageBox.information(self, "Success", f"Animation saved to:\n{video_output}")
        except Exception as e:
            QMessageBox.warning(self, "FFmpeg Error", 
                              f"Could not compile video. Frames saved to:\n{output_dir}\n\nError: {str(e)}")
    
    def export_data(self):
        """Export data and/or plots based on selected options"""
        if not self.db_path or self.data is None:
            return
        
        # Create output directory with naming convention: <db_name>_fntUwbAnalysis
        db_dir = os.path.dirname(self.db_path)
        db_filename = os.path.basename(self.db_path)
        db_name = os.path.splitext(db_filename)[0]  # Remove extension
        output_dir = os.path.join(db_dir, f"{db_name}_fntUwbAnalysis")
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        export_csv = self.chk_export_csv.isChecked()
        save_plots = self.chk_save_plots.isChecked()
        save_animation = self.chk_save_animation.isChecked()
        
        if not export_csv and not save_plots and not save_animation:
            QMessageBox.warning(self, "No Export Selected", "Please select at least one export option (CSV, Plots, or Animation)")
            return
        
        try:
            self.log_message(f"Starting export to {output_dir}")
            
            # Handle overwrite setting
            if self.chk_overwrite.isChecked():
                # Delete all files in analysis folder EXCEPT the config file
                self.log_message("Clearing existing files (preserving config)...")
                config_path = os.path.join(output_dir, 'fnt_config.json')
                
                for filename in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, filename)
                    # Skip the config file and skip directories (we'll handle them separately)
                    if filename == 'fnt_config.json':
                        continue
                    
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"Warning: Could not delete {filename}: {str(e)}")
                    elif os.path.isdir(file_path):
                        # Remove directories like animation_frames
                        try:
                            shutil.rmtree(file_path, ignore_errors=True)
                        except Exception as e:
                            print(f"Warning: Could not delete directory {filename}: {str(e)}")
            
            # Export CSV if requested
            if export_csv:
                self.log_message("Exporting CSV...")
                csv_filename = f'{db_name}_{self.table_name}_processed.csv'
                csv_path = os.path.join(output_dir, csv_filename)
                
                # Check if file exists and overwrite setting
                if not self.chk_overwrite.isChecked() and os.path.exists(csv_path):
                    self.log_message(f"Skipped CSV (already exists): {csv_filename}")
                else:
                    # Ensure sex and identity columns are in the data
                    csv_data = self.data.copy()
                    if 'sex' not in csv_data.columns or 'identity' not in csv_data.columns:
                        if self.tag_identities:
                            csv_data['sex'] = csv_data['shortid'].map(lambda x: self.tag_identities.get(x, {}).get('sex', 'M'))
                            csv_data['identity'] = csv_data['shortid'].map(lambda x: self.tag_identities.get(x, {}).get('identity', f'Tag{x}'))
                        else:
                            csv_data['sex'] = 'M'
                            csv_data['identity'] = csv_data['shortid'].apply(lambda x: f'Tag{x}')
                    
                    csv_data.to_csv(csv_path, index=False)
                    self.log_message(f"✓ CSV exported: {csv_filename}")
            
            # Save configuration file
            self.save_config(output_dir)
            
            # Export animation if requested
            if save_animation:
                self.generate_animation(output_dir)
            
            # Export plots if requested
            if save_plots:
                selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]
                
                # Get selected plot types - include ALL plot types
                plot_types = {
                    'daily_paths': self.plot_type_checkboxes['daily_paths'].isChecked(),
                    'trajectory_overview': self.plot_type_checkboxes['trajectory_overview'].isChecked(),
                    'battery_levels': self.plot_type_checkboxes['battery_levels'].isChecked(),
                    '3d_occupancy': self.plot_type_checkboxes['3d_occupancy'].isChecked(),
                    'activity_timeline': self.plot_type_checkboxes['activity_timeline'].isChecked(),
                    'velocity_distribution': self.plot_type_checkboxes['velocity_distribution'].isChecked()
                }
                
                # Get overwrite setting
                overwrite = self.chk_overwrite.isChecked()
                
                # Get rolling window value
                rolling_window = self.spin_rolling_window.value()
                
                self.btn_export.setEnabled(False)
                self.log_message("Starting plot generation in background...")
                
                self.worker = PlotSaverWorker(
                    self.db_path, 
                    self.table_name, 
                    selected_tags,
                    self.chk_downsample.isChecked(),
                    self.combo_smoothing.currentText(),
                    plot_types,
                    overwrite,
                    rolling_window,
                    self.combo_timezone.currentText(),
                    self.tag_identities,
                    bool(self.tag_identities)  # Use identities if any are configured
                )
                self.worker.progress.connect(self.update_status)
                self.worker.finished.connect(self.export_finished)
                self.worker.start()
            elif export_csv or save_animation:
                # If only CSV or animation was exported (no plots), show success message
                self.log_message("✓ Export completed successfully")
                msg = "Export completed:\n"
                if export_csv:
                    msg += f"- CSV: {csv_path}\n"
                if save_animation:
                    msg += f"- Animation: {output_dir}\n"
                QMessageBox.information(self, "Success", msg)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
            self.log_message(f"✗ Export failed: {str(e)}")
    
    def update_status(self, message):
        """Update status label and messages window"""
        self.log_message(message)
    
    def export_finished(self, success, message):
        """Handle export completion"""
        self.btn_export.setEnabled(True)
        
        if success:
            self.log_message("✓ Export completed successfully")
            QMessageBox.information(self, "Success", message)
        else:
            self.log_message(f"✗ Export failed: {message}")
            QMessageBox.critical(self, "Error", message)
        
        self.log_message("Ready for next operation")


def main():
    app = QApplication(sys.argv)
    window = UWBQuickVisualizationWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
