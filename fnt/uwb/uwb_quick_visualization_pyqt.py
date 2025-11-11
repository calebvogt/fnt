import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import pytz
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
                             QSpinBox, QSplitter, QFrame, QSlider)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont


class PlotSaverWorker(QThread):
    """Worker thread for saving plots without blocking the UI"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, db_path, table_name, selected_tags, downsample, smoothing_method, 
                 plot_types=None, overwrite=True, rolling_window=10, timezone='US/Mountain'):
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
            data = data.sort_values(by=['shortid', 'Timestamp'])
            
            # Filter to selected tags
            if self.selected_tags:
                data = data[data['shortid'].isin(self.selected_tags)]
            
            # Downsample if requested
            if self.downsample:
                data = self.apply_downsampling(data)
            
            # Apply smoothing if requested
            if self.smoothing_method != "None":
                data = self.apply_smoothing(data, self.smoothing_method)
            
            # Get output directory (same as database)
            output_dir = os.path.dirname(self.db_path)
            
            # Generate and save plots based on selection
            generated_count = 0
            skipped_count = 0
            
            if self.plot_types.get('daily_paths', False):
                result = self.save_daily_paths_per_tag(data, output_dir)
                if result:
                    generated_count += result
            
            if self.plot_types.get('trajectory_overview', False):
                result = self.save_trajectory_overview(data, output_dir)
                if result:
                    generated_count += 1
                else:
                    skipped_count += 1
            
            if self.plot_types.get('battery_levels', False):
                result = self.save_battery_levels(data, output_dir)
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
    
    def save_daily_paths_per_tag(self, data, output_dir):
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
            output_path = os.path.join(output_dir, f'uwb_DailyPaths_Tag{tag}.png')
            
            # Check if file exists and overwrite is False
            if not self.overwrite and os.path.exists(output_path):
                self.progress.emit(f"Skipped (exists): uwb_DailyPaths_Tag{tag}.png")
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
                ax.invert_yaxis()
            
            fig.suptitle(f'Daily Paths - Tag {tag}', fontsize=14, fontweight='bold')
            fig.tight_layout()
            
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            generated += 1
            
            self.progress.emit(f"Saved: uwb_DailyPaths_Tag{tag}.png")
        
        return generated
    
    def save_trajectory_overview(self, data, output_dir):
        """Save trajectory overview
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating trajectory overview...")
        
        output_path = os.path.join(output_dir, 'uwb_TrajectoryOverview.png')
        
        # Check if file exists and overwrite is False
        if not self.overwrite and os.path.exists(output_path):
            self.progress.emit("Skipped (exists): uwb_TrajectoryOverview.png")
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
        ax.invert_yaxis()
        fig.tight_layout()
        
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self.progress.emit("Saved: uwb_TrajectoryOverview.png")
        return True
    
    def save_battery_levels(self, data, output_dir):
        """Save battery levels plot
        Returns: True if generated, False if skipped or no battery data"""
        self.progress.emit("Generating battery levels...")
        
        output_path = os.path.join(output_dir, 'uwb_BatteryLevels.png')
        
        # Check if file exists and overwrite is False
        if not self.overwrite and os.path.exists(output_path):
            self.progress.emit("Skipped (exists): uwb_BatteryLevels.png")
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
        
        self.progress.emit("Saved: uwb_BatteryLevels.png")
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
        
        self.initUI()
        
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
        
        # Downsample
        self.chk_downsample = QCheckBox("Downsample to 1Hz")
        self.chk_downsample.setChecked(True)
        self.chk_downsample.stateChanged.connect(self.mark_options_changed)
        options_layout.addWidget(self.chk_downsample)
        
        # Smoothing
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
        self.chk_save_plots.setChecked(True)
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
            ("battery_levels", "Battery Levels", "Battery voltage over time")
        ]
        
        for key, plot_name, plot_desc in plot_types:
            cb = QCheckBox(plot_name)
            cb.setChecked(True)
            cb.setToolTip(plot_desc)
            self.plot_type_checkboxes[key] = cb
            plot_types_layout.addWidget(cb)
        
        # Overwrite checkbox (also indented)
        self.chk_overwrite = QCheckBox("Overwrite existing files")
        self.chk_overwrite.setChecked(False)
        self.chk_overwrite.setToolTip("If unchecked, will skip files that already exist")
        plot_types_layout.addWidget(self.chk_overwrite)
        
        self.plot_types_widget.setLayout(plot_types_layout)
        export_layout.addWidget(self.plot_types_widget)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Export button
        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self.export_data)
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet("padding: 10px; font-size: 12px; font-weight: bold;")
        layout.addWidget(self.btn_export)
        
        # Status
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #666666; font-style: italic; font-size: 10px;")
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
            
            if len(tables) == 1:
                self.combo_table.setCurrentIndex(0)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open database: {str(e)}")
    
    def on_table_selected(self, table_name):
        """Handle table selection"""
        if table_name:
            self.table_name = table_name
            self.load_tags_from_table()
    
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
        self.plot_types_widget.setEnabled(enabled)
        for cb in self.plot_type_checkboxes.values():
            cb.setEnabled(enabled)
        self.chk_overwrite.setEnabled(enabled)
    
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
            self.tag_checkboxes[tag] = cb
            self.tag_layout.insertWidget(self.tag_layout.count() - 1, cb)
    
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
            self.lbl_status.setText("Loading data...")
            
            # Load data
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT * FROM {self.table_name}"
            self.data = pd.read_sql_query(query, conn)
            conn.close()
            
            # Process data
            self.data['Timestamp'] = pd.to_datetime(self.data['timestamp'], unit='ms', origin='unix', utc=True)
            tz = pytz.timezone(self.combo_timezone.currentText())
            self.data['Timestamp'] = self.data['Timestamp'].dt.tz_convert(tz)
            
            self.data['location_x'] *= 0.0254
            self.data['location_y'] *= 0.0254
            self.data = self.data.sort_values(by=['shortid', 'Timestamp'])
            
            # Filter tags
            self.data = self.data[self.data['shortid'].isin(selected_tags)]
            
            # Downsample if requested
            if self.chk_downsample.isChecked():
                self.data['time_sec'] = (self.data['Timestamp'].astype(np.int64) // 1_000_000_000).astype(int)
                self.data = self.data.groupby(['shortid', 'time_sec']).first().reset_index()
            
            # Apply smoothing if requested
            if self.combo_smoothing.currentText() != "None":
                self.apply_smoothing()
            
            # Setup slider based on unique timestamps
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
            self.lbl_status.setText(f"Loaded {len(self.data)} data points across {len(unique_times)} unique timestamps")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load preview: {str(e)}")
            self.lbl_status.setText("Error loading data")
    
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
        self.ax.invert_yaxis()
        
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
    
    def export_data(self):
        """Export data and/or plots based on selected options"""
        if not self.db_path or self.data is None:
            return
        
        output_dir = os.path.dirname(self.db_path)
        export_csv = self.chk_export_csv.isChecked()
        save_plots = self.chk_save_plots.isChecked()
        
        if not export_csv and not save_plots:
            QMessageBox.warning(self, "No Export Selected", "Please select at least one export option (CSV or Plots)")
            return
        
        try:
            # Export CSV if requested
            if export_csv:
                self.lbl_status.setText("Exporting CSV...")
                csv_path = os.path.join(output_dir, f'{self.table_name}_processed.csv')
                self.data.to_csv(csv_path, index=False)
                self.lbl_status.setText(f"CSV exported to {csv_path}")
            
            # Export plots if requested
            if save_plots:
                selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]
                
                # Get selected plot types
                plot_types = {
                    'daily_paths': self.plot_type_checkboxes['daily_paths'].isChecked(),
                    'trajectory_overview': self.plot_type_checkboxes['trajectory_overview'].isChecked(),
                    'battery_levels': self.plot_type_checkboxes['battery_levels'].isChecked()
                }
                
                # Get overwrite setting
                overwrite = self.chk_overwrite.isChecked()
                
                # Get rolling window value
                rolling_window = self.spin_rolling_window.value()
                
                self.btn_export.setEnabled(False)
                self.lbl_status.setText("Generating plots...")
                
                self.worker = PlotSaverWorker(
                    self.db_path, 
                    self.table_name, 
                    selected_tags,
                    self.chk_downsample.isChecked(),
                    self.combo_smoothing.currentText(),
                    plot_types,
                    overwrite,
                    rolling_window,
                    self.combo_timezone.currentText()
                )
                self.worker.progress.connect(self.update_status)
                self.worker.finished.connect(self.export_finished)
                self.worker.start()
            elif export_csv:
                # If only CSV was exported (no plots), show success message
                QMessageBox.information(self, "Success", f"CSV exported to {csv_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
            self.lbl_status.setText("Export failed")
    
    def update_status(self, message):
        """Update status label"""
        self.lbl_status.setText(message)
    
    def export_finished(self, success, message):
        """Handle export completion"""
        self.btn_export.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)
        
        self.lbl_status.setText("Ready")


def main():
    app = QApplication(sys.argv)
    window = UWBQuickVisualizationWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
