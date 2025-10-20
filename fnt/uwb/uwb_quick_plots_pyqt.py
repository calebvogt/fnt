import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QMessageBox, 
                             QGroupBox, QCheckBox, QScrollArea, QComboBox,
                             QSpinBox, QSplitter, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class PlotGeneratorWorker(QThread):
    """Worker thread for generating plots without blocking the UI"""
    progress = pyqtSignal(str)
    plot_ready = pyqtSignal(object, str)  # (figure, plot_name)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, db_path, table_name, selected_tags, plot_types, downsample, smoothing_method):
        super().__init__()
        self.db_path = db_path
        self.table_name = table_name
        self.selected_tags = selected_tags
        self.plot_types = plot_types
        self.downsample = downsample
        self.smoothing_method = smoothing_method
        
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
            data['location_x'] *= 0.0254  # Convert to meters
            data['location_y'] *= 0.0254
            data = data.sort_values(by=['shortid', 'Timestamp'])
            
            # Filter to selected tags
            if self.selected_tags:
                data = data[data['shortid'].isin(self.selected_tags)]
                self.progress.emit(f"Filtered to {len(self.selected_tags)} selected tags")
            
            if data.empty:
                self.finished.emit(False, "No data available for selected tags")
                return
            
            # Downsample to 1Hz if requested (before smoothing)
            if self.downsample:
                self.progress.emit("Downsampling to 1Hz...")
                data = self.apply_downsampling(data)
            
            # Apply smoothing if requested (after downsampling)
            if self.smoothing_method != "None":
                self.progress.emit(f"Applying {self.smoothing_method} smoothing...")
                data = self.apply_smoothing(data, self.smoothing_method)
            
            # Generate requested plots
            for plot_type, enabled in self.plot_types.items():
                if enabled:
                    self.progress.emit(f"Generating {plot_type}...")
                    fig = self.generate_plot(data, plot_type)
                    if fig is not None:
                        self.plot_ready.emit(fig, plot_type)
            
            self.finished.emit(True, "All plots generated successfully!")
            
        except Exception as e:
            self.finished.emit(False, f"Error generating plots: {str(e)}")
    
    def apply_downsampling(self, data):
        """Downsample data to 1Hz (1 sample per second per tag)"""
        data = data.copy()
        data['time_sec'] = (data['Timestamp'].astype(np.int64) // 1_000_000_000).astype(int)
        # Take first sample in each second for each tag
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
            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(
                lambda x: x.rolling(30, min_periods=1).mean())
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(
                lambda x: x.rolling(30, min_periods=1).mean())
        
        return data
    
    def generate_plot(self, data, plot_type):
        """Generate individual plot based on type"""
        try:
            if plot_type == "Battery Levels":
                return self.plot_battery_levels(data)
            elif plot_type == "Trajectory Overview":
                return self.plot_trajectory_overview(data)
            elif plot_type == "Daily Trajectories":
                return self.plot_daily_trajectories(data)
            elif plot_type == "3D Occupancy Heatmap":
                return self.plot_3d_occupancy(data)
            elif plot_type == "Activity Timeline":
                return self.plot_activity_timeline(data)
            elif plot_type == "Velocity Distribution":
                return self.plot_velocity_distribution(data)
            elif plot_type == "Tag Summary Stats":
                return self.plot_summary_stats(data)
            elif plot_type == "Data Quality":
                return self.plot_data_quality(data)
            return None
        except Exception as e:
            self.progress.emit(f"Error in {plot_type}: {str(e)}")
            return None
    
    def plot_battery_levels(self, data):
        """Plot battery levels over time for each tag"""
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Search for battery column - try common variations
        battery_col = None
        possible_names = ['battery_voltage', 'vbat', 'battery', 'bat', 'voltage', 'Battery', 'VBAT', 'Voltage']
        
        for col_name in possible_names:
            if col_name in data.columns:
                battery_col = col_name
                break
        
        if battery_col is None:
            # Show available columns to help debug
            col_list = ', '.join(data.columns[:10].tolist())
            if len(data.columns) > 10:
                col_list += f"... ({len(data.columns)} total columns)"
            ax.text(0.5, 0.5, f'Battery data not available in database\n\nAvailable columns: {col_list}', 
                   ha='center', va='center', fontsize=10, wrap=True)
            ax.axis('off')
            return fig
        
        for tag in data['shortid'].unique():
            tag_data = data[data['shortid'] == tag]
            # Downsample to hourly for cleaner plot
            tag_data_hourly = tag_data.set_index('Timestamp').resample('1H')[battery_col].mean()
            ax.plot(tag_data_hourly.index, tag_data_hourly.values, 
                   label=f'Tag {tag}', marker='o', markersize=2, linewidth=1)
        
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel(f'{battery_col} (V)', fontsize=10)
        ax.set_title('Battery Levels Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig
    
    def plot_trajectory_overview(self, data):
        """Plot trajectory overview for all tags"""
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(data['shortid'].unique())))
        
        for i, tag in enumerate(data['shortid'].unique()):
            tag_data = data[data['shortid'] == tag]
            ax.plot(tag_data[x_col], tag_data[y_col], 
                   label=f'Tag {tag}', alpha=0.6, linewidth=0.8, color=colors[i])
        
        ax.set_xlabel('X Position (m)', fontsize=10)
        ax.set_ylabel('Y Position (m)', fontsize=10)
        ax.set_title('Trajectory Overview - All Tags', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        fig.tight_layout()
        return fig
    
    def plot_daily_trajectories(self, data):
        """Plot daily trajectories for each tag"""
        # Add Date column if not present
        data = data.copy()
        if 'Date' not in data.columns:
            data['Date'] = data['Timestamp'].dt.date
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Get unique dates and tags
        unique_dates = sorted(data['Date'].unique())
        unique_tags = sorted(data['shortid'].unique())
        
        # Create subplots: one row per tag, columns for days
        num_tags = len(unique_tags)
        num_days = len(unique_dates)
        
        fig = Figure(figsize=(min(4 * num_days, 20), 3 * num_tags))
        
        for tag_idx, tag in enumerate(unique_tags):
            tag_data = data[data['shortid'] == tag]
            
            for day_idx, date in enumerate(unique_dates):
                day_data = tag_data[tag_data['Date'] == date]
                
                ax = fig.add_subplot(num_tags, num_days, tag_idx * num_days + day_idx + 1)
                
                if not day_data.empty:
                    ax.plot(day_data[x_col], day_data[y_col], 
                           linewidth=1, alpha=0.7, color='blue')
                    ax.scatter(day_data[x_col].iloc[0], day_data[y_col].iloc[0], 
                              c='green', s=50, marker='o', label='Start', zorder=5)
                    ax.scatter(day_data[x_col].iloc[-1], day_data[y_col].iloc[-1], 
                              c='red', s=50, marker='s', label='End', zorder=5)
                
                ax.set_xlabel('X (m)', fontsize=8)
                ax.set_ylabel('Y (m)', fontsize=8)
                ax.set_title(f'Tag {tag} - {date}', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                if tag_idx == 0 and day_idx == 0:
                    ax.legend(fontsize=7)
        
        fig.suptitle('Daily Trajectories by Tag', fontsize=14, fontweight='bold')
        fig.tight_layout()
        return fig
    
    def plot_3d_occupancy(self, data):
        """Plot 3D occupancy heatmap for each tag"""
        from mpl_toolkits.mplot3d import Axes3D
        
        # Add Date column if not present
        data = data.copy()
        if 'Date' not in data.columns:
            data['Date'] = data['Timestamp'].dt.date
        
        # Add Day column
        unique_dates = sorted(data['Date'].unique())
        date_to_day = {date: i+1 for i, date in enumerate(unique_dates)}
        data['Day'] = data['Date'].map(date_to_day)
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        unique_tags = sorted(data['shortid'].unique())
        num_tags = len(unique_tags)
        
        # Create subplot for each tag
        fig = Figure(figsize=(8 * min(num_tags, 2), 6 * ((num_tags + 1) // 2)))
        
        for idx, tag in enumerate(unique_tags):
            tag_data = data[data['shortid'] == tag]
            unique_days = sorted(tag_data['Day'].unique())
            
            ax = fig.add_subplot((num_tags + 1) // 2, min(num_tags, 2), idx + 1, projection='3d')
            
            # Create 2D histogram for overall occupancy
            x = tag_data[x_col]
            y = tag_data[y_col]
            
            if len(x) > 0 and len(y) > 0:
                occupancy, xedges, yedges = np.histogram2d(x, y, bins=30)
                
                # Create meshgrid
                xcenters = (xedges[:-1] + xedges[1:]) / 2
                ycenters = (yedges[:-1] + yedges[1:]) / 2
                X, Y = np.meshgrid(xcenters, ycenters)
                
                # Plot surface
                surf = ax.plot_surface(X, Y, occupancy.T, cmap='viridis', 
                                      edgecolor='none', alpha=0.8)
                
                ax.set_xlabel('X Position (m)', fontsize=9)
                ax.set_ylabel('Y Position (m)', fontsize=9)
                ax.set_zlabel('Occupancy Count', fontsize=9)
                ax.set_title(f'Tag {tag} - 3D Occupancy', fontsize=10, fontweight='bold')
                
                # Add colorbar
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        fig.suptitle('3D Occupancy Heatmaps', fontsize=14, fontweight='bold')
        fig.tight_layout()
        return fig
    
    def plot_activity_timeline(self, data):
        """Plot activity timeline showing data points per hour"""
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        # Count data points per hour for each tag
        for tag in data['shortid'].unique():
            tag_data = data[data['shortid'] == tag]
            hourly_counts = tag_data.set_index('Timestamp').resample('1H').size()
            ax.plot(hourly_counts.index, hourly_counts.values, 
                   label=f'Tag {tag}', marker='o', markersize=3, linewidth=1)
        
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Data Points per Hour', fontsize=10)
        ax.set_title('Activity Timeline - Data Points Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig
    
    def plot_velocity_distribution(self, data):
        """Plot velocity distribution for each tag"""
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Calculate velocity
        data = data.copy()
        data['time_diff'] = data.groupby('shortid')['Timestamp'].diff().dt.total_seconds()
        data['distance'] = np.sqrt(
            (data['location_x'] - data.groupby('shortid')['location_x'].shift())**2 +
            (data['location_y'] - data.groupby('shortid')['location_y'].shift())**2
        )
        data['velocity'] = data['distance'] / data['time_diff']
        
        # Filter out unrealistic velocities
        data = data[(data['velocity'] <= 2) | (data['velocity'].isna())]
        
        for tag in data['shortid'].unique():
            tag_data = data[data['shortid'] == tag]
            velocities = tag_data['velocity'].dropna()
            if len(velocities) > 0:
                ax.hist(velocities, bins=50, alpha=0.5, label=f'Tag {tag}', density=True)
        
        ax.set_xlabel('Velocity (m/s)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title('Velocity Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    
    def plot_summary_stats(self, data):
        """Plot summary statistics table"""
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Calculate summary stats for each tag
        stats_data = []
        for tag in sorted(data['shortid'].unique()):
            tag_data = data[data['shortid'] == tag]
            
            duration = (tag_data['Timestamp'].max() - tag_data['Timestamp'].min()).total_seconds() / 3600
            n_points = len(tag_data)
            
            # Calculate total distance
            tag_data = tag_data.sort_values('Timestamp')
            distances = np.sqrt(
                (tag_data['location_x'].diff())**2 + 
                (tag_data['location_y'].diff())**2
            )
            total_distance = distances.sum()
            
            stats_data.append([
                f'Tag {tag}',
                f'{n_points:,}',
                f'{duration:.1f} hrs',
                f'{total_distance:.1f} m',
                f'{tag_data["Timestamp"].min():%Y-%m-%d %H:%M}',
                f'{tag_data["Timestamp"].max():%Y-%m-%d %H:%M}'
            ])
        
        # Create table
        table = ax.table(cellText=stats_data,
                        colLabels=['Tag ID', 'Data Points', 'Duration', 'Total Distance', 'Start Time', 'End Time'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.12, 0.15, 0.15, 0.18, 0.2, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Tag Summary Statistics', fontsize=12, fontweight='bold', pad=20)
        fig.tight_layout()
        return fig
    
    def plot_data_quality(self, data):
        """Plot data quality metrics"""
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Calculate data gaps (time between consecutive points)
        quality_data = []
        for tag in sorted(data['shortid'].unique()):
            tag_data = data[data['shortid'] == tag].sort_values('Timestamp')
            time_diffs = tag_data['Timestamp'].diff().dt.total_seconds()
            
            # Statistics
            median_gap = time_diffs.median()
            max_gap = time_diffs.max()
            gaps_over_60s = (time_diffs > 60).sum()
            
            quality_data.append([
                f'Tag {tag}',
                f'{median_gap:.2f}s',
                f'{max_gap:.0f}s',
                f'{gaps_over_60s}'
            ])
        
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=quality_data,
                        colLabels=['Tag ID', 'Median Gap', 'Max Gap', 'Gaps >60s'],
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
        return fig


class UWBQuickPlotsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.db_path = None
        self.table_name = None
        self.available_tags = []
        self.plot_canvases = {}  # Store references to plot canvases
        self.worker = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("UWB Quick Plots")
        self.setGeometry(50, 50, 1400, 900)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Settings
        left_panel = self.create_settings_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Plot display
        right_panel = self.create_plot_panel()
        splitter.addWidget(right_panel)
        
        # Set initial sizes (30% left, 70% right)
        splitter.setSizes([400, 1000])
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
    def create_settings_panel(self):
        """Create the left settings panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("UWB Quick Plots")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Generate quick visualization plots from UWB tracking data")
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
        table_select_layout = QHBoxLayout()
        table_select_layout.addWidget(QLabel("Table:"))
        self.combo_table = QComboBox()
        self.combo_table.setEnabled(False)
        self.combo_table.currentTextChanged.connect(self.on_table_selected)
        table_select_layout.addWidget(self.combo_table)
        db_layout.addLayout(table_select_layout)
        
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
        
        self.tag_group.setLayout(self.tag_layout)
        layout.addWidget(self.tag_group)
        
        # Processing options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        
        # Downsample option
        self.chk_downsample = QCheckBox("Downsample to 1Hz")
        self.chk_downsample.setChecked(True)
        self.chk_downsample.setToolTip("Downsample data to 1 sample per second before smoothing")
        options_layout.addWidget(self.chk_downsample)
        
        # Smoothing method
        options_layout.addWidget(QLabel("Smoothing method:"))
        self.combo_smoothing = QComboBox()
        self.combo_smoothing.addItems(["None", "Savitzky-Golay", "Rolling Average"])
        options_layout.addWidget(self.combo_smoothing)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Plot type selection
        plot_group = QGroupBox("Plot Types")
        plot_layout = QVBoxLayout()
        
        self.plot_checkboxes = {}
        plot_types = [
            ("Battery Levels", "Battery voltage over time for each tag"),
            ("Trajectory Overview", "Combined spatial trajectories"),
            ("Daily Trajectories", "Individual trajectory plots for each day and tag"),
            ("3D Occupancy Heatmap", "3D surface plots showing spatial usage over time"),
            ("Activity Timeline", "Data points collected over time"),
            ("Velocity Distribution", "Speed distribution histograms"),
            ("Tag Summary Stats", "Summary statistics table"),
            ("Data Quality", "Data quality metrics and gaps")
        ]
        
        for plot_name, plot_desc in plot_types:
            cb = QCheckBox(plot_name)
            cb.setToolTip(plot_desc)
            cb.setChecked(True)
            self.plot_checkboxes[plot_name] = cb
            plot_layout.addWidget(cb)
        
        # Select All/None buttons
        plot_btn_layout = QHBoxLayout()
        btn_plot_all = QPushButton("Select All")
        btn_plot_all.clicked.connect(self.select_all_plots)
        btn_plot_none = QPushButton("Select None")
        btn_plot_none.clicked.connect(self.select_none_plots)
        plot_btn_layout.addWidget(btn_plot_all)
        plot_btn_layout.addWidget(btn_plot_none)
        plot_layout.addLayout(plot_btn_layout)
        
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)
        
        # Generate button
        self.btn_generate = QPushButton("Generate Quick Report")
        self.btn_generate.clicked.connect(self.generate_plots)
        self.btn_generate.setEnabled(False)
        self.btn_generate.setStyleSheet("padding: 10px; font-size: 12px; font-weight: bold;")
        layout.addWidget(self.btn_generate)
        
        # Status label
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #666666; font-style: italic;")
        layout.addWidget(self.lbl_status)
        
        layout.addStretch()
        panel.setLayout(layout)
        
        # Make panel scrollable
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(350)
        
        return scroll
        
    def create_plot_panel(self):
        """Create the right plot display panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Title and buttons
        header_layout = QHBoxLayout()
        
        title = QLabel("Generated Plots")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        self.btn_save_all = QPushButton("Save All Plots")
        self.btn_save_all.clicked.connect(self.save_all_plots)
        self.btn_save_all.setEnabled(False)
        header_layout.addWidget(self.btn_save_all)
        
        self.btn_clear_all = QPushButton("Clear All Plots")
        self.btn_clear_all.clicked.connect(self.clear_all_plots)
        self.btn_clear_all.setEnabled(False)
        header_layout.addWidget(self.btn_clear_all)
        
        layout.addLayout(header_layout)
        
        # Scrollable plot area
        self.plot_scroll = QScrollArea()
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout()
        self.plot_container.setLayout(self.plot_layout)
        self.plot_scroll.setWidget(self.plot_container)
        self.plot_scroll.setWidgetResizable(True)
        
        # Initial message
        self.lbl_no_plots = QLabel("No plots generated yet.\n\nConfigure settings on the left and click 'Generate Quick Report'")
        self.lbl_no_plots.setAlignment(Qt.AlignCenter)
        self.lbl_no_plots.setStyleSheet("color: #999999; font-size: 12px; padding: 50px;")
        self.plot_layout.addWidget(self.lbl_no_plots)
        
        layout.addWidget(self.plot_scroll)
        
        panel.setLayout(layout)
        return panel
    
    def select_database(self):
        """Select SQLite database file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SQLite Database",
            "",
            "SQLite Files (*.sqlite *.db *.sql);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            self.db_path = file_path
            self.lbl_db.setText(f"{os.path.basename(file_path)}")
            self.lbl_db.setStyleSheet("color: black;")
            
            # Get available tables
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if not tables:
                QMessageBox.warning(self, "No Tables", "No tables found in the database.")
                return
            
            # Populate table dropdown
            self.combo_table.clear()
            self.combo_table.addItems(tables)
            self.combo_table.setEnabled(True)
            
            # If only one table, select it automatically
            if len(tables) == 1:
                self.table_name = tables[0]
                self.load_tags_from_table()
            else:
                self.lbl_status.setText(f"Found {len(tables)} tables. Please select one.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load database:\n{str(e)}")
    
    def on_table_selected(self, table_name):
        """Handle table selection from dropdown"""
        if table_name:
            self.table_name = table_name
            self.btn_preview_table.setEnabled(True)
            self.load_tags_from_table()
    
    def preview_table(self):
        """Preview the selected table in a new window"""
        if not self.db_path or not self.table_name:
            QMessageBox.warning(self, "No Table", "Please select a table first.")
            return
        
        try:
            # Load first 20 rows
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT * FROM {self.table_name} LIMIT 20"
            df = pd.read_sql_query(query, conn)
            
            # Also get column names and types
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({self.table_name})")
            columns_info = cursor.fetchall()
            conn.close()
            
            # Create preview dialog
            from PyQt5.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QHeaderView
            
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Preview: {self.table_name}")
            dialog.setGeometry(100, 100, 1000, 600)
            
            layout = QVBoxLayout()
            
            # Info label
            info_label = QLabel(f"Showing first 20 rows of table '{self.table_name}'\n"
                               f"Total columns: {len(df.columns)} | Column types shown below table")
            info_label.setStyleSheet("font-weight: bold; margin: 10px;")
            layout.addWidget(info_label)
            
            # Create table widget
            table = QTableWidget()
            table.setRowCount(len(df))
            table.setColumnCount(len(df.columns))
            table.setHorizontalHeaderLabels(df.columns.tolist())
            
            # Populate table
            for i in range(len(df)):
                for j in range(len(df.columns)):
                    value = str(df.iloc[i, j])
                    item = QTableWidgetItem(value)
                    table.setItem(i, j, item)
            
            # Auto-resize columns
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            
            layout.addWidget(table)
            
            # Column info
            col_info_text = "Column Types: " + " | ".join([f"{col[1]}: {col[2]}" for col in columns_info])
            col_info_label = QLabel(col_info_text)
            col_info_label.setWordWrap(True)
            col_info_label.setStyleSheet("color: #666666; font-size: 9px; margin: 10px;")
            layout.addWidget(col_info_label)
            
            # Close button
            btn_close = QPushButton("Close")
            btn_close.clicked.connect(dialog.close)
            layout.addWidget(btn_close)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to preview table:\n{str(e)}")
    
    def load_tags_from_table(self):
        """Load tags from the selected table"""
        if not self.db_path or not self.table_name:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            tag_query = f"SELECT DISTINCT shortid FROM {self.table_name} ORDER BY shortid"
            tags_df = pd.read_sql_query(tag_query, conn)
            conn.close()
            
            self.available_tags = list(tags_df['shortid'])
            
            # Update tag selection UI
            self.update_tag_selection()
            
            self.btn_generate.setEnabled(True)
            self.lbl_status.setText(f"Table '{self.table_name}': Found {len(self.available_tags)} tags")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load tags from table:\n{str(e)}")
            self.btn_generate.setEnabled(False)
    
    def update_tag_selection(self):
        """Update tag selection checkboxes"""
        # Clear existing checkboxes
        for cb in self.tag_checkboxes.values():
            cb.deleteLater()
        self.tag_checkboxes.clear()
        
        if self.lbl_no_tags:
            self.lbl_no_tags.deleteLater()
            self.lbl_no_tags = None
        
        # Add checkboxes for each tag
        for tag in self.available_tags:
            cb = QCheckBox(f"Tag {tag}")
            cb.setChecked(True)
            self.tag_checkboxes[tag] = cb
            self.tag_layout.insertWidget(self.tag_layout.count() - 1, cb)  # Insert before buttons
    
    def select_all_tags(self):
        """Select all tags"""
        for cb in self.tag_checkboxes.values():
            cb.setChecked(True)
    
    def select_none_tags(self):
        """Deselect all tags"""
        for cb in self.tag_checkboxes.values():
            cb.setChecked(False)
    
    def select_all_plots(self):
        """Select all plot types"""
        for cb in self.plot_checkboxes.values():
            cb.setChecked(True)
    
    def select_none_plots(self):
        """Deselect all plot types"""
        for cb in self.plot_checkboxes.values():
            cb.setChecked(False)
    
    def generate_plots(self):
        """Generate plots based on selections"""
        if not self.db_path:
            QMessageBox.warning(self, "No Database", "Please select a database first.")
            return
        
        if not self.table_name:
            QMessageBox.warning(self, "No Table", "Please select a table first.")
            return
        
        # Get selected tags
        selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]
        if not selected_tags:
            QMessageBox.warning(self, "No Tags", "Please select at least one tag.")
            return
        
        # Get selected plot types
        selected_plots = {name: cb.isChecked() for name, cb in self.plot_checkboxes.items()}
        if not any(selected_plots.values()):
            QMessageBox.warning(self, "No Plots", "Please select at least one plot type.")
            return
        
        # Clear existing plots
        self.clear_all_plots()
        
        # Disable controls during generation
        self.btn_generate.setEnabled(False)
        self.lbl_status.setText("Generating plots...")
        
        # Get processing options
        downsample = self.chk_downsample.isChecked()
        smoothing = self.combo_smoothing.currentText()
        
        # Create and start worker thread
        self.worker = PlotGeneratorWorker(self.db_path, self.table_name, selected_tags, selected_plots, downsample, smoothing)
        self.worker.progress.connect(self.update_status)
        self.worker.plot_ready.connect(self.add_plot)
        self.worker.finished.connect(self.generation_finished)
        self.worker.start()
    
    def update_status(self, message):
        """Update status label"""
        self.lbl_status.setText(message)
    
    def add_plot(self, figure, plot_name):
        """Add a generated plot to the display area"""
        # Remove "no plots" message if it exists
        if self.lbl_no_plots and self.lbl_no_plots.parent():
            self.lbl_no_plots.deleteLater()
            self.lbl_no_plots = None
        
        # Create plot widget
        plot_widget = QWidget()
        plot_layout = QVBoxLayout()
        
        # Add canvas with fixed height
        canvas = FigureCanvas(figure)
        canvas.setMinimumHeight(500)  # Set minimum height for each plot
        toolbar = NavigationToolbar(canvas, self)
        
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(canvas)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        plot_layout.addWidget(separator)
        
        plot_widget.setLayout(plot_layout)
        self.plot_layout.addWidget(plot_widget)
        
        # Store reference
        self.plot_canvases[plot_name] = (canvas, figure)
        
        # Enable save/clear buttons
        self.btn_save_all.setEnabled(True)
        self.btn_clear_all.setEnabled(True)
    
    def generation_finished(self, success, message):
        """Handle plot generation completion"""
        self.btn_generate.setEnabled(True)
        
        if success:
            self.lbl_status.setText(f"✅ {message}")
        else:
            self.lbl_status.setText(f"❌ {message}")
            QMessageBox.critical(self, "Error", message)
    
    def save_all_plots(self):
        """Save all generated plots"""
        if not self.plot_canvases:
            return
        
        # Select directory
        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Plots")
        if not save_dir:
            return
        
        try:
            for plot_name, (canvas, figure) in self.plot_canvases.items():
                filename = f"{plot_name.replace(' ', '_').lower()}.png"
                filepath = os.path.join(save_dir, filename)
                figure.savefig(filepath, dpi=300, bbox_inches='tight')
            
            QMessageBox.information(
                self,
                "Success",
                f"Saved {len(self.plot_canvases)} plots to:\n{save_dir}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plots:\n{str(e)}")
    
    def clear_all_plots(self):
        """Clear all generated plots"""
        # Clear plot canvases
        for canvas, figure in self.plot_canvases.values():
            plt.close(figure)
        
        self.plot_canvases.clear()
        
        # Clear plot layout
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add "no plots" message back
        self.lbl_no_plots = QLabel("No plots generated yet.\n\nConfigure settings on the left and click 'Generate Quick Report'")
        self.lbl_no_plots.setAlignment(Qt.AlignCenter)
        self.lbl_no_plots.setStyleSheet("color: #999999; font-size: 12px; padding: 50px;")
        self.plot_layout.addWidget(self.lbl_no_plots)
        
        # Disable save/clear buttons
        self.btn_save_all.setEnabled(False)
        self.btn_clear_all.setEnabled(False)
        
        self.lbl_status.setText("Plots cleared")


def main():
    app = QApplication(sys.argv)
    window = UWBQuickPlotsWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
