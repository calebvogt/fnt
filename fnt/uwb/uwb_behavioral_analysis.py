"""
Standalone UWB Behavioral Analysis Tool

This script provides a complete GUI-based workflow for analyzing behaviors from UWB tracking data.
Based on the behavioral_classifier.py module and reusing GUI patterns from uwb_plots.py and uwb_animate.py.

Features:
- SQLite database selection
- Data preprocessing (downsampling, smoothing, timezone)
- Tag selection and metadata
- Behavioral parameter configuration
- Real-time behavioral analysis
- Results visualization and export

Author: AI Assistant
Date: 2025-07-21
"""

import tkinter as tk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import sqlite3
import os
import json
import traceback
from datetime import datetime
import sys

# Import our behavioral classifier module
# TODO: Implement behavioral_classifier module
try:
    # from behavioral_classifier import BehavioralClassifier, BehaviorConfig
    print("⚠️ Behavioral classifier module not yet implemented - skipping import")
    BEHAVIORAL_CLASSIFIER_AVAILABLE = False
except ImportError as e:
    print(f"❌ Failed to import behavioral_classifier: {e}")
    print("Make sure behavioral_classifier.py is in the same directory")
    BEHAVIORAL_CLASSIFIER_AVAILABLE = False

def uwb_behavioral_analysis():
    """
    Main function for standalone behavioral analysis with GUI prompts
    """
    print("🧠 Starting UWB Behavioral Analysis Tool...")
    print("=" * 60)

    # ===== STEP 1: DATABASE SELECTION =====
    print("Step 1: Database Selection")
    root = tk.Tk()
    root.withdraw()
    print("Please select your SQLite database file...")
    file_path = filedialog.askopenfilename(
        title="Select UWB SQLite Database", 
        filetypes=[("SQLite Files", "*.sqlite"), ("Database Files", "*.db"), ("All Files", "*.*")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"Selected database: {os.path.basename(file_path)}")

    # ===== STEP 2: DATA PREPROCESSING OPTIONS =====
    print("\nStep 2: Data Preprocessing Configuration")
    
    # Downsampling option
    print("Configuring downsampling option...")
    downsample = messagebox.askyesno(
        "Downsample Data", 
        "Do you want to downsample the data to 1Hz?\n\n" +
        "Recommended: Yes for large datasets or faster processing\n" +
        "Choose No to keep original sampling rate"
    )
    print(f"Downsampling: {'Enabled' if downsample else 'Disabled'}")

    # Smoothing method selection
    print("Configuring smoothing method...")
    smoothing_window = tk.Tk()
    smoothing_window.title("Select Smoothing Method")
    smoothing_window.geometry("400x300")

    smoothing_choice = tk.StringVar(value="none")

    def set_smoothing(choice):
        smoothing_choice.set(choice)
        smoothing_window.quit()
        smoothing_window.destroy()

    # Smoothing options UI
    tk.Label(smoothing_window, text="Choose a smoothing method:", 
             font=("Arial", 12, "bold")).pack(pady=15)
    
    tk.Label(smoothing_window, text="Smoothing can help reduce noise in movement data", 
             font=("Arial", 10, "italic")).pack(pady=(0, 20))
    
    tk.Button(smoothing_window, text="Savitzky-Golay Filter", 
              command=lambda: set_smoothing("savitzky-golay"), 
              width=20, height=2).pack(pady=5)
    
    tk.Button(smoothing_window, text="Rolling Average", 
              command=lambda: set_smoothing("rolling-average"), 
              width=20, height=2).pack(pady=5)
    
    tk.Button(smoothing_window, text="No Smoothing", 
              command=lambda: set_smoothing("none"), 
              width=20, height=2).pack(pady=5)

    smoothing_window.mainloop()
    smoothing = smoothing_choice.get()
    print(f"Smoothing method: {smoothing}")

    # Timezone selection
    print("Configuring timezone...")
    timezone_window = tk.Tk()
    timezone_window.title("Select Timezone")
    timezone_window.geometry("300x400")
    
    timezone_choice = tk.StringVar(value="UTC")
    
    def set_timezone(tz):
        timezone_choice.set(tz)
        timezone_window.quit()
        timezone_window.destroy()
    
    tk.Label(timezone_window, text="Select timezone for timestamps:", 
             font=("Arial", 12, "bold")).pack(pady=15)
    
    timezones = [
        ("Eastern (ET)", "US/Eastern"),
        ("Central (CT)", "US/Central"), 
        ("Mountain (MT)", "US/Mountain"),
        ("Pacific (PT)", "US/Pacific"),
        ("Alaska (AKT)", "US/Alaska"),
        ("Hawaii (HST)", "US/Hawaii"),
        ("Keep UTC", "UTC")
    ]
    
    for display_name, tz_code in timezones:
        tk.Button(timezone_window, text=display_name, 
                  command=lambda tz=tz_code: set_timezone(tz), 
                  width=20).pack(pady=3)
    
    timezone_window.mainloop()
    selected_timezone = timezone_choice.get()
    print(f"Selected timezone: {selected_timezone}")

    # ===== STEP 3: BEHAVIORAL ANALYSIS PARAMETERS =====
    print("\nStep 3: Behavioral Analysis Configuration")
    
    behavior_config = configure_behavioral_parameters()
    print("Behavioral parameters configured successfully")

    # ===== STEP 4: TAG SELECTION =====
    print("\nStep 4: Tag Selection and Metadata")
    
    # Quick database query to get available tags
    print("Scanning database for available tags...")
    try:
        conn = sqlite3.connect(file_path)
        tag_query = "SELECT DISTINCT shortid FROM data ORDER BY shortid"
        available_tags = pd.read_sql_query(tag_query, conn)
        conn.close()
        
        if available_tags.empty:
            print("❌ No tags found in database. Exiting.")
            return
            
        print(f"Found {len(available_tags)} tags in database: {list(available_tags['shortid'])}")
    
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return

    # Tag selection and metadata
    tag_results = configure_tag_selection(available_tags['shortid'].tolist())
    selected_tags = [tag_id for tag_id, data in tag_results.items() if data['include']]
    
    if not selected_tags:
        print("❌ No tags selected. Exiting.")
        return
        
    print(f"Selected {len(selected_tags)} tags for analysis: {selected_tags}")

    # ===== STEP 4.5: DAY SELECTION =====
    print("\nStep 4.5: Day Selection")
    print("Scanning database for available dates...")
    
    try:
        # Quick query to get available dates
        conn = sqlite3.connect(file_path)
        date_query = """
            SELECT DISTINCT DATE(datetime(timestamp/1000, 'unixepoch')) as date 
            FROM data 
            WHERE shortid IN ({}) 
            ORDER BY date
        """.format(','.join('?' * len(selected_tags)))
        available_dates = pd.read_sql_query(date_query, conn, params=selected_tags)
        conn.close()
        
        if available_dates.empty:
            print("❌ No dates found for selected tags. Exiting.")
            return
            
        unique_dates = [pd.to_datetime(date).date() for date in available_dates['date']]
        print(f"Found {len(unique_dates)} days of data: {unique_dates}")
        
        # Day selection window (using the same pattern as uwb_animate.py)
        print("Opening day selection window...")
        
        day_window = tk.Tk()
        day_window.title("Select Days for Behavioral Analysis")
        day_window.geometry("400x500")
        
        # Use a simple dictionary to track selections
        day_selections = {date: False for date in unique_dates}
        
        # Header
        tk.Label(day_window, text="Select Days for Behavioral Analysis", 
                 font=("Arial", 14, "bold")).pack(pady=10)
        tk.Label(day_window, text="Tip: Select fewer days for faster processing", 
                 font=("Arial", 10, "italic")).pack(pady=(0, 10))
        
        # Create scrollable frame for days
        canvas = tk.Canvas(day_window)
        scrollbar = tk.Scrollbar(day_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create checkboxes with callback functions
        def make_toggle_callback(date):
            def toggle():
                day_selections[date] = not day_selections[date]
                print(f"  {date} {'selected' if day_selections[date] else 'deselected'}")
            return toggle
        
        for date in unique_dates:
            checkbox = tk.Checkbutton(
                scrollable_frame, 
                text=f"{date} (Day {unique_dates.index(date) + 1})",
                command=make_toggle_callback(date),
                font=("Arial", 10)
            )
            checkbox.pack(anchor="w", padx=10, pady=2)
        
        # Buttons frame
        button_frame = tk.Frame(day_window)
        button_frame.pack(fill="x", pady=10)
        
        def on_day_submit():
            print("Collecting day selections...")
            for date, selected in day_selections.items():
                if selected:
                    print(f"  {date}: Selected")
            
            day_window.quit()
            day_window.destroy()
            print("Day selection complete.")
        
        tk.Button(button_frame, text="Continue Analysis", command=on_day_submit,
                  font=("Arial", 10, "bold")).pack(pady=10)
        
        # Pack scrollable components
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0))
        scrollbar.pack(side="right", fill="y")
        
        day_window.mainloop()
        
        # Use the captured results
        selected_date_list = [date for date, selected in day_selections.items() if selected]
        print(f"Selected {len(selected_date_list)} days for analysis: {selected_date_list}")
        
        if not selected_date_list:
            print("❌ No days selected. Exiting.")
            return
    
    except Exception as e:
        print(f"❌ Error with day selection: {e}")
        return

    # ===== STEP 5: DATA LOADING AND PROCESSING =====
    print("\nStep 5: Data Loading and Processing")
    print("=" * 30)
    
    try:
        # Load data from database for selected tags and days
        print("Loading data from database...")
        conn = sqlite3.connect(file_path)
        
        # Create date filter for SQL query
        date_strings = [date.strftime('%Y-%m-%d') for date in selected_date_list]
        date_filter = ' OR '.join([f"DATE(datetime(timestamp/1000, 'unixepoch')) = '{date}'" for date in date_strings])
        
        query = f"""
            SELECT * FROM data 
            WHERE shortid IN ({','.join('?' * len(selected_tags))}) 
            AND ({date_filter})
            ORDER BY shortid, timestamp
        """
        
        print(f"Query: {query[:100]}...")  # Print first part of query for debugging
        data = pd.read_sql_query(query, conn, params=selected_tags)
        conn.close()
        
        if data.empty:
            print("❌ No data found for selected tags and dates. Exiting.")
            return
            
        print(f"✅ Loaded {len(data)} records for {len(selected_tags)} tags over {len(selected_date_list)} days")

        # Process timestamps and coordinates
        print("Processing timestamps and coordinates...")
        data['Timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', origin='unix', utc=True)
        
        # Convert timezone if not UTC
        if selected_timezone != "UTC":
            print(f"Converting timestamps to {selected_timezone}...")
            data['Timestamp'] = data['Timestamp'].dt.tz_convert(selected_timezone)
        
        # Convert coordinates to meters
        data['location_x'] *= 0.0254
        data['location_y'] *= 0.0254
        data = data.sort_values(by=['shortid', 'Timestamp'])
        data['Date'] = data['Timestamp'].dt.date

        print(f"✅ Data processed. Time range: {data['Timestamp'].min()} to {data['Timestamp'].max()}")

        # Apply downsampling if selected
        if downsample:
            print("Applying 1Hz downsampling...")
            original_count = len(data)
            data['time_sec'] = (data['Timestamp'].astype(np.int64) // 1_000_000_000).astype(int)
            data = data.groupby(['shortid', 'time_sec']).first().reset_index()
            print(f"✅ Downsampled from {original_count} to {len(data)} records")

        # Apply smoothing if selected
        if smoothing == 'savitzky-golay':
            print("Applying Savitzky-Golay smoothing...")
            def apply_savgol_filter(group):
                window_length = min(31, len(group))
                if window_length % 2 == 0:
                    window_length -= 1
                polyorder = min(2, window_length - 1)
                if window_length >= polyorder + 1:
                    return savgol_filter(group, window_length=window_length, polyorder=polyorder)
                else:
                    return group  # Return original if not enough points

            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(apply_savgol_filter)
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(apply_savgol_filter)
            print("✅ Savitzky-Golay smoothing applied")

        elif smoothing == 'rolling-average':
            print("Applying rolling-average smoothing...")
            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(lambda x: x.rolling(30, min_periods=1).mean())
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(lambda x: x.rolling(30, min_periods=1).mean())
            print("✅ Rolling-average smoothing applied")

        # Set up color map and labels based on tag metadata
        print("Setting up tag colors and labels...")
        tag_color_map, tag_label_map = setup_tag_display(data['shortid'].unique(), tag_results)

    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return

    # ===== STEP 6: BEHAVIORAL ANALYSIS =====
    print("\nStep 6: Behavioral Analysis")
    print("=" * 25)
    
    try:
        # Create classifier with configured parameters
        classifier = BehavioralClassifier(behavior_config)
        
        # Run behavioral analysis
        print("🧠 Running behavioral analysis...")
        behaviors = classifier.analyze_session(data, method='rule_based')
        
        if behaviors.empty:
            print("⚠️  No behaviors were classified. Check your data and parameters.")
            return
            
        print(f"✅ Analysis complete! Classified {len(behaviors)} time windows")
        
        # Show behavior summary
        print("\n📊 Behavior Summary:")
        behavior_summary = behaviors.groupby(['animal_id', 'behavior']).size().unstack(fill_value=0)
        print(behavior_summary)
        
        # Calculate behavior statistics
        print("\n📈 Behavior Statistics:")
        total_windows = len(behaviors)
        for behavior in behaviors['behavior'].unique():
            count = len(behaviors[behaviors['behavior'] == behavior])
            percentage = 100 * count / total_windows
            print(f"  {behavior_config.behavior_labels.get(behavior, behavior)}: {count} windows ({percentage:.1f}%)")

    except Exception as e:
        print(f"❌ Behavioral analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ===== STEP 7: RESULTS VISUALIZATION AND EXPORT =====
    print("\nStep 7: Results and Export")
    print("=" * 22)
    
    # Offer visualization and export options
    show_results = messagebox.askyesno(
        "Show Results", 
        "Would you like to view behavioral analysis results?\n\n" +
        "This will create plots showing:\n" +
        "• Behavior timelines for each animal\n" +
        "• Behavior proportion charts\n" +
        "• Movement trajectories with behavior overlays"
    )
    
    if show_results:
        print("Creating visualization plots...")
        create_behavior_visualizations(data, behaviors, classifier.config, tag_color_map, tag_label_map)
    
    # Offer to export results
    export_results = messagebox.askyesno(
        "Export Results",
        "Would you like to export the behavioral analysis results?\n\n" +
        "This will save:\n" +
        "• Behavior timeline data (CSV)\n" +
        "• Analysis parameters (JSON)\n" +
        "• Summary statistics (TXT)"
    )
    
    if export_results:
        print("Exporting results...")
        export_behavioral_results(behaviors, behavior_config, file_path, tag_results)

    print("\n" + "=" * 60)
    print("🎉 UWB Behavioral Analysis Complete!")
    print(f"✅ Analyzed {len(selected_tags)} animals over {len(behaviors)} time windows")
    print(f"📊 Identified {len(behaviors['behavior'].unique())} different behaviors")
    
    # Fix timedelta handling
    duration_seconds = pd.Timedelta(behaviors['timestamp'].max() - behaviors['timestamp'].min()).total_seconds()
    print(f"⏱️  Analysis covered {duration_seconds/3600:.2f} hours of data")
    print("=" * 60)

def configure_behavioral_parameters():
    """Configure behavioral analysis parameters through GUI"""
    print("Opening behavioral parameter configuration...")
    
    param_window = tk.Tk()
    param_window.title("Behavioral Analysis Parameters")
    param_window.geometry("600x800")
    
    # Create scrollable frame
    canvas = tk.Canvas(param_window)
    scrollbar = tk.Scrollbar(param_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Parameter variables
    window_size = tk.DoubleVar(value=10.0)
    overlap = tk.DoubleVar(value=5.0)
    min_duration = tk.DoubleVar(value=2.0)
    speed_rest = tk.DoubleVar(value=0.05)
    speed_active = tk.DoubleVar(value=0.3)
    dist_huddle = tk.DoubleVar(value=0.15)
    dist_social = tk.DoubleVar(value=0.5)
    
    # Title
    tk.Label(scrollable_frame, text="Behavioral Analysis Parameters", 
             font=("Arial", 16, "bold")).pack(pady=10)
    
    # Time window parameters
    time_frame = tk.LabelFrame(scrollable_frame, text="Time Window Settings", font=("Arial", 12, "bold"))
    time_frame.pack(fill="x", padx=20, pady=10)
    
    tk.Label(time_frame, text="Analysis Window Size (seconds):").pack(anchor="w", pady=2)
    tk.Label(time_frame, text="Duration of each analysis window (default: 10.0)", font=("Arial", 9, "italic")).pack(anchor="w")
    tk.Entry(time_frame, textvariable=window_size, width=10).pack(anchor="w", pady=(2,10))
    
    tk.Label(time_frame, text="Window Overlap (seconds):").pack(anchor="w", pady=2)
    tk.Label(time_frame, text="Overlap between consecutive windows (default: 5.0)", font=("Arial", 9, "italic")).pack(anchor="w")
    tk.Entry(time_frame, textvariable=overlap, width=10).pack(anchor="w", pady=(2,10))
    
    tk.Label(time_frame, text="Minimum Behavior Duration (seconds):").pack(anchor="w", pady=2)
    tk.Label(time_frame, text="Shortest duration to consider a behavior valid (default: 2.0)", font=("Arial", 9, "italic")).pack(anchor="w")
    tk.Entry(time_frame, textvariable=min_duration, width=10).pack(anchor="w", pady=(2,10))
    
    # Speed thresholds
    speed_frame = tk.LabelFrame(scrollable_frame, text="Speed Thresholds (m/s)", font=("Arial", 12, "bold"))
    speed_frame.pack(fill="x", padx=20, pady=10)
    
    tk.Label(speed_frame, text="Resting Speed Threshold:").pack(anchor="w", pady=2)
    tk.Label(speed_frame, text="Speed below which animal is considered resting (default: 0.05)", font=("Arial", 9, "italic")).pack(anchor="w")
    tk.Entry(speed_frame, textvariable=speed_rest, width=10).pack(anchor="w", pady=(2,10))
    
    tk.Label(speed_frame, text="Active Movement Threshold:").pack(anchor="w", pady=2)
    tk.Label(speed_frame, text="Speed above which animal is considered highly active (default: 0.3)", font=("Arial", 9, "italic")).pack(anchor="w")
    tk.Entry(speed_frame, textvariable=speed_active, width=10).pack(anchor="w", pady=(2,10))
    
    # Distance thresholds
    dist_frame = tk.LabelFrame(scrollable_frame, text="Social Distance Thresholds (meters)", font=("Arial", 12, "bold"))
    dist_frame.pack(fill="x", padx=20, pady=10)
    
    tk.Label(dist_frame, text="Huddling Distance:").pack(anchor="w", pady=2)
    tk.Label(dist_frame, text="Maximum distance for animals to be considered huddling (default: 0.15)", font=("Arial", 9, "italic")).pack(anchor="w")
    tk.Entry(dist_frame, textvariable=dist_huddle, width=10).pack(anchor="w", pady=(2,10))
    
    tk.Label(dist_frame, text="Social Interaction Distance:").pack(anchor="w", pady=2)
    tk.Label(dist_frame, text="Maximum distance for social interactions (default: 0.5)", font=("Arial", 9, "italic")).pack(anchor="w")
    tk.Entry(dist_frame, textvariable=dist_social, width=10).pack(anchor="w", pady=(2,10))
    
    # Buttons
    button_frame = tk.Frame(scrollable_frame)
    button_frame.pack(fill="x", padx=20, pady=20)
    
    config_result = {}
    
    def save_defaults():
        """Save current parameters as defaults"""
        try:
            config = BehaviorConfig(
                window_size_seconds=window_size.get(),
                overlap_seconds=overlap.get(),
                min_behavior_duration=min_duration.get(),
                speed_threshold_rest=speed_rest.get(),
                speed_threshold_active=speed_active.get(),
                distance_threshold_huddle=dist_huddle.get(),
                distance_threshold_social=dist_social.get()
            )
            
            # Save to file
            config_path = os.path.join(os.path.dirname(__file__), 'user_behavior_config.json')
            classifier = BehavioralClassifier(config)
            classifier.save_config(config_path)
            
            messagebox.showinfo("Saved", f"Parameters saved as defaults to:\n{config_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save parameters: {e}")
    
    def load_defaults():
        """Load default parameters"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'user_behavior_config.json')
            if os.path.exists(config_path):
                classifier = BehavioralClassifier.load_config(config_path)
                config = classifier.config
                
                window_size.set(config.window_size_seconds)
                overlap.set(config.overlap_seconds)
                min_duration.set(config.min_behavior_duration)
                speed_rest.set(config.speed_threshold_rest)
                speed_active.set(config.speed_threshold_active)
                dist_huddle.set(config.distance_threshold_huddle)
                dist_social.set(config.distance_threshold_social)
                
                messagebox.showinfo("Loaded", "Default parameters loaded successfully")
            else:
                messagebox.showinfo("No Defaults", "No saved defaults found. Using built-in defaults.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load parameters: {e}")
    
    def on_param_submit():
        try:
            # Validate parameters
            if window_size.get() <= 0:
                raise ValueError("Window size must be positive")
            if overlap.get() < 0 or overlap.get() >= window_size.get():
                raise ValueError("Overlap must be non-negative and less than window size")
            if speed_rest.get() >= speed_active.get():
                raise ValueError("Resting speed threshold must be less than active speed threshold")
                
            config_result['config'] = BehaviorConfig(
                window_size_seconds=window_size.get(),
                overlap_seconds=overlap.get(),
                min_behavior_duration=min_duration.get(),
                speed_threshold_rest=speed_rest.get(),
                speed_threshold_active=speed_active.get(),
                distance_threshold_huddle=dist_huddle.get(),
                distance_threshold_social=dist_social.get()
            )
            
            param_window.quit()
            param_window.destroy()
            
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", str(e))
    
    # Button layout
    tk.Button(button_frame, text="Load Defaults", command=load_defaults).pack(side="left", padx=5)
    tk.Button(button_frame, text="Save as Defaults", command=save_defaults).pack(side="left", padx=5)
    tk.Button(button_frame, text="Continue Analysis", command=on_param_submit, 
              font=("Arial", 10, "bold")).pack(side="right", padx=5)
    
    # Pack scrollable components
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    param_window.mainloop()
    
    return config_result.get('config', BehaviorConfig())

def configure_tag_selection(available_tags):
    """Configure tag selection and metadata - reused from uwb_plots.py"""
    print("Opening tag selection window...")
    
    tag_window = tk.Tk()
    tag_window.title("Tag Selection and Metadata")
    tag_window.geometry("500x400")
    
    # Create scrollable frame
    canvas = tk.Canvas(tag_window)
    scrollbar = tk.Scrollbar(tag_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Header
    tk.Label(scrollable_frame, text="Tag Selection and Metadata", 
             font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=4, pady=10)
    tk.Label(scrollable_frame, text="(Optional: Enter sex M/F and display name for custom colors/labels)", 
             font=("Arial", 9, "italic")).grid(row=1, column=0, columnspan=4, pady=(0,5))
    
    # Column headers
    headers = ["Include", "Tag ID", "Sex (M/F)", "Display ID"]
    for i, header in enumerate(headers):
        tk.Label(scrollable_frame, text=header, font=("Arial", 10, "bold")).grid(
            row=2, column=i, padx=5, pady=5)
    
    # Store tag metadata and widgets
    tag_metadata = {}
    tag_widgets = {}
    
    for i, tag_id in enumerate(available_tags):
        row = i + 3
        
        # Include checkbox
        include_var = tk.BooleanVar()
        include_var.set(True)  # Default to include all tags
        include_check = tk.Checkbutton(scrollable_frame, variable=include_var)
        include_check.grid(row=row, column=0, padx=5, pady=2)
        
        # Tag ID label
        tk.Label(scrollable_frame, text=str(tag_id)).grid(row=row, column=1, padx=5, pady=2)
        
        # Sex entry
        sex_var = tk.StringVar()
        sex_entry = tk.Entry(scrollable_frame, textvariable=sex_var, width=8)
        sex_entry.grid(row=row, column=2, padx=5, pady=2)
        
        # Display ID entry
        display_var = tk.StringVar()
        display_entry = tk.Entry(scrollable_frame, textvariable=display_var, width=15)
        display_entry.grid(row=row, column=3, padx=5, pady=2)
        
        tag_metadata[tag_id] = {
            'include': include_var,
            'sex': sex_var,
            'display_id': display_var
        }
        
        tag_widgets[tag_id] = {
            'include': include_check,
            'sex_entry': sex_entry,
            'display_entry': display_entry
        }
    
    tag_results = {}
    
    def on_tag_submit():
        for tag_id, widgets in tag_widgets.items():
            include = tag_metadata[tag_id]['include'].get()
            sex_raw = widgets['sex_entry'].get()
            display_raw = widgets['display_entry'].get()
            
            sex = sex_raw.strip().upper() if sex_raw else ""
            display_id = display_raw.strip() if display_raw else ""
            
            # Validate sex input
            if sex and sex not in ['M', 'F']:
                sex = ""
            
            tag_results[tag_id] = {
                'include': include,
                'sex': sex,
                'display_id': display_id
            }
        
        tag_window.quit()
        tag_window.destroy()
    
    tk.Button(scrollable_frame, text="Continue", command=on_tag_submit).grid(
        row=len(available_tags)+4, column=0, columnspan=4, pady=10)
    
    # Pack scrollable components
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    tag_window.mainloop()
    
    return tag_results

def setup_tag_display(unique_tags, tag_results):
    """Set up color and label mapping for tags"""
    tag_color_map = {}
    tag_label_map = {}
    
    default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan', 'magenta']
    
    for i, tag in enumerate(unique_tags):
        tag_key = int(tag) if isinstance(tag, (str, float)) else tag
        
        if tag_key in tag_results:
            metadata = tag_results[tag_key]
            sex = metadata['sex']
            display_id = metadata['display_id']
            
            # Set color based on sex if provided
            if sex == 'M':
                color = 'blue'
            elif sex == 'F':
                color = 'red'
            else:
                color = default_colors[i % len(default_colors)]
            
            tag_color_map[tag] = color
            
            # Create label
            if sex and display_id:
                label = f"{sex}-{display_id}"
            elif sex:
                label = f"{sex}-{tag}"
            elif display_id:
                label = display_id
            else:
                label = f"Tag {tag}"
            
            tag_label_map[tag] = label
        else:
            # Fallback
            color = default_colors[i % len(default_colors)]
            tag_color_map[tag] = color
            tag_label_map[tag] = f"Tag {tag}"
    
    return tag_color_map, tag_label_map

def create_behavior_visualizations(data, behaviors, config, tag_color_map, tag_label_map):
    """Create visualizations of behavioral analysis results"""
    print("Creating behavioral analysis visualizations...")
    
    # Set matplotlib to non-blocking mode
    plt.ion()
    
    try:
        # 1. Behavior timeline plots for each animal
        print("  Creating behavior timeline plots...")
        for animal_id in behaviors['animal_id'].unique():
            animal_behaviors = behaviors[behaviors['animal_id'] == animal_id].sort_values('timestamp')
            
            if animal_behaviors.empty:
                continue
                
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
            
            # Top plot: Behavior timeline
            behavior_colors = config.behavior_colors
            y_pos = 0
            
            for _, row in animal_behaviors.iterrows():
                behavior = row['behavior']
                timestamp = row['timestamp']
                color = behavior_colors.get(behavior, '#808080')
                
                # Create bar for behavior duration (approximate)
                duration = config.window_size_seconds / 3600  # Convert to hours for display
                ax1.barh(y_pos, duration, left=timestamp.hour + timestamp.minute/60, 
                        color=color, alpha=0.8, height=0.8)
            
            ax1.set_xlabel('Time of Day (hours)')
            ax1.set_ylabel('Behavior Timeline')
            ax1.set_title(f'Behavior Timeline - {tag_label_map.get(animal_id, f"Animal {animal_id}")}')
            ax1.grid(True, alpha=0.3)
            
            # Create custom legend
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=config.behavior_labels.get(behavior, behavior)) 
                             for behavior, color in behavior_colors.items() 
                             if behavior in animal_behaviors['behavior'].values]
            ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            
            # Bottom plot: Speed over time (if available)
            if 'speed_mean' in animal_behaviors.columns:
                ax2.plot(animal_behaviors['timestamp'], animal_behaviors['speed_mean'], 
                        color=tag_color_map.get(animal_id, 'black'), linewidth=2)
                ax2.axhline(y=config.speed_threshold_rest, color='red', linestyle='--', 
                           label=f'Rest threshold ({config.speed_threshold_rest} m/s)')
                ax2.axhline(y=config.speed_threshold_active, color='orange', linestyle='--', 
                           label=f'Active threshold ({config.speed_threshold_active} m/s)')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Speed (m/s)')
                ax2.set_title('Movement Speed Over Time')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show(block=False)
        
        # 2. Behavior proportion pie charts
        print("  Creating behavior proportion charts...")
        unique_behaviors = behaviors['behavior'].unique()
        
        for animal_id in behaviors['animal_id'].unique():
            animal_behaviors = behaviors[behaviors['animal_id'] == animal_id]
            behavior_counts = animal_behaviors['behavior'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            colors = [config.behavior_colors.get(behavior, '#808080') for behavior in behavior_counts.index]
            labels = [config.behavior_labels.get(behavior, behavior) for behavior in behavior_counts.index]
            
            wedges, texts, autotexts = ax.pie(behavior_counts.values, labels=labels, colors=colors, 
                                            autopct='%1.1f%%', startangle=90)
            
            ax.set_title(f'Behavior Distribution - {tag_label_map.get(animal_id, f"Animal {animal_id}")}', 
                        fontsize=14, fontweight='bold')
            
            plt.show(block=False)
        
        # 3. Overall behavior summary
        print("  Creating overall behavior summary...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        behavior_summary = behaviors.groupby(['animal_id', 'behavior']).size().unstack(fill_value=0)
        
        behavior_summary.plot(kind='bar', stacked=True, ax=ax, 
                            color=[config.behavior_colors.get(b, '#808080') for b in behavior_summary.columns])
        
        ax.set_xlabel('Animal ID')
        ax.set_ylabel('Number of Time Windows')
        ax.set_title('Behavior Summary - All Animals')
        ax.legend(title='Behaviors', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels to use tag labels
        x_labels = [tag_label_map.get(animal_id, f"Animal {animal_id}") 
                   for animal_id in behavior_summary.index]
        ax.set_xticklabels(x_labels, rotation=45)
        
        plt.tight_layout()
        plt.show(block=False)
        
        print("✅ Visualization plots created successfully")
        
    except Exception as e:
        print(f"⚠️  Some visualizations failed: {e}")

def export_behavioral_results(behaviors, config, database_path, tag_results):
    """Export behavioral analysis results to files"""
    try:
        # Set up export directory
        db_dir = os.path.dirname(database_path)
        db_name = os.path.splitext(os.path.basename(database_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        export_dir = os.path.join(db_dir, f"behavioral_analysis_{db_name}_{timestamp}")
        os.makedirs(export_dir, exist_ok=True)
        
        # 1. Export behavior timeline CSV
        behavior_file = os.path.join(export_dir, "behavior_timeline.csv")
        behaviors.to_csv(behavior_file, index=False)
        print(f"✅ Behavior timeline saved: {behavior_file}")
        
        # 2. Export configuration JSON
        config_file = os.path.join(export_dir, "analysis_parameters.json")
        classifier = BehavioralClassifier(config)
        classifier.save_config(config_file)
        print(f"✅ Analysis parameters saved: {config_file}")
        
        # 3. Export summary statistics
        stats_file = os.path.join(export_dir, "behavior_summary.txt")
        with open(stats_file, 'w') as f:
            f.write(f"UWB Behavioral Analysis Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Database: {os.path.basename(database_path)}\n\n")
            
            f.write(f"Analysis Parameters:\n")
            f.write(f"  Window Size: {config.window_size_seconds} seconds\n")
            f.write(f"  Overlap: {config.overlap_seconds} seconds\n")
            f.write(f"  Min Duration: {config.min_behavior_duration} seconds\n")
            f.write(f"  Speed Thresholds: {config.speed_threshold_rest} - {config.speed_threshold_active} m/s\n")
            f.write(f"  Distance Thresholds: huddle={config.distance_threshold_huddle}m, social={config.distance_threshold_social}m\n\n")
            
            f.write(f"Results Summary:\n")
            f.write(f"  Total Time Windows: {len(behaviors)}\n")
            f.write(f"  Animals Analyzed: {len(behaviors['animal_id'].unique())}\n")
            f.write(f"  Time Range: {behaviors['timestamp'].min()} to {behaviors['timestamp'].max()}\n")
            # Fix timedelta handling
            duration_seconds = pd.Timedelta(behaviors['timestamp'].max() - behaviors['timestamp'].min()).total_seconds()
            f.write(f"  Duration: {duration_seconds/3600:.2f} hours\n\n")
            
            f.write("Behavior Distribution:\n")
            behavior_counts = behaviors['behavior'].value_counts()
            total_windows = len(behaviors)
            for behavior, count in behavior_counts.items():
                percentage = 100 * count / total_windows
                f.write(f"  {config.behavior_labels.get(behavior, behavior)}: {count} windows ({percentage:.1f}%)\n")
            
            f.write(f"\nPer-Animal Summary:\n")
            for animal_id in sorted(behaviors['animal_id'].unique()):
                animal_behaviors = behaviors[behaviors['animal_id'] == animal_id]
                f.write(f"  Animal {animal_id}: {len(animal_behaviors)} windows\n")
                animal_counts = animal_behaviors['behavior'].value_counts()
                for behavior, count in animal_counts.items():
                    percentage = 100 * count / len(animal_behaviors)
                    f.write(f"    {config.behavior_labels.get(behavior, behavior)}: {count} ({percentage:.1f}%)\n")
        
        print(f"✅ Summary statistics saved: {stats_file}")
        
        # 4. Export tag metadata
        if tag_results:
            tag_file = os.path.join(export_dir, "tag_metadata.json")
            with open(tag_file, 'w') as f:
                json.dump(tag_results, f, indent=2)
            print(f"✅ Tag metadata saved: {tag_file}")
        
        print(f"\n📁 All results exported to: {export_dir}")
        
        # Show completion message
        messagebox.showinfo(
            "Export Complete", 
            f"Behavioral analysis results exported to:\n\n{export_dir}\n\n" +
            "Files created:\n" +
            "• behavior_timeline.csv - Raw behavior classifications\n" +
            "• analysis_parameters.json - Analysis settings\n" +
            "• behavior_summary.txt - Summary statistics\n" +
            "• tag_metadata.json - Tag information"
        )
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        messagebox.showerror("Export Error", f"Failed to export results:\n{e}")

if __name__ == "__main__":
    uwb_behavioral_analysis()
