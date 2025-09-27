import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import sqlite3
import os
import subprocess
import shutil
from tqdm import tqdm
import gc
import xml.etree.ElementTree as ET

def parse_wiser_xml_zones(xml_file_path):
    """
    Parse Wiser XML configuration file to extract zone coordinates.
    
    Parameters:
    xml_file_path (str): Path to the Wiser XML configuration file
    
    Returns:
    pd.DataFrame: DataFrame with columns ['zone', 'x', 'y'] containing zone coordinates in meters
    """
    try:
        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Find the Zones section
        zones_element = root.find('Zones')
        if zones_element is None:
            print("No Zones section found in XML file")
            return None
        
        zone_data = []
        
        # Iterate through each zone
        for zone in zones_element.findall('Zone'):
            zone_name = zone.get('name')
            if zone_name is None:
                continue
                
            # Find the Shape element
            shape = zone.find('Shape')
            if shape is None:
                continue
                
            # Extract all points for this zone
            for point in shape.findall('Point'):
                x_str = point.get('x')
                y_str = point.get('y')
                
                if x_str is not None and y_str is not None:
                    try:
                        # Convert coordinates to meters (same as location data: inches to meters)
                        x_meters = float(x_str) * 0.0254  # Convert inches to meters
                        y_meters = float(y_str) * 0.0254  # Convert inches to meters
                        
                        zone_data.append({
                            'zone': zone_name,
                            'x': x_meters,
                            'y': y_meters
                        })
                    except ValueError:
                        print(f"Warning: Could not convert coordinates for zone {zone_name}: x={x_str}, y={y_str}")
                        continue
        
        if not zone_data:
            print("No valid zone coordinates found in XML file")
            return None
        
        # Create DataFrame
        df_zones = pd.DataFrame(zone_data)
        print(f"Parsed {len(df_zones['zone'].unique())} zones with {len(df_zones)} total coordinate points")
        return df_zones
        
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"Error reading XML file: {e}")
        return None

def uwb_animate_paths():
    """
    Standalone function to create animated videos of UWB tag paths with optional downsampling and smoothing.
    """
    print("Starting UWB Tag Path Animation...")

    # ===== COLLECT ALL USER CHOICES UPFRONT =====
    print("Collecting user preferences...")
    
    # 1. Prompt user to select SQLite file
    root = tk.Tk()
    root.withdraw()
    print("Please select your SQLite database file...")
    file_path = filedialog.askopenfilename(title="Select SQLite File", filetypes=[("SQLite Files", "*.sqlite")])

    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"Selected file: {os.path.basename(file_path)}")

    # 2. Ask user if they want to downsample to 1Hz
    print("Asking user about downsampling...")
    downsample = messagebox.askyesno("Downsample Data", "Do you want to downsample the data to 1Hz?")
    print(f"Downsampling: {'Enabled' if downsample else 'Disabled'}")

    # 3. Smoothing method selection
    print("Asking user about smoothing method...")
    smoothing_window = tk.Tk()
    smoothing_window.title("Select Smoothing Method")

    smoothing_choice = tk.StringVar(value="none")

    def set_smoothing(choice):
        smoothing_choice.set(choice)
        smoothing_window.quit()
        smoothing_window.destroy()

    tk.Label(smoothing_window, text="Choose a smoothing method:").pack(pady=10)
    tk.Button(smoothing_window, text="Savitzky-Golay", command=lambda: set_smoothing("savitzky-golay")).pack(pady=5)
    tk.Button(smoothing_window, text="Rolling-Average", command=lambda: set_smoothing("rolling-average")).pack(pady=5)
    tk.Button(smoothing_window, text="None", command=lambda: set_smoothing("none")).pack(pady=5)

    smoothing_window.mainloop()

    smoothing = smoothing_choice.get()
    print(f"Smoothing method: {smoothing}")

    # 4. Timezone selection
    print("Asking user about timezone...")
    timezone_window = tk.Tk()
    timezone_window.title("Select US Timezone")
    
    timezone_choice = tk.StringVar(value="UTC")
    
    def set_timezone(tz):
        timezone_choice.set(tz)
        timezone_window.quit()
        timezone_window.destroy()
    
    tk.Label(timezone_window, text="Select your timezone for timestamp conversion:").pack(pady=10)
    tk.Button(timezone_window, text="Eastern (ET)", command=lambda: set_timezone("US/Eastern")).pack(pady=3)
    tk.Button(timezone_window, text="Central (CT)", command=lambda: set_timezone("US/Central")).pack(pady=3)
    tk.Button(timezone_window, text="Mountain (MT)", command=lambda: set_timezone("US/Mountain")).pack(pady=3)
    tk.Button(timezone_window, text="Pacific (PT)", command=lambda: set_timezone("US/Pacific")).pack(pady=3)
    tk.Button(timezone_window, text="Alaska (AKT)", command=lambda: set_timezone("US/Alaska")).pack(pady=3)
    tk.Button(timezone_window, text="Hawaii (HST)", command=lambda: set_timezone("US/Hawaii")).pack(pady=3)
    tk.Button(timezone_window, text="Keep UTC", command=lambda: set_timezone("UTC")).pack(pady=3)
    
    timezone_window.mainloop()
    
    selected_timezone = timezone_choice.get()
    print(f"Selected timezone: {selected_timezone}")

    # 5. Playback speed selection
    print("Asking user about playback speed...")
    speed_window = tk.Tk()
    speed_window.title("Select Playback Speed")
    speed_window.geometry("300x500")  # Make window taller to accommodate more buttons
    
    speed_choice = tk.StringVar(value="1")
    
    def set_speed(speed):
        speed_choice.set(speed)
        speed_window.quit()
        speed_window.destroy()
    
    tk.Label(speed_window, text="Choose playback speed:", font=("Arial", 12, "bold")).pack(pady=10)
    
    # Standard speeds
    tk.Label(speed_window, text="Standard Speeds:", font=("Arial", 10, "bold")).pack(pady=(10,5))
    tk.Button(speed_window, text="1x (Real-time)", command=lambda: set_speed("1"), width=20).pack(pady=2)
    tk.Button(speed_window, text="2x", command=lambda: set_speed("2"), width=20).pack(pady=2)
    tk.Button(speed_window, text="5x", command=lambda: set_speed("5"), width=20).pack(pady=2)
    tk.Button(speed_window, text="10x", command=lambda: set_speed("10"), width=20).pack(pady=2)
    
    # High speeds
    tk.Label(speed_window, text="High Speeds:", font=("Arial", 10, "bold")).pack(pady=(15,5))
    tk.Button(speed_window, text="20x", command=lambda: set_speed("20"), width=20).pack(pady=2)
    tk.Button(speed_window, text="40x", command=lambda: set_speed("40"), width=20).pack(pady=2)
    tk.Button(speed_window, text="60x", command=lambda: set_speed("60"), width=20).pack(pady=2)
    tk.Button(speed_window, text="80x", command=lambda: set_speed("80"), width=20).pack(pady=2)
    tk.Button(speed_window, text="100x", command=lambda: set_speed("100"), width=20).pack(pady=2)
    
    speed_window.mainloop()
    
    playback_speed = int(speed_choice.get())
    print(f"Playback speed: {playback_speed}x")

    # 6. Arena/Zone coordinates from XML configuration file
    print("Asking user about arena zone coordinates...")
    arena_coords_window = tk.Tk()
    arena_coords_window.title("Arena Zone Coordinates")
    arena_coords_window.geometry("400x200")
    
    arena_coords_choice = tk.StringVar(value="no")
    
    def set_arena_coords(choice):
        arena_coords_choice.set(choice)
        arena_coords_window.quit()
        arena_coords_window.destroy()
    
    tk.Label(arena_coords_window, text="Do you want to provide Wiser zone coordinates", font=("Arial", 12, "bold")).pack(pady=10)
    tk.Label(arena_coords_window, text="via an XML configuration file?", font=("Arial", 12, "bold")).pack(pady=(0,10))
    tk.Button(arena_coords_window, text="Yes - Select XML File", command=lambda: set_arena_coords("yes"), width=20).pack(pady=5)
    tk.Button(arena_coords_window, text="No - Continue without zones", command=lambda: set_arena_coords("no"), width=20).pack(pady=5)
    
    arena_coords_window.mainloop()
    
    arena_coordinates = None
    if arena_coords_choice.get() == "yes":
        print("User selected to provide XML configuration file...")
        xml_file_path = filedialog.askopenfilename(title="Select Wiser XML Configuration File", filetypes=[("XML Files", "*.xml")])
        
        if xml_file_path:
            print(f"Selected XML file: {os.path.basename(xml_file_path)}")
            arena_coordinates = parse_wiser_xml_zones(xml_file_path)
            if arena_coordinates is not None:
                print(f"Successfully parsed {len(arena_coordinates)} zone coordinate points from XML")
            else:
                print("Failed to parse XML file - continuing without zone coordinates")
        else:
            print("No XML file selected - continuing without zone coordinates")
    else:
        print("User chose to continue without zone coordinates")

    # Quick database query to get available dates and tags without loading all data
    print("Checking database for available tags and dates...")
    conn = sqlite3.connect(file_path)
    
    # Query just timestamps to determine available dates
    date_query = "SELECT DISTINCT date(datetime(timestamp/1000, 'unixepoch')) as date FROM data ORDER BY date"
    available_dates = pd.read_sql_query(date_query, conn)
    
    # Query unique tags
    tag_query = "SELECT DISTINCT shortid FROM data ORDER BY shortid"
    available_tags = pd.read_sql_query(tag_query, conn)
    conn.close()
    
    if available_dates.empty:
        print("No data found in database. Exiting.")
        return
        
    print(f"Found {len(available_tags)} unique tags in database")

    # 6. Tag selection and metadata window
    print("Opening tag selection and metadata window...")
    unique_tag_ids = list(available_tags['shortid'])
    
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
    tk.Label(scrollable_frame, text="Tag Selection and Metadata", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=4, pady=10)
    tk.Label(scrollable_frame, text="(Optional: Enter sex M/F and display name for custom colors/labels)", font=("Arial", 9, "italic")).grid(row=1, column=0, columnspan=4, pady=(0,5))
    tk.Label(scrollable_frame, text="Include", font=("Arial", 10, "bold")).grid(row=2, column=0, padx=5, pady=5)
    tk.Label(scrollable_frame, text="Tag ID", font=("Arial", 10, "bold")).grid(row=2, column=1, padx=5, pady=5)
    tk.Label(scrollable_frame, text="Sex (M/F)", font=("Arial", 10, "bold")).grid(row=2, column=2, padx=5, pady=5)
    tk.Label(scrollable_frame, text="Display ID", font=("Arial", 10, "bold")).grid(row=2, column=3, padx=5, pady=5)
    
    # Store tag metadata
    tag_metadata = {}
    tag_widgets = {}  # Store direct references to Entry widgets
    
    print(f"Creating GUI elements for {len(unique_tag_ids)} tags...")
    for i, tag_id in enumerate(unique_tag_ids):
        row = i + 3  # Changed from i + 2 to i + 3 to account for the new instruction row
        
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
        
        # Store direct widget references as backup
        tag_widgets[tag_id] = {
            'include': include_check,
            'sex_entry': sex_entry,
            'display_entry': display_entry
        }
        
        print(f"  Created GUI for tag {tag_id} at row {row}")
    
    print("GUI creation complete. Waiting for user input...")
    
    # Store results
    tag_results = {}
    
    def on_tag_submit():
        print("Collecting tag metadata from user input...")
        print("Method 1 - Raw metadata check from StringVar:")
        
        # First, let's check what's actually in the StringVar objects
        for tag_id, vars in tag_metadata.items():
            sex_raw = vars['sex'].get()
            display_raw = vars['display_id'].get()
            print(f"  Tag {tag_id} RAW: sex='{sex_raw}' (len={len(sex_raw)}), display='{display_raw}' (len={len(display_raw)})")
        
        print("Method 2 - Direct from Entry widgets:")
        # Try getting values directly from Entry widgets
        for tag_id, widgets in tag_widgets.items():
            sex_direct = widgets['sex_entry'].get()
            display_direct = widgets['display_entry'].get()
            print(f"  Tag {tag_id} DIRECT: sex='{sex_direct}' (len={len(sex_direct)}), display='{display_direct}' (len={len(display_direct)})")
        
        print("Processing metadata using direct method:")
        for tag_id, widgets in tag_widgets.items():
            include = tag_metadata[tag_id]['include'].get()
            sex_raw = widgets['sex_entry'].get()
            display_raw = widgets['display_entry'].get()
            
            # More careful string processing
            sex = sex_raw.strip().upper() if sex_raw else ""
            display_id = display_raw.strip() if display_raw else ""
            
            print(f"  Tag {tag_id} PROCESSED: sex_raw='{sex_raw}' -> sex='{sex}', display_raw='{display_raw}' -> display_id='{display_id}'")
            
            # Validate sex input
            if sex and sex not in ['M', 'F']:
                print(f"    WARNING: Invalid sex '{sex}' for tag {tag_id}, clearing...")
                sex = ""
            
            tag_results[tag_id] = {
                'include': include,
                'sex': sex,
                'display_id': display_id
            }
            print(f"  Tag {tag_id} FINAL: include={include}, sex='{sex}', display_id='{display_id}'")
        
        tag_window.quit()
        tag_window.destroy()
        print("Tag metadata collection complete.")
    
    tk.Button(scrollable_frame, text="Continue", command=on_tag_submit).grid(row=len(unique_tag_ids)+4, column=0, columnspan=4, pady=10)
    
    # Pack the scrollable components
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    tag_window.mainloop()
    
    # Process tag results
    selected_tags = [tag_id for tag_id, data in tag_results.items() if data['include']]
    print(f"Selected {len(selected_tags)} tags for processing")
    
    if not selected_tags:
        print("No tags selected. Exiting.")
        return

    # 7. Day selection window (using quickly queried dates)
    print("Opening day selection window...")
    unique_dates = [pd.to_datetime(date).date() for date in available_dates['date']]
    
    day_window = tk.Tk()
    day_window.title("Select Days to Animate")
    
    # Use a simple dictionary to track selections
    day_selections = {date: False for date in unique_dates}
    
    tk.Label(day_window, text="Select days to create videos for:").pack(pady=10)
    
    # Create checkboxes with callback functions
    def make_toggle_callback(date):
        def toggle():
            day_selections[date] = not day_selections[date]
            print(f"DEBUG: {date} toggled to {'Selected' if day_selections[date] else 'Not selected'}")
        return toggle
    
    for date in unique_dates:
        checkbox = tk.Checkbutton(
            day_window, 
            text=str(date),
            command=make_toggle_callback(date)
        )
        checkbox.pack(anchor="w")
    
    def on_day_submit():
        print("Collecting day selections...")
        for date, selected in day_selections.items():
            print(f"  {date}: {'Selected' if selected else 'Not selected'}")
        
        day_window.quit()
        day_window.destroy()
        print("Day selection complete.")
    
    tk.Button(day_window, text="Continue", command=on_day_submit).pack(pady=10)
    day_window.mainloop()
    
    # Use the captured results
    selected_date_list = [date for date, selected in day_selections.items() if selected]
    print(f"Will create videos for {len(selected_date_list)} days")

    if not selected_date_list:
        print("No days selected. Exiting.")
        return

    print("Loading and processing full dataset...")

    # ===== FULL DATABASE OPERATIONS AND DATA PROCESSING =====
    
    # Connect to SQLite database and query data
    print("Querying all data from database...")
    conn = sqlite3.connect(file_path)
    query = "SELECT * FROM data"
    data = pd.read_sql_query(query, conn)
    conn.close()
    print("Data loaded successfully.")

    print("Processing timestamps and coordinates...")
    data['Timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', origin='unix', utc=True)
    
    # Convert timezone if not UTC
    if selected_timezone != "UTC":
        print(f"Converting timestamps to {selected_timezone}...")
        data['Timestamp'] = data['Timestamp'].dt.tz_convert(selected_timezone)
    
    data['location_x'] *= 0.0254
    data['location_y'] *= 0.0254
    data = data.sort_values(by=['shortid', 'Timestamp'])
    data['Date'] = data['Timestamp'].dt.date

    # Filter data to only selected dates and tags
    print(f"Filtering data to {len(selected_date_list)} days and {len(selected_tags)} tags...")
    data = data[data['Date'].isin(selected_date_list)]
    data = data[data['shortid'].isin(selected_tags)]
    print(f"Filtered dataset contains {len(data)} records")

    # Apply downsampling if selected
    if downsample:
        print("Applying 1Hz downsampling...")
        data['time_sec'] = (data['Timestamp'].astype(np.int64) // 1_000_000_000).astype(int)
        data = data.groupby(['shortid', 'time_sec']).first().reset_index()
        print(f"Downsampled to {len(data)} records")

    # Apply smoothing if selected
    if smoothing == 'savitzky-golay':
        print("Applying Savitzky-Golay smoothing...")
        def apply_savgol_filter(group):
            window_length = min(31, len(group))
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(2, window_length - 1)
            return savgol_filter(group, window_length=window_length, polyorder=polyorder)

        data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(apply_savgol_filter)
        data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(apply_savgol_filter)
        print("Smoothing applied.")

    elif smoothing == 'rolling-average':
        print("Applying rolling-average smoothing...")
        data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(lambda x: x.rolling(30, min_periods=1).mean())
        data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(lambda x: x.rolling(30, min_periods=1).mean())
        print("Smoothing applied.")

    # Set column names for plotting
    x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
    y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'

    # Set up output directory - with better desktop detection
    # Try multiple possible desktop locations
    possible_desktops = [
        os.path.join(os.path.expanduser("~"), "Desktop"),  # Standard user desktop
        os.path.join(os.environ.get('USERPROFILE', ''), "Desktop"),  # Windows user profile desktop
        os.path.join(os.environ.get('USERPROFILE', ''), "OneDrive", "Desktop"),  # OneDrive desktop
        "C:\\Users\\caleb\\Desktop",  # Direct path to your desktop
    ]
    
    # Set up output directory
    print("Setting up output directories...")
    # Use the first existing desktop, or fall back to the SQL file directory
    desktop_path = None
    for desktop in possible_desktops:
        if os.path.exists(desktop):
            desktop_path = desktop
            break
    
    if desktop_path is None:
        desktop_path = os.path.dirname(file_path)
    
    output_dir = os.path.join(desktop_path, "uwb_animation_frames")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set final video destination to same folder as SQLite file
    sql_file_dir = os.path.dirname(file_path)
    print(f"Videos will be saved to: {sql_file_dir}")

    # Calculate global axis limits
    x_min, x_max = data[x_col].min(), data[x_col].max()
    y_min, y_max = data[y_col].min(), data[y_col].max()

    # Create color map and labels based on tag metadata
    print("Setting up tag colors and labels...")
    unique_tags = data['shortid'].unique()
    tag_color_map = {}
    tag_label_map = {}
    
    # Default color palette for tags without metadata
    default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan', 'magenta']
    
    print("Applying tag metadata and colors...")
    for i, tag in enumerate(unique_tags):
        # Convert tag to int if it's not already, to match tag_results keys
        tag_key = int(tag) if isinstance(tag, (str, float)) else tag
        
        if tag_key in tag_results:
            metadata = tag_results[tag_key]
            sex = metadata['sex']
            display_id = metadata['display_id']
            
            # Set color based on sex if provided, otherwise use default color
            if sex == 'M':
                color = 'blue'
            elif sex == 'F':
                color = 'red'
            else:
                # Use default color scheme if no sex specified
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
            print(f"  Tag {tag}: color={color}, label='{label}'")
        else:
            # Fallback for tags not in results (shouldn't happen with current logic)
            color = default_colors[i % len(default_colors)]
            tag_color_map[tag] = color
            tag_label_map[tag] = f"Tag {tag}"
            print(f"  Tag {tag}: color={color}, label='Tag {tag}' (fallback)")

    def create_daily_video(day_data, date_str, arena_coords=None):
        """Creates a video for a single day."""
        print(f"Creating video for {date_str}...")
        
        # Clean the directory
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Calculate actual real-world time parameters
        start_time = day_data['Timestamp'].min()
        end_time = day_data['Timestamp'].max()
        total_real_duration = (end_time - start_time).total_seconds()
        
        print(f"  Real-world data duration: {total_real_duration:.1f} seconds ({total_real_duration/3600:.2f} hours)")
        
        # Set video parameters for accurate time representation
        # We want each video second to represent (playback_speed) real-world seconds
        video_fps = 30  # Standard video frame rate
        real_seconds_per_video_second = playback_speed  # 1x = 1 real second per video second, 2x = 2 real seconds per video second, etc.
        real_seconds_per_frame = real_seconds_per_video_second / video_fps  # How much real time each frame represents
        
        # Calculate trailing window (how much historical data to show in each frame)
        # Show the last 5 minutes of real time, but scale it appropriately
        trailing_window_real_seconds = min(300, total_real_duration * 0.1)  # 5 minutes or 10% of total duration, whichever is smaller
        
        print(f"  Playback speed: {playback_speed}x real-time")
        print(f"  Each video second = {real_seconds_per_video_second} real-world seconds")
        print(f"  Each frame = {real_seconds_per_frame:.2f} real-world seconds")
        print(f"  Trailing window: {trailing_window_real_seconds:.1f} real-world seconds")
        
        # Create time intervals based on real-world time progression
        time_starts = []
        current_time = start_time
        
        while current_time < end_time:
            time_starts.append(current_time)
            current_time += pd.Timedelta(seconds=real_seconds_per_frame)
        
        print(f"  Will create {len(time_starts)} frames for {total_real_duration/playback_speed:.1f} seconds of video")

        # Create frames
        with tqdm(total=len(time_starts), desc=f"Creating frames for {date_str}") as pbar:
            for i, frame_start in enumerate(time_starts):
                frame_end = frame_start + pd.Timedelta(seconds=trailing_window_real_seconds)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.grid(True, alpha=0.3)

                # Plot arena coordinates if provided
                if arena_coords is not None:
                    for zone in arena_coords['zone'].unique():
                        zone_coords = arena_coords[arena_coords['zone'] == zone][['x', 'y']].values
                        if len(zone_coords) >= 3:  # Need at least 3 points to make a polygon
                            polygon = Polygon(zone_coords, closed=True, edgecolor='black', facecolor='none', linewidth=1.5, alpha=0.8)
                            ax.add_patch(polygon)
                            
                            # Add zone label at centroid
                            centroid_x = zone_coords[:, 0].mean()
                            centroid_y = zone_coords[:, 1].mean()
                            ax.text(centroid_x, centroid_y, zone, fontsize=8, ha='center', va='center', 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

                # Plot each tag
                for tag in unique_tags:
                    tag_data = day_data[(day_data['shortid'] == tag) & 
                                       (day_data['Timestamp'] >= frame_start) & 
                                       (day_data['Timestamp'] <= frame_end)]
                    
                    if tag_data.empty:
                        continue

                    color = tag_color_map[tag]
                    label = tag_label_map[tag]
                    
                    # Plot trailing path
                    ax.plot(tag_data[x_col], tag_data[y_col], color=color, alpha=0.6, linewidth=1.5)
                    
                    # Plot current position (most recent point within a small time window)
                    # Get the most recent data point within the last few seconds of real time
                    current_window_start = frame_start
                    current_data = tag_data[tag_data['Timestamp'] >= current_window_start]
                    
                    if not current_data.empty:
                        current_x = current_data[x_col].iloc[-1]
                        current_y = current_data[y_col].iloc[-1]
                        ax.plot(current_x, current_y, 'o', color=color, markersize=8)
                        ax.text(current_x, current_y + 0.2, label, fontsize=8, ha='center', color=color)

                # Set plot properties
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel('X Coordinate (meters)')
                ax.set_ylabel('Y Coordinate (meters)')
                
                # Calculate elapsed time for title
                elapsed_real_seconds = (frame_start - start_time).total_seconds()
                elapsed_hours = int(elapsed_real_seconds // 3600)
                elapsed_minutes = int((elapsed_real_seconds % 3600) // 60)
                elapsed_secs = int(elapsed_real_seconds % 60)
                
                ax.set_title(f'UWB Tag Paths - {date_str} ({playback_speed}x speed)\nReal Time: {frame_start.strftime("%H:%M:%S")} (+{elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_secs:02d})')

                # Save frame
                filename = os.path.join(output_dir, f"frame_{i:04d}.png")
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close(fig)
                gc.collect()

                pbar.update(1)

        # Create video using ffmpeg with the standard frame rate
        video_name = f"uwb_animation_{date_str}_{playback_speed}x.mp4"
        video_output_path = os.path.join(output_dir, video_name)
        
        print(f"Rendering video with ffmpeg at {video_fps} fps...")
        subprocess.call([
            'ffmpeg', '-framerate', str(video_fps), '-i', os.path.join(output_dir, 'frame_%04d.png'),
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p', 
            video_output_path, '-y'
        ])

        # Move video to SQLite file directory
        final_video_path = os.path.join(sql_file_dir, video_name)
        shutil.move(video_output_path, final_video_path)
        
        expected_video_duration = total_real_duration / playback_speed
        print(f"Video saved: {video_name}")
        print(f"Video duration: {expected_video_duration:.1f} seconds ({expected_video_duration/60:.1f} minutes)")

        # Clean up PNG files
        print("Cleaning up temporary frame files...")
        for filename in os.listdir(output_dir):
            if filename.endswith(".png"):
                os.remove(os.path.join(output_dir, filename))

    # Create videos for selected days
    print(f"Creating {len(selected_date_list)} animation videos...")
    for date in selected_date_list:
        day_data = data[data['Date'] == date]
        if not day_data.empty:
            create_daily_video(day_data, str(date), arena_coordinates)

    print("All animations complete!")

if __name__ == "__main__":
    uwb_animate_paths()
