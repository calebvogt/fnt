import os
import subprocess
import glob
import re
import tkinter as tk
from tkinter import filedialog

def video_reencode():
    """Allows the user to select a folder containing videos and re-encodes them while preserving quality and ensuring seekability."""

    # Open file dialog to select a directory
    root = tk.Tk()
    root.withdraw()  # Hide the main Tk window
    input_directory = filedialog.askdirectory(title="Select Folder with Videos")

    # Check if the user selected a folder
    if not input_directory:
        print("No folder selected. Exiting.")
        return

    # Directory to store converted videos (inside the selected folder)
    out_dir = os.path.join(input_directory, "proc")
    os.makedirs(out_dir, exist_ok=True)  # Create directory if it doesn't exist

    # List of supported video file extensions
    video_extensions = ["*.avi", "*.mp4", "*.mov"]

    # Loop over each video extension type
    for video_extension in video_extensions:
        # List all video files in the selected directory
        video_files = glob.glob(os.path.join(input_directory, video_extension))

        # Loop over each video file
        for video_file in video_files:
            # Get the video filename without extension
            video_filename = os.path.basename(video_file)
            video_filename_no_ext = re.sub(r'\.avi|\.mp4|\.mov', '', video_filename)

            # Output file path
            output_file = os.path.join(out_dir, video_filename_no_ext + '.mp4')

            print(f"\nProcessing: {video_file} → {output_file}")

            # FFmpeg command
            cmd = [
                "ffmpeg",
                "-i", video_file,               # Input file
                "-vcodec", "libx265",          # Efficient compression
                "-preset", "ultrafast",        # Fast encoding
                "-crf", "15",                  # High quality (default 23, 15 is near-lossless)
                "-pix_fmt", "yuv420p",         # Standard pixel format
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame,"
                       "pad=1920:1080:-1:-1:color=black,format=gray",  # Resize & grayscale
                "-an",                         # Remove audio
                "-r", "30",                    # Force 30 FPS
                "-max_muxing_queue_size", "10000000",  # Ensures smooth encoding
                output_file
            ]

            # Run FFmpeg with live progress output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line.strip())

            process.wait()  # Wait for process to complete

    print("\n✅ All videos have been re-encoded and saved in the 'proc' folder.")

# Only run if the script is executed directly
if __name__ == "__main__":
    video_reencode()
