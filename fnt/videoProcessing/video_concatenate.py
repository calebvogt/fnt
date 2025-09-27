## TODO:
# Add a second ffmpeg step that moves the start atom to the beginning of the concatenated file. Not necessary unless doing so for sleap, but still might prove useful. 
# ffmpeg -i output.mp4 -movflags +faststart -c copy output_seekable.mp4 



import os
import subprocess
import tkinter as tk
from tkinter import filedialog

def video_concatenate():
    """Prompts user to select a folder and concatenates all video files within it using FFmpeg."""
    
    # Open file dialog to select a directory
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    input_directory = filedialog.askdirectory(title="Select Folder with Videos")

    # Check if the user selected a folder
    if not input_directory:
        print("No folder selected. Exiting.")
        return

    os.chdir(input_directory)  # Move into the selected directory

    # List of supported video file extensions
    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".MP4")

    # Normalize filenames (removing spaces)
    for f in os.listdir("."):
        if f.endswith(VIDEO_EXTENSIONS):
            new_name = f.replace(" ", "")  # Removes spaces
            if new_name != f:
                os.rename(f, new_name)

    # Generate a list of video files in sorted order
    video_files = sorted([f for f in os.listdir(".") if f.endswith(VIDEO_EXTENSIONS)])

    if not video_files:
        print("No valid video files found in the selected folder.")
        return

    # Create a text file with the video filenames for FFmpeg
    with open("mylist.txt", "w") as fp:
        for video in video_files:
            fp.write(f"file '{video}'\n")  # Correctly format filenames

    # Run FFmpeg to concatenate videos and print real-time output
    command = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", "mylist.txt", "-c", "copy", "concatenated_output.mp4"]
    print("\nConcatenating videos... This may take a moment.\n")

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line.strip())

    process.wait()

    # Delete mylist.txt after processing
    os.remove("mylist.txt")
    print("Concatenation complete! Output file: output.mp4")
    print("Temporary file mylist.txt deleted.")

# Only run if the script is executed directly
if __name__ == "__main__":
    video_concatenate()
