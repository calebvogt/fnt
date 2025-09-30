import os
import subprocess
import tkinter as tk
from tkinter import filedialog

def video_downsample(gpu=False):
    # Prompt user to select folder
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select folder with video files")
    if not folder:
        print("No folder selected. Exiting.")
        return

    # Create 'proc' subdirectory
    proc_dir = os.path.join(folder, "proc")
    os.makedirs(proc_dir, exist_ok=True)

    # Supported video formats
    video_exts = (".avi", ".mp4", ".mov")

    # Loop through video files
    for filename in os.listdir(folder):
        if filename.lower().endswith(video_exts):
            input_path = os.path.join(folder, filename)
            output_path = os.path.join(proc_dir, f"{os.path.splitext(filename)[0]}_proc.mp4")

            # Choose CPU or GPU encoding
            if gpu:
                print(f"→ Using GPU encoding for {filename}")
                cmd = [
                    "ffmpeg", "-hwaccel", "cuda", "-i", input_path,
                    "-vcodec", "hevc_nvenc", # gpu acceleration
                    "-preset", "hq", # 
                    "-rc:v", "vbr",     # variable bitrate mode
                    "-cq:v", "30",      # qualtiy (15-32): lower is better
                    "-b:v", "0.8M",        # target average bitrate
                    "-maxrate", "0.8M",    # maximum bitrate
                    "-bufsize", "1.6M",    # buffer size
                    "-pix_fmt", "yuv420p",
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame,"
                           "pad=1920:1080:-1:-1:color=black,format=gray",
                    "-r", "30",
                    "-vsync", "cfr",
                    "-an",
                    "-movflags", "+faststart",
                    "-max_muxing_queue_size", "10000000",
                    output_path
                ]
            else:
                print(f"→ Using CPU encoding for {filename}")
                cmd = [
                    "ffmpeg", "-i", input_path,
                    "-vcodec", "libx265",
                    "-preset", "fast",
                    "-crf", "25",
                    "-pix_fmt", "yuv420p",
                    "-vf", "fps=30,scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame,"
                           "pad=1920:1080:-1:-1:color=black,format=gray",
                    # "-r", "30",
                    # "-vsync", "cfr",
                    "-an",
                    "-movflags", "+faststart",
                    "-max_muxing_queue_size", "10000000",
                    output_path
                ]

            print(f"\nProcessing: {filename}")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end='')

    print("\n✅ All videos processed and saved to 'proc' folder.")

if __name__ == "__main__":
    video_downsample()
