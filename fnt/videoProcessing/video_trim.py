# TODO: have single ffmpeg line printed that updates, instead of printing a new line for every speed/frame update etc. 

import os
import cv2
import tkinter as tk
from tkinter import filedialog
import subprocess
import numpy as np

def video_trim():
    """Opens a file dialog to select a video file and provides a GUI for trimming with previews."""
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        _process_video(file_path)

def _process_video(video_path):
    """Creates a GUI with sliders to trim the video using FFmpeg, showing start and end previews."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    # Trim window
    root = tk.Tk()
    root.title("Trim Video")
    root.state("zoomed")  # Maximize window on open

    # Variables for start and end times
    start_time = tk.DoubleVar(value=0)
    end_time = tk.DoubleVar(value=duration)

    # Create canvas for video previews
    start_canvas = tk.Canvas(root, width=640, height=360, bg="black")
    end_canvas = tk.Canvas(root, width=640, height=360, bg="black")

    start_canvas.grid(row=0, column=0, columnspan=6, padx=10, pady=5)
    end_canvas.grid(row=0, column=6, columnspan=6, padx=10, pady=5)

    def update_preview(time_var, canvas):
        """Updates the preview frame based on the trim selection."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_var.get() * fps))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 360))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = cv2.imencode('.png', frame)[1].tobytes()
            img = tk.PhotoImage(data=img)
            canvas.create_image(0, 0, anchor="nw", image=img)
            canvas.image = img  # Keep reference

    # Labels
    tk.Label(root, text="Start Time (seconds)").grid(row=1, column=0, columnspan=6)
    tk.Label(root, text="End Time (seconds)").grid(row=1, column=6, columnspan=6)

    # Sliders
    start_slider = tk.Scale(root, variable=start_time, from_=0, to=duration, resolution=0.1, orient="horizontal",
                            length=400, command=lambda x: update_preview(start_time, start_canvas))
    start_slider.grid(row=2, column=0, columnspan=6, padx=10, pady=5)

    end_slider = tk.Scale(root, variable=end_time, from_=0, to=duration, resolution=0.1, orient="horizontal",
                          length=400, command=lambda x: update_preview(end_time, end_canvas))
    end_slider.grid(row=2, column=6, columnspan=6, padx=10, pady=5)

    def adjust_slider(time_var, seconds):
        """Adjusts the slider position by the given second count."""
        new_time = max(0, min(duration, time_var.get() + seconds))
        time_var.set(new_time)
        update_preview(time_var, start_canvas if time_var == start_time else end_canvas)

    # Buttons Layout
    button_spacing = {"padx": 2, "pady": 5}  # Reduced horizontal spacing

    # Buttons for Start Time
    start_button_frame = tk.Frame(root)
    start_button_frame.grid(row=3, column=0, columnspan=6, pady=5)

    tk.Button(start_button_frame, text="-60s", command=lambda: adjust_slider(start_time, -60)).pack(side="left", **button_spacing)
    tk.Button(start_button_frame, text="-30s", command=lambda: adjust_slider(start_time, -30)).pack(side="left", **button_spacing)
    tk.Button(start_button_frame, text="-1s", command=lambda: adjust_slider(start_time, -1)).pack(side="left", **button_spacing)
    tk.Button(start_button_frame, text="+1s", command=lambda: adjust_slider(start_time, 1)).pack(side="left", **button_spacing)
    tk.Button(start_button_frame, text="+30s", command=lambda: adjust_slider(start_time, 30)).pack(side="left", **button_spacing)
    tk.Button(start_button_frame, text="+60s", command=lambda: adjust_slider(start_time, 60)).pack(side="left", **button_spacing)

    # Buttons for End Time
    end_button_frame = tk.Frame(root)
    end_button_frame.grid(row=3, column=6, columnspan=6, pady=5)

    tk.Button(end_button_frame, text="-60s", command=lambda: adjust_slider(end_time, -60)).pack(side="left", **button_spacing)
    tk.Button(end_button_frame, text="-30s", command=lambda: adjust_slider(end_time, -30)).pack(side="left", **button_spacing)
    tk.Button(end_button_frame, text="-1s", command=lambda: adjust_slider(end_time, -1)).pack(side="left", **button_spacing)
    tk.Button(end_button_frame, text="+1s", command=lambda: adjust_slider(end_time, 1)).pack(side="left", **button_spacing)
    tk.Button(end_button_frame, text="+30s", command=lambda: adjust_slider(end_time, 30)).pack(side="left", **button_spacing)
    tk.Button(end_button_frame, text="+60s", command=lambda: adjust_slider(end_time, 60)).pack(side="left", **button_spacing)

    def trim_and_close():
        """Executes FFmpeg to trim the video, prints output, and closes GUI."""
        start = start_time.get()
        end = end_time.get()
        if end <= start:
            print("Error: End time must be greater than start time.")
            return

        # Generate output filename
        base_name, ext = os.path.splitext(video_path)
        output_file = f"{base_name}_trimmed.mp4"

        # FFmpeg command
        command = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(start),
            "-to", str(end),
            "-c", "copy",
            output_file
        ]

        print(f"Trimming video... This may take a moment.\n")

        # Run FFmpeg and print output in real-time
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line.strip())

        process.wait()
        print(f"Trimmed video saved as: {output_file}")

        # Close the GUI after trimming is complete
        root.destroy()

    # Trim button
    trim_button = tk.Button(root, text="Trim Video", command=trim_and_close)
    trim_button.grid(row=4, column=3, columnspan=6, pady=20)

    root.mainloop()
    cap.release()
    cv2.destroyAllWindows()

# Only run if the script is executed directly
if __name__ == "__main__":
    video_trim()
