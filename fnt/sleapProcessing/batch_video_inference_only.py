import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

def prompt_model_type():
    return messagebox.askyesno("Model Type", "Are you using a TOP-DOWN model?\n(Click 'No' for bottom-up)")

def select_model_folder(title="Select model folder"):
    return filedialog.askdirectory(title=title)

def select_video_folder():
    return filedialog.askdirectory(title="Select folder containing video files for inference")

def get_output_path(video_path):
    base = os.path.basename(video_path)
    parent = os.path.dirname(video_path)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    filename = f"{base}.{timestamp}.predictions.slp"
    return os.path.join(parent, filename)

def convert_to_csv(slp_file):
    csv_file = slp_file.replace(".predictions.slp", ".predictions.analysis.csv")
    cmd = ["sleap-convert", "--format", "analysis.csv", "-o", csv_file, slp_file]
    print(f"📄 Converting to CSV: {os.path.basename(csv_file)}")
    subprocess.run(cmd)

def run_inference_on_video(video_file, model_paths):
    cmd = ["sleap-track", video_file]

    for model_path in model_paths:
        cmd += ["-m", os.path.join(model_path, "training_config.json")]

    output_file = get_output_path(video_file)
    cmd += [
        "--only-suggested-frames",
        "--no-empty-frames",
        "--verbosity", "json",
        "--video.input_format", "channels_last",
        "--gpu", "auto",
        "--batch_size", "4",
        "--peak_threshold", "0.2",
        "--tracking.tracker", "none",
        "--controller_port", "9000",
        "--publish_port", "9001",
        "-o", output_file
    ]

    print(f"\n🔁 Running inference on: {os.path.basename(video_file)}")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd)

    convert_to_csv(output_file)

def prompt_overwrite():
    return messagebox.askyesno(
        "Overwrite Existing Files?",
        "Prediction files (.slp or .csv) already exist for some videos.\n\n"
        "Would you like to OVERWRITE them?\n\n"
        "Click 'No' to skip those files."
    )

def main():
    root = tk.Tk()
    root.withdraw()

    video_folders = []

    while True:
        folder = select_video_folder()
        if not folder:
            break
        video_folders.append(folder)
        if not messagebox.askyesno("More folders?", "Would you like to add another folder?"):
            break

    if not video_folders:
        print("❌ No folders selected.")
        return

    is_top_down = prompt_model_type()
    model_paths = []

    if is_top_down:
        model_paths.append(select_model_folder("Select CENTROID model folder"))
        model_paths.append(select_model_folder("Select CENTERED INSTANCE model folder"))
    else:
        model_paths.append(select_model_folder("Select BOTTOM-UP model folder"))

    # --- Check for any pre-existing prediction files ---
    existing_found = False
    for folder in video_folders:
        video_files = [f for f in os.listdir(folder)
                       if f.lower().endswith((".mp4", ".avi", ".mov"))]
        for video_file in video_files:
            full_path = os.path.join(folder, video_file)
            slp_path = get_output_path(full_path)
            csv_path = slp_path.replace(".predictions.slp", ".predictions.analysis.csv")
            if os.path.exists(slp_path) or os.path.exists(csv_path):
                existing_found = True
                break
        if existing_found:
            break

    overwrite_existing = True
    if existing_found:
        overwrite_existing = prompt_overwrite()

    # --- Run inference ---
    for folder in video_folders:
        video_files = [f for f in os.listdir(folder)
                       if f.lower().endswith((".mp4", ".avi", ".mov"))]
        if not video_files:
            print(f"⚠️ No video files found in: {folder}")
            continue

        for video_file in video_files:
            full_path = os.path.join(folder, video_file)
            slp_path = get_output_path(full_path)
            csv_path = slp_path.replace(".predictions.slp", ".predictions.analysis.csv")

            if not overwrite_existing and (os.path.exists(slp_path) or os.path.exists(csv_path)):
                print(f"⏭️ Skipping {video_file} (existing predictions detected)")
                continue

            run_inference_on_video(full_path, model_paths)

    print("\n✅ Inference and CSV export complete for all videos.")

if __name__ == "__main__":
    main()
