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

def ask_export_format():
    selected_formats = []

    def submit():
        selected_formats.clear()
        if var_csv.get():
            selected_formats.append("csv")
        if var_h5.get():
            selected_formats.append("h5")
        if not selected_formats:
            messagebox.showwarning("Nothing selected", "Please select at least one format.")
            return
        win.destroy()

    win = tk.Toplevel()
    win.title("Choose Export Format")
    win.geometry("250x150")
    win.grab_set()

    var_csv = tk.BooleanVar()
    var_h5 = tk.BooleanVar()

    tk.Label(win, text="Select format(s) to export:").pack(pady=10)
    tk.Checkbutton(win, text=".CSV", variable=var_csv).pack(anchor="w", padx=20)
    tk.Checkbutton(win, text=".H5", variable=var_h5).pack(anchor="w", padx=20)
    tk.Button(win, text="OK", command=submit).pack(pady=10)
    win.wait_window()

    return selected_formats

def ask_max_instances():
    result = {"max_instances": None}

    def submit():
        val = entry.get().strip()
        if val:
            try:
                result["max_instances"] = int(val)
            except ValueError:
                messagebox.showwarning("Invalid input", "Please enter an integer value.")
                return
        win.destroy()

    def no_limit():
        result["max_instances"] = None
        win.destroy()

    win = tk.Toplevel()
    win.title("Max Instances")
    win.geometry("300x130")
    win.grab_set()

    tk.Label(win, text="Enter max number of instances (or leave blank for no limit):").pack(pady=10)
    entry = tk.Entry(win)
    entry.pack(pady=5)

    tk.Button(win, text="OK", command=submit).pack(side="left", padx=20, pady=10)
    tk.Button(win, text="No Max", command=no_limit).pack(side="right", padx=20, pady=10)

    win.wait_window()
    return result["max_instances"]

def ask_skip_existing(count_existing, total):
    return messagebox.askyesno(
        "Skip Already Processed Videos?",
        f"{count_existing} out of {total} videos already have associated .slp/.csv/.h5 tracking files.\n\n"
        "Do you want to skip these videos?"
    )

def ask_move_empty_videos():
    return messagebox.askyesno(
        "Move Videos with No Detections?",
        "Place videos with no detected instances into a subfolder?\n\n(They will be moved to a folder named 'no_instances_detected')"
    )

def get_output_path(video_path):
    base = os.path.basename(video_path)
    parent = os.path.dirname(video_path)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    filename = f"{base}.{timestamp}.predictions.slp"
    return os.path.join(parent, filename)

def is_slp_empty(slp_path):
    try:
        import sleap
        labels = sleap.load_file(slp_path)
        return not labels.instances or len(labels.skeletons) == 0
    except Exception as e:
        print(f"⚠️ Failed to inspect {os.path.basename(slp_path)}: {e}")
        return True

def convert_slp_to_format(slp_path, fmt):
    ext = f".analysis.{fmt}"
    output_file = slp_path.replace(".slp", ext)
    if os.path.exists(output_file):
        print(f"⏭️ Skipping {os.path.basename(output_file)} (already exists)")
        return False

    print(f"\n🔁 Now converting: {os.path.basename(slp_path)} to .{fmt} format...")
    cmd = ["sleap-convert", "--format", f"analysis.{fmt}", "-o", output_file, slp_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("🔧 STDOUT:\n", result.stdout)
    print("❗ STDERR:\n", result.stderr)
    return True

def run_inference_and_convert(video_file, model_paths, max_instances, formats, move_empty):
    cmd = ["sleap-track", video_file]
    for model_path in model_paths:
        cmd += ["-m", os.path.join(model_path, "training_config.json")]

    output_file = get_output_path(video_file)
    cmd += [
        "--no-empty-frames",
        "--verbosity", "rich",
        "--video.input_format", "channels_last",
        "--gpu", "auto",
        "--batch_size", "4",
        "--peak_threshold", "0.2",
        "--tracking.tracker", "simple",
        "--controller_port", "9000",
        "--publish_port", "9001",
        "-o", output_file
    ]
    if max_instances is not None:
        cmd += ["--max_instances", str(max_instances)]

    print(f"\n🔁 Running inference + tracking on: {os.path.basename(video_file)}")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd)

    if move_empty and is_slp_empty(output_file):
        print(f"📭 No detections in {os.path.basename(video_file)} — moving to subfolder.")
        dest_folder = os.path.join(os.path.dirname(video_file), "no_instances_detected")
        os.makedirs(dest_folder, exist_ok=True)
        os.rename(video_file, os.path.join(dest_folder, os.path.basename(video_file)))
        os.rename(output_file, os.path.join(dest_folder, os.path.basename(output_file)))
        return

    for fmt in formats:
        convert_slp_to_format(output_file, fmt)

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

    max_instances = ask_max_instances()
    formats = ask_export_format()
    if not formats:
        print("⚠️ No export format selected. Exiting.")
        return

    move_empty = ask_move_empty_videos()

    all_videos = []
    existing_videos = []

    for folder in video_folders:
        for f in os.listdir(folder):
            if f.lower().endswith((".mp4", ".avi", ".mov")):
                full_path = os.path.join(folder, f)
                all_videos.append(full_path)

                base = os.path.splitext(f)[0]
                slp = [x for x in os.listdir(folder) if x.startswith(base) and x.endswith(".slp")]
                csv = [x for x in os.listdir(folder) if x.startswith(base) and x.endswith(".analysis.csv")]
                h5 = [x for x in os.listdir(folder) if x.startswith(base) and x.endswith(".analysis.h5")]
                if slp or csv or h5:
                    existing_videos.append(full_path)

    skip_existing = ask_skip_existing(len(existing_videos), len(all_videos))

    for video in all_videos:
        if skip_existing and video in existing_videos:
            print(f"⏭️ Skipping {os.path.basename(video)} (existing tracking files detected)")
            continue
        run_inference_and_convert(video, model_paths, max_instances, formats, move_empty)

    print("\n✅ Inference, tracking, and export complete for all videos.")

if __name__ == "__main__":
    main()
