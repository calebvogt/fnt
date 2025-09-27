import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ---------- GUI for setting tracking parameters ----------
class TrackingParamsGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("SLEAP Re-Tracking Parameters")

        self.params = {}

        # Dropdown for tracker type
        tk.Label(master, text="Tracking Tracker:").grid(row=0, column=0, sticky="w")
        self.tracker_var = tk.StringVar(value="simple")
        tracker_options = ["simple", "flow", "simplemaxtracks", "flowmaxtracks", "none"]
        ttk.Combobox(master, textvariable=self.tracker_var, values=tracker_options, state="readonly").grid(row=0, column=1)

        # Entry fields for numeric params
        self.add_entry("tracking.max_tracks", "Max Tracks", row=1)
        self.add_entry("tracking.target_instance_count", "Target Instance Count", row=2)
        self.add_entry("tracking.clean_instance_count", "Clean Instance Count", row=3)
        self.add_entry("tracking.clean_iou_threshold", "Clean IOU Threshold", row=4)
        self.add_entry("tracking.track_window", "Track Window", row=5)

        # Checkboxes for boolean flags
        self.max_tracking_var = tk.BooleanVar()
        tk.Checkbutton(master, text="Enable Max Tracking (--tracking.max_tracking)", variable=self.max_tracking_var)\
            .grid(row=6, columnspan=2, sticky="w")

        # Run button
        tk.Button(master, text="Run Re-Tracking", command=self.submit).grid(row=7, columnspan=2, pady=10)

    def add_entry(self, key, label, row):
        tk.Label(self.master, text=f"{label}:").grid(row=row, column=0, sticky="w")
        var = tk.StringVar()
        tk.Entry(self.master, textvariable=var).grid(row=row, column=1)
        self.params[key] = var

    def submit(self):
        self.tracking_args = []

        if self.tracker_var.get() != "none":
            self.tracking_args += ["--tracking.tracker", self.tracker_var.get()]

        for key, var in self.params.items():
            val = var.get().strip()
            if val != "":
                self.tracking_args += [f"--{key}", val]

        if self.max_tracking_var.get():
            self.tracking_args += ["--tracking.max_tracking", "1"]

        self.master.destroy()


# ---------- Run sleap-track for each file ----------
def run_retracking(slp_file, args, output_suffix="_retracked"):
    output_file = slp_file.replace(".slp", f"{output_suffix}.slp")
    cmd = ["sleap-track", "-o", output_file] + args + [slp_file]

    print(f"🔁 Re-tracking: {os.path.basename(slp_file)}")
    print("CMD:", " ".join(cmd))
    subprocess.run(cmd)


# ---------- File selector ----------
def select_slp_files():
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(
        title="Select .slp prediction files to re-track",
        filetypes=[("SLEAP prediction files", "*.slp")]
    )
    return root.tk.splitlist(files)


# ---------- Entry point ----------
def main():
    slp_files = select_slp_files()
    if not slp_files:
        print("❌ No files selected.")
        return

    # Launch GUI for tracking parameters
    root = tk.Tk()
    app = TrackingParamsGUI(root)
    root.mainloop()

    for slp_file in slp_files:
        run_retracking(slp_file, app.tracking_args)

    print("\n✅ Re-tracking complete.")

if __name__ == "__main__":
    main()
