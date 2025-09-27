import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

# Converts .slp files to .csv and/or .h5 formats using the sleap-convert command-line tool.


# ✅ Initialize global root for Tkinter dialogs and windows
root = tk.Tk()
root.withdraw()

def select_folders():
    folders = []

    while True:
        folder = filedialog.askdirectory(title="Select folder with .slp files")
        if not folder:
            break
        folders.append(folder)
        if not messagebox.askyesno("More folders?", "Would you like to add another folder?"):
            break

    return folders

def ask_export_format():
    selected_formats = []

    def submit():
        # Clear and repopulate to avoid side-effects if resubmitted
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

def convert_slp_to_format(slp_path, fmt):
    ext = f".analysis.{fmt}"
    output_file = slp_path.replace(".slp", ext)
    if os.path.exists(output_file):
        print(f"⏭️ Skipping {os.path.basename(output_file)} (already exists)")
        return False

    print(f"\n🔁 Now converting: {os.path.basename(slp_path)} to .{fmt} format...")
    cmd = ["sleap-convert", "--format", f"analysis.{fmt}", "-o", output_file, slp_path]
    print("CMD:", cmd, flush=True)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Print both outputs immediately ## these can be deleted
    print("🔧 STDOUT:\n", result.stdout)
    print("❗ STDERR:\n", result.stderr)
    return True

def main():
    print("📂 Opening folder selection dialog...", flush=True)
    folders = select_folders()  
    print("✅ Folder selection done.", flush=True)
    if not folders:
        print("❌ No folders selected.")
        return

    print("🛠️ Opening export format selector...", flush=True)
    formats = ask_export_format()
    if not formats:
        print("⚠️ No export format selected. Exiting.")
        return

    print(f"\n📁 Export formats selected: {formats}")
    print("--------------------------------------------------\n")

    total_converted = 0
    total_skipped = 0
    total_files = 0

    for folder in folders:
        slp_files = [f for f in os.listdir(folder) if f.endswith(".slp")]
        if not slp_files:
            print(f"⚠️ No .slp files found in: {folder}")
            continue

        for slp_file in slp_files:
            total_files += 1
            full_path = os.path.join(folder, slp_file)

            for fmt in formats:
                success = convert_slp_to_format(full_path, fmt)
                if success:
                    total_converted += 1
                else:
                    total_skipped += 1

    print("\n✅ Conversion complete.")
    print("--------------------------------------------------")
    print(f"🧾 Total .slp files scanned: {total_files}")
    print(f"✅ Files converted: {total_converted}")
    print(f"⏭️ Files skipped (already exist): {total_skipped}")

if __name__ == "__main__":
    main()
