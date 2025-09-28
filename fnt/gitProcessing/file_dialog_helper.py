#!/usr/bin/env python3
"""
Alternative File Dialog Helper for FNT
Uses tkinter file dialogs as a fallback if PyQt dialogs have issues
"""

import tkinter as tk
from tkinter import filedialog
import os


def select_files_tkinter(title="Select files", filetypes=None):
    """
    Select multiple files using tkinter (more reliable on some systems)
    
    Args:
        title (str): Dialog title
        filetypes (list): List of file type tuples
    
    Returns:
        list: Selected file paths
    """
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    if filetypes is None:
        filetypes = [
            ("All files", "*.*"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx;*.xls"),
            ("Data files", "*.dat;*.txt"),
        ]
    
    files = filedialog.askopenfilenames(
        title=title,
        filetypes=filetypes
    )
    
    root.destroy()
    return list(files) if files else []


def select_file_tkinter(title="Select file", filetypes=None):
    """
    Select a single file using tkinter
    
    Args:
        title (str): Dialog title
        filetypes (list): List of file type tuples
    
    Returns:
        str: Selected file path or empty string
    """
    root = tk.Tk()
    root.withdraw()
    
    if filetypes is None:
        filetypes = [
            ("Split info files", "*.split_info.txt"),
            ("Part files", "*.part001.*"),
            ("All files", "*.*"),
        ]
    
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )
    
    root.destroy()
    return file_path if file_path else ""


if __name__ == "__main__":
    # Test the file dialogs
    print("Testing tkinter file dialogs...")
    
    print("\n1. Testing multi-file selection:")
    files = select_files_tkinter("Select files to split")
    print(f"Selected {len(files)} files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    print("\n2. Testing single file selection:")
    file_path = select_file_tkinter("Select split info or part file")
    if file_path:
        print(f"Selected: {os.path.basename(file_path)}")
    else:
        print("No file selected")