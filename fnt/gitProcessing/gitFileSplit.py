#!/usr/bin/env python3
"""
Git File Splitter for FieldNeuroToolbox (FNT)

This module provides functionality to split large files into smaller chunks
to comply with GitHub's 50MB file size limit. It also provides functionality
to rejoin the split files.

Usage:
    python gitFileSplit.py

The script will prompt you to:
1. Select files to split
2. Choose maximum chunk size (default: 45MB for safety)

The split files will be created in the same directory as the original file.
A .split_info.txt file will be created to help with rejoining.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path


def select_files_to_split():
    """Select multiple files to split using tkinter file dialog"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    files = filedialog.askopenfilenames(
        title="Select files to split for GitHub",
        filetypes=[
            ("All files", "*.*"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx"),
            ("Data files", "*.dat"),
            ("Text files", "*.txt"),
        ]
    )
    
    root.destroy()
    return list(files) if files else []


def get_max_size_mb():
    """Get maximum file size from user"""
    root = tk.Tk()
    root.withdraw()
    
    max_size = simpledialog.askinteger(
        "File Size Limit",
        "Enter maximum file size in MB:\n" +
        "(GitHub limit is 50MB)\n" +
        "Recommended: 45MB for safety",
        initialvalue=45,
        minvalue=1,
        maxvalue=100
    )
    
    root.destroy()
    return max_size


def split_file(file_path, max_size_bytes):
    """
    Split a file into chunks of specified maximum size
    
    Args:
        file_path (str): Path to the file to split
        max_size_bytes (int): Maximum size of each chunk in bytes
    
    Returns:
        int: Number of chunks created
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    base_name, ext = os.path.splitext(file_name)
    
    chunk_number = 1
    total_size = os.path.getsize(file_path)
    
    print(f"Splitting {file_name} ({total_size/1024/1024:.1f}MB)...")
    
    with open(file_path, 'rb') as input_file:
        while True:
            chunk_data = input_file.read(max_size_bytes)
            if not chunk_data:
                break
            
            # Create chunk filename: original.part001.ext
            chunk_filename = f"{base_name}.part{chunk_number:03d}{ext}"
            chunk_path = os.path.join(file_dir, chunk_filename)
            
            # Write chunk
            with open(chunk_path, 'wb') as chunk_file:
                chunk_file.write(chunk_data)
            
            chunk_size_mb = len(chunk_data) / 1024 / 1024
            print(f"  Created: {chunk_filename} ({chunk_size_mb:.1f}MB)")
            chunk_number += 1
    
    total_chunks = chunk_number - 1
    
    # Create info file for reconstruction
    info_filename = f"{base_name}.split_info.txt"
    info_path = os.path.join(file_dir, info_filename)
    
    with open(info_path, 'w') as info_file:
        info_file.write(f"Original file: {file_name}\n")
        info_file.write(f"Original size: {total_size} bytes ({total_size/1024/1024:.1f}MB)\n")
        info_file.write(f"Total chunks: {total_chunks}\n")
        info_file.write(f"Max chunk size: {max_size_bytes} bytes ({max_size_bytes/1024/1024:.1f}MB)\n")
        info_file.write(f"Split date: {os.path.getctime(file_path)}\n")
        info_file.write(f"\nChunk files:\n")
        for i in range(1, total_chunks + 1):
            chunk_name = f"{base_name}.part{i:03d}{ext}"
            info_file.write(f"  {chunk_name}\n")
        info_file.write(f"\nTo rejoin files:\n")
        info_file.write(f"1. Use FNT GUI: GitHub Preprocessing -> File Joiner\n")
        info_file.write(f"2. Or run: python gitFileSplit.py --join {info_filename}\n")
        info_file.write(f"3. Or manually (Windows): copy /b {base_name}.part001{ext}+{base_name}.part002{ext}+... {file_name}\n")
        info_file.write(f"4. Or manually (Mac/Linux): cat {base_name}.part*{ext} > {file_name}\n")
    
    print(f"✅ Split complete: {total_chunks} chunks created")
    print(f"📝 Info file: {info_filename}")
    
    return total_chunks


def join_files_from_info(info_path):
    """
    Rejoin files using the split info file
    
    Args:
        info_path (str): Path to the .split_info.txt file
    """
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Info file not found: {info_path}")
    
    file_dir = os.path.dirname(info_path)
    
    # Read info file
    with open(info_path, 'r') as info_file:
        lines = info_file.readlines()
    
    # Parse info
    original_name = None
    total_chunks = None
    
    for line in lines:
        if line.startswith("Original file:"):
            original_name = line.split(': ', 1)[1].strip()
        elif line.startswith("Total chunks:"):
            total_chunks = int(line.split(': ')[1])
    
    if not original_name or not total_chunks:
        raise ValueError("Invalid info file format")
    
    base_name, ext = os.path.splitext(original_name)
    output_path = os.path.join(file_dir, f"{base_name}_rejoined{ext}")
    
    print(f"Rejoining {original_name} from {total_chunks} chunks...")
    
    # Join chunks
    with open(output_path, 'wb') as output_file:
        for chunk_num in range(1, total_chunks + 1):
            chunk_filename = f"{base_name}.part{chunk_num:03d}{ext}"
            chunk_path = os.path.join(file_dir, chunk_filename)
            
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f"Missing chunk: {chunk_filename}")
            
            with open(chunk_path, 'rb') as chunk_file:
                data = chunk_file.read()
                output_file.write(data)
                chunk_size_mb = len(data) / 1024 / 1024
                print(f"  Added: {chunk_filename} ({chunk_size_mb:.1f}MB)")
    
    output_size = os.path.getsize(output_path)
    print(f"✅ Rejoin complete: {os.path.basename(output_path)} ({output_size/1024/1024:.1f}MB)")
    
    return output_path


def select_info_file():
    """Select a split info file for rejoining"""
    root = tk.Tk()
    root.withdraw()
    
    info_file = filedialog.askopenfilename(
        title="Select split info file",
        filetypes=[
            ("Split info files", "*.split_info.txt"),
            ("All files", "*.*"),
        ]
    )
    
    root.destroy()
    return info_file


def main():
    """Main function for command-line usage"""
    print("🔧 Git File Splitter for FieldNeuroToolbox")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--join" and len(sys.argv) > 2:
            # Join mode
            info_path = sys.argv[2]
            try:
                output_path = join_files_from_info(info_path)
                print(f"\n✅ Files successfully rejoined: {os.path.basename(output_path)}")
            except Exception as e:
                print(f"\n❌ Error rejoining files: {e}")
            return
    
    # Interactive mode
    print("Choose operation:")
    print("1. Split files")
    print("2. Join files")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            # Split mode
            files = select_files_to_split()
            if not files:
                print("No files selected. Exiting.")
                return
            
            max_size_mb = get_max_size_mb()
            if not max_size_mb:
                print("No size specified. Exiting.")
                return
            
            max_size_bytes = max_size_mb * 1024 * 1024
            
            print(f"\nProcessing {len(files)} file(s)...")
            print(f"Maximum chunk size: {max_size_mb}MB")
            
            total_split = 0
            total_skipped = 0
            
            for file_path in files:
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size <= max_size_bytes:
                        print(f"⏭️  Skipping {os.path.basename(file_path)}: already under size limit")
                        total_skipped += 1
                        continue
                    
                    chunks = split_file(file_path, max_size_bytes)
                    total_split += 1
                    
                except Exception as e:
                    print(f"❌ Error processing {os.path.basename(file_path)}: {e}")
            
            print(f"\n📊 Summary:")
            print(f"   Files split: {total_split}")
            print(f"   Files skipped: {total_skipped}")
            print(f"   Total processed: {len(files)}")
        
        elif choice == "2":
            # Join mode
            info_file = select_info_file()
            if not info_file:
                print("No info file selected. Exiting.")
                return
            
            try:
                output_path = join_files_from_info(info_file)
                print(f"\n✅ Files successfully rejoined: {os.path.basename(output_path)}")
            except Exception as e:
                print(f"\n❌ Error rejoining files: {e}")
        
        else:
            print("Invalid choice. Exiting.")
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()