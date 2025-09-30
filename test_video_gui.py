#!/usr/bin/env python3
"""
Quick test to launch the new Video Processing GUI
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.abspath('.'))

def test_video_processing_gui():
    """Test that the new video processing GUI can be launched"""
    try:
        from fnt.videoProcessing.videoProcessing import main
        print("✅ Video Processing GUI imported successfully")
        print("🚀 Launching Video Processing GUI...")
        print("(Close the window to complete the test)")
        
        # Launch the GUI
        main()
        
    except Exception as e:
        print(f"❌ Error launching Video Processing GUI: {e}")
        return False

if __name__ == "__main__":
    test_video_processing_gui()