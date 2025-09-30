#!/usr/bin/env python3
"""
Test script to verify the new video processing functionality
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """Test that all video processing modules can be imported"""
    print("🧪 Testing video processing imports...")
    
    try:
        # Test main GUI import
        from fnt.gui_pyqt import FNTMainWindow
        print("✅ Successfully imported FNTMainWindow")
        
        # Test new video processing tool
        from fnt.videoProcessing.videoProcessing import VideoProcessingGUI
        print("✅ Successfully imported VideoProcessingGUI")
        
        # Test original video processing modules
        from fnt.videoProcessing.videoDownsample import video_downsample
        print("✅ Successfully imported video_downsample")
        
        from fnt.videoProcessing.video_reencode import video_reencode
        print("✅ Successfully imported video_reencode")
        
        print("\\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_gui_methods():
    """Test that GUI methods exist and are properly defined"""
    print("\\n🧪 Testing GUI methods...")
    
    try:
        from fnt.gui_pyqt import FNTMainWindow
        
        # Create a minimal app to test GUI creation
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Create main window instance
        window = FNTMainWindow()
        
        # Check that all video processing methods exist
        methods_to_check = [
            'run_video_processing',
            'run_video_downsample', 
            'run_video_reencode',
            'run_video_trim',
            'run_video_concatenate'
        ]
        
        for method_name in methods_to_check:
            if hasattr(window, method_name):
                print(f"✅ Method {method_name} exists")
            else:
                print(f"❌ Method {method_name} missing")
                return False
        
        print("\\n🎉 All GUI methods found!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing GUI methods: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("VIDEO PROCESSING FUNCTIONALITY TEST")
    print("=" * 60)
    
    success1 = test_imports()
    success2 = test_gui_methods()
    
    print("\\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("\\nNew features added:")
        print("• Combined Video Processing tool with PyQt interface")
        print("• Frame rate and grayscale customization")
        print("• GPU acceleration option")
        print("• Fixed import paths for existing tools")
    else:
        print("❌ SOME TESTS FAILED!")
    
    print("=" * 60)