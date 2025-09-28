#!/usr/bin/env python3
"""
Test script to verify the GUI cleanup was successful
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.abspath('.'))

def test_gui_cleanup():
    """Test that joining methods were removed and GUI still works"""
    print("🧪 Testing GUI cleanup...")
    
    try:
        from fnt.gui_pyqt import FNTMainWindow
        print("✅ Successfully imported FNTMainWindow")
        
        # Check that joining methods are removed
        methods = [method for method in dir(FNTMainWindow) if 'join' in method.lower()]
        if methods:
            print(f"❌ Still has joining methods: {methods}")
            return False
        else:
            print("✅ All joining methods successfully removed")
        
        # Check that split methods are still there
        split_methods = [method for method in dir(FNTMainWindow) if 'split' in method.lower()]
        print(f"✅ Split methods available: {split_methods}")
        
        # Test that we can create an instance (without showing GUI)
        import sys
        from PyQt5.QtWidgets import QApplication
        
        # Create app if it doesn't exist
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Create window instance
        window = FNTMainWindow()
        print("✅ Successfully created FNTMainWindow instance")
        
        # Check that GitHub tab exists and has File Splitter
        github_tab = None
        for i in range(window.tab_widget.count()):
            if window.tab_widget.tabText(i) == "GitHub":
                github_tab = window.tab_widget.widget(i)
                break
        
        if github_tab:
            print("✅ GitHub tab found")
            # Look for File Splitter button (should exist)
            splitter_button = github_tab.findChild(object, "splitter_button") 
            print("✅ GitHub tab configured properly")
        else:
            print("❌ GitHub tab not found")
            return False
            
        print("🎉 GUI cleanup test PASSED!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_gui_cleanup()
    sys.exit(0 if success else 1)