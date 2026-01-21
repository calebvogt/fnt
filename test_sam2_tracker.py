#!/usr/bin/env python3
"""
SAM 2 Tracker Test Script

Simple test to verify SAM 2 multi-object tracking works on your videos.
Draw bounding boxes manually in code, then track multiple objects through video.

Usage:
    python test_sam2_tracker.py

Author: FieldNeuroToolbox Contributors
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add fnt to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from fnt.videoTracking.sam2_tracker import SAM2MultiObjectTracker


def visualize_tracking(
    tracker: SAM2MultiObjectTracker,
    video_path: str,
    output_path: str = None
):
    """
    Visualize tracking results with overlays.
    
    Args:
        tracker: Initialized tracker with objects added
        video_path: Path to input video
        output_path: Optional path to save visualization video
    """
    cap = cv2.VideoCapture(video_path)
    
    # Setup video writer if saving
    if output_path:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Starting tracking visualization...")
    print("Press 'q' to stop early, any other key to continue")
    
    # Colors for different objects
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
    ]
    
    frame_count = 0
    
    # Track through video
    for frame_idx, object_ids, video_masks in tracker.track_objects():
        # Read corresponding frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw masks and trajectories for each object
        for i, obj_id in enumerate(object_ids):
            color = colors[i % len(colors)]
            
            # Get mask
            mask = video_masks[obj_id][0].cpu().numpy()
            
            # Create colored overlay
            mask_overlay = np.zeros_like(frame)
            mask_overlay[mask > 0] = color
            
            # Blend with frame
            frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.3, 0)
            
            # Draw contour
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, contours, -1, color, 2)
            
            # Get centroid
            moments = cv2.moments(mask.astype(np.uint8))
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                
                # Draw centroid
                cv2.circle(frame, (cx, cy), 8, color, -1)
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
                
                # Draw label
                label = tracker.object_labels[obj_id]
                cv2.putText(
                    frame,
                    label,
                    (cx + 15, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
        
        # Draw frame number
        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        # Show frame
        cv2.imshow('SAM 2 Tracking', frame)
        
        # Save if requested
        if output_path:
            out.write(frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Stopping early...")
            break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Tracking complete! Processed {frame_count} frames")


def main():
    """Main test function."""
    print("=" * 60)
    print("SAM 2 Multi-Object Tracker Test")
    print("=" * 60)
    
    # ============================================
    # CONFIGURATION - EDIT THESE
    # ============================================
    
    # Video path
    VIDEO_PATH = "path/to/your/video.mp4"  # CHANGE THIS
    
    # SAM 2 checkpoint (will auto-detect in SAM_models/)
    SAM_MODELS_DIR = Path(__file__).parent / "SAM_models"
    checkpoints = list(SAM_MODELS_DIR.glob("sam2*.pt"))
    
    if not checkpoints:
        print("\n❌ ERROR: No SAM 2 checkpoints found!")
        print(f"Please download a checkpoint to: {SAM_MODELS_DIR}")
        print("\nRun FNT GUI and use the Mask Tracker to download automatically,")
        print("or manually download from:")
        print("https://dl.fbaipublicfiles.com/segment_anything_2/092824/")
        return
    
    CHECKPOINT_PATH = str(checkpoints[0])
    print(f"\n✓ Using checkpoint: {checkpoints[0].name}")
    
    # Auto-detect config
    ckpt_name = checkpoints[0].name.lower()
    if "tiny" in ckpt_name:
        CONFIG_NAME = "sam2.1_hiera_t.yaml"
    elif "small" in ckpt_name:
        CONFIG_NAME = "sam2.1_hiera_s.yaml"
    elif "base" in ckpt_name:
        CONFIG_NAME = "sam2.1_hiera_b+.yaml"
    elif "large" in ckpt_name:
        CONFIG_NAME = "sam2.1_hiera_l.yaml"
    else:
        CONFIG_NAME = "sam2.1_hiera_l.yaml"
    
    print(f"✓ Using config: {CONFIG_NAME}")
    
    # Check video exists
    if not Path(VIDEO_PATH).exists():
        print(f"\n❌ ERROR: Video not found: {VIDEO_PATH}")
        print("\nPlease edit VIDEO_PATH in this script to point to your video.")
        return
    
    # ============================================
    # DEFINE BOUNDING BOXES FOR OBJECTS
    # ============================================
    # Format: [x1, y1, x2, y2] in pixel coordinates
    # You can get these by viewing first frame in any image viewer
    
    # Example: Rodent in top-left region
    RODENT_BOX = np.array([100, 100, 300, 300], dtype=np.float32)
    
    # Example: Water tower in top-right region  
    TOWER_BOX = np.array([500, 100, 700, 300], dtype=np.float32)
    
    # NOTE: To find correct coordinates, you can:
    # 1. Open first frame of video in a viewer
    # 2. Note pixel coordinates by hovering
    # 3. Or add a helper below to click and print coords
    
    print("\n" + "=" * 60)
    print("Starting SAM 2 Tracker...")
    print("=" * 60)
    
    # ============================================
    # INITIALIZE TRACKER
    # ============================================
    try:
        tracker = SAM2MultiObjectTracker(
            checkpoint_path=CHECKPOINT_PATH,
            config_name=CONFIG_NAME,
            device="cuda"  # Change to "cpu" if no GPU
        )
        
        print("\n1. Initializing video...")
        tracker.init_video(VIDEO_PATH)
        
        print("\n2. Adding object prompts...")
        
        # Add rodent
        tracker.add_box_prompt(
            box=RODENT_BOX,
            frame_idx=0,
            obj_id=1,
            label="Rodent"
        )
        
        # Add water tower
        tracker.add_box_prompt(
            box=TOWER_BOX,
            frame_idx=0,
            obj_id=2,
            label="Water Tower"
        )
        
        print(f"\n✓ Added {len(tracker.object_prompts)} objects")
        print("\n3. Running tracking...")
        
        # ============================================
        # RUN TRACKING WITH VISUALIZATION
        # ============================================
        output_video = Path(VIDEO_PATH).parent / f"{Path(VIDEO_PATH).stem}_tracked.mp4"
        
        visualize_tracking(
            tracker,
            VIDEO_PATH,
            output_path=str(output_video)
        )
        
        # ============================================
        # EXPORT TRAJECTORIES
        # ============================================
        print("\n4. Exporting trajectories...")
        trajectories = tracker.export_trajectories()
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        for obj_id, df in trajectories.items():
            label = tracker.object_labels[obj_id]
            print(f"\n{label} (ID {obj_id}):")
            print(f"  - Frames tracked: {len(df)}")
            print(f"  - Valid detections: {df['x'].notna().sum()}")
            print(f"  - Average speed: {df['speed'].mean():.2f} px/s")
            print(f"  - CSV saved: {label.lower().replace(' ', '_')}_trajectory.csv")
        
        print(f"\n✓ Tracking video saved: {output_video}")
        print("\n" + "=" * 60)
        print("TEST COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return


def get_box_coordinates_helper(video_path: str):
    """
    Helper function to interactively get bounding box coordinates.
    
    Click two opposite corners of a box to get [x1, y1, x2, y2].
    Press 'q' when done.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read video")
        return
    
    print("\nClick two opposite corners of each bounding box")
    print("Press 'r' to reset current box")
    print("Press 'q' when done")
    
    clicks = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            print(f"Point {len(clicks)}: ({x}, {y})")
            
            # Draw point
            cv2.circle(frame_display, (x, y), 5, (0, 255, 0), -1)
            
            # If two points, draw box
            if len(clicks) == 2:
                x1, y1 = clicks[0]
                x2, y2 = clicks[1]
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                print(f"Box: [{min(x1,x2)}, {min(y1,y2)}, {max(x1,x2)}, {max(y1,y2)}]")
                print("Press 'r' to reset or 'q' to quit")
            
            cv2.imshow('Select Bounding Box', frame_display)
    
    frame_display = frame.copy()
    cv2.namedWindow('Select Bounding Box')
    cv2.setMouseCallback('Select Bounding Box', mouse_callback)
    cv2.imshow('Select Bounding Box', frame_display)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            clicks = []
            frame_display = frame.copy()
            cv2.imshow('Select Bounding Box', frame_display)
            print("Reset - click new box")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    
    # Uncomment to use box coordinate helper:
    # get_box_coordinates_helper("path/to/your/video.mp4")
