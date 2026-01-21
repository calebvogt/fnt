#!/usr/bin/env python3
"""
SAM 2 Video Tracker

Uses SAM 2 for multi-object video tracking with bounding box prompts.
Supports tracking multiple objects simultaneously in a single pass.

Author: FieldNeuroToolbox Contributors
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM 2 not available. Install with: pip install git+https://github.com/facebookresearch/sam2.git")


class SAM2MultiObjectTracker:
    """
    Multi-object video tracker using SAM 2.
    
    Supports:
    - Bounding box prompts
    - Point prompts  
    - Multiple objects tracked simultaneously
    - Automatic trajectory extraction
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_name: str,
        device: str = "cuda" if TORCH_AVAILABLE and torch and torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize SAM 2 tracker.
        
        Args:
            checkpoint_path: Path to SAM 2 checkpoint (.pt file)
            config_name: Config file name (e.g., "sam2.1_hiera_l.yaml")
            device: "cuda" or "cpu"
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for SAM 2 tracking. Install with: pip install torch")
        if not SAM2_AVAILABLE:
            raise ImportError("SAM 2 is required. Install with: pip install git+https://github.com/facebookresearch/sam2.git")
            
        self.checkpoint_path = checkpoint_path
        self.config_name = config_name
        self.device = device
        
        self.predictor = None
        self.inference_state = None
        self.video_path = None
        
        # Object tracking data
        self.object_prompts = {}  # {obj_id: {"type": "box/point", "data": ...}}
        self.object_masks = {}    # {obj_id: {frame_idx: mask}}
        self.object_labels = {}   # {obj_id: "label_name"}
        
        # Video properties
        self.total_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        
    def load_model(self):
        """Load SAM 2 video predictor."""
        if not SAM2_AVAILABLE:
            raise ImportError("SAM 2 not installed")
            
        print(f"Loading SAM 2 model: {self.config_name}")
        
        # Build SAM 2 config path - look in sam2 package configs directory
        import sam2
        sam2_pkg_dir = Path(sam2.__file__).parent
        
        # Try multiple common config locations
        possible_paths = [
            sam2_pkg_dir / "configs" / self.config_name,
            sam2_pkg_dir / "configs" / "sam2.1" / self.config_name,
            sam2_pkg_dir / self.config_name,
        ]
        
        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError(f"Could not find SAM 2 config '{self.config_name}' in sam2 package. Tried: {possible_paths}")
            
        self.predictor = build_sam2_video_predictor(
            str(config_path),
            self.checkpoint_path,
            device=self.device
        )
        
        print(f"SAM 2 model loaded successfully on {self.device}")
        
    def init_video(self, video_path: str):
        """
        Initialize video for tracking.
        
        Args:
            video_path: Path to video file or JPEG folder
        """
        self.video_path = video_path
        
        # Get video properties
        if Path(video_path).is_file():
            cap = cv2.VideoCapture(video_path)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        
        # Initialize SAM 2 inference state
        if self.predictor is None:
            self.load_model()
        
        if TORCH_AVAILABLE and torch:
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                self.inference_state = self.predictor.init_state(video_path=video_path)
        else:
            self.inference_state = self.predictor.init_state(video_path=video_path)
            
        print(f"Video initialized: {self.width}x{self.height} @ {self.fps:.1f} fps, {self.total_frames} frames")
        
    def add_box_prompt(
        self,
        box: np.ndarray,
        frame_idx: int = 0,
        obj_id: Optional[int] = None,
        label: str = ""
    ) -> int:
        """
        Add bounding box prompt for object.
        
        Args:
            box: [x1, y1, x2, y2] bounding box coordinates
            frame_idx: Frame index to initialize tracking
            obj_id: Optional object ID (auto-assigned if None)
            label: Human-readable label for this object
            
        Returns:
            Object ID
        """
        if self.inference_state is None:
            raise ValueError("Must initialize video first with init_video()")
            
        # Auto-assign object ID if not provided
        if obj_id is None:
            obj_id = len(self.object_prompts) + 1
            
        # Add prompt to SAM 2
        if TORCH_AVAILABLE and torch:
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=box
                )
        else:
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box
            )
        
        # Store prompt info
        self.object_prompts[obj_id] = {
            "type": "box",
            "data": box,
            "frame_idx": frame_idx
        }
        self.object_labels[obj_id] = label or f"Object {obj_id}"
        
        print(f"Added box prompt for {self.object_labels[obj_id]} (ID: {obj_id})")
        
        return obj_id
        
    def add_point_prompt(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        frame_idx: int = 0,
        obj_id: Optional[int] = None,
        label: str = ""
    ) -> int:
        """
        Add point prompt for object.
        
        Args:
            points: Nx2 array of [x, y] coordinates
            labels: N array of labels (1=foreground, 0=background)
            frame_idx: Frame index
            obj_id: Optional object ID
        if TORCH_AVAILABLE and torch:
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels
                )
        else
            
        Returns:
            Object ID
        """
        if self.inference_state is None:
            raise ValueError("Must initialize video first with init_video()")
            
        if obj_id is None:
            obj_id = len(self.object_prompts) + 1
            
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels
            )
        
        self.object_prompts[obj_id] = {
            "type": "point",
            "data": {"points": points, "labels": labels},
            "frame_idx": frame_idx
        }
        self.object_labels[obj_id] = label or f"Object {obj_id}"
        
        print(f"Added point prompt for {self.object_labels[obj_id]} (ID: {obj_id})")
        
        return obj_id
    
    def track_objects(self):
        """
        Run SAM 2 video tracking on all added prompts.
        
        Yields:
            Tuple of (frame_idx, object_ids, video_masks)
        """
        if self.inference_state is None:
            raise ValueError("Must initialize video first with init_video()")
        
        if not self.object_prompts:
            raise ValueError("Must add at least one prompt (box or point) before tracking")
        
        print(f"Starting tracking for {len(self.object_prompts)} object(s)...")
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            for frame_idx, object_ids, video_masks in self.predictor.propagate_in_video(
                self.inference_state
            ):
                # Store masks - video_masks is indexed by position, not object ID
                for i, obj_id in enumerate(object_ids):
                    if obj_id not in self.object_masks:
                        self.object_masks[obj_id] = {}
                    
                    # Extract mask for this object (use position index i, not obj_id)
                    mask = video_masks[i][0].cpu().numpy()
                    self.object_masks[obj_id][frame_idx] = mask
                    
                yield frame_idx, object_ids, video_masks
                
        print("Tracking complete!")
        
    def get_centroids(self, obj_id: int) -> np.ndarray:
        """
        Get centroid trajectory for object.
        
        Args:
            obj_id: Object ID
            
        Returns:
            Nx2 array of [x, y] centroids for each frame
        """
        if obj_id not in self.object_masks:
            return np.array([])
            
        centroids = []
        max_frame = max(self.object_masks[obj_id].keys())
        
        for frame_idx in range(max_frame + 1):
            mask = self.object_masks[obj_id].get(frame_idx)
            
            if mask is not None and mask.sum() > 0:
                # Calculate centroid
                moments = cv2.moments(mask.astype(np.uint8))
                if moments["m00"] > 0:
                    cx = moments["m10"] / moments["m00"]
                    cy = moments["m01"] / moments["m00"]
                    centroids.append([cx, cy])
                else:
                    centroids.append([np.nan, np.nan])
            else:
                centroids.append([np.nan, np.nan])
                
        return np.array(centroids)
        
    def export_trajectories(self, output_dir: Optional[str] = None) -> Dict[int, pd.DataFrame]:
        """
        Export trajectories for all objects to CSV files.
        
        Args:
            output_dir: Output directory (defaults to video directory)
            
        Returns:
            Dictionary mapping obj_id to DataFrame
        """
        if output_dir is None:
            output_dir = Path(self.video_path).parent
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_stem = Path(self.video_path).stem
        trajectories = {}
        
        for obj_id in self.object_prompts.keys():
            centroids = self.get_centroids(obj_id)
            
            if len(centroids) == 0:
                continue
                
            # Create DataFrame
            df = pd.DataFrame({
                'frame': np.arange(len(centroids)),
                'x': centroids[:, 0],
                'y': centroids[:, 1],
                'object_id': obj_id,
                'label': self.object_labels[obj_id]
            })
            
            # Add time and velocity
            df['time_s'] = df['frame'] / self.fps if self.fps > 0 else df['frame']
            df['vx'] = df['x'].diff() * self.fps
            df['vy'] = df['y'].diff() * self.fps
            df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)
            
            # Save to CSV
            label_slug = self.object_labels[obj_id].lower().replace(' ', '_')
            output_path = output_dir / f"{video_stem}_{label_slug}_trajectory.csv"
            df.to_csv(output_path, index=False)
            
            trajectories[obj_id] = df
            print(f"Exported trajectory for {self.object_labels[obj_id]}: {output_path}")
            
        return trajectories
        
    def reset(self):
        """Reset tracker state."""
        self.inference_state = None
        self.object_prompts.clear()
        self.object_masks.clear()
        self.object_labels.clear()
