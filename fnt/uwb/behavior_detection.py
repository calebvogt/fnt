"""
Behavioral Detection Module for UWB Quick Visualization

Streamlined behavioral analysis for real-time detection and summary export.
Focuses on practical behavior categories with configurable thresholds.

Author: Integrated from behavioral_classifier.py
Date: 2025-11-18
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from typing import Dict, List, Tuple
import time


class BehaviorDetector:
    """Detect and classify animal behaviors from UWB tracking data"""
    
    def __init__(self, 
                 speed_threshold_rest=0.05,      # m/s - below this is resting
                 speed_threshold_active=0.3,     # m/s - above this is active movement
                 distance_threshold_social=0.5,  # m - within this is social interaction
                 window_seconds=10.0,            # seconds per analysis window
                 overlap_seconds=5.0):           # seconds of overlap between windows
        
        self.speed_threshold_rest = speed_threshold_rest
        self.speed_threshold_active = speed_threshold_active
        self.distance_threshold_social = distance_threshold_social
        self.window_seconds = window_seconds
        self.overlap_seconds = overlap_seconds
        
        # Behavior categories we detect
        self.behavior_categories = [
            'resting',           # Low movement, alone
            'social_interaction', # Close proximity with other animal(s)
            'exploring',         # Moderate movement, alone
            'active',            # High movement, alone
            'unknown'            # Can't classify confidently
        ]
    
    def analyze_behaviors(self, data: pd.DataFrame, log_callback=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze behaviors in tracking data
        
        Args:
            data: DataFrame with columns: Timestamp, shortid, smoothed_x/location_x, smoothed_y/location_y
            log_callback: Optional function to call with log messages
            
        Returns:
            Tuple of (behavior_timeline, social_interactions)
            - behavior_timeline: DataFrame with timestamp, animal_id, behavior, confidence
            - social_interactions: DataFrame with timestamp, animal1, animal2, distance, duration
        """
        
        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)
        
        log("Starting behavioral analysis...")
        start_time = time.time()
        
        # Determine coordinate columns
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Get time bounds
        start_timestamp = data['Timestamp'].min()
        end_timestamp = data['Timestamp'].max()
        
        # Create sliding time windows
        current_time = start_timestamp
        window_td = pd.Timedelta(seconds=self.window_seconds)
        step_td = pd.Timedelta(seconds=self.overlap_seconds)
        
        behavior_records = []
        social_interaction_records = []
        
        window_count = 0
        
        while current_time < end_timestamp:
            window_end = current_time + window_td
            
            # Get data for this window
            window_data = data[
                (data['Timestamp'] >= current_time) & 
                (data['Timestamp'] < window_end)
            ].copy()
            
            if len(window_data) > 0:
                # Analyze this window
                window_behaviors, window_interactions = self._analyze_window(
                    window_data, current_time, x_col, y_col
                )
                behavior_records.extend(window_behaviors)
                social_interaction_records.extend(window_interactions)
                
                window_count += 1
                if window_count % 50 == 0:
                    log(f"Processed {window_count} time windows...")
            
            current_time += step_td
        
        # Create DataFrames from records
        if behavior_records:
            behavior_timeline = pd.DataFrame(behavior_records)
        else:
            behavior_timeline = pd.DataFrame(columns=['timestamp', 'animal_id', 'behavior', 'confidence'])
        
        if social_interaction_records:
            social_interactions = pd.DataFrame(social_interaction_records)
        else:
            social_interactions = pd.DataFrame(columns=['timestamp', 'animal1', 'animal2', 'distance', 'duration'])
        
        elapsed = time.time() - start_time
        log(f"Behavioral analysis complete: {window_count} windows in {elapsed:.1f}s")
        log(f"Detected {len(behavior_records)} behavior events")
        log(f"Detected {len(social_interaction_records)} social interactions")
        
        return behavior_timeline, social_interactions
    
    def _analyze_window(self, window_data: pd.DataFrame, timestamp: pd.Timestamp, 
                       x_col: str, y_col: str) -> Tuple[List[Dict], List[Dict]]:
        """Analyze a single time window"""
        
        behavior_records = []
        interaction_records = []
        
        # Get unique animals in this window
        animals = window_data['shortid'].unique()
        
        # Calculate features for each animal
        animal_features = {}
        for animal_id in animals:
            animal_data = window_data[window_data['shortid'] == animal_id].sort_values('Timestamp')
            features = self._calculate_individual_features(animal_data, x_col, y_col)
            animal_features[animal_id] = features
        
        # Calculate pairwise distances and detect social interactions
        animal_positions = {}  # Store mean position for each animal
        for animal_id in animals:
            animal_data = window_data[window_data['shortid'] == animal_id]
            mean_x = animal_data[x_col].mean()
            mean_y = animal_data[y_col].mean()
            animal_positions[animal_id] = (mean_x, mean_y)
        
        # Check all pairs for social interactions
        for i, animal1 in enumerate(animals):
            for animal2 in animals[i+1:]:
                pos1 = animal_positions[animal1]
                pos2 = animal_positions[animal2]
                distance = euclidean(pos1, pos2)
                
                if distance < self.distance_threshold_social:
                    # Social interaction detected
                    interaction_records.append({
                        'timestamp': timestamp,
                        'animal1': int(animal1),
                        'animal2': int(animal2),
                        'distance': float(distance),
                        'duration': float(self.window_seconds)  # Window duration
                    })
        
        # Classify behavior for each animal
        for animal_id in animals:
            features = animal_features[animal_id]
            
            # Check if this animal is in social interaction
            in_social_interaction = False
            min_social_distance = np.inf
            
            for other_id in animals:
                if other_id != animal_id:
                    pos1 = animal_positions[animal_id]
                    pos2 = animal_positions[other_id]
                    dist = euclidean(pos1, pos2)
                    if dist < self.distance_threshold_social:
                        in_social_interaction = True
                        min_social_distance = min(min_social_distance, dist)
            
            # Classify behavior based on speed and social context
            speed = features['speed_mean']
            
            if in_social_interaction:
                behavior = 'social_interaction'
                confidence = 0.9
            elif speed < self.speed_threshold_rest:
                behavior = 'resting'
                confidence = 0.85
            elif speed > self.speed_threshold_active:
                behavior = 'active'
                confidence = 0.8
            elif speed > self.speed_threshold_rest:
                behavior = 'exploring'
                confidence = 0.75
            else:
                behavior = 'unknown'
                confidence = 0.5
            
            behavior_records.append({
                'timestamp': timestamp,
                'animal_id': int(animal_id),
                'behavior': behavior,
                'confidence': float(confidence),
                'speed_mean': float(speed),
                'min_social_distance': float(min_social_distance) if min_social_distance != np.inf else None
            })
        
        return behavior_records, interaction_records
    
    def _calculate_individual_features(self, animal_data: pd.DataFrame, 
                                      x_col: str, y_col: str) -> Dict:
        """Calculate movement features for one animal in a time window"""
        
        features = {}
        
        if len(animal_data) < 2:
            # Not enough data points
            features['speed_mean'] = 0.0
            features['speed_max'] = 0.0
            features['total_distance'] = 0.0
            return features
        
        # Extract coordinates and timestamps
        x_coords = animal_data[x_col].values
        y_coords = animal_data[y_col].values
        timestamps = animal_data['Timestamp'].values
        
        # Calculate distances and speeds
        distances = []
        speeds = []
        
        for i in range(1, len(x_coords)):
            # Distance between consecutive points
            dist = euclidean([x_coords[i-1], y_coords[i-1]], [x_coords[i], y_coords[i]])
            distances.append(dist)
            
            # Time difference
            time_diff_td = timestamps[i] - timestamps[i-1]
            if hasattr(time_diff_td, 'total_seconds'):
                time_diff = time_diff_td.total_seconds()
            else:
                time_diff = time_diff_td / pd.Timedelta(seconds=1)
            
            # Speed
            if time_diff > 0:
                speed = dist / time_diff
                speeds.append(speed)
        
        # Calculate features
        if speeds:
            features['speed_mean'] = np.mean(speeds)
            features['speed_max'] = np.max(speeds)
        else:
            features['speed_mean'] = 0.0
            features['speed_max'] = 0.0
        
        if distances:
            features['total_distance'] = np.sum(distances)
        else:
            features['total_distance'] = 0.0
        
        return features
    
    def generate_behavior_summary(self, behavior_timeline: pd.DataFrame, 
                                  social_interactions: pd.DataFrame,
                                  tag_identities: Dict = None) -> pd.DataFrame:
        """
        Generate summary statistics for each animal's behaviors
        
        Returns:
            DataFrame with columns: animal_id, behavior_category, total_duration_seconds, percentage, event_count
        """
        
        if behavior_timeline.empty:
            return pd.DataFrame(columns=['animal_id', 'behavior', 'duration_seconds', 'percentage', 'event_count'])
        
        # Calculate duration for each behavior per animal
        summary_records = []
        
        for animal_id in behavior_timeline['animal_id'].unique():
            animal_behaviors = behavior_timeline[behavior_timeline['animal_id'] == animal_id]
            total_windows = len(animal_behaviors)
            
            # Count each behavior
            behavior_counts = animal_behaviors['behavior'].value_counts()
            
            for behavior, count in behavior_counts.items():
                duration_seconds = count * self.window_seconds
                percentage = 100 * count / total_windows
                
                summary_records.append({
                    'animal_id': int(animal_id),
                    'behavior': behavior,
                    'duration_seconds': float(duration_seconds),
                    'percentage': float(percentage),
                    'event_count': int(count)
                })
        
        summary_df = pd.DataFrame(summary_records)
        
        # Add identity labels if available
        if tag_identities and not summary_df.empty:
            summary_df['identity'] = summary_df['animal_id'].apply(
                lambda x: f"{tag_identities[x]['sex']}-{tag_identities[x]['identity']}" 
                if x in tag_identities else f"Tag_{x}"
            )
            # Reorder columns
            cols = ['animal_id', 'identity', 'behavior', 'duration_seconds', 'percentage', 'event_count']
            summary_df = summary_df[cols]
        
        return summary_df
    
    def generate_social_interaction_summary(self, social_interactions: pd.DataFrame,
                                           tag_identities: Dict = None) -> pd.DataFrame:
        """
        Generate summary of social interactions between animals
        
        Returns:
            DataFrame with columns: animal1, animal2, interaction_count, total_duration_seconds, mean_distance
        """
        
        if social_interactions.empty:
            return pd.DataFrame(columns=['animal1', 'animal2', 'interaction_count', 
                                        'total_duration_seconds', 'mean_distance'])
        
        # Group by animal pairs
        summary_records = []
        
        # Create a sorted pair column to group bidirectional interactions
        social_interactions['pair'] = social_interactions.apply(
            lambda row: tuple(sorted([row['animal1'], row['animal2']])), axis=1
        )
        
        for pair, group in social_interactions.groupby('pair'):
            animal1, animal2 = pair
            
            summary_records.append({
                'animal1': int(animal1),
                'animal2': int(animal2),
                'interaction_count': int(len(group)),
                'total_duration_seconds': float(group['duration'].sum()),
                'mean_distance': float(group['distance'].mean()),
                'min_distance': float(group['distance'].min())
            })
        
        summary_df = pd.DataFrame(summary_records)
        
        # Add identity labels if available
        if tag_identities and not summary_df.empty:
            summary_df['identity1'] = summary_df['animal1'].apply(
                lambda x: f"{tag_identities[x]['sex']}-{tag_identities[x]['identity']}" 
                if x in tag_identities else f"Tag_{x}"
            )
            summary_df['identity2'] = summary_df['animal2'].apply(
                lambda x: f"{tag_identities[x]['sex']}-{tag_identities[x]['identity']}" 
                if x in tag_identities else f"Tag_{x}"
            )
            # Reorder columns
            cols = ['animal1', 'identity1', 'animal2', 'identity2', 'interaction_count', 
                   'total_duration_seconds', 'mean_distance', 'min_distance']
            summary_df = summary_df[cols]
        
        return summary_df
