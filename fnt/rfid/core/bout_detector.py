"""
Movement Bout Detector - Stage 2 of pipeline.

Detects movement bouts from RFID reads:
- Identifies continuous presence in a zone
- Classifies reads as START, STOP, or SINGLE_READ
- Calculates bout durations

Equivalent to R script: 2_create_ALLTRIAL_MOVEBOUT.R
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable

from ..config import RFIDConfig


class BoutDetector:
    """
    Detector for movement bouts from RFID data.

    Identifies continuous bouts of presence in zones based on time threshold.
    """

    def __init__(self, config: RFIDConfig):
        """
        Initialize bout detector with configuration.

        Args:
            config: RFID configuration object
        """
        self.config = config

    def detect_bouts(
        self,
        rfid_df: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Detect movement bouts from RFID data.

        Algorithm:
        1. Group by (trial, animal, day)
        2. Sort by time
        3. Identify bout boundaries:
           - New bout starts when: time gap > threshold OR zone changes
           - Bout ends when next bout starts or at last read
        4. Classify reads: START, STOP, or SINGLE_READ
        5. Calculate bout durations

        Args:
            rfid_df: Processed RFID DataFrame
            progress_callback: Optional callback function(message: str)

        Returns:
            DataFrame with bout information added

        Raises:
            ValueError: If required columns are missing
        """
        # Validate input
        required_cols = ['trial', 'name', 'noon_day', 'time_sec', 'zone_id']
        missing_cols = [col for col in required_cols if col not in rfid_df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if progress_callback:
            progress_callback("Detecting movement bouts...")

        # Process each group separately
        bout_dfs = []

        # Group by trial, animal, day
        grouped = rfid_df.groupby(['trial', 'name', 'noon_day'], dropna=False)
        total_groups = len(grouped)

        for i, ((trial, animal, day), group_df) in enumerate(grouped):
            if progress_callback and i % 100 == 0:
                progress_callback(f"Processing group {i+1}/{total_groups} ({trial}, {animal}, day {day})")

            # Process this group
            bout_df = self._process_group(group_df)
            bout_dfs.append(bout_df)

        # Combine all results
        result_df = pd.concat(bout_dfs, ignore_index=True)

        if progress_callback:
            progress_callback(f"Detected {result_df['bout_type'].value_counts().to_dict()}")

        # Save output
        output_path = self._save_output(result_df, progress_callback)

        if progress_callback:
            progress_callback(f"Saved movement bouts to: {output_path}")

        return result_df

    def _process_group(self, group_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single group (trial, animal, day) to detect bouts.

        Args:
            group_df: DataFrame for one group

        Returns:
            DataFrame with bout information
        """
        if len(group_df) == 0:
            return group_df

        # Sort by time
        group_df = group_df.sort_values('time_sec').copy()

        # Calculate time differences between consecutive reads
        group_df['time_diff'] = group_df['time_sec'].diff()

        # Detect zone changes
        group_df['zone_changed'] = (group_df['zone_id'] != group_df['zone_id'].shift(1))

        # Identify bout starts
        # A bout starts when:
        # 1. It's the first read (time_diff is NaN)
        # 2. Time gap > threshold
        # 3. Zone changed
        bout_threshold = self.config.bout_threshold_sec

        group_df['bout_start'] = (
            group_df['time_diff'].isna() |
            (group_df['time_diff'] > bout_threshold) |
            group_df['zone_changed']
        )

        # Assign bout IDs
        group_df['bout_id'] = group_df['bout_start'].cumsum()

        # Calculate bout statistics
        bout_groups = group_df.groupby('bout_id')

        # Bout start and end times
        bout_start_times = bout_groups['time_sec'].first()
        bout_end_times = bout_groups['time_sec'].last()
        bout_sizes = bout_groups.size()

        # Create bout info mapping
        group_df['bout_start_time'] = group_df['bout_id'].map(bout_start_times)
        group_df['bout_end_time'] = group_df['bout_id'].map(bout_end_times)
        group_df['bout_size'] = group_df['bout_id'].map(bout_sizes)

        # Calculate bout duration
        group_df['bout_duration'] = group_df['bout_end_time'] - group_df['bout_start_time']

        # Classify reads
        group_df['bout_type'] = 'WITHIN_BOUT'  # Default

        # Mark bout starts and ends
        for bout_id in group_df['bout_id'].unique():
            bout_mask = group_df['bout_id'] == bout_id
            bout_indices = group_df[bout_mask].index

            if len(bout_indices) == 1:
                # Single read - mark as SINGLE_READ
                group_df.loc[bout_indices, 'bout_type'] = 'SINGLE_READ'
                # Set duration to min_duration_sec for single reads
                group_df.loc[bout_indices, 'bout_duration'] = self.config.min_duration_sec
            else:
                # Multi-read bout - mark START and STOP
                group_df.loc[bout_indices[0], 'bout_type'] = 'START'
                group_df.loc[bout_indices[-1], 'bout_type'] = 'STOP'

        # Apply minimum duration
        # Zero-duration bouts get min_duration_sec
        zero_duration_mask = group_df['bout_duration'] == 0
        group_df.loc[zero_duration_mask, 'bout_duration'] = self.config.min_duration_sec

        # Clean up temporary columns
        group_df = group_df.drop(columns=['time_diff', 'zone_changed', 'bout_start'])

        return group_df

    def _save_output(
        self,
        df: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Save movement bout data to file.

        Args:
            df: Movement bout DataFrame
            progress_callback: Optional callback function

        Returns:
            Path to output file
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "ALLTRIAL_MOVEBOUT.csv"

        df.to_csv(output_path, index=False)

        if progress_callback:
            progress_callback(f"Saved {len(df):,} reads with bout information")

        return str(output_path)

    def get_bout_statistics(self, bout_df: pd.DataFrame) -> dict:
        """
        Calculate summary statistics for movement bouts.

        Args:
            bout_df: Movement bout DataFrame

        Returns:
            Dictionary of summary statistics
        """
        stats = {
            'total_reads': len(bout_df),
            'total_bouts': bout_df['bout_id'].nunique(),
            'bout_type_counts': bout_df['bout_type'].value_counts().to_dict(),
        }

        # Only analyze actual bouts (not single reads)
        bout_reads = bout_df[bout_df['bout_type'].isin(['START', 'STOP'])]

        if len(bout_reads) > 0:
            # Get bout-level statistics (one row per bout - use START reads)
            bout_starts = bout_reads[bout_reads['bout_type'] == 'START']

            stats['bout_duration_stats'] = {
                'mean': bout_starts['bout_duration'].mean(),
                'median': bout_starts['bout_duration'].median(),
                'min': bout_starts['bout_duration'].min(),
                'max': bout_starts['bout_duration'].max(),
                'std': bout_starts['bout_duration'].std()
            }

            # Per-animal bout counts
            if 'name' in bout_df.columns:
                stats['bouts_per_animal'] = bout_starts.groupby('name').size().to_dict()

            # Per-zone bout counts
            if 'zone_id' in bout_df.columns:
                stats['bouts_per_zone'] = bout_starts.groupby('zone_id').size().to_dict()

        return stats
