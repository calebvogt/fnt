"""
Displacement Event Detector - Stage 3d of pipeline.

Detects male displacement events (1 male → 2 males → 1 male pattern).

Equivalent to R script: 3c_create_ALLTRIAL_MOVEBOUT_GBI_displace.R
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable

from ..config import RFIDConfig


class DisplacementDetector:
    """
    Detector for displacement events in RFID data.
    """

    def __init__(self, config: RFIDConfig):
        """
        Initialize displacement detector with configuration.

        Args:
            config: RFID configuration object
        """
        self.config = config

    def detect_displacements(
        self,
        movebout_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Detect displacement events (males only).

        Pattern: 1 male → 2 males → 1 male in a zone

        Args:
            movebout_df: Movement bout DataFrame
            metadata_df: Metadata DataFrame with animal information
            progress_callback: Optional callback function(message: str)

        Returns:
            DataFrame with displacement events
        """
        if progress_callback:
            progress_callback("Detecting displacement events...")

        # Filter for males only
        males = metadata_df[metadata_df['sex'] == 'M']['name'].unique()
        male_bouts = movebout_df[movebout_df['name'].isin(males)].copy()

        if len(male_bouts) == 0:
            if progress_callback:
                progress_callback("No male data found")
            return pd.DataFrame()

        displacements = []

        # Process each trial and zone separately
        for trial in self.config.trial_ids:
            trial_bouts = male_bouts[male_bouts['trial'] == trial]

            for zone in trial_bouts['zone_id'].unique():
                zone_bouts = trial_bouts[trial_bouts['zone_id'] == zone]

                # Sort by time
                zone_bouts = zone_bouts.sort_values('bout_start_time')

                # Detect displacement pattern
                zone_displacements = self._detect_zone_displacements(zone_bouts, trial, zone)
                displacements.extend(zone_displacements)

        # Convert to DataFrame
        if displacements:
            displacement_df = pd.DataFrame(displacements)

            # Save output
            output_path = self._save_output(displacement_df, progress_callback)

            if progress_callback:
                progress_callback(f"Detected {len(displacement_df)} displacement events")

            return displacement_df
        else:
            if progress_callback:
                progress_callback("No displacement events detected")
            return pd.DataFrame()

    def _detect_zone_displacements(
        self,
        zone_bouts: pd.DataFrame,
        trial: str,
        zone: int
    ) -> list:
        """
        Detect displacement events in a single zone.

        Args:
            zone_bouts: Bouts for one zone
            trial: Trial identifier
            zone: Zone identifier

        Returns:
            List of displacement event dictionaries
        """
        displacements = []

        # Group by time windows to find overlapping bouts
        zone_bouts = zone_bouts.sort_values('bout_start_time').reset_index(drop=True)

        for i in range(len(zone_bouts) - 1):
            bout1 = zone_bouts.iloc[i]

            # Find overlapping bouts
            overlapping = zone_bouts[
                (zone_bouts['bout_start_time'] <= bout1['bout_end_time']) &
                (zone_bouts['bout_end_time'] >= bout1['bout_start_time']) &
                (zone_bouts.index != i)
            ]

            # Check for 1 → 2 → 1 pattern
            if len(overlapping) > 0:
                # Two males present
                for j, bout2 in overlapping.iterrows():
                    # Find the male that remains
                    overlap_start = max(bout1['bout_start_time'], bout2['bout_start_time'])
                    overlap_end = min(bout1['bout_end_time'], bout2['bout_end_time'])

                    # Determine winner (who stays longer)
                    if bout1['bout_end_time'] > bout2['bout_end_time']:
                        winner = bout1['name']
                        loser = bout2['name']
                    else:
                        winner = bout2['name']
                        loser = bout1['name']

                    displacement = {
                        'trial': trial,
                        'zone_id': zone,
                        'time': overlap_start,
                        'winner': winner,
                        'loser': loser,
                        'duration': overlap_end - overlap_start
                    }

                    displacements.append(displacement)

        return displacements

    def _save_output(
        self,
        displacement_df: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Save displacement events to file.

        Args:
            displacement_df: Displacement DataFrame
            progress_callback: Optional callback function

        Returns:
            Path to output file
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "ALLTRIAL_MOVEBOUT_GBI_displace.csv"

        displacement_df.to_csv(output_path, index=False)

        return str(output_path)
