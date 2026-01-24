"""
Group-By-Individual (GBI) Matrix Generator - Stage 3a of pipeline.

Creates binary presence matrices showing which animals were present
in each grouping event (co-occurrence in zones).

Equivalent to R script: 3_create_ALLTRIAL_MOVEBOUT_GBI.R
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple
from itertools import combinations

from ..config import RFIDConfig


class GBIGenerator:
    """
    Generator for Group-By-Individual matrices.

    Creates binary matrices showing co-occurrence patterns in zones.
    """

    def __init__(self, config: RFIDConfig):
        """
        Initialize GBI generator with configuration.

        Args:
            config: RFID configuration object
        """
        self.config = config

    def create_gbi_matrices(
        self,
        movebout_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Create GBI matrices for all trials.

        Args:
            movebout_df: Movement bout DataFrame
            metadata_df: Metadata DataFrame with animal information
            progress_callback: Optional callback function(message: str)

        Returns:
            Dictionary mapping trial_id to GBI DataFrame

        Raises:
            ValueError: If required columns are missing
        """
        # Validate input
        required_cols = ['trial', 'name', 'zone_id', 'bout_type', 'bout_id',
                        'bout_start_time', 'bout_end_time']
        missing_cols = [col for col in required_cols if col not in movebout_df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if progress_callback:
            progress_callback("Creating GBI matrices...")

        # Process each trial separately
        gbi_dict = {}

        for trial_id in self.config.trial_ids:
            if progress_callback:
                progress_callback(f"Processing trial: {trial_id}")

            trial_df = movebout_df[movebout_df['trial'] == trial_id]
            trial_metadata = metadata_df[metadata_df['trial'] == trial_id]

            if len(trial_df) == 0:
                if progress_callback:
                    progress_callback(f"Warning: No data for trial {trial_id}")
                continue

            # Create GBI for this trial
            gbi_df = self._create_trial_gbi(trial_df, trial_metadata, progress_callback)

            gbi_dict[trial_id] = gbi_df

            # Save trial-specific GBI
            output_path = self._save_trial_gbi(trial_id, gbi_df, progress_callback)

            if progress_callback:
                progress_callback(f"Saved {len(gbi_df)} grouping events for {trial_id}")

        return gbi_dict

    def _create_trial_gbi(
        self,
        trial_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Create GBI matrix for a single trial.

        Args:
            trial_df: Movement bout data for one trial
            metadata_df: Metadata for one trial
            progress_callback: Optional callback function

        Returns:
            GBI DataFrame with grouping events
        """
        # Get list of all animals in this trial
        animals = sorted(metadata_df['name'].unique())

        # Get sex information for summary columns
        animal_sex = metadata_df.set_index('name')['sex'].to_dict()

        # Extract bout information (START and STOP reads)
        bout_reads = trial_df[trial_df['bout_type'].isin(['START', 'STOP'])].copy()

        # Get bout-level information
        bouts = self._extract_bouts(bout_reads)

        if len(bouts) == 0:
            # No bouts found, return empty GBI
            return pd.DataFrame()

        # Find overlapping bouts by zone
        grouping_events = []

        # Process each zone separately
        zones = bouts['zone_id'].unique()

        for zone_id in zones:
            zone_bouts = bouts[bouts['zone_id'] == zone_id]

            # Find overlapping time windows
            overlaps = self._find_overlapping_bouts(zone_bouts)

            for overlap_group in overlaps:
                # Create grouping event
                animals_present = overlap_group['animals']
                start_time = overlap_group['start_time']
                end_time = overlap_group['end_time']
                center_time = (start_time + end_time) / 2

                # Create binary presence vector
                presence = {animal: (1 if animal in animals_present else 0) for animal in animals}

                # Add event metadata
                event = {
                    'zone_id': zone_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'center_time': center_time,
                    'duration': end_time - start_time,
                    'group_size': len(animals_present)
                }

                # Add presence columns
                event.update(presence)

                # Add sex-based counts
                males = [a for a in animals_present if animal_sex.get(a) == 'M']
                females = [a for a in animals_present if animal_sex.get(a) == 'F']

                event['m_sum'] = len(males)
                event['f_sum'] = len(females)
                event['mf_sum'] = len(males) + len(females)

                grouping_events.append(event)

        # Convert to DataFrame
        if grouping_events:
            gbi_df = pd.DataFrame(grouping_events)

            # Reorder columns: metadata first, then animal presence
            metadata_cols = ['zone_id', 'start_time', 'end_time', 'center_time',
                            'duration', 'group_size', 'm_sum', 'f_sum', 'mf_sum']
            animal_cols = sorted(animals)

            # Ensure all columns exist
            for col in metadata_cols + animal_cols:
                if col not in gbi_df.columns:
                    if col in metadata_cols:
                        gbi_df[col] = np.nan
                    else:
                        gbi_df[col] = 0

            gbi_df = gbi_df[metadata_cols + animal_cols]

            return gbi_df
        else:
            return pd.DataFrame()

    def _extract_bouts(self, bout_reads: pd.DataFrame) -> pd.DataFrame:
        """
        Extract bout-level information from START/STOP reads.

        Args:
            bout_reads: DataFrame with START and STOP reads

        Returns:
            DataFrame with one row per bout
        """
        bouts = []

        # Group by bout_id to get bout information
        for bout_id, group in bout_reads.groupby('bout_id'):
            start_reads = group[group['bout_type'] == 'START']

            if len(start_reads) == 0:
                continue

            # Use first START read for bout info (should only be one)
            start_read = start_reads.iloc[0]

            bout_info = {
                'bout_id': bout_id,
                'animal': start_read['name'],
                'zone_id': start_read['zone_id'],
                'start_time': start_read['bout_start_time'],
                'end_time': start_read['bout_end_time'],
                'duration': start_read['bout_duration']
            }

            bouts.append(bout_info)

        return pd.DataFrame(bouts)

    def _find_overlapping_bouts(self, zone_bouts: pd.DataFrame) -> List[Dict]:
        """
        Find overlapping bouts in a zone (co-occurrence events).

        Uses the "center time" method: bouts overlap if their centers
        fall within each other's duration.

        Args:
            zone_bouts: Bouts in a single zone

        Returns:
            List of overlapping bout groups
        """
        if len(zone_bouts) == 0:
            return []

        zone_bouts = zone_bouts.copy()
        zone_bouts['center_time'] = (zone_bouts['start_time'] + zone_bouts['end_time']) / 2

        # Sort by center time
        zone_bouts = zone_bouts.sort_values('center_time')

        # Find overlapping groups
        overlap_groups = []

        for idx, bout in zone_bouts.iterrows():
            # Check if this bout's center falls within any existing bout
            overlapping_bouts = zone_bouts[
                (zone_bouts['start_time'] <= bout['center_time']) &
                (zone_bouts['end_time'] >= bout['center_time'])
            ]

            if len(overlapping_bouts) > 0:
                # Create grouping event from overlapping bouts
                animals_present = set(overlapping_bouts['animal'].unique())

                # Use the overlap period
                overlap_start = overlapping_bouts['start_time'].max()
                overlap_end = overlapping_bouts['end_time'].min()

                # Only create event if there's actual overlap
                if overlap_end > overlap_start:
                    overlap_group = {
                        'animals': animals_present,
                        'start_time': overlap_start,
                        'end_time': overlap_end
                    }

                    # Check if this is a new unique group (avoid duplicates)
                    is_duplicate = False
                    for existing_group in overlap_groups:
                        if (existing_group['animals'] == animals_present and
                            abs(existing_group['start_time'] - overlap_start) < 0.1 and
                            abs(existing_group['end_time'] - overlap_end) < 0.1):
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        overlap_groups.append(overlap_group)

        return overlap_groups

    def _save_trial_gbi(
        self,
        trial_id: str,
        gbi_df: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Save GBI matrix for a trial.

        Args:
            trial_id: Trial identifier
            gbi_df: GBI DataFrame
            progress_callback: Optional callback function

        Returns:
            Path to output file
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{trial_id}_MOVEBOUT_GBI.csv"

        gbi_df.to_csv(output_path, index=False)

        return str(output_path)

    def get_gbi_statistics(self, gbi_dict: Dict[str, pd.DataFrame]) -> dict:
        """
        Calculate summary statistics for GBI matrices.

        Args:
            gbi_dict: Dictionary of GBI DataFrames by trial

        Returns:
            Dictionary of summary statistics
        """
        stats = {
            'num_trials': len(gbi_dict),
            'trial_stats': {}
        }

        for trial_id, gbi_df in gbi_dict.items():
            trial_stats = {
                'num_events': len(gbi_df),
                'group_size_stats': {
                    'mean': gbi_df['group_size'].mean() if len(gbi_df) > 0 else 0,
                    'median': gbi_df['group_size'].median() if len(gbi_df) > 0 else 0,
                    'min': gbi_df['group_size'].min() if len(gbi_df) > 0 else 0,
                    'max': gbi_df['group_size'].max() if len(gbi_df) > 0 else 0
                },
                'duration_stats': {
                    'mean': gbi_df['duration'].mean() if len(gbi_df) > 0 else 0,
                    'median': gbi_df['duration'].median() if len(gbi_df) > 0 else 0,
                    'total': gbi_df['duration'].sum() if len(gbi_df) > 0 else 0
                }
            }

            stats['trial_stats'][trial_id] = trial_stats

        return stats
