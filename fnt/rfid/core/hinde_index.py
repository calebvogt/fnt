"""
Hinde Index Calculator - Stage 3e-f of pipeline.

Calculates Hinde indices and summary statistics:
- Broad: All contact events
- Narrow: Only 2-individual contact events
- Summary: Individual-level movement statistics

Equivalent to R scripts: 3d, 3e, 3f
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple

from ..config import RFIDConfig


class HindeIndexCalculator:
    """
    Calculator for Hinde indices and summary statistics.
    """

    def __init__(self, config: RFIDConfig):
        """
        Initialize Hinde index calculator with configuration.

        Args:
            config: RFID configuration object
        """
        self.config = config

    def calculate_hinde_indices(
        self,
        gbi_dict: Dict[str, pd.DataFrame],
        movebout_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Calculate Hinde indices and summary statistics.

        Args:
            gbi_dict: Dictionary mapping trial_id to GBI DataFrame
            movebout_df: Movement bout DataFrame
            metadata_df: Metadata DataFrame
            progress_callback: Optional callback function(message: str)

        Returns:
            Tuple of (hinde_broad_df, hinde_narrow_df, summary_df)
        """
        if progress_callback:
            progress_callback("Calculating Hinde indices...")

        # Calculate broad index (all contacts)
        hinde_broad = self._calculate_broad_index(gbi_dict, progress_callback)

        # Calculate narrow index (2-individual contacts only)
        hinde_narrow = self._calculate_narrow_index(gbi_dict, progress_callback)

        # Calculate summary statistics
        summary = self._calculate_summary(movebout_df, metadata_df, progress_callback)

        # Save outputs
        self._save_outputs(hinde_broad, hinde_narrow, summary, progress_callback)

        return hinde_broad, hinde_narrow, summary

    def _calculate_broad_index(
        self,
        gbi_dict: Dict[str, pd.DataFrame],
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Calculate broad Hinde index (all contact events).

        Args:
            gbi_dict: Dictionary of GBI DataFrames
            progress_callback: Optional callback function

        Returns:
            DataFrame with broad Hinde indices
        """
        if progress_callback:
            progress_callback("Calculating broad Hinde index...")

        all_contacts = []

        for trial_id, gbi_df in gbi_dict.items():
            if len(gbi_df) == 0:
                continue

            # Get animal columns
            metadata_cols = ['zone_id', 'start_time', 'end_time', 'center_time',
                            'duration', 'group_size', 'm_sum', 'f_sum', 'mf_sum']
            animal_cols = [col for col in gbi_df.columns if col not in metadata_cols]

            # Process each event
            for idx, row in gbi_df.iterrows():
                animals_present = [col for col in animal_cols if row[col] == 1]

                # Count all contacts (all group sizes)
                for i in range(len(animals_present)):
                    for j in range(i+1, len(animals_present)):
                        contact = {
                            'trial': trial_id,
                            'animal1': animals_present[i],
                            'animal2': animals_present[j],
                            'zone_id': row['zone_id'],
                            'time': row['center_time'],
                            'duration': row['duration'],
                            'group_size': row['group_size']
                        }
                        all_contacts.append(contact)

        return pd.DataFrame(all_contacts) if all_contacts else pd.DataFrame()

    def _calculate_narrow_index(
        self,
        gbi_dict: Dict[str, pd.DataFrame],
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Calculate narrow Hinde index (only 2-individual contacts).

        Args:
            gbi_dict: Dictionary of GBI DataFrames
            progress_callback: Optional callback function

        Returns:
            DataFrame with narrow Hinde indices
        """
        if progress_callback:
            progress_callback("Calculating narrow Hinde index...")

        narrow_contacts = []

        for trial_id, gbi_df in gbi_dict.items():
            if len(gbi_df) == 0:
                continue

            # Filter for group size = 2
            dyads = gbi_df[gbi_df['group_size'] == 2]

            # Get animal columns
            metadata_cols = ['zone_id', 'start_time', 'end_time', 'center_time',
                            'duration', 'group_size', 'm_sum', 'f_sum', 'mf_sum']
            animal_cols = [col for col in dyads.columns if col not in metadata_cols]

            # Process each dyadic event
            for idx, row in dyads.iterrows():
                animals_present = [col for col in animal_cols if row[col] == 1]

                if len(animals_present) == 2:
                    contact = {
                        'trial': trial_id,
                        'animal1': animals_present[0],
                        'animal2': animals_present[1],
                        'zone_id': row['zone_id'],
                        'time': row['center_time'],
                        'duration': row['duration']
                    }
                    narrow_contacts.append(contact)

        return pd.DataFrame(narrow_contacts) if narrow_contacts else pd.DataFrame()

    def _calculate_summary(
        self,
        movebout_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Calculate individual-level summary statistics.

        Args:
            movebout_df: Movement bout DataFrame
            metadata_df: Metadata DataFrame
            progress_callback: Optional callback function

        Returns:
            DataFrame with individual summaries
        """
        if progress_callback:
            progress_callback("Calculating summary statistics...")

        # Get bout-level data (START reads)
        bout_starts = movebout_df[movebout_df['bout_type'] == 'START'].copy()

        if len(bout_starts) == 0:
            return pd.DataFrame()

        # Calculate per-animal statistics
        summary_stats = []

        for (trial, animal), group in bout_starts.groupby(['trial', 'name']):
            stats = {
                'trial': trial,
                'name': animal,
                'num_bouts': len(group),
                'total_time': group['bout_duration'].sum(),
                'mean_bout_duration': group['bout_duration'].mean(),
                'median_bout_duration': group['bout_duration'].median(),
                'num_zones_visited': group['zone_id'].nunique()
            }

            # Add zone-specific counts
            for zone in range(1, self.config.num_zones + 1):
                zone_bouts = group[group['zone_id'] == zone]
                stats[f'zone_{zone}_bouts'] = len(zone_bouts)
                stats[f'zone_{zone}_time'] = zone_bouts['bout_duration'].sum()

            summary_stats.append(stats)

        summary_df = pd.DataFrame(summary_stats)

        # Merge with metadata
        summary_df = summary_df.merge(
            metadata_df[['trial', 'name', 'sex', 'phase']],
            on=['trial', 'name'],
            how='left'
        )

        return summary_df

    def _save_outputs(
        self,
        hinde_broad: pd.DataFrame,
        hinde_narrow: pd.DataFrame,
        summary: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> None:
        """
        Save Hinde index outputs.

        Args:
            hinde_broad: Broad Hinde index DataFrame
            hinde_narrow: Narrow Hinde index DataFrame
            summary: Summary statistics DataFrame
            progress_callback: Optional callback function
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save broad index
        if len(hinde_broad) > 0:
            broad_path = output_dir / "ALLTRIAL_MOVEBOUT_GBI_hinde_broad.csv"
            hinde_broad.to_csv(broad_path, index=False)

            if progress_callback:
                progress_callback(f"Saved broad Hinde index: {len(hinde_broad)} contacts")

        # Save narrow index
        if len(hinde_narrow) > 0:
            narrow_path = output_dir / "ALLTRIAL_MOVEBOUT_GBI_hinde_narrow.csv"
            hinde_narrow.to_csv(narrow_path, index=False)

            if progress_callback:
                progress_callback(f"Saved narrow Hinde index: {len(hinde_narrow)} dyadic contacts")

        # Save summary
        if len(summary) > 0:
            summary_path = output_dir / "ALLTRIAL_MOVEBOUT_GBI_summary.csv"
            summary.to_csv(summary_path, index=False)

            if progress_callback:
                progress_callback(f"Saved summary statistics: {len(summary)} animals")
