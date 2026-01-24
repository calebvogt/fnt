"""
Edgelist Generator - Stage 3c of pipeline.

Creates dyadic interaction records from GBI matrices.

Equivalent to R script: 3b_create_ALLTRIAL_MOVEBOUT_GBI_edgelist.R
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict

from ..config import RFIDConfig


class EdgelistGenerator:
    """
    Generator for dyadic edgelists from GBI matrices.
    """

    def __init__(self, config: RFIDConfig):
        """
        Initialize edgelist generator with configuration.

        Args:
            config: RFID configuration object
        """
        self.config = config

    def create_edgelist(
        self,
        gbi_dict: Dict[str, pd.DataFrame],
        movebout_df: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Create edgelist from GBI matrices.

        Args:
            gbi_dict: Dictionary mapping trial_id to GBI DataFrame
            movebout_df: Movement bout DataFrame
            progress_callback: Optional callback function(message: str)

        Returns:
            Edgelist DataFrame with dyadic interactions
        """
        if progress_callback:
            progress_callback("Creating edgelist...")

        all_edges = []

        for trial_id, gbi_df in gbi_dict.items():
            if progress_callback:
                progress_callback(f"Processing edgelist for trial: {trial_id}")

            if len(gbi_df) == 0:
                continue

            # Get animal columns
            metadata_cols = ['zone_id', 'start_time', 'end_time', 'center_time',
                            'duration', 'group_size', 'm_sum', 'f_sum', 'mf_sum']
            animal_cols = [col for col in gbi_df.columns if col not in metadata_cols]

            # Process each grouping event
            for idx, row in gbi_df.iterrows():
                # Find animals present in this event
                animals_present = [col for col in animal_cols if row[col] == 1]

                # Create edges for all pairs
                for i in range(len(animals_present)):
                    for j in range(i+1, len(animals_present)):
                        edge = {
                            'trial': trial_id,
                            'animal1': animals_present[i],
                            'animal2': animals_present[j],
                            'zone_id': row['zone_id'],
                            'start_time': row['start_time'],
                            'end_time': row['end_time'],
                            'duration': row['duration']
                        }
                        all_edges.append(edge)

        # Convert to DataFrame
        if all_edges:
            edgelist_df = pd.DataFrame(all_edges)

            # Concatenate continuous interactions (same dyad, same zone)
            edgelist_df = self._concatenate_continuous_interactions(edgelist_df)

            # Save output
            output_path = self._save_output(edgelist_df, progress_callback)

            if progress_callback:
                progress_callback(f"Created {len(edgelist_df)} dyadic interactions")

            return edgelist_df
        else:
            return pd.DataFrame()

    def _concatenate_continuous_interactions(self, edgelist_df: pd.DataFrame) -> pd.DataFrame:
        """
        Concatenate continuous co-occurrence events.

        Args:
            edgelist_df: Raw edgelist DataFrame

        Returns:
            Edgelist with continuous interactions merged
        """
        if len(edgelist_df) == 0:
            return edgelist_df

        # Sort by dyad, zone, and time
        edgelist_df = edgelist_df.sort_values(['trial', 'animal1', 'animal2', 'zone_id', 'start_time'])

        # Group by dyad and zone
        grouped = edgelist_df.groupby(['trial', 'animal1', 'animal2', 'zone_id'])

        concatenated = []

        for (trial, animal1, animal2, zone), group in grouped:
            group = group.sort_values('start_time').reset_index(drop=True)

            if len(group) == 1:
                concatenated.append(group.iloc[0])
                continue

            # Merge continuous interactions
            current_start = group.iloc[0]['start_time']
            current_end = group.iloc[0]['end_time']

            for i in range(1, len(group)):
                next_start = group.iloc[i]['start_time']
                next_end = group.iloc[i]['end_time']

                # If next interaction starts before current ends, extend
                if next_start <= current_end:
                    current_end = max(current_end, next_end)
                else:
                    # Save current interaction
                    concatenated.append({
                        'trial': trial,
                        'animal1': animal1,
                        'animal2': animal2,
                        'zone_id': zone,
                        'start_time': current_start,
                        'end_time': current_end,
                        'duration': current_end - current_start
                    })

                    # Start new interaction
                    current_start = next_start
                    current_end = next_end

            # Save last interaction
            concatenated.append({
                'trial': trial,
                'animal1': animal1,
                'animal2': animal2,
                'zone_id': zone,
                'start_time': current_start,
                'end_time': current_end,
                'duration': current_end - current_start
            })

        return pd.DataFrame(concatenated)

    def _save_output(
        self,
        edgelist_df: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Save edgelist to file.

        Args:
            edgelist_df: Edgelist DataFrame
            progress_callback: Optional callback function

        Returns:
            Path to output file
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "ALLTRIAL_MOVEBOUT_GBI_edgelist.csv"

        edgelist_df.to_csv(output_path, index=False)

        return str(output_path)
