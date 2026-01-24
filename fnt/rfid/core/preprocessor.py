"""
RFID Preprocessor - Stage 1 of pipeline.

Processes raw RFID data files and creates consolidated dataset with:
- Animal identity mapping via metadata
- Zone and coordinate assignments
- Temporal variables (field_time, noon_day, time_sec)
- Cross-paddock read filtering

Equivalent to R script: 1_create_ALLTRIAL_RFID.R
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable
import os

from ..config import RFIDConfig
from .file_readers import read_all_trials
from .utils import (
    parse_metadata,
    create_tag_mapping,
    assign_zone_info,
    create_temporal_variables,
    filter_cross_paddock_reads
)


class RFIDPreprocessor:
    """
    Preprocessor for raw RFID data.

    Converts raw RFID files into consolidated dataset with animal identities,
    spatial assignments, and temporal variables.
    """

    def __init__(self, config: RFIDConfig):
        """
        Initialize preprocessor with configuration.

        Args:
            config: RFID configuration object
        """
        self.config = config
        self.metadata_df = None
        self.tag_mapping = None

    def load_metadata(self, progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Load and parse metadata file.

        Args:
            progress_callback: Optional callback function(message: str)

        Returns:
            Metadata DataFrame

        Raises:
            FileNotFoundError: If metadata file doesn't exist
            ValueError: If metadata is invalid
        """
        if progress_callback:
            progress_callback("Loading metadata...")

        self.metadata_df = parse_metadata(
            self.config.metadata_file_path,
            tag_columns=self.config.tag_columns,
            strain_prefixes=self.config.strain_prefixes
        )

        # Create tag-to-animal mapping
        self.tag_mapping = create_tag_mapping(
            self.metadata_df,
            tag_columns=self.config.tag_columns
        )

        if progress_callback:
            progress_callback(f"Loaded metadata for {len(self.metadata_df)} animals")

        return self.metadata_df

    def process_raw_rfid(self, progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Process raw RFID data files through complete pipeline.

        Steps:
        1. Load metadata and create tag mappings
        2. Read all trial RFID files
        3. Merge with metadata to link tags to animals
        4. Assign zone IDs and coordinates
        5. Create temporal variables
        6. Filter cross-paddock reads
        7. Save output file(s)

        Args:
            progress_callback: Optional callback function(message: str)

        Returns:
            Processed RFID DataFrame

        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If processing fails
        """
        # Step 1: Load metadata
        if progress_callback:
            progress_callback("Step 1/7: Loading metadata...")

        if self.metadata_df is None:
            self.load_metadata(progress_callback)

        # Step 2: Read all RFID files
        if progress_callback:
            progress_callback("Step 2/7: Reading RFID files...")

        rfid_df = read_all_trials(
            self.config.input_dir,
            self.config.trial_ids,
            progress_callback=progress_callback
        )

        if progress_callback:
            progress_callback(f"Loaded {len(rfid_df):,} RFID reads")

        # Step 3: Merge with metadata to link tags to animals
        if progress_callback:
            progress_callback("Step 3/7: Merging with metadata...")

        rfid_df = self._merge_with_metadata(rfid_df, progress_callback)

        # Step 4: Assign zone IDs and coordinates
        if progress_callback:
            progress_callback("Step 4/7: Assigning zones and coordinates...")

        rfid_df = assign_zone_info(
            rfid_df,
            self.config.antenna_zone_map,
            self.config.zone_coordinates,
            antenna_col='antenna_id'
        )

        # Step 5: Create temporal variables
        if progress_callback:
            progress_callback("Step 5/7: Creating temporal variables...")

        rfid_df = create_temporal_variables(
            rfid_df,
            scan_date_col='scan_date',
            scan_time_col='scan_time',
            day_origin_time=self.config.day_origin_time,
            trial_col='trial'
        )

        # Step 6: Filter cross-paddock reads
        if progress_callback:
            progress_callback("Step 6/7: Filtering cross-paddock reads...")

        initial_count = len(rfid_df)
        rfid_df = filter_cross_paddock_reads(rfid_df, self.config.trial_reader_map)
        filtered_count = initial_count - len(rfid_df)

        if progress_callback:
            progress_callback(f"Filtered {filtered_count:,} cross-paddock reads")

        # Step 7: Save output
        if progress_callback:
            progress_callback("Step 7/7: Saving output...")

        output_path = self._save_output(rfid_df, progress_callback)

        if progress_callback:
            progress_callback(f"Saved output to: {output_path}")

        return rfid_df

    def _merge_with_metadata(
        self,
        rfid_df: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Merge RFID data with metadata to add animal identities.

        Args:
            rfid_df: Raw RFID DataFrame
            progress_callback: Optional callback function

        Returns:
            Merged DataFrame with animal information
        """
        # Merge RFID data with tag mapping
        merged_df = rfid_df.merge(
            self.tag_mapping,
            on=['tag_id', 'trial'],
            how='left'
        )

        # Check for unmatched tags
        unmatched = merged_df[merged_df['name'].isna()]

        if len(unmatched) > 0:
            unmatched_tags = unmatched['tag_id'].unique()
            if progress_callback:
                progress_callback(
                    f"Warning: {len(unmatched):,} reads from {len(unmatched_tags)} unmatched tags"
                )

        return merged_df

    def _save_output(
        self,
        df: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Save processed RFID data to file(s).

        Implements chunked saving for large datasets.

        Args:
            df: Processed RFID DataFrame
            progress_callback: Optional callback function

        Returns:
            Path to output file (or first chunk if chunked)
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_filename = "ALLTRIAL_RFID_DATA"
        output_path = output_dir / f"{base_filename}.csv"

        # Estimate file size (rough: 100 bytes per row)
        estimated_size_mb = len(df) * 100 / (1024 * 1024)
        max_size_mb = 100  # Maximum file size before chunking

        if estimated_size_mb > max_size_mb:
            # Save in chunks
            if progress_callback:
                progress_callback(f"Large dataset detected ({estimated_size_mb:.1f} MB). Saving in chunks...")

            chunk_size = int(len(df) * max_size_mb / estimated_size_mb)
            num_chunks = int(np.ceil(len(df) / chunk_size))

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(df))
                chunk_df = df.iloc[start_idx:end_idx]

                chunk_filename = f"{base_filename}_part{i+1:02d}.csv"
                chunk_path = output_dir / chunk_filename

                chunk_df.to_csv(chunk_path, index=False)

                if progress_callback:
                    progress_callback(f"Saved chunk {i+1}/{num_chunks}: {chunk_filename}")

            return str(output_dir / f"{base_filename}_part01.csv")
        else:
            # Save as single file
            df.to_csv(output_path, index=False)

            if progress_callback:
                progress_callback(f"Saved {len(df):,} rows to {output_path.name}")

            return str(output_path)

    def get_summary_statistics(self, df: pd.DataFrame) -> dict:
        """
        Calculate summary statistics for processed RFID data.

        Args:
            df: Processed RFID DataFrame

        Returns:
            Dictionary of summary statistics
        """
        stats = {
            'total_reads': len(df),
            'num_trials': df['trial'].nunique(),
            'num_animals': df['name'].nunique() if 'name' in df.columns else 0,
            'num_zones': df['zone_id'].nunique() if 'zone_id' in df.columns else 0,
            'date_range': (
                df['field_time'].min(),
                df['field_time'].max()
            ) if 'field_time' in df.columns else (None, None),
            'day_range': (
                df['noon_day'].min(),
                df['noon_day'].max()
            ) if 'noon_day' in df.columns else (None, None)
        }

        # Per-trial statistics
        if 'trial' in df.columns:
            stats['reads_per_trial'] = df.groupby('trial').size().to_dict()

        # Per-animal statistics
        if 'name' in df.columns:
            stats['reads_per_animal'] = df.groupby('name').size().to_dict()

        return stats
