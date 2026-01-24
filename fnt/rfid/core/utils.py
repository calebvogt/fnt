"""
Utility functions for RFID preprocessing.

Provides helper functions for antenna mapping, coordinate assignment,
metadata parsing, and temporal variable creation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


def map_antenna_to_zone(antenna_id: int, antenna_zone_map: Dict[int, int]) -> Optional[int]:
    """
    Map antenna ID to zone ID.

    Args:
        antenna_id: Antenna identifier
        antenna_zone_map: Dictionary mapping antenna IDs to zone IDs

    Returns:
        Zone ID, or None if antenna_id not in map
    """
    return antenna_zone_map.get(antenna_id)


def get_zone_coordinates(zone_id: int, zone_coords: List[Dict]) -> Optional[Tuple[float, float, str]]:
    """
    Get coordinates for a zone.

    Args:
        zone_id: Zone identifier
        zone_coords: List of zone coordinate dictionaries

    Returns:
        Tuple of (x, y, location) or None if zone not found
    """
    for zone_info in zone_coords:
        if zone_info['zone'] == zone_id:
            return (zone_info['x'], zone_info['y'], zone_info['location'])
    return None


def parse_metadata(
    metadata_path: str,
    tag_columns: List[str] = None,
    strain_prefixes: List[str] = None
) -> pd.DataFrame:
    """
    Parse metadata file and create tag-to-animal mapping.

    Args:
        metadata_path: Path to metadata Excel or CSV file
        tag_columns: List of column names containing tag IDs (default: ["tag_1", "tag_2"])
        strain_prefixes: List of strain prefixes to remove from names (default: ["OB-M-", "OB-F-"])

    Returns:
        DataFrame with metadata and tag-to-animal mapping

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        ValueError: If required columns are missing
    """
    if tag_columns is None:
        tag_columns = ["tag_1", "tag_2"]

    if strain_prefixes is None:
        strain_prefixes = ["OB-M-", "OB-F-"]

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Read metadata file (try Excel first, then CSV)
    try:
        # Try reading as Excel
        metadata_df = pd.read_excel(metadata_path)
    except Exception:
        try:
            # Try reading as CSV
            metadata_df = pd.read_csv(metadata_path)
        except Exception as e:
            raise ValueError(f"Could not read metadata file: {e}")

    # Check for required columns
    required_cols = ['trial', 'name', 'sex'] + tag_columns
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]

    if missing_cols:
        raise ValueError(f"Metadata file missing required columns: {missing_cols}")

    # Remove strain prefixes from names if present
    if 'name' in metadata_df.columns:
        for prefix in strain_prefixes:
            metadata_df['name'] = metadata_df['name'].str.replace(prefix, '', regex=False)

    # Create sex-phase column if phase exists
    if 'phase' in metadata_df.columns:
        metadata_df['sex_phase'] = metadata_df['sex'] + '-' + metadata_df['phase']

    return metadata_df


def create_tag_mapping(metadata_df: pd.DataFrame, tag_columns: List[str]) -> pd.DataFrame:
    """
    Create a mapping table from tag IDs to animal information.

    Handles dual-tag system where each animal has multiple tags.

    Args:
        metadata_df: Metadata DataFrame
        tag_columns: List of column names containing tag IDs

    Returns:
        DataFrame with columns: tag_id, trial, name, sex, (other metadata columns)
    """
    tag_mapping_dfs = []

    for tag_col in tag_columns:
        if tag_col in metadata_df.columns:
            # Create mapping for this tag column
            tag_df = metadata_df.copy()
            tag_df = tag_df.rename(columns={tag_col: 'tag_id'})

            # Keep only rows where tag_id is not null
            tag_df = tag_df[tag_df['tag_id'].notna()]

            # Remove other tag columns
            cols_to_drop = [col for col in tag_columns if col != tag_col and col in tag_df.columns]
            tag_df = tag_df.drop(columns=cols_to_drop)

            tag_mapping_dfs.append(tag_df)

    # Combine all tag mappings
    if tag_mapping_dfs:
        tag_mapping = pd.concat(tag_mapping_dfs, ignore_index=True)

        # Convert tag_id to string to handle potential formatting issues
        tag_mapping['tag_id'] = tag_mapping['tag_id'].astype(str).str.strip()

        # Remove duplicates (same tag might appear in multiple columns)
        tag_mapping = tag_mapping.drop_duplicates(subset=['tag_id', 'trial'])

        return tag_mapping
    else:
        raise ValueError("No valid tag columns found in metadata")


def filter_cross_paddock_reads(df: pd.DataFrame, trial_reader_map: Dict[str, int]) -> pd.DataFrame:
    """
    Filter out cross-paddock reads (reads from wrong reader for a trial).

    Args:
        df: DataFrame with 'trial' and 'reader_id' columns
        trial_reader_map: Dictionary mapping trial IDs to correct reader IDs

    Returns:
        Filtered DataFrame with only valid reads
    """
    if 'trial' not in df.columns or 'reader_id' not in df.columns:
        raise ValueError("DataFrame must have 'trial' and 'reader_id' columns")

    # Create filter conditions
    valid_reads = pd.Series([True] * len(df), index=df.index)

    for trial_id, correct_reader_id in trial_reader_map.items():
        # For this trial, keep only reads from the correct reader
        trial_mask = df['trial'] == trial_id
        reader_mask = df['reader_id'] == correct_reader_id

        # Mark invalid reads (wrong reader for this trial)
        valid_reads = valid_reads & (~trial_mask | reader_mask)

    return df[valid_reads].copy()


def create_temporal_variables(
    df: pd.DataFrame,
    scan_date_col: str = 'scan_date',
    scan_time_col: str = 'scan_time',
    day_origin_time: str = "12:00:00",
    trial_col: str = 'trial'
) -> pd.DataFrame:
    """
    Create temporal variables for RFID data.

    Adds:
    - field_time: Combined date/time as datetime
    - noon_day: Day number based on day_origin_time (24hr periods starting at origin)
    - time_sec: Seconds from trial start

    Args:
        df: DataFrame with date and time columns
        scan_date_col: Name of date column
        scan_time_col: Name of time column
        day_origin_time: Time of day boundaries (default "12:00:00" for noon)
        trial_col: Name of trial column

    Returns:
        DataFrame with added temporal variables
    """
    df = df.copy()

    # Create field_time (combined datetime)
    if scan_date_col in df.columns and scan_time_col in df.columns:
        # Handle different date/time formats
        df['field_time'] = pd.to_datetime(
            df[scan_date_col].astype(str) + ' ' + df[scan_time_col].astype(str),
            errors='coerce'
        )
    else:
        raise ValueError(f"Columns {scan_date_col} and {scan_time_col} required")

    # Parse day origin time
    origin_hour, origin_min, origin_sec = map(int, day_origin_time.split(':'))

    # Calculate noon_day (day number based on origin time)
    # Day boundaries are at the origin time (e.g., noon)
    def calculate_noon_day(timestamp):
        if pd.isna(timestamp):
            return np.nan

        # Get the date of the timestamp
        date = timestamp.date()

        # Create the origin datetime for this date
        origin_dt = datetime.combine(date, datetime.min.time().replace(
            hour=origin_hour, minute=origin_min, second=origin_sec
        ))

        # If timestamp is before origin time, it belongs to previous day
        if timestamp < origin_dt:
            origin_dt = origin_dt - timedelta(days=1)

        # Find the first origin datetime for this trial
        return origin_dt

    # Group by trial to calculate trial-specific day numbers
    if trial_col in df.columns:
        df_with_days = []

        for trial_id in df[trial_col].unique():
            trial_df = df[df[trial_col] == trial_id].copy()

            # Get first occurrence time for this trial
            first_time = trial_df['field_time'].min()

            if pd.isna(first_time):
                trial_df['noon_day'] = np.nan
                trial_df['time_sec'] = np.nan
                df_with_days.append(trial_df)
                continue

            # Calculate the first day origin for this trial
            first_date = first_time.date()
            first_origin = datetime.combine(first_date, datetime.min.time().replace(
                hour=origin_hour, minute=origin_min, second=origin_sec
            ))

            # If first time is before origin, use previous day's origin
            if first_time < first_origin:
                first_origin = first_origin - timedelta(days=1)

            # Calculate noon_day: number of days since first origin
            trial_df['noon_day'] = trial_df['field_time'].apply(
                lambda x: np.floor((x - first_origin).total_seconds() / 86400) + 1 if pd.notna(x) else np.nan
            )

            # Calculate time_sec: seconds from first occurrence
            trial_df['time_sec'] = (trial_df['field_time'] - first_time).dt.total_seconds()

            df_with_days.append(trial_df)

        df = pd.concat(df_with_days, ignore_index=True)
    else:
        # No trial column, calculate for entire dataset
        first_time = df['field_time'].min()

        if not pd.isna(first_time):
            first_date = first_time.date()
            first_origin = datetime.combine(first_date, datetime.min.time().replace(
                hour=origin_hour, minute=origin_min, second=origin_sec
            ))

            if first_time < first_origin:
                first_origin = first_origin - timedelta(days=1)

            df['noon_day'] = df['field_time'].apply(
                lambda x: np.floor((x - first_origin).total_seconds() / 86400) + 1 if pd.notna(x) else np.nan
            )

            df['time_sec'] = (df['field_time'] - first_time).dt.total_seconds()
        else:
            df['noon_day'] = np.nan
            df['time_sec'] = np.nan

    return df


def assign_zone_info(
    df: pd.DataFrame,
    antenna_zone_map: Dict[int, int],
    zone_coords: List[Dict],
    antenna_col: str = 'antenna_id'
) -> pd.DataFrame:
    """
    Assign zone IDs and coordinates to RFID reads based on antenna ID.

    Args:
        df: DataFrame with antenna_id column
        antenna_zone_map: Dictionary mapping antenna IDs to zone IDs
        zone_coords: List of zone coordinate dictionaries
        antenna_col: Name of antenna column

    Returns:
        DataFrame with added columns: zone_id, x, y, location
    """
    df = df.copy()

    # Map antenna to zone
    df['zone_id'] = df[antenna_col].map(antenna_zone_map)

    # Create zone coordinates lookup
    zone_coords_dict = {
        zc['zone']: (zc['x'], zc['y'], zc['location'])
        for zc in zone_coords
    }

    # Assign coordinates
    df[['x', 'y', 'location']] = df['zone_id'].apply(
        lambda z: pd.Series(zone_coords_dict.get(z, (np.nan, np.nan, '')))
    )

    return df
