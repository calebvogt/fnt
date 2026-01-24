"""
File readers for RFID data.

Supports automatic detection and reading of RFID data in multiple formats:
- Excel (.xlsx)
- CSV (.csv)
- Plain text (.txt)
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
import re


def detect_file_format(filepath: str) -> str:
    """
    Detect file format from file extension.

    Args:
        filepath: Path to file

    Returns:
        Format string: 'xlsx', 'csv', or 'txt'

    Raises:
        ValueError: If file format is not supported
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    format_map = {
        '.xlsx': 'xlsx',
        '.xls': 'xlsx',
        '.csv': 'csv',
        '.txt': 'txt'
    }

    if suffix in format_map:
        return format_map[suffix]
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: {list(format_map.keys())}")


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to match expected format.

    Expected columns:
    - scan_date: Date of RFID read
    - scan_time: Time of RFID read
    - reader_id: Reader/paddock identifier
    - antenna_id: Antenna identifier
    - tag_id: RFID tag ID

    Args:
        df: DataFrame with RFID data

    Returns:
        DataFrame with standardized column names

    Raises:
        ValueError: If required columns cannot be mapped
    """
    df = df.copy()

    # Column name mapping (various possible names → standard name)
    column_mappings = {
        'scan_date': ['scan date', 'date', 'scan_date', 'scandate'],
        'scan_time': ['scan time', 'time', 'scan_time', 'scantime'],
        'reader_id': ['reader id', 'reader', 'reader_id', 'readerid', 'paddock'],
        'antenna_id': ['antenna id', 'antenna', 'antenna_id', 'antennaid', 'ant'],
        'tag_id': ['dec tag id', 'tag id', 'tag_id', 'tagid', 'tag', 'dec_tag_id']
    }

    # Create lowercase column mapping
    current_cols_lower = {col.lower(): col for col in df.columns}

    # Map columns
    rename_dict = {}
    for standard_name, possible_names in column_mappings.items():
        found = False
        for possible_name in possible_names:
            if possible_name.lower() in current_cols_lower:
                rename_dict[current_cols_lower[possible_name.lower()]] = standard_name
                found = True
                break

        if not found and standard_name not in df.columns:
            # Try partial matches
            for col in df.columns:
                col_lower = col.lower()
                for possible_name in possible_names:
                    if possible_name in col_lower:
                        rename_dict[col] = standard_name
                        found = True
                        break
                if found:
                    break

    # Apply renaming
    df = df.rename(columns=rename_dict)

    # Check for required columns
    required_cols = ['scan_date', 'scan_time', 'reader_id', 'antenna_id', 'tag_id']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Could not map required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def read_xlsx_file(filepath: str) -> pd.DataFrame:
    """
    Read RFID data from Excel file.

    Args:
        filepath: Path to Excel file

    Returns:
        DataFrame with standardized column names
    """
    df = pd.read_excel(filepath)
    df = standardize_column_names(df)
    return df


def read_csv_file(filepath: str) -> pd.DataFrame:
    """
    Read RFID data from CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with standardized column names
    """
    # Try different delimiters
    for delimiter in [',', '\t', ';', '|']:
        try:
            df = pd.read_csv(filepath, delimiter=delimiter)
            if len(df.columns) > 1:  # Valid multi-column file
                df = standardize_column_names(df)
                return df
        except Exception:
            continue

    raise ValueError(f"Could not read CSV file with common delimiters: {filepath}")


def read_txt_file(filepath: str) -> pd.DataFrame:
    """
    Read RFID data from plain text file.

    Attempts to detect delimiter and format automatically.

    Args:
        filepath: Path to text file

    Returns:
        DataFrame with standardized column names
    """
    # Try tab-delimited first (common for RFID readers)
    try:
        df = pd.read_csv(filepath, delimiter='\t')
        if len(df.columns) > 1:
            df = standardize_column_names(df)
            return df
    except Exception:
        pass

    # Try comma-delimited
    try:
        df = pd.read_csv(filepath, delimiter=',')
        if len(df.columns) > 1:
            df = standardize_column_names(df)
            return df
    except Exception:
        pass

    # Try whitespace-delimited
    try:
        df = pd.read_csv(filepath, delim_whitespace=True)
        if len(df.columns) > 1:
            df = standardize_column_names(df)
            return df
    except Exception:
        pass

    raise ValueError(f"Could not read text file with common delimiters: {filepath}")


def read_rfid_file(filepath: str, format: Optional[str] = None) -> pd.DataFrame:
    """
    Read RFID data file with automatic format detection.

    Args:
        filepath: Path to RFID data file
        format: File format ('xlsx', 'csv', 'txt'). If None, auto-detect from extension.

    Returns:
        DataFrame with standardized columns:
        - scan_date: Date of read
        - scan_time: Time of read
        - reader_id: Reader identifier
        - antenna_id: Antenna identifier
        - tag_id: Tag identifier

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported or cannot be parsed
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Auto-detect format if not specified
    if format is None:
        format = detect_file_format(str(filepath))

    # Read based on format
    if format == 'xlsx':
        df = read_xlsx_file(str(filepath))
    elif format == 'csv':
        df = read_csv_file(str(filepath))
    elif format == 'txt':
        df = read_txt_file(str(filepath))
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Convert tag_id to string (to handle long numeric IDs)
    df['tag_id'] = df['tag_id'].astype(str).str.strip()

    # Convert reader_id and antenna_id to int
    df['reader_id'] = pd.to_numeric(df['reader_id'], errors='coerce').astype('Int64')
    df['antenna_id'] = pd.to_numeric(df['antenna_id'], errors='coerce').astype('Int64')

    return df


def read_rfid_directory(
    directory_path: str,
    trial_id: str,
    file_pattern: str = '*',
    progress_callback=None
) -> pd.DataFrame:
    """
    Read all RFID files from a directory and concatenate.

    Args:
        directory_path: Path to directory containing RFID files
        trial_id: Trial identifier to add to data
        file_pattern: Glob pattern for files (default: all files)
        progress_callback: Optional callback function(message: str) for progress updates

    Returns:
        Concatenated DataFrame with trial column added

    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no valid files found
    """
    directory_path = Path(directory_path)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    # Find all matching files
    supported_extensions = ['.xlsx', '.xls', '.csv', '.txt']
    files = []

    for ext in supported_extensions:
        pattern = file_pattern if file_pattern.endswith(ext) else f"{file_pattern}{ext}"
        files.extend(directory_path.glob(pattern))

    # Remove duplicates and sort
    files = sorted(set(files))

    if not files:
        raise ValueError(f"No RFID files found in {directory_path} matching pattern '{file_pattern}'")

    # Read and concatenate all files
    dfs = []

    for i, file in enumerate(files):
        if progress_callback:
            progress_callback(f"Reading file {i+1}/{len(files)}: {file.name}")

        try:
            df = read_rfid_file(str(file))
            dfs.append(df)
        except Exception as e:
            # Log warning but continue with other files
            if progress_callback:
                progress_callback(f"Warning: Could not read {file.name}: {e}")

    if not dfs:
        raise ValueError(f"No valid RFID files could be read from {directory_path}")

    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Add trial column
    combined_df['trial'] = trial_id

    if progress_callback:
        progress_callback(f"Loaded {len(dfs)} files with {len(combined_df)} total reads for trial {trial_id}")

    return combined_df


def read_all_trials(
    base_directory: str,
    trial_ids: List[str],
    progress_callback=None
) -> pd.DataFrame:
    """
    Read RFID data from multiple trial directories.

    Expects directory structure:
    base_directory/
        trial_id_1/
            rfid_files...
        trial_id_2/
            rfid_files...

    Args:
        base_directory: Base directory containing trial subdirectories
        trial_ids: List of trial identifiers
        progress_callback: Optional callback function(message: str) for progress updates

    Returns:
        Concatenated DataFrame with all trials

    Raises:
        FileNotFoundError: If base directory doesn't exist
        ValueError: If no valid data found
    """
    base_directory = Path(base_directory)

    if not base_directory.exists():
        raise FileNotFoundError(f"Base directory not found: {base_directory}")

    all_trials_dfs = []

    for trial_id in trial_ids:
        if progress_callback:
            progress_callback(f"Reading trial: {trial_id}")

        trial_dir = base_directory / trial_id

        if not trial_dir.exists():
            if progress_callback:
                progress_callback(f"Warning: Trial directory not found: {trial_dir}")
            continue

        try:
            trial_df = read_rfid_directory(
                str(trial_dir),
                trial_id,
                progress_callback=progress_callback
            )
            all_trials_dfs.append(trial_df)
        except Exception as e:
            if progress_callback:
                progress_callback(f"Warning: Could not read trial {trial_id}: {e}")

    if not all_trials_dfs:
        raise ValueError(f"No valid trial data found in {base_directory}")

    # Concatenate all trials
    combined_df = pd.concat(all_trials_dfs, ignore_index=True)

    if progress_callback:
        progress_callback(f"Loaded {len(all_trials_dfs)} trials with {len(combined_df)} total reads")

    return combined_df
