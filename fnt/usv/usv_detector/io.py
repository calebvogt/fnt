"""
I/O utilities for USV detection results.

Provides functions for saving and loading USV annotations in DAS-compatible
CSV format, as well as generating summary statistics.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime


def save_das_format(
    calls: List[Dict],
    output_path: str,
    include_extended: bool = True
) -> str:
    """
    Save detected calls in DAS-compatible CSV format.

    The DAS format has columns: start_seconds, stop_seconds, name
    Extended format adds: peak_freq_hz, mean_freq_hz, duration_ms, etc.

    Args:
        calls: List of call dictionaries
        output_path: Path for output CSV file
        include_extended: Whether to include extended columns

    Returns:
        Path to saved file
    """
    if not calls:
        # Create empty file with headers
        df = pd.DataFrame(columns=['start_seconds', 'stop_seconds', 'name'])
    else:
        df = pd.DataFrame(calls)

    # Ensure required DAS columns exist
    required_cols = ['start_seconds', 'stop_seconds', 'name']
    for col in required_cols:
        if col not in df.columns:
            df[col] = '' if col == 'name' else 0.0

    # Define column order
    if include_extended:
        # Extended format with additional features
        col_order = [
            'start_seconds', 'stop_seconds', 'name',
            'peak_freq_hz', 'mean_freq_hz', 'freq_bandwidth_hz',
            'duration_ms', 'mean_power_db', 'max_power_db'
        ]
    else:
        # Basic DAS format
        col_order = ['start_seconds', 'stop_seconds', 'name']

    # Keep only columns that exist
    col_order = [c for c in col_order if c in df.columns]

    # Add any remaining columns not in the order list
    remaining = [c for c in df.columns if c not in col_order and not c.startswith('_')]
    col_order.extend(remaining)

    df = df[col_order]

    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(6)

    # Save to CSV
    df.to_csv(output_path, index=False)

    return output_path


def load_das_annotations(filepath: str) -> pd.DataFrame:
    """
    Load DAS annotation CSV file.

    Args:
        filepath: Path to annotation CSV

    Returns:
        DataFrame with annotations
    """
    df = pd.read_csv(filepath)

    # Calculate duration if not present
    if 'duration_ms' not in df.columns and 'start_seconds' in df.columns and 'stop_seconds' in df.columns:
        df['duration_ms'] = (df['stop_seconds'] - df['start_seconds']) * 1000

    return df


def generate_summary(calls: List[Dict]) -> Dict:
    """
    Generate summary statistics from detected calls.

    Args:
        calls: List of call dictionaries

    Returns:
        Dictionary with summary statistics
    """
    if not calls:
        return {
            'total_calls': 0,
            'total_duration_s': 0,
            'mean_duration_ms': 0,
            'std_duration_ms': 0,
            'calls_per_minute': 0,
        }

    df = pd.DataFrame(calls)

    # Basic statistics
    summary = {
        'total_calls': len(df),
        'total_duration_s': df['duration_ms'].sum() / 1000 if 'duration_ms' in df.columns else 0,
        'mean_duration_ms': df['duration_ms'].mean() if 'duration_ms' in df.columns else 0,
        'std_duration_ms': df['duration_ms'].std() if 'duration_ms' in df.columns else 0,
    }

    # Frequency statistics
    if 'peak_freq_hz' in df.columns:
        summary['mean_peak_freq_hz'] = df['peak_freq_hz'].mean()
        summary['min_peak_freq_hz'] = df['peak_freq_hz'].min()
        summary['max_peak_freq_hz'] = df['peak_freq_hz'].max()

    # Power statistics
    if 'max_power_db' in df.columns:
        summary['mean_max_power_db'] = df['max_power_db'].mean()

    # Call type distribution
    if 'name' in df.columns:
        for call_type, count in df['name'].value_counts().items():
            summary[f'{call_type}_count'] = int(count)
            summary[f'{call_type}_percentage'] = float(count / len(df) * 100)

    # Temporal statistics
    if len(df) > 1 and 'start_seconds' in df.columns:
        df_sorted = df.sort_values('start_seconds')
        time_span = df_sorted['stop_seconds'].max() - df_sorted['start_seconds'].min()
        if time_span > 0:
            summary['calls_per_minute'] = len(df) / (time_span / 60)

            # Inter-call intervals
            intervals = df_sorted['start_seconds'].diff().dropna()
            summary['mean_interval_s'] = intervals.mean()
            summary['median_interval_s'] = intervals.median()

    return summary


def generate_batch_summary(
    results: List[Dict],
    output_path: str
) -> pd.DataFrame:
    """
    Generate summary across multiple files.

    Args:
        results: List of result dictionaries from batch processing
        output_path: Path for output summary CSV

    Returns:
        DataFrame with per-file summaries
    """
    summaries = []

    for result in results:
        if 'error' in result:
            summaries.append({
                'filename': os.path.basename(result.get('input_file', 'unknown')),
                'status': 'error',
                'error': result['error'],
            })
        else:
            file_summary = result.get('summary', {})
            file_summary['filename'] = os.path.basename(result.get('input_file', 'unknown'))
            file_summary['status'] = 'success'
            file_summary['output_file'] = os.path.basename(result.get('output_file', ''))
            summaries.append(file_summary)

    df = pd.DataFrame(summaries)

    # Reorder columns
    priority_cols = ['filename', 'status', 'total_calls', 'calls_per_minute',
                     'mean_duration_ms', 'mean_peak_freq_hz']
    col_order = [c for c in priority_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in col_order]
    df = df[col_order + remaining]

    # Save to CSV
    df.to_csv(output_path, index=False)

    return df


def merge_annotations(
    annotation_files: List[str],
    output_path: str,
    add_source_column: bool = True
) -> pd.DataFrame:
    """
    Merge multiple annotation files into one.

    Args:
        annotation_files: List of annotation CSV paths
        output_path: Path for merged output
        add_source_column: Whether to add a column indicating source file

    Returns:
        Merged DataFrame
    """
    dfs = []

    for filepath in annotation_files:
        try:
            df = load_das_annotations(filepath)
            if add_source_column:
                df['source_file'] = os.path.basename(filepath)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")

    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(output_path, index=False)

    return merged


def export_for_praat(
    calls: List[Dict],
    output_path: str,
    tier_name: str = "USV"
) -> str:
    """
    Export annotations in Praat TextGrid format.

    Args:
        calls: List of call dictionaries
        output_path: Path for output TextGrid file
        tier_name: Name for the annotation tier

    Returns:
        Path to saved file
    """
    if not calls:
        xmin, xmax = 0, 1
    else:
        xmin = min(c['start_seconds'] for c in calls)
        xmax = max(c['stop_seconds'] for c in calls)

    # Build TextGrid content
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        '',
        f'xmin = {xmin}',
        f'xmax = {xmax}',
        'tiers? <exists>',
        'size = 1',
        'item []:',
        '    item [1]:',
        '        class = "IntervalTier"',
        f'        name = "{tier_name}"',
        f'        xmin = {xmin}',
        f'        xmax = {xmax}',
        f'        intervals: size = {len(calls) * 2 + 1}',
    ]

    # Add intervals (alternating between silence and calls)
    interval_idx = 1
    prev_end = xmin

    for call in sorted(calls, key=lambda x: x['start_seconds']):
        # Silence before call
        if call['start_seconds'] > prev_end:
            lines.extend([
                f'        intervals [{interval_idx}]:',
                f'            xmin = {prev_end}',
                f'            xmax = {call["start_seconds"]}',
                '            text = ""',
            ])
            interval_idx += 1

        # The call
        lines.extend([
            f'        intervals [{interval_idx}]:',
            f'            xmin = {call["start_seconds"]}',
            f'            xmax = {call["stop_seconds"]}',
            f'            text = "{call.get("name", "USV")}"',
        ])
        interval_idx += 1
        prev_end = call['stop_seconds']

    # Final silence
    if prev_end < xmax:
        lines.extend([
            f'        intervals [{interval_idx}]:',
            f'            xmin = {prev_end}',
            f'            xmax = {xmax}',
            '            text = ""',
        ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path
