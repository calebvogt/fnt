"""
Batch processing utilities for USV detection.

Provides functions for processing multiple audio files with progress tracking.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import USVDetectorConfig
from .dsp_detector import DSPDetector
from .io import save_das_format, generate_summary, generate_batch_summary
from .spectrogram import get_audio_info


def process_single_file(
    wav_path: str,
    config: USVDetectorConfig,
    output_dir: Optional[str] = None,
    include_extended: bool = True
) -> Dict:
    """
    Process a single WAV file for USV detection.

    Args:
        wav_path: Path to WAV file
        config: Detection configuration
        output_dir: Output directory (defaults to same as input)
        include_extended: Include extended columns in output

    Returns:
        Dictionary with processing results:
            - input_file: Input file path
            - output_file: Output annotation file path
            - num_calls: Number of detected calls
            - summary: Summary statistics dictionary
            - error: Error message if processing failed
    """
    try:
        # Create detector
        detector = DSPDetector(config)

        # Detect calls
        calls = detector.detect_file(wav_path)

        # Determine output path
        if output_dir is None:
            output_dir = os.path.dirname(wav_path)
        os.makedirs(output_dir, exist_ok=True)

        base_name = Path(wav_path).stem
        output_path = os.path.join(output_dir, f"{base_name}{config.output_suffix}.csv")

        # Save annotations
        save_das_format(calls, output_path, include_extended=include_extended)

        # Generate summary
        summary = generate_summary(calls)

        # Add file info to summary
        audio_info = get_audio_info(wav_path)
        if 'duration' in audio_info:
            summary['file_duration_s'] = audio_info['duration']
            if summary['total_calls'] > 0 and audio_info['duration'] > 0:
                summary['calls_per_minute'] = summary['total_calls'] / (audio_info['duration'] / 60)

        return {
            'input_file': wav_path,
            'output_file': output_path,
            'num_calls': len(calls),
            'summary': summary,
        }

    except Exception as e:
        return {
            'input_file': wav_path,
            'error': str(e),
        }


def batch_process(
    input_folder: str,
    config: Optional[USVDetectorConfig] = None,
    output_folder: Optional[str] = None,
    file_pattern: str = "*.wav",
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    max_workers: int = 1,
    include_extended: bool = True
) -> List[Dict]:
    """
    Process all WAV files in a folder.

    Args:
        input_folder: Folder containing WAV files
        config: Detection configuration (uses prairie vole defaults if None)
        output_folder: Output folder for annotations (defaults to input folder)
        file_pattern: Glob pattern for finding WAV files
        progress_callback: Callback function(filename, current, total) for progress updates
        max_workers: Number of parallel workers (1 for sequential processing)
        include_extended: Include extended columns in output

    Returns:
        List of result dictionaries (one per file)
    """
    if config is None:
        from .config import get_prairie_vole_config
        config = get_prairie_vole_config()

    # Find WAV files
    wav_files = list(Path(input_folder).glob(file_pattern))
    if not wav_files:
        # Also try case-insensitive
        wav_files = list(Path(input_folder).glob(file_pattern.upper()))

    if not wav_files:
        return []

    # Set up output folder
    if output_folder is None:
        output_folder = input_folder
    os.makedirs(output_folder, exist_ok=True)

    results = []
    total_files = len(wav_files)

    if max_workers > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    process_single_file,
                    str(wav_path),
                    config,
                    output_folder,
                    include_extended
                ): wav_path
                for wav_path in wav_files
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_file)):
                wav_path = future_to_file[future]
                if progress_callback:
                    progress_callback(wav_path.name, i + 1, total_files)

                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'input_file': str(wav_path),
                        'error': str(e),
                    })
    else:
        # Sequential processing
        for i, wav_path in enumerate(wav_files):
            if progress_callback:
                progress_callback(wav_path.name, i + 1, total_files)

            result = process_single_file(
                str(wav_path),
                config,
                output_folder,
                include_extended
            )
            results.append(result)

    # Generate batch summary
    summary_path = os.path.join(output_folder, "usv_batch_summary.csv")
    generate_batch_summary(results, summary_path)

    return results


def batch_process_with_logging(
    input_folder: str,
    config: Optional[USVDetectorConfig] = None,
    output_folder: Optional[str] = None,
    log_file: Optional[str] = None
) -> List[Dict]:
    """
    Process files with console logging.

    Args:
        input_folder: Input folder path
        config: Detection configuration
        output_folder: Output folder path
        log_file: Optional log file path

    Returns:
        List of results
    """
    import time
    from datetime import datetime

    start_time = time.time()
    log_messages = []

    def log(msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        log_messages.append(log_msg)

    def progress(filename: str, current: int, total: int):
        log(f"Processing {current}/{total}: {filename}")

    log(f"Starting batch processing of {input_folder}")
    log(f"Output folder: {output_folder or input_folder}")

    results = batch_process(
        input_folder,
        config=config,
        output_folder=output_folder,
        progress_callback=progress
    )

    # Summary
    elapsed = time.time() - start_time
    successful = sum(1 for r in results if 'error' not in r)
    failed = sum(1 for r in results if 'error' in r)
    total_calls = sum(r.get('num_calls', 0) for r in results if 'error' not in r)

    log(f"Batch processing complete in {elapsed:.1f} seconds")
    log(f"Files processed: {len(results)} ({successful} successful, {failed} failed)")
    log(f"Total USV calls detected: {total_calls}")

    # Save log if requested
    if log_file:
        with open(log_file, 'w') as f:
            f.write('\n'.join(log_messages))

    return results


def validate_files(folder: str, file_pattern: str = "*.wav") -> Dict:
    """
    Validate audio files before processing.

    Args:
        folder: Folder to scan
        file_pattern: Glob pattern for files

    Returns:
        Dictionary with validation results
    """
    wav_files = list(Path(folder).glob(file_pattern))

    results = {
        'total_files': len(wav_files),
        'valid_files': [],
        'invalid_files': [],
        'total_duration_s': 0,
        'sample_rates': set(),
    }

    for wav_path in wav_files:
        try:
            info = get_audio_info(str(wav_path))
            if 'error' not in info:
                results['valid_files'].append({
                    'path': str(wav_path),
                    'duration': info.get('duration', 0),
                    'sample_rate': info.get('sample_rate', 0),
                })
                results['total_duration_s'] += info.get('duration', 0)
                results['sample_rates'].add(info.get('sample_rate', 0))
            else:
                results['invalid_files'].append({
                    'path': str(wav_path),
                    'error': info['error'],
                })
        except Exception as e:
            results['invalid_files'].append({
                'path': str(wav_path),
                'error': str(e),
            })

    results['sample_rates'] = list(results['sample_rates'])

    return results
