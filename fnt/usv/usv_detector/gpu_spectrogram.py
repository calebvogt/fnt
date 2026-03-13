"""
GPU-accelerated spectrogram computation using PyTorch.

Provides a drop-in replacement for compute_spectrogram() that uses
torch.stft() on CUDA or MPS devices for significant speedup on the
FFT computation (the main bottleneck in USV detection).

Returns identical NumPy array types as the scipy-based version.
"""

import numpy as np
from typing import Optional, Tuple
import warnings


def compute_spectrogram_gpu(
    audio: np.ndarray,
    sr: int,
    nperseg: int = 512,
    noverlap: int = 384,
    nfft: int = 1024,
    window: str = 'hann',
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram using GPU-accelerated torch.stft().

    Drop-in replacement for spectrogram.compute_spectrogram().
    Same signature, same return types (NumPy arrays).

    Args:
        audio: Audio signal array (1D float)
        sr: Sample rate
        nperseg: Length of each segment (FFT window size)
        noverlap: Number of points to overlap between segments
        nfft: Length of FFT (zero-padded)
        window: Window function name (currently only 'hann' supported)
        min_freq: Minimum frequency to include (Hz)
        max_freq: Maximum frequency to include (Hz)
        device: PyTorch device string ("cuda:0", "mps", etc.)

    Returns:
        Tuple of (frequencies, times, Sxx_db) as NumPy arrays
    """
    import torch

    hop_length = nperseg - noverlap

    # Create window tensor on GPU
    if window == 'hann':
        window_tensor = torch.hann_window(nperseg, device=device)
    elif window == 'hamming':
        window_tensor = torch.hamming_window(nperseg, device=device)
    else:
        # Fallback: create on CPU and transfer
        from scipy.signal import get_window
        win_np = get_window(window, nperseg).astype(np.float32)
        window_tensor = torch.from_numpy(win_np).to(device)

    # Transfer audio to GPU
    audio_tensor = torch.from_numpy(audio.astype(np.float32)).to(device)

    # Compute STFT on GPU
    # Returns complex tensor of shape (nfft//2 + 1, n_frames)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="An output with one or more elements was resized")
        stft_result = torch.stft(
            audio_tensor,
            n_fft=nfft,
            hop_length=hop_length,
            win_length=nperseg,
            window=window_tensor,
            return_complex=True,
            center=False,  # Match scipy's default (no padding)
        )
    # stft_result shape: (n_freqs, n_frames)

    # Power spectral density — match scipy scaling='density'
    # scipy density scaling: Sxx = |X|^2 / (fs * sum(win^2))
    win_norm = (window_tensor ** 2).sum().item()
    Sxx = (stft_result.real ** 2 + stft_result.imag ** 2) / (sr * win_norm)

    # Convert to dB on GPU (avoids large CPU transfer of linear values)
    Sxx_db = 10.0 * torch.log10(Sxx + 1e-10)

    # Build frequency and time arrays (CPU — these are small)
    n_freqs = nfft // 2 + 1
    frequencies = np.linspace(0, sr / 2, n_freqs)

    n_frames = stft_result.shape[1]
    # Match scipy time centers: first center at nperseg/2, then every hop_length
    times = np.arange(n_frames) * hop_length / sr + nperseg / (2.0 * sr)

    # Apply frequency mask before GPU→CPU transfer (reduces transfer size)
    if min_freq is not None or max_freq is not None:
        min_f = min_freq if min_freq is not None else 0
        max_f = max_freq if max_freq is not None else sr / 2
        freq_mask = (frequencies >= min_f) & (frequencies <= max_f)
        frequencies = frequencies[freq_mask]
        # Apply mask on GPU tensor
        freq_mask_tensor = torch.from_numpy(freq_mask).to(device)
        Sxx_db = Sxx_db[freq_mask_tensor, :]

    # Transfer result to CPU
    Sxx_db_np = Sxx_db.cpu().numpy()

    # Free GPU memory explicitly
    del audio_tensor, stft_result, Sxx, Sxx_db, window_tensor
    if device.startswith('cuda'):
        torch.cuda.empty_cache()

    return frequencies, times, Sxx_db_np
