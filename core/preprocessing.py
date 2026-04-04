"""
preprocessing.py
----------------
Takes raw RF signals and cleans them before feature extraction.

Why this matters:
  Real CSI data is dirty — hardware noise, phase drift, sudden spikes.
  Even our simulated signal has noise added. Before we can extract
  meaningful features or run edge detection, we need to:

    1. Remove high-frequency noise   (low-pass filter)
    2. Remove DC offset / drift      (mean subtraction)
    3. Normalize to a fixed range    (so heatmaps are always comparable)
    4. Detect and remove spikes      (outlier clipping)

  This is standard signal processing — exactly what research papers do
  in their preprocessing pipeline before any ML or detection step.

Imports from:
  - numpy, scipy only (no internal imports needed)

Used by:
  - feature_extract.py
  - web/pages/1_live_demo.py  (directly for display)
"""

import numpy as np
from scipy.signal import butter, filtfilt, medfilt


# ─────────────────────────────────────────────
# 1. LOW-PASS BUTTERWORTH FILTER
# ─────────────────────────────────────────────

def lowpass_filter(signal, cutoff=0.1, order=4):
    """
    Removes high-frequency noise from the signal.

    A Butterworth filter is the standard choice in CSI research because
    it has a maximally flat frequency response — it doesn't distort the
    signal shape while removing noise.

    Think of it like this: the object disturbance is a slow, wide bump.
    Noise is rapid random jitter. The filter keeps the bump, kills the jitter.

    Args:
        signal: 1D numpy array — raw signal
        cutoff: frequency cutoff (0.1 = keep only slow variations)
                lower = smoother but may lose detail
                higher = more detail but more noise passes through
        order:  filter steepness (4 = good balance, standard choice)

    Returns:
        1D numpy array — filtered signal, same shape as input
    """
    # butter() designs the filter coefficients
    # 'low' = low-pass (keep low frequencies, cut high frequencies)
    # Wn=cutoff normalized between 0 and 1 (1 = Nyquist frequency)
    b, a = butter(order, cutoff, btype='low', analog=False)

    # filtfilt() applies the filter TWICE — forward and backward
    # This eliminates phase shift (the signal doesn't get time-shifted)
    # Phase preservation matters because we care about WHERE the bump is
    filtered = filtfilt(b, a, signal)

    return filtered


# ─────────────────────────────────────────────
# 2. MEDIAN FILTER (spike removal)
# ─────────────────────────────────────────────

def remove_spikes(signal, kernel_size=5):
    """
    Removes sudden spikes from the signal using a median filter.

    Spikes in CSI happen due to packet loss, hardware glitches, or
    interference bursts. A median filter replaces each point with the
    median of its neighbors — spikes (extreme outliers) get smoothed out
    while edges and bumps are preserved better than with averaging.

    Args:
        signal:      1D numpy array
        kernel_size: how many neighbors to consider (must be odd)
                     5 = each point looks at 2 neighbors each side

    Returns:
        1D numpy array — spike-free signal
    """
    return medfilt(signal, kernel_size=kernel_size)


# ─────────────────────────────────────────────
# 3. MEAN SUBTRACTION (DC offset removal)
# ─────────────────────────────────────────────

def remove_dc_offset(signal):
    """
    Removes the DC offset (constant baseline shift) from the signal.

    In real CSI, the mean amplitude varies between rooms, distances,
    and hardware. Subtracting the mean centers the signal around zero,
    making it comparable across different environments and runs.

    This is called 'DC offset removal' because in signal processing,
    a constant shift = a zero-frequency (DC) component.

    Args:
        signal: 1D numpy array

    Returns:
        1D numpy array — zero-centered signal
    """
    return signal - np.mean(signal)


# ─────────────────────────────────────────────
# 4. NORMALIZATION
# ─────────────────────────────────────────────

def normalize_signal(signal, method='minmax'):
    """
    Scales signal values to a fixed range.

    Why: without normalization, a strong signal and a weak signal produce
    heatmaps at completely different scales — the colormap becomes meaningless.
    After normalization, the heatmap always represents relative intensity.

    Two methods:
      'minmax' → scales to [0, 1]  — best for visualization and heatmaps
      'zscore' → scales to mean=0, std=1  — best before ML classification

    Args:
        signal: 1D numpy array
        method: 'minmax' or 'zscore'

    Returns:
        1D numpy array — normalized signal
    """
    if method == 'minmax':
        s_min = signal.min()
        s_max = signal.max()

        # Avoid division by zero if signal is flat (all same value)
        if s_max - s_min == 0:
            return np.zeros_like(signal)

        return (signal - s_min) / (s_max - s_min)

    elif method == 'zscore':
        std = signal.std()

        if std == 0:
            return np.zeros_like(signal)

        return (signal - signal.mean()) / std

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'minmax' or 'zscore'.")


# ─────────────────────────────────────────────
# 5. OUTLIER CLIPPING
# ─────────────────────────────────────────────

def clip_outliers(signal, n_std=3.0):
    """
    Clips extreme values that are more than n_std standard deviations
    from the mean.

    This is a softer alternative to spike removal — instead of replacing
    outliers with neighbors, we just cap them at a maximum value.
    Useful as a final safety step after filtering.

    Args:
        signal: 1D numpy array
        n_std:  clip threshold in standard deviations
                3.0 = keep 99.7% of values (standard choice)

    Returns:
        1D numpy array — clipped signal
    """
    mean = signal.mean()
    std  = signal.std()

    lower = mean - n_std * std
    upper = mean + n_std * std

    return np.clip(signal, lower, upper)


# ─────────────────────────────────────────────
# 6. FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

def preprocess(signal, normalize=True, method='minmax'):
    """
    Runs the complete preprocessing pipeline on a raw signal.

    This is the single function everyone else calls — it runs all
    the steps in the correct order:

      raw signal
          → remove spikes        (median filter)
          → remove DC offset     (mean subtraction)
          → smooth noise         (low-pass Butterworth)
          → clip extreme values  (outlier clipping)
          → normalize            (minmax or zscore)

    Order matters: filter before normalize, clip before normalize.

    Args:
        signal:    1D numpy array — raw signal from signal_engine.py
        normalize: whether to normalize at the end (True for visualization)
        method:    normalization method ('minmax' or 'zscore')

    Returns:
        1D numpy array — clean, processed signal ready for feature extraction
    """
    s = signal.copy()    # never modify the original signal

    s = remove_spikes(s)
    s = remove_dc_offset(s)
    s = lowpass_filter(s)
    s = clip_outliers(s)

    if normalize:
        s = normalize_signal(s, method=method)

    return s


# ─────────────────────────────────────────────
# 7. BATCH PREPROCESSING (for dataset loading)
# ─────────────────────────────────────────────

def preprocess_batch(signals, normalize=True, method='minmax'):
    """
    Preprocesses multiple signals at once.

    Used when loading a batch of CSI samples from Widar3.0 or any
    dataset — processes each signal independently through the full pipeline.

    Args:
        signals: 2D numpy array of shape (n_samples, n_subcarriers)
                 OR a list of 1D arrays
        normalize: whether to normalize each signal
        method:    normalization method

    Returns:
        2D numpy array of shape (n_samples, n_subcarriers) — all cleaned
    """
    signals = np.array(signals)
    return np.array([preprocess(s, normalize, method) for s in signals])


# ─────────────────────────────────────────────
# 8. QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run directly to verify:
        python core/preprocessing.py
    """
    # Import here only for testing — keeps the file self-contained
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.signal_engine import generate_rf_scene

    print("Testing preprocessing.py...\n")

    # Generate a noisy scene
    raw = generate_rf_scene(object_size=0.6, object_pos=50, noise_level=0.15)
    print(f"Raw signal      — min: {raw.min():.3f}, max: {raw.max():.3f}, std: {raw.std():.3f}")

    # Run full pipeline
    clean = preprocess(raw, normalize=True, method='minmax')
    print(f"Clean signal    — min: {clean.min():.3f}, max: {clean.max():.3f}, std: {clean.std():.3f}")
    print(f"Range check     — should be [0.0, 1.0]: [{clean.min():.1f}, {clean.max():.1f}]")

    # Test zscore
    zscored = preprocess(raw, normalize=True, method='zscore')
    print(f"Z-scored signal — mean: {zscored.mean():.3f} (≈0), std: {zscored.std():.3f} (≈1)")

    # Test batch
    batch_raw = np.array([generate_rf_scene(object_size=i*0.2) for i in range(5)])
    batch_clean = preprocess_batch(batch_raw)
    print(f"\nBatch input  — shape: {batch_raw.shape}")
    print(f"Batch output — shape: {batch_clean.shape}")

    # Verify disturbance survived filtering
    empty_clean = preprocess(generate_rf_scene(object_size=0.0))
    obj_clean   = preprocess(generate_rf_scene(object_size=0.6, object_pos=50))
    diff = np.abs(obj_clean - empty_clean).mean()
    print(f"\nPost-preprocessing diff (empty vs object): {diff:.4f}")
    print("(should be > 0 — disturbance survived filtering)\n")

    