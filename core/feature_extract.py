"""
feature_extract.py
------------------
Converts a clean preprocessed signal into a feature vector.

Why this file exists:
  A raw signal is just 100 numbers. A classifier (SVM, random forest)
  can't reason about raw numbers well — it needs *meaningful* numbers.
  Features are hand-crafted measurements that capture what matters:

    - Is there a big disturbance somewhere? (energy, variance)
    - Where is the disturbance? (peak position)
    - How spread out is it? (kurtosis, spread)
    - What frequencies are dominant? (FFT features)
    - What does the spatial map look like? (heatmap stats)

  This is called "feature engineering" — one of the most important
  skills in signal processing ML. Research papers spend entire sections
  justifying their feature choices. You'll do the same in your pitch.

Pipeline position:
  signal_engine.py → preprocessing.py → feature_extract.py → model/

Imports from:
  - numpy, scipy
  - core.signal_engine  (for heatmap conversion)
"""

import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import fft

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.signal_engine import signal_to_heatmap


# ─────────────────────────────────────────────
# 1. TIME-DOMAIN FEATURES
# ─────────────────────────────────────────────

def time_domain_features(signal):
    """
    Extracts statistical features directly from the signal amplitude.

    These are the simplest features — just statistics of the raw values.
    They capture the overall energy and shape of the disturbance.

    Features extracted:
      - mean:     average amplitude (shifts up when object is present)
      - std:      standard deviation (spread — larger with object)
      - variance: std squared (emphasizes large disturbances)
      - energy:   sum of squares (total signal power)
      - peak:     maximum value (height of the disturbance bump)
      - peak_pos: where the peak is (object position estimate)
      - rms:      root mean square (another power measure)
      - range:    max - min (how much the signal varies)

    Args:
        signal: 1D numpy array — preprocessed signal

    Returns:
        dict of feature_name → float value
    """
    return {
        'mean':     float(np.mean(signal)),
        'std':      float(np.std(signal)),
        'variance': float(np.var(signal)),
        'energy':   float(np.sum(signal ** 2)),
        'peak':     float(np.max(signal)),
        'peak_pos': float(np.argmax(signal)),   # index of max value
        'rms':      float(np.sqrt(np.mean(signal ** 2))),
        'range':    float(np.max(signal) - np.min(signal)),
    }


# ─────────────────────────────────────────────
# 2. SHAPE FEATURES
# ─────────────────────────────────────────────

def shape_features(signal):
    """
    Extracts features that describe the *shape* of the signal distribution.

    These capture whether the disturbance is symmetric, peaked, or skewed —
    useful for distinguishing object sizes and types.

    Features extracted:
      - kurtosis: how peaked the distribution is
                  high kurtosis = sharp narrow bump = small/far object
                  low kurtosis  = flat wide bump = large/close object
      - skewness: how asymmetric the bump is
                  non-zero = object not centered in the space
      - spread:   how many points are above the median
                  (wider spread = larger object footprint)

    Args:
        signal: 1D numpy array

    Returns:
        dict of feature_name → float value
    """
    median_val = np.median(signal)

    return {
        'kurtosis': float(kurtosis(signal)),
        'skewness': float(skew(signal)),
        'spread':   float(np.sum(signal > median_val)),
    }


# ─────────────────────────────────────────────
# 3. FREQUENCY DOMAIN FEATURES (FFT)
# ─────────────────────────────────────────────

def frequency_features(signal, n_components=10):
    """
    Extracts features from the frequency domain using FFT.

    The Fast Fourier Transform (FFT) decomposes the signal into its
    frequency components. Object presence creates specific frequency
    patterns in the CSI that are distinct from empty-room patterns.

    Think of it like: the bump has a characteristic "shape signature"
    that shows up as specific frequency components. The FFT finds those.

    Features extracted:
      - top n_components FFT magnitude values (dominant frequencies)
      - spectral energy: total power in frequency domain
      - spectral centroid: where the "center of mass" of frequency is

    Args:
        signal:       1D numpy array
        n_components: how many FFT components to include as features

    Returns:
        dict of feature_name → float value
    """
    # Compute FFT magnitudes (we only need the positive frequencies)
    fft_vals  = np.abs(fft(signal))
    half      = fft_vals[:len(fft_vals) // 2]   # positive frequencies only

    # Top frequency components
    fft_features = {
        f'fft_{i}': float(half[i])
        for i in range(min(n_components, len(half)))
    }

    # Spectral energy
    fft_features['spectral_energy']    = float(np.sum(half ** 2))

    # Spectral centroid — weighted average of frequencies
    freqs = np.arange(len(half))
    if half.sum() > 0:
        fft_features['spectral_centroid'] = float(np.sum(freqs * half) / half.sum())
    else:
        fft_features['spectral_centroid'] = 0.0

    return fft_features


# ─────────────────────────────────────────────
# 4. SPATIAL FEATURES (from 2D heatmap)
# ─────────────────────────────────────────────

def spatial_features(signal):
    """
    Converts signal to 2D heatmap and extracts spatial statistics.

    Once we have a 2D representation, we can ask spatial questions:
    Where is the energy concentrated? Is it in the center or the edges?
    How much of the grid is "active" (above threshold)?

    These features connect directly to object detection — an object
    creates a concentrated high-energy region in the heatmap.

    Features extracted:
      - spatial_mean:    average intensity across the grid
      - spatial_std:     variation across grid cells
      - active_cells:    how many cells exceed 50% of max intensity
      - center_energy:   energy in the center 4×4 region
      - edge_energy:     energy in the outer ring
      - center_ratio:    center_energy / total (is object in center?)

    Args:
        signal: 1D numpy array (length 100)

    Returns:
        dict of feature_name → float value
    """
    heatmap = signal_to_heatmap(signal)   # shape (10, 10)

    # Overall heatmap stats
    spatial_mean = float(np.mean(heatmap))
    spatial_std  = float(np.std(heatmap))

    # Active cells (above half of max intensity)
    threshold    = 0.5 * heatmap.max()
    active_cells = float(np.sum(heatmap > threshold))

    # Center region energy (rows 3-7, cols 3-7 = inner 4×4 grid)
    center       = heatmap[3:7, 3:7]
    center_energy = float(np.sum(center ** 2))

    # Edge energy (everything outside center)
    total_energy  = float(np.sum(heatmap ** 2))
    edge_energy   = float(total_energy - center_energy)

    # Center ratio — if object is in center this will be high
    center_ratio  = float(center_energy / total_energy) if total_energy > 0 else 0.0

    return {
        'spatial_mean':   spatial_mean,
        'spatial_std':    spatial_std,
        'active_cells':   active_cells,
        'center_energy':  center_energy,
        'edge_energy':    edge_energy,
        'center_ratio':   center_ratio,
    }


# ─────────────────────────────────────────────
# 5. FULL FEATURE VECTOR
# ─────────────────────────────────────────────

def extract_features(signal, include_fft=True):
    """
    Extracts the complete feature vector from a preprocessed signal.

    Combines all feature groups into a single flat dictionary.
    This is what gets fed into the SVM/classifier.

    Total features:
      - 8  time-domain features
      - 3  shape features
      - 12 frequency features (10 FFT + 2 spectral) [if include_fft=True]
      - 6  spatial features
      = 29 features total (or 17 without FFT)

    Args:
        signal:      1D numpy array — preprocessed signal
        include_fft: whether to include FFT features
                     (True for classification, False for quick tests)

    Returns:
        dict of all feature_name → float value
    """
    features = {}

    features.update(time_domain_features(signal))
    features.update(shape_features(signal))
    features.update(spatial_features(signal))

    if include_fft:
        features.update(frequency_features(signal))

    return features


def extract_feature_vector(signal, include_fft=True):
    """
    Same as extract_features but returns a numpy array instead of dict.

    This is what sklearn SVM/RandomForest actually needs as input —
    a flat array of numbers, not a dictionary.

    Returns:
        1D numpy array of feature values (in consistent order)
        list of feature names (same order, useful for debugging)
    """
    feat_dict = extract_features(signal, include_fft)

    names  = list(feat_dict.keys())
    values = np.array(list(feat_dict.values()), dtype=np.float32)

    return values, names


# ─────────────────────────────────────────────
# 6. BATCH FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_features_batch(signals, include_fft=True):
    """
    Extracts features from multiple signals at once.

    Used when preparing training data for the classifier —
    processes an entire dataset of CSI samples in one call.

    Args:
        signals: 2D numpy array (n_samples, n_subcarriers)
                 or list of 1D arrays

    Returns:
        X: 2D numpy array (n_samples, n_features) — feature matrix
        feature_names: list of feature names (column labels)
    """
    results = [extract_feature_vector(s, include_fft) for s in signals]

    X             = np.array([r[0] for r in results])
    feature_names = results[0][1]    # same names for every sample

    return X, feature_names


# ─────────────────────────────────────────────
# 7. QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run directly to verify:
        python core/feature_extract.py
    """
    from core.preprocessing import preprocess

    print("Testing feature_extract.py...\n")

    # Generate and preprocess two scenes
    from core.signal_engine import generate_rf_scene

    raw_empty  = generate_rf_scene(object_size=0.0)
    raw_object = generate_rf_scene(object_size=0.6, object_pos=50)

    clean_empty  = preprocess(raw_empty)
    clean_object = preprocess(raw_object)

    # Extract feature dicts
    feat_empty  = extract_features(clean_empty)
    feat_object = extract_features(clean_object)

    print("Feature comparison (empty room vs object present):")
    print(f"{'Feature':<22} {'Empty':>10} {'Object':>10} {'Diff':>10}")
    print("-" * 56)

    # Show the most meaningful features for comparison
    key_features = ['mean', 'std', 'energy', 'peak', 'kurtosis',
                    'active_cells', 'center_ratio', 'spectral_energy']

    for k in key_features:
        e = feat_empty.get(k, 0)
        o = feat_object.get(k, 0)
        d = o - e
        print(f"{k:<22} {e:>10.4f} {o:>10.4f} {d:>+10.4f}")

    # Extract as vector
    vec, names = extract_feature_vector(clean_object)
    print(f"\nFeature vector shape: {vec.shape}")
    print(f"Total features: {len(names)}")
    print(f"Feature names: {names[:8]} ...")

    # Batch extraction
    from core.preprocessing import preprocess_batch
    raw_batch   = np.array([generate_rf_scene(object_size=i*0.2) for i in range(6)])
    clean_batch = preprocess_batch(raw_batch)
    X, names    = extract_features_batch(clean_batch)
    print(f"\nBatch feature matrix shape: {X.shape}")
    print(f"(6 samples × {X.shape[1]} features)\n")

