"""
signal_engine.py
----------------
This is the heart of WiSense.

Everything in this project starts here — we either simulate an RF signal
or load a real one from a dataset. Every other file imports from this one.

Key concepts:
  - WiFi CSI = how a signal changes as it travels through space
  - A clean signal = empty room (no disturbance)
  - A disturbed signal = object present (reflecting/absorbing the wave)
  - We model that disturbance as a Gaussian bump on top of the base signal

Two modes:
  1. 1D signal → reshaped to 2D  (used by feature_extract, preprocessing)
  2. Native 2D heatmap           (used by edge_detect — better for visualization)
"""

import numpy as np


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

NUM_SUBCARRIERS = 100
HEATMAP_ROWS    = 10
HEATMAP_COLS    = 10
HEATMAP_2D_SIZE = 64    # native 2D heatmap resolution (64×64 grid)
SIGNAL_MAX      = 1.2   # expected max amplitude — used for fixed-scale normalization


# ─────────────────────────────────────────────
# 1. BASE SIGNAL (1D)
# ─────────────────────────────────────────────

def generate_base_signal(n=NUM_SUBCARRIERS):
    """
    Generates a clean WiFi-like signal — empty room.
    Two overlapping sine waves mimic real CSI multipath patterns.
    """
    x = np.linspace(0, 10, n)
    return np.sin(x) + 0.5 * np.sin(2 * x)


# ─────────────────────────────────────────────
# 2. 1D DISTURBANCE MODEL
# ─────────────────────────────────────────────

def generate_disturbance(n=NUM_SUBCARRIERS, object_size=0.5, object_pos=50):
    """
    Simulates how an object disturbs the 1D WiFi signal.
    Gaussian bump: taller + wider with larger objects.
    """
    if object_size == 0:
        return np.zeros(n)
    positions = np.arange(n)
    width = 200 / (object_size * 10 + 1)
    return object_size * np.exp(-((positions - object_pos) ** 2) / width)


# ─────────────────────────────────────────────
# 3. NOISE
# ─────────────────────────────────────────────

def generate_noise(n=NUM_SUBCARRIERS, level=0.08):
    """Gaussian white noise — simulates real indoor RF environment."""
    return level * np.random.randn(n)


# ─────────────────────────────────────────────
# 4. FULL 1D SCENE (used by feature_extract + preprocessing)
# ─────────────────────────────────────────────

def generate_rf_scene(
    object_size=0.0,
    object_pos=50,
    noise_level=0.08,
    n=NUM_SUBCARRIERS
):
    """
    Generates a complete 1D RF scene: base + disturbance + noise.
    Used by preprocessing.py and feature_extract.py.

    Args:
        object_size:  0.0 = empty room, 0.3 = small, 0.6 = medium, 1.0 = large
        object_pos:   0-100, position of object
        noise_level:  0.08 is realistic indoor noise
        n:            number of subcarrier points

    Returns:
        1D numpy array shape (n,)
    """
    base        = generate_base_signal(n)
    disturbance = generate_disturbance(n, object_size, object_pos)
    noise       = generate_noise(n, noise_level)
    return base + disturbance + noise


# ─────────────────────────────────────────────
# 5. 1D SIGNAL → 2D HEATMAP (small, for features)
# ─────────────────────────────────────────────

def signal_to_heatmap(signal, rows=HEATMAP_ROWS, cols=HEATMAP_COLS):
    """
    Reshapes a 1D signal into a 2D spatial heatmap (10×10).
    Used by feature_extract.py for spatial feature computation.

    NOTE: for edge detection and visualization use generate_2d_heatmap()
    instead — it produces a cleaner, higher-resolution 64×64 map.
    """
    expected = rows * cols
    return signal[:expected].reshape(rows, cols)


# ─────────────────────────────────────────────
# 6. NATIVE 2D HEATMAP (64×64, used by edge_detect + Streamlit)
# ─────────────────────────────────────────────

def generate_2d_heatmap(
    object_size=0.0,
    object_x=0.5,
    object_y=0.5,
    rows=HEATMAP_2D_SIZE,
    cols=HEATMAP_2D_SIZE,
    noise_level=0.02
):
    """
    Generates a native 2D spatial heatmap directly — no 1D reshape.

    This is the correct approach for visualization and edge detection.
    The object appears as a 2D Gaussian blob at position (object_x, object_y).

    Why this works better than signal_to_heatmap():
      - The disturbance is a proper 2D blob — physically meaningful
      - Empty room = near-zero everywhere = no edges
      - Object present = clear bright blob = detectable edges
      - Fixed amplitude scale means empty room stays dark (not stretched)

    Args:
        object_size:  0.0 = empty, 0.3 = small, 0.6 = medium, 1.0 = large
        object_x:     0.0-1.0 horizontal position (0=left, 1=right)
        object_y:     0.0-1.0 vertical position (0=top, 1=bottom)
        rows, cols:   grid resolution (64×64 default)
        noise_level:  0.02 is realistic for this scale

    Returns:
        2D numpy array shape (rows, cols), values roughly in [0, object_size]
    """
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    y_coords = np.linspace(0, 1, rows)
    x_coords = np.linspace(0, 1, cols)
    xx, yy   = np.meshgrid(x_coords, y_coords)

    if object_size > 0:
        # Larger object → wider Gaussian blob
        sigma = 0.10 + object_size * 0.05
        disturbance = object_size * np.exp(
            -(((xx - object_x) ** 2 + (yy - object_y) ** 2) / (2 * sigma ** 2))
        )
    else:
        disturbance = np.zeros((rows, cols))

    noise = noise_level * np.random.randn(rows, cols)
    return disturbance + noise



def generate_motion_sequence(
    object_size=0.6,
    start_x=0.1,
    end_x=0.9,
    steps=20,
    noise_level=0.02
):
    """
    Generates animation frames of an object moving left → right.
    Returns list of 2D heatmaps (one per frame).
    Used by the Streamlit animation demo.
    """
    frames    = []
    positions = np.linspace(start_x, end_x, steps)

    for x_pos in positions:
        heatmap = generate_2d_heatmap(
            object_size=object_size,
            object_x=x_pos,
            object_y=0.5,
            noise_level=noise_level
        )
        frames.append(heatmap)

    return frames


# ─────────────────────────────────────────────
# 8. QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing signal_engine.py...\n")

    # 1D signal tests
    empty = generate_rf_scene(object_size=0.0)
    obj   = generate_rf_scene(object_size=0.6, object_pos=50)
    print(f"1D empty signal  — shape: {empty.shape}, mean: {empty.mean():.3f}")
    print(f"1D object signal — shape: {obj.shape},   mean: {obj.mean():.3f}")

    heatmap_1d = signal_to_heatmap(obj)
    print(f"1D→heatmap shape — {heatmap_1d.shape}  (should be (10, 10))")

    # 2D heatmap tests
    hm_empty = generate_2d_heatmap(object_size=0.0)
    hm_large = generate_2d_heatmap(object_size=1.0)
    print(f"\n2D empty heatmap — shape: {hm_empty.shape}, max: {hm_empty.max():.3f}")
    print(f"2D large heatmap — shape: {hm_large.shape}, max: {hm_large.max():.3f}")
    assert hm_large.max() > hm_empty.max(), "Object heatmap should be brighter"
    print("Assertion passed — object heatmap is brighter than empty room.")

    # Motion sequence
    frames = generate_motion_sequence(steps=5)
    print(f"\nMotion sequence — {len(frames)} frames, each {frames[0].shape}")

    print("\nAll tests passed. signal_engine.py is ready.")
