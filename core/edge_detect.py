"""
edge_detect.py
--------------
Takes a 2D heatmap and finds the boundaries of objects within it.

Why the previous approach failed — and what we fixed:
  WRONG: using NORM_MINMAX on the image before Canny.
    NORM_MINMAX stretches ANY image to full 0-255 contrast — including a
    flat noise-only (empty room) image. So empty room noise got stretched
    to look like a high-contrast image, and Canny found thousands of
    'edges' in pure noise. The detector couldn't tell rooms apart.

  RIGHT: use a FIXED scale (clip to known max amplitude, scale to 0-255).
    Empty room: near-zero values → stays dark → almost no edges.
    Object present: bright Gaussian blob → clear ring of edges.
    Now edge count is physically meaningful.

  Also fixed: switched from 1D reshape heatmap to native 2D heatmap
  (generate_2d_heatmap) — the blob is a proper 2D Gaussian, not a
  1D wave pattern folded into a grid.

Pipeline:
  generate_2d_heatmap()
      → clip to [0, SIGNAL_MAX]     (remove negative noise, fixed scale)
      → scale to 0-255 uint8        (fixed scale, not NORM_MINMAX)
      → GaussianBlur (5×5)          (smooth pixel noise)
      → Canny(20, 60)               (find blob boundary)
      → edge count → detection
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.signal_engine import signal_to_heatmap, generate_2d_heatmap, SIGNAL_MAX


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

CANNY_LOW  = 20
CANNY_HIGH = 60


# ─────────────────────────────────────────────
# 1. HEATMAP → UINT8 (fixed scale — critical)
# ─────────────────────────────────────────────

def heatmap_to_image(heatmap, signal_max=SIGNAL_MAX):
    """
    Converts float heatmap to uint8 using a FIXED amplitude scale.

    This is the key fix. We do NOT use NORM_MINMAX because:
      - An empty room (near-zero heatmap) would get stretched to full
        0-255 contrast, making noise look like edges.
      - Instead we clip to [0, signal_max] and scale linearly.
      - Empty room stays near 0 → dark image → no edges.
      - Object blob stays at its true amplitude → bright region → edges.

    Args:
        heatmap:    2D float numpy array from generate_2d_heatmap()
        signal_max: max expected amplitude (1.2 for object_size=1.0)

    Returns:
        2D uint8 numpy array
    """
    h = heatmap.astype(np.float32)

    # Remove negative noise floor — only positive disturbance matters
    h = np.clip(h, 0, signal_max)

    # Fixed linear scale: 0..signal_max → 0..255
    img = (h / signal_max * 255).astype(np.uint8)

    return img


# ─────────────────────────────────────────────
# 2. FULL PIPELINE
# ─────────────────────────────────────────────

def process_heatmap(heatmap):
    """
    Full pipeline: 2D float heatmap → edge map + metadata.

    Call this with output from generate_2d_heatmap() — NOT from
    preprocessing.preprocess(). The preprocessing module compresses
    contrast via NORM_MINMAX which breaks fixed-scale edge detection.

    Args:
        heatmap: 2D numpy float array (from generate_2d_heatmap)

    Returns:
        dict:
          'heatmap_raw'  — original float heatmap (for 'hot' colormap)
          'heatmap_img'  — processed uint8 image
          'edges'        — binary edge map (0 or 255)
          'edge_count'   — white pixel count (object indicator)
    """
    # Step 1: fixed-scale conversion to uint8
    img = heatmap_to_image(heatmap)

    # Step 2: smooth noise before edge detection
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Step 3: Canny edge detection
    edges = cv2.Canny(img, CANNY_LOW, CANNY_HIGH)

    edge_count = int(np.sum(edges > 0))

    return {
        'heatmap_raw': heatmap,
        'heatmap_img': img,
        'edges':       edges,
        'edge_count':  edge_count,
    }


# ─────────────────────────────────────────────
# 3. OBJECT DETECTION DECISION
# ─────────────────────────────────────────────

def is_object_present(edge_result, threshold=30):
    """
    Rule-based detection: object present if edge count exceeds threshold.

    With fixed-scale normalization:
      - Empty room  → ~0   edge pixels
      - Small object → ~60  edge pixels
      - Large object → ~130 edge pixels

    Threshold of 30 sits cleanly between 0 and 60.

    Returns:
        (bool, float) — detected, confidence 0.0-1.0
    """
    count      = edge_result['edge_count']
    confidence = min(1.0, count / 200.0)
    return count >= threshold, round(confidence, 2)


# ─────────────────────────────────────────────
# 4. MATPLOTLIB FIGURE BUILDERS
# ─────────────────────────────────────────────

def make_signal_figure(signal, title="RF Signal"):
    """Panel 1: 1D signal plot."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(signal, color='#3B8BD4', linewidth=1.5, alpha=0.9)
    ax.fill_between(range(len(signal)), signal, alpha=0.15, color='#3B8BD4')
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Subcarrier index", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, len(signal) - 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def make_heatmap_figure(heatmap, title="RF Heatmap"):
    """Panel 2: 2D heatmap with 'hot' colormap."""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear',
                   aspect='auto', vmin=0, vmax=SIGNAL_MAX)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Spatial X", fontsize=9)
    ax.set_ylabel("Spatial Y", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def make_edge_figure(edges, title="Edge Detection"):
    """Panel 3: binary edge map."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(edges, cmap='gray', aspect='auto')
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Spatial X", fontsize=9)
    ax.set_ylabel("Spatial Y", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def make_three_panel_figure(heatmap, edges, detected=False, confidence=0.0):
    """
    Three panels side by side: heatmap (hot) | heatmap (gray) | edges.
    For pitch deck screenshots and the notebook.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1: Hot colormap heatmap
    im = axes[0].imshow(heatmap, cmap='hot', interpolation='bilinear',
                        vmin=0, vmax=SIGNAL_MAX)
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title("RF Heatmap (hot)", fontsize=11)
    axes[0].set_xlabel("Spatial X", fontsize=9)
    axes[0].set_ylabel("Spatial Y", fontsize=9)

    # Panel 2: Viridis colormap (easier to see blob shape)
    im2 = axes[1].imshow(heatmap, cmap='viridis', interpolation='bilinear',
                          vmin=0, vmax=SIGNAL_MAX)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("RF Heatmap (viridis)", fontsize=11)
    axes[1].set_xlabel("Spatial X", fontsize=9)
    axes[1].set_ylabel("Spatial Y", fontsize=9)

    # Panel 3: Edge map + detection status
    axes[2].imshow(edges, cmap='gray')
    status = "OBJECT DETECTED" if detected else "EMPTY ROOM"
    color  = "#2ecc71" if detected else "#95a5a6"
    axes[2].set_title(
        f"Edge Detection\n{status} ({confidence:.0%})",
        fontsize=11, color=color
    )
    axes[2].set_xlabel("Spatial X", fontsize=9)
    axes[2].set_ylabel("Spatial Y", fontsize=9)

    fig.suptitle("WiSense — RF Object Detection Pipeline", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 5. QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing edge_detect.py...\n")

    sizes = [
        ("Empty room  ", 0.0),
        ("Small object", 0.3),
        ("Medium obj  ", 0.6),
        ("Large object", 1.0),
    ]

    counts = []
    for label, size in sizes:
        heatmap  = generate_2d_heatmap(object_size=size, object_x=0.5,
                                        object_y=0.5, noise_level=0.02)
        result   = process_heatmap(heatmap)
        det, cf  = is_object_present(result)
        counts.append(result['edge_count'])
        print(f"{label} — edge pixels: {result['edge_count']:>5} | "
              f"detected: {str(det):<5} | confidence: {cf:.2f}")

    print(f"\nEdge count trend (empty → small → medium → large):")
    print(f"  {counts[0]} → {counts[1]} → {counts[2]} → {counts[3]}")

    assert counts[3] > counts[0], \
        "Large object must produce more edges than empty room"
    assert counts[0] == 0 or counts[0] < 30, \
        "Empty room should produce near-zero edges"
    print("Assertions passed.\n")

    # Save three-panel figure
    heatmap = generate_2d_heatmap(object_size=1.0, object_x=0.5,
                                   object_y=0.5, noise_level=0.02)
    result  = process_heatmap(heatmap)
    det, cf = is_object_present(result)

    fig = make_three_panel_figure(
        result['heatmap_raw'],
        result['edges'],
        detected=det,
        confidence=cf
    )
    fig.savefig("test_output.png", dpi=120, bbox_inches='tight')
    print("Saved test_output.png — open it to see the three panels.\n")
    print("All tests passed. edge_detect.py is ready.")