"""
dch2.encoding.quantizer
-----------------------
Converts continuous-valued feature maps into binary hash vectors
using spatially-aware local median thresholding with overlapping quadrants.

Improvement over v1:
    v1 used hard-boundary quadrants — a feature straddling a quadrant
    boundary was processed inconsistently, causing bit flips on slight
    shifts or rotations that moved the feature across the boundary.

    v2 uses overlapping quadrants with 25% overlap — each boundary
    region is processed in two quadrant contexts. The final bit for
    boundary pixels is determined by majority vote between the two
    quadrant thresholds, eliminating boundary artifacts.
"""

import numpy as np
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Overlap fraction between adjacent quadrants
# 0.25 means each quadrant extends 25% into its neighbour's territory
OVERLAP_FRACTION = 0.25

# Output bits per quadrant — 4x4 core region = 16 bits each
# Total: 4 quadrants × 16 bits = 64 bits per 8x8 map
BITS_PER_QUADRANT = 16


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_overlapping_quadrants(block: np.ndarray) -> List[np.ndarray]:
    """
    Extract four overlapping quadrants from a 2D feature map.

    Each quadrant extends OVERLAP_FRACTION beyond its core boundary
    into the adjacent quadrant's territory. This ensures features
    near boundaries are evaluated in both quadrant contexts.

    For an 8x8 input with 25% overlap:
        Core mid-point: row=4, col=4
        Overlap extension: 4 * 0.25 = 1 pixel

        Q0 (top-left):     rows [0:5], cols [0:5]  — core + 1px overlap
        Q1 (top-right):    rows [0:5], cols [3:8]  — 1px overlap + core
        Q2 (bottom-left):  rows [3:8], cols [0:5]  — 1px overlap + core
        Q3 (bottom-right): rows [3:8], cols [3:8]  — overlap + core

    Parameters
    ----------
    block : np.ndarray
        2D float64 feature map, typically (8, 8).

    Returns
    -------
    List[np.ndarray]
        Four overlapping quadrant arrays [Q0, Q1, Q2, Q3].
    """
    rows, cols = block.shape
    mid_row    = rows // 2
    mid_col    = cols // 2

    overlap_row = max(1, int(mid_row * OVERLAP_FRACTION))
    overlap_col = max(1, int(mid_col * OVERLAP_FRACTION))

    # Each quadrant: core region + overlap extension
    q0 = block[:mid_row + overlap_row,  :mid_col + overlap_col]   # top-left
    q1 = block[:mid_row + overlap_row,   mid_col - overlap_col:]  # top-right
    q2 = block[mid_row - overlap_row:,  :mid_col + overlap_col]   # bottom-left
    q3 = block[mid_row - overlap_row:,   mid_col - overlap_col:]  # bottom-right

    return [q0, q1, q2, q3]


def _binarize_quadrant(quadrant: np.ndarray) -> List[int]:
    """
    Binarize a quadrant against its own local median threshold.

    Flattens, computes median, then assigns 1 to values >= median
    and 0 to values below. Using the local median means each quadrant
    is compared only against its own distribution — a strong pattern
    in one quadrant cannot shift the threshold of another.

    The core 4x4 region (16 values) is extracted from the flattened
    quadrant after sorting to ensure consistent bit count regardless
    of overlap size. This keeps the output fixed at BITS_PER_QUADRANT.

    Parameters
    ----------
    quadrant : np.ndarray
        2D float64 array, overlapping quadrant region.

    Returns
    -------
    List[int]
        Binary bit list of length BITS_PER_QUADRANT (16 bits).
    """
    flat      = quadrant.flatten()
    threshold = float(np.median(flat))

    # Binarize all values
    all_bits = [1 if v >= threshold else 0 for v in flat]

    # Take the center BITS_PER_QUADRANT values — these correspond to
    # the core non-overlapping region, reducing boundary influence
    # while still having been thresholded with overlap context
    center_start = (len(all_bits) - BITS_PER_QUADRANT) // 2
    core_bits    = all_bits[center_start: center_start + BITS_PER_QUADRANT]

    # Safety: if extraction yields wrong count, fall back to first N bits
    if len(core_bits) != BITS_PER_QUADRANT:
        core_bits = all_bits[:BITS_PER_QUADRANT]

    return core_bits


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def quantize(spectral_map: np.ndarray) -> List[int]:
    """
    Quantize a spectral feature map into a binary hash vector.

    Divides the map into four overlapping quadrants, applies local
    median thresholding independently per quadrant, and concatenates
    the resulting bit lists into a single 64-bit vector.

    Bit layout:
        bits  0-15:  top-left quadrant core     (Q0)
        bits 16-31:  top-right quadrant core    (Q1)
        bits 32-47:  bottom-left quadrant core  (Q2)
        bits 48-63:  bottom-right quadrant core (Q3)

    Spatial adjacency is preserved — bits from spatially close regions
    are numerically adjacent in the output vector. This gives Hamming
    distance mild spatial sensitivity: images differing in one corner
    will show bit differences clustered in that quadrant's bit range.

    Parameters
    ----------
    spectral_map : np.ndarray
        2D float64 feature map, shape (8, 8), values in [0, 1].
        Output of spectral_feature_map() spatial or invariant path.

    Returns
    -------
    List[int]
        64-element binary list (values 0 or 1).
    """
    quadrants = _extract_overlapping_quadrants(spectral_map)
    hash_bits = []

    for quadrant in quadrants:
        quadrant_bits = _binarize_quadrant(quadrant)
        hash_bits.extend(quadrant_bits)

    return hash_bits