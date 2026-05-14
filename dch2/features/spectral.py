"""
dch2.features.spectral
----------------------
Wavelet decomposition and spatial/invariant map generation.

Two paths per feature map:
    Spatial Map  — WHERE structures are (crop/scale robust)
    Invariant Map — WHAT energy distribution exists (rotation/flip robust)
"""

import cv2
import numpy as np
import pywt


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Wavelet basis — SWT (Stationary Wavelet Transform) for shift invariance
# SWT eliminates the downsampling step of standard DWT, making coefficients
# shift-invariant: a 1-pixel translation produces proportionally shifted
# coefficients rather than completely different coefficient patterns.
# This directly addresses the crop robustness limitation of Haar DWT in v1.
WAVELET_BASIS  = 'haar'
SWT_LEVEL      = 1

# Default output dimension for spatial and invariant maps
TARGET_DIMENSION = 8


# ─────────────────────────────────────────────────────────────────────────────
# HH energy — drives adaptive DoG weight in filters.py
# ─────────────────────────────────────────────────────────────────────────────

def compute_hh_energy(feature_map: np.ndarray) -> float:
    """
    Compute mean absolute energy of the HH (diagonal detail) subband.

    Used by pipeline.py to compute the adaptive DoG blend weight.
    HH energy is a reliable proxy for image texture density.

    Parameters
    ----------
    feature_map : np.ndarray
        Single channel float64 image in [0, 1].

    Returns
    -------
    float
        Mean absolute HH subband energy.
    """
    coeffs       = pywt.swt2(feature_map, WAVELET_BASIS, level=SWT_LEVEL)
    _, (_, _, HH) = coeffs[0]
    return float(np.mean(np.abs(HH)))


# ─────────────────────────────────────────────────────────────────────────────
# Radial Energy Profile — invariant map
# ─────────────────────────────────────────────────────────────────────────────

def _compute_radial_energy_profile(
    magnitude_map:    np.ndarray,
    target_dimension: int = TARGET_DIMENSION
) -> np.ndarray:
    """
    Compute Radial Energy Profile (REP) from a magnitude map.

    Rotation and flip invariant because radial distance from center
    is preserved under all these transforms — only angle changes.

    Bins pixels into concentric rings, computes mean energy per ring,
    normalizes to [0, 1], reshapes to (target_dimension, target_dimension).

    Parameters
    ----------
    magnitude_map : np.ndarray
        Float64 magnitude map from wavelet decomposition.
    target_dimension : int
        Output grid size. Default 8 → 8x8 = 64 bits after quantization.

    Returns
    -------
    np.ndarray
        Float64 array shape (target_dimension, target_dimension) in [0, 1].
    """
    rows, cols   = magnitude_map.shape
    center_row   = rows / 2.0
    center_col   = cols / 2.0
    max_radius   = np.sqrt((rows / 2.0) ** 2 + (cols / 2.0) ** 2)

    row_idx, col_idx = np.indices((rows, cols))
    distance_map     = np.sqrt(
        (row_idx - center_row) ** 2 +
        (col_idx - center_col) ** 2
    )

    num_rings    = target_dimension * target_dimension
    ring_energies = np.zeros(num_rings, dtype=np.float64)

    for i in range(num_rings):
        inner = (i / num_rings) * max_radius
        outer = ((i + 1) / num_rings) * max_radius
        mask  = (distance_map >= inner) & (distance_map < outer)

        if np.any(mask):
            ring_energies[i] = np.mean(magnitude_map[mask])

    max_energy = np.max(ring_energies)
    if max_energy > 0:
        ring_energies = ring_energies / max_energy

    return ring_energies.reshape(target_dimension, target_dimension)


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def spectral_feature_map(
    feature_map:      np.ndarray,
    target_dimension: int = TARGET_DIMENSION
) -> tuple:
    """
    Decompose a feature map into spatial and invariant maps.

    Uses Stationary Wavelet Transform (SWT) instead of standard DWT.
    SWT retains full spatial resolution by removing the downsampling
    step — coefficients are shift-invariant, meaning small translations
    (crops, slight misalignments) produce proportionally shifted
    coefficients rather than completely different patterns.

    Subband weighting:
        Equal weights (0.33 each) across LH, HL, HH subbands.
        Kept equal because the adaptive DoG blending in filters.py
        already handles texture emphasis upstream — boosting HH
        in the magnitude map on top of that would double-weight
        texture and destabilize the invariant REP path.

    Parameters
    ----------
    feature_map : np.ndarray
        Blended feature map from filters.blend_texture().
        Single channel float64, shape (H, W), values in [0, 1].

    target_dimension : int
        Output size for both maps. Default 8 → 8x8 = 64 bits each.

    Returns
    -------
    tuple (spatial_map, invariant_map)
        spatial_map   : np.ndarray (target_dimension, target_dimension)
            Resized magnitude map. Spatially aware — good for crops/scale.
        invariant_map : np.ndarray (target_dimension, target_dimension)
            Radial Energy Profile. Rotation and flip invariant.
        Both float64, values in [0, 1].
    """
    # Stationary Wavelet Transform — shift invariant
    coeffs         = pywt.swt2(feature_map, WAVELET_BASIS, level=SWT_LEVEL)
    _, (LH, HL, HH) = coeffs[0]

    # Equal-weight magnitude map
    magnitude_map = np.sqrt(
        (0.33 * np.power(LH, 2)) +
        (0.33 * np.power(HL, 2)) +
        (0.33 * np.power(HH, 2))
    )

    # Path A: Spatial map — resize preserves spatial layout
    spatial_map = cv2.resize(
        magnitude_map,
        (target_dimension, target_dimension),
        interpolation=cv2.INTER_AREA
    )

    # Normalize spatial map to [0, 1]
    s_max = np.max(spatial_map)
    if s_max > 0:
        spatial_map = spatial_map / s_max

    # Path B: Invariant map — REP discards spatial position, keeps energy distribution
    invariant_map = _compute_radial_energy_profile(magnitude_map, target_dimension)

    return spatial_map, invariant_map