"""
dch2.similarity.stats
---------------------
Structural statistical moment fingerprinting.

Transform-invariant rescue signal for the compounded transform case
where both spatial and invariant hash paths lose signal simultaneously
(e.g. mirror + arbitrary rotation + heavy crop applied together).

Why statistical moments survive all transforms:
    Mirror   → same edges exist, reflected    → distribution unchanged
    Rotation → same edges exist, rotated      → distribution unchanged
    Crop     → dominant content still present → distribution similar

Six-moment fingerprint per image:
    [sobel_mean, sobel_variance, sobel_skewness,
     lap_mean,   lap_variance,   lap_skewness]

Similarity uses normalized absolute difference per component,
averaged across all six. This is scale-independent — the ratio
of difference matters, not absolute magnitude.

Note on Mahalanobis distance:
    The IEEE report suggested Mahalanobis distance which accounts for
    covariance between moments. However Mahalanobis requires a reference
    covariance matrix computed from a training dataset — this violates
    the zero-model constraint. Normalized absolute difference achieves
    comparable discrimination while remaining fully deterministic.
"""

import cv2
import numpy as np
from scipy.stats import skew
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Processing resolution for stats — lower than pipeline resolution
# Stats are distribution-level measurements, not spatial — 128px is sufficient
# and reduces compute time for this secondary signal
STATS_TARGET_SIZE = 128

# Stats rescue threshold — sim_stats must exceed this to trigger rescue
# Calibrated so same-image-different-transform pairs score above this
# while visually-similar-but-different pairs (different city, same aesthetic)
# score below it, preserving a meaningful safety margin
RESCUE_THRESHOLD = 0.82

# Blend weights when stats rescue activates
RESCUE_WEIGHT_FINAL = 0.70   # existing final score weight
RESCUE_WEIGHT_STATS = 0.30   # stats signal weight


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def compute_structural_stats(image_path: str) -> np.ndarray:
    """
    Compute six statistical moments from Sobel and Laplacian feature maps.

    Uses grayscale (not L+H) for stats computation. Statistical moments
    describe energy distribution, not color content — grayscale is more
    stable for moment computation and avoids hue channel variance affecting
    the distribution shape.

    Parameters
    ----------
    image_path : str
        Path to the source image.

    Returns
    -------
    np.ndarray
        6-element float64 array:
        [sobel_mean, sobel_var, sobel_skew, lap_mean, lap_var, lap_skew]
        Returns zeros if image cannot be loaded.
    """
    image = cv2.imread(image_path)

    if image is None:
        return np.zeros(6, dtype=np.float64)

    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    resized = cv2.resize(gray, (STATS_TARGET_SIZE, STATS_TARGET_SIZE),
                         interpolation=cv2.INTER_AREA)

    # Sobel edge map
    sobel_x   = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y   = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    sobel_map = np.abs(sobel_x) + np.abs(sobel_y)

    # Laplacian map
    lap_map = np.abs(cv2.Laplacian(resized, cv2.CV_64F))

    sobel_flat = sobel_map.flatten()
    lap_flat   = lap_map.flatten()

    return np.array([
        np.mean(sobel_flat),
        np.var(sobel_flat),
        float(skew(sobel_flat)),
        np.mean(lap_flat),
        np.var(lap_flat),
        float(skew(lap_flat)),
    ], dtype=np.float64)


def compute_stats_similarity(
    stats_a: np.ndarray,
    stats_b: np.ndarray
) -> float:
    """
    Compute normalized similarity between two statistical fingerprints.

    Uses normalized absolute difference per component — scale-independent
    so a mean of 0.5 vs 0.6 is treated proportionally the same regardless
    of the absolute magnitude of the feature maps.

    Parameters
    ----------
    stats_a : np.ndarray
        6-element stats vector from compute_structural_stats().
    stats_b : np.ndarray
        6-element stats vector from compute_structural_stats().

    Returns
    -------
    float
        Similarity score in [0, 1].
        1.0 — identical statistical distributions.
        0.0 — completely different distributions.
    """
    components = []

    for a, b in zip(stats_a, stats_b):
        max_val = max(abs(a), abs(b))
        if max_val == 0:
            components.append(1.0)
        else:
            diff = abs(a - b) / max_val
            components.append(max(0.0, 1.0 - diff))

    return float(np.mean(components))


def apply_stats_rescue(
    current_final: float,
    sim_stats:     float,
    threshold:     float
) -> tuple:
    """
    Apply stats rescue when hash paths have both failed.

    Only activates when:
        1. current_final is below the acceptance threshold
        2. sim_stats exceeds RESCUE_THRESHOLD (0.82)

    When active, blends stats signal into the final score at 30% weight.
    This is conservative — stats alone cannot force a SELECT, they can
    only nudge a borderline case above threshold.

    Parameters
    ----------
    current_final : float
        Final score from hash paths after coherence adjustment.
    sim_stats : float
        Statistical similarity from compute_stats_similarity().
    threshold : float
        Acceptance threshold used by the comparator.

    Returns
    -------
    tuple (rescued_score, was_rescued)
        rescued_score : float — final score after rescue attempt
        was_rescued   : bool  — True if rescue elevated score above threshold
    """
    if current_final >= threshold or sim_stats < RESCUE_THRESHOLD:
        return current_final, False

    rescued_score = (RESCUE_WEIGHT_FINAL * current_final) + \
                    (RESCUE_WEIGHT_STATS * sim_stats)

    was_rescued = rescued_score >= threshold

    return rescued_score, was_rescued