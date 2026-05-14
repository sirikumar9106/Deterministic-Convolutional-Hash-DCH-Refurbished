"""
dch2.core.preprocess
--------------------
Image loading and perceptual channel conversion.

Converts a raw BGR image into a single-channel float64 map in [0, 1]
that serves as input to all downstream feature extractors.

The key design decision here is the channel used for feature extraction.
Grayscale (used in traditional perceptual hashing) discards all color
information — two images that differ only in color become structurally
identical to edge and gradient filters. This causes false positives
between images that share similar structure but different color content.

DCH2 replaces grayscale with a Saturation-Adaptive L+H channel:

    combined = (L_weight * L) + (H_weight * H * S)

Where:
    L  = lightness    — preserves structural gradients (dominant)
    H  = hue          — adds color discrimination (adaptive)
    S  = saturation   — per-pixel suppression of unreliable hue values
    H_weight — scales dynamically with mean image saturation

This is a two-layer hue noise suppression system:
    Layer 1 (per-pixel): H multiplied by S — unreliable hue in
        individual low-saturation pixels is zeroed out locally.
    Layer 2 (per-image): H_weight scales with mean saturation —
        globally desaturated images (black and white photos, fog,
        low light) get a reduced H contribution at the image level,
        preventing diffuse hue noise from accumulating across all pixels.

The output is dimensionally identical to grayscale — a single channel
float64 array — so all downstream components receive the same format
regardless of which channel strategy is used.
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Maximum H contribution when image is fully saturated
H_WEIGHT_MAX = 0.20

# Minimum H contribution even for near-greyscale images
# Keeps a trace of color signal without introducing pure noise
H_WEIGHT_MIN = 0.05

# Saturation threshold below which image is considered near-greyscale
# Mean saturation below this → H weight interpolates toward H_WEIGHT_MIN
SATURATION_THRESHOLD = 0.25

# Standard resize dimension for all images entering the pipeline
TARGET_SIZE = 256


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_image(image_path: str) -> np.ndarray:
    """
    Load image from disk as BGR.

    Raises
    ------
    ValueError
        If the image cannot be loaded — bad path, corrupt file,
        or unsupported format.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not load image at path: '{image_path}'. "
            f"Check that the file exists and is a supported image format."
        )

    return image


def _resize(image: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    """
    Resize image to a square of target_size x target_size pixels.

    Uses INTER_AREA interpolation — the correct choice for downscaling
    as it computes the average of the source pixels covered by each
    destination pixel, preventing aliasing artifacts that would
    introduce spurious high-frequency energy into the feature maps.
    """
    return cv2.resize(
        image,
        (target_size, target_size),
        interpolation=cv2.INTER_AREA
    )


def _compute_adaptive_h_weight(s_normalized: np.ndarray) -> float:
    """
    Compute the adaptive H weight based on mean image saturation.

    Maps mean saturation linearly between H_WEIGHT_MIN and H_WEIGHT_MAX:

        mean_sat >= SATURATION_THRESHOLD  →  H_weight = H_WEIGHT_MAX (0.20)
        mean_sat == 0.0                   →  H_weight = H_WEIGHT_MIN (0.05)
        mean_sat in between               →  H_weight interpolated linearly

    This ensures:
        - Richly colored images get full hue discrimination benefit
        - Near-greyscale images are protected from hue noise accumulation
        - The transition is smooth, not a hard cutoff

    Parameters
    ----------
    s_normalized : np.ndarray
        Saturation channel normalized to [0, 1].

    Returns
    -------
    float
        Adaptive H weight in range [H_WEIGHT_MIN, H_WEIGHT_MAX].
    """
    mean_saturation = float(np.mean(s_normalized))

    # Clamp mean saturation to [0, SATURATION_THRESHOLD] for interpolation
    clamped = min(mean_saturation, SATURATION_THRESHOLD)

    # Linear interpolation between H_WEIGHT_MIN and H_WEIGHT_MAX
    t = clamped / SATURATION_THRESHOLD
    h_weight = H_WEIGHT_MIN + t * (H_WEIGHT_MAX - H_WEIGHT_MIN)

    return h_weight


def _convert_to_lh_channel(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to saturation-adaptive L+H single channel.

    Steps
    -----
    1. Convert BGR → HLS (OpenCV channel order: H=0, L=1, S=2)
    2. Normalize H to [0, 1] (OpenCV range is 0-180)
    3. Normalize L to [0, 1] (OpenCV range is 0-255)
    4. Normalize S to [0, 1] (OpenCV range is 0-255)
    5. Compute adaptive H weight from mean saturation (per-image)
    6. Weight H by S per-pixel (per-pixel noise suppression)
    7. Combine: (1 - h_weight) * L + h_weight * (H * S)

    Parameters
    ----------
    image : np.ndarray
        BGR image, uint8.

    Returns
    -------
    np.ndarray
        Single channel float64 array in [0, 1], shape (H, W).
    """
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # Extract and normalize each channel
    h = hls[:, :, 0].astype(np.float64) / 180.0   # Hue:        0-180 → 0-1
    l = hls[:, :, 1].astype(np.float64) / 255.0   # Lightness:  0-255 → 0-1
    s = hls[:, :, 2].astype(np.float64) / 255.0   # Saturation: 0-255 → 0-1

    # Layer 1 — per-image adaptive weight based on mean saturation
    h_weight = _compute_adaptive_h_weight(s)
    l_weight = 1.0 - h_weight

    # Layer 2 — per-pixel hue suppression in low-saturation regions
    # Multiplying H by S zeroes out hue where saturation is near zero
    # (near-grey pixels report arbitrary hue values — this silences them)
    h_weighted = h * s

    # Combine L and H into single channel
    combined = (l_weight * l) + (h_weight * h_weighted)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_path: str, target_size: int = TARGET_SIZE) -> np.ndarray:
    """
    Load and preprocess an image for DCH2 feature extraction.

    This is the single entry point for all image loading in the pipeline.
    Every image entering DCH2 passes through this function exactly once.

    Pipeline
    --------
    1. Load from disk (BGR)
    2. Resize to target_size × target_size using INTER_AREA
    3. Convert to saturation-adaptive L+H single channel
    4. Output float64 array in [0, 1]

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file.

    target_size : int, optional
        Square resize dimension. Default 256.
        Must match across all images being compared.

    Returns
    -------
    np.ndarray
        Single channel float64 image, shape (target_size, target_size),
        values in [0, 1]. Ready for feature extraction.

    Raises
    ------
    ValueError
        If image cannot be loaded from image_path.

    Notes
    -----
    Output is already normalized to [0, 1] by the L+H conversion.
    Do NOT apply additional normalization after this function —
    dividing by 255 again would crush all values to near-zero.
    """
    image  = _load_image(image_path)
    image  = _resize(image, target_size)
    result = _convert_to_lh_channel(image)

    return result


def get_preprocessing_info(image_path: str) -> dict:
    """
    Diagnostic function — returns preprocessing metadata for an image.

    Useful for understanding how the adaptive H weight behaved
    on a specific image. Not used in the main pipeline.

    Returns
    -------
    dict with keys:
        image_path    : str
        mean_sat      : float   — mean saturation of the image [0, 1]
        h_weight      : float   — adaptive H weight applied [0.05, 0.20]
        l_weight      : float   — L weight applied [0.80, 0.95]
        output_min    : float   — minimum value in output channel
        output_max    : float   — maximum value in output channel
        output_mean   : float   — mean value in output channel
    """
    image = _load_image(image_path)
    image = _resize(image)

    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s   = hls[:, :, 2].astype(np.float64) / 255.0

    h_weight  = _compute_adaptive_h_weight(s)
    output    = _convert_to_lh_channel(image)

    return {
        "image_path":  image_path,
        "mean_sat":    round(float(np.mean(s)), 4),
        "h_weight":    round(h_weight, 4),
        "l_weight":    round(1.0 - h_weight, 4),
        "output_min":  round(float(np.min(output)), 4),
        "output_max":  round(float(np.max(output)), 4),
        "output_mean": round(float(np.mean(output)), 4),
    }