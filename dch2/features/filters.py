"""
dch2.features.filters
---------------------
Structural feature extraction from preprocessed images.

Three complementary filters capture different frequency bands
of structural information:

    Sobel   — high frequency sharp edges and boundaries
              responds to: object outlines, hard transitions
              mathematically: first-order gradient magnitude

    Laplacian — fine structural transitions and detail
                responds to: texture boundaries, fine edges
                mathematically: second-order derivative (∇²f)

    DoG     — mid-frequency texture blobs and surface patterns
              responds to: fur, foliage, fabric, cloud texture
              mathematically: band-pass filter via Gaussian difference

Why three filters:
    No single filter captures all structurally meaningful information.
    Sobel misses smooth texture. Laplacian misses broad gradients.
    DoG misses sharp edges. Together they form a complete structural
    fingerprint that is robust across diverse image content types.

Content-Adaptive DoG Blending:
    DoG is blended into the primary feature maps (Sobel and Laplacian)
    rather than used as a standalone path. The blend weight is not fixed —
    it is passed in by the pipeline after being computed from the HH
    wavelet subband energy of the image:

        High HH energy  → dense texture present → higher dog_weight
        Low HH energy   → smooth surfaces       → lower dog_weight

    This means textured images (fur, foliage, stripes) get stronger
    texture discrimination while smooth images (sky, water, walls)
    are not contaminated by DoG noise amplification.

    The pipeline (core/pipeline.py) is responsible for computing and
    passing the dog_weight. This file accepts it as a parameter.
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# DoG blend weight bounds — pipeline passes a value within this range
DOG_WEIGHT_MIN     = 0.05   # smooth images — DoG contributes minimally
DOG_WEIGHT_MAX     = 0.25   # textured images — DoG contributes strongly
DOG_WEIGHT_DEFAULT = 0.15   # fallback when no adaptive weight provided

# Sobel kernel size — 3x3 is standard for edge detection at this scale
SOBEL_KERNEL_SIZE = 3

# DoG Gaussian parameters
# Small kernel: tight blur, captures fine texture boundaries
DOG_SIGMA_SMALL   = 0.5
DOG_KERNEL_SMALL  = 3

# Large kernel: loose blur, captures coarser texture regions
DOG_SIGMA_LARGE   = 2.0
DOG_KERNEL_LARGE  = 9


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_map(feature_map: np.ndarray) -> np.ndarray:
    """
    Normalize a feature map to [0, 1].

    Required before blending maps from different filters.
    Each filter produces values on a different numerical scale —
    Sobel sums absolute gradients, Laplacian computes second derivatives,
    DoG computes Gaussian differences. Blending raw values without
    normalization would let whichever filter produces larger absolute
    values dominate the blend regardless of the blend ratio.

    After normalization the blend ratio means what it says:
    dog_weight=0.15 means 15% DoG character, 85% primary character.

    Handles flat maps (all values equal) by returning zeros.
    """
    map_min = np.min(feature_map)
    map_max = np.max(feature_map)

    if map_max - map_min == 0:
        return np.zeros_like(feature_map, dtype=np.float64)

    return (feature_map - map_min) / (map_max - map_min)


# ─────────────────────────────────────────────────────────────────────────────
# Sobel edge map
# ─────────────────────────────────────────────────────────────────────────────

def compute_edge_map(image: np.ndarray) -> np.ndarray:
    """
    Compute direction-agnostic edge magnitude map using Sobel filters.

    Computes gradient magnitude as |SobelX| + |SobelY| rather than
    sqrt(SobelX² + SobelY²). The sum of absolutes is computationally
    cheaper and produces equivalent structural information for hashing
    purposes — the exact magnitude value is less important than the
    spatial distribution of edge energy.

    Parameters
    ----------
    image : np.ndarray
        Single channel float64 image in [0, 1], shape (H, W).
        Output of preprocess_image().

    Returns
    -------
    np.ndarray
        Float64 edge magnitude map, same shape as input.
        Values are non-negative (absolute values summed).
        Not normalized — call _normalize_map() before blending.
    """
    sobel_x  = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
    sobel_y  = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=SOBEL_KERNEL_SIZE)
    edge_map = np.abs(sobel_x) + np.abs(sobel_y)

    return edge_map


# ─────────────────────────────────────────────────────────────────────────────
# Laplacian structure map
# ─────────────────────────────────────────────────────────────────────────────

def compute_laplacian_map(image: np.ndarray) -> np.ndarray:
    """
    Compute absolute Laplacian map for fine structural change detection.

    The Laplacian (∇²f) computes the second derivative of intensity,
    producing strong responses at fine structural transitions that
    first-order Sobel filters may miss — fine texture edges, subtle
    detail boundaries, and rapid local intensity changes.

    Absolute value is taken to remove sign sensitivity — a bright
    region transitioning to dark and dark to bright both represent
    structural boundaries and should be treated equivalently.

    Parameters
    ----------
    image : np.ndarray
        Single channel float64 image in [0, 1], shape (H, W).

    Returns
    -------
    np.ndarray
        Float64 absolute Laplacian map, same shape as input.
        Values are non-negative.
        Not normalized — call _normalize_map() before blending.
    """
    laplacian_map = cv2.Laplacian(image, cv2.CV_64F)
    return np.abs(laplacian_map)


# ─────────────────────────────────────────────────────────────────────────────
# DoG texture map
# ─────────────────────────────────────────────────────────────────────────────

def compute_texture_map(image: np.ndarray) -> np.ndarray:
    """
    Compute absolute Difference of Gaussians (DoG) texture map.

    DoG approximates the Laplacian of Gaussian (LoG) and acts as a
    band-pass filter — it suppresses both very high frequencies (noise)
    and very low frequencies (broad gradients), leaving mid-frequency
    texture information. This is the frequency band that contains
    surface texture patterns: fur, feathers, foliage, fabric, stripes.

    Two Gaussian blurs of different scales are subtracted:
        Small blur (σ=0.5, k=3) — preserves fine texture detail
        Large blur (σ=2.0, k=9) — captures coarser structure
        DoG = small_blur - large_blur

    The difference isolates the band between the two blur scales.
    Absolute value removes sign — both positive and negative DoG
    responses represent texture activity and should be treated equally.

    Parameters
    ----------
    image : np.ndarray
        Single channel float64 image in [0, 1], shape (H, W).

    Returns
    -------
    np.ndarray
        Float64 absolute DoG texture map, same shape as input.
        Values are non-negative.
        Not normalized — call _normalize_map() before blending.
    """
    blur_small = cv2.GaussianBlur(
        image,
        (DOG_KERNEL_SMALL, DOG_KERNEL_SMALL),
        DOG_SIGMA_SMALL
    )
    blur_large = cv2.GaussianBlur(
        image,
        (DOG_KERNEL_LARGE, DOG_KERNEL_LARGE),
        DOG_SIGMA_LARGE
    )
    dog_map = blur_small - blur_large
    return np.abs(dog_map)


# ─────────────────────────────────────────────────────────────────────────────
# Content-adaptive blending
# ─────────────────────────────────────────────────────────────────────────────

def blend_texture(
    primary_map: np.ndarray,
    texture_map: np.ndarray,
    dog_weight:  float = DOG_WEIGHT_DEFAULT
) -> np.ndarray:
    """
    Blend a primary feature map with the DoG texture map.

    Both maps are normalized to [0, 1] before blending so the
    dog_weight ratio reflects actual character contribution rather
    than numerical scale. The primary map remains dominant.

        blended = (1 - dog_weight) * primary_norm + dog_weight * texture_norm

    The dog_weight is computed externally by the pipeline from the
    HH wavelet subband energy and passed in here. This function does
    not compute the weight — it only applies it. This separation keeps
    the filter layer independent from the spectral layer.

    Parameters
    ----------
    primary_map : np.ndarray
        Edge or Laplacian feature map (output of compute_edge_map
        or compute_laplacian_map). Float64, non-negative.

    texture_map : np.ndarray
        DoG texture map (output of compute_texture_map).
        Float64, non-negative.

    dog_weight : float, optional
        Blend weight for texture map. Range [DOG_WEIGHT_MIN, DOG_WEIGHT_MAX].
        Default DOG_WEIGHT_DEFAULT (0.15) used when pipeline does not
        provide an adaptive weight.
        Higher value → stronger texture sensitivity.
        Lower value  → primary filter dominates more strongly.

    Returns
    -------
    np.ndarray
        Blended feature map, float64, values in [0, 1].
        Ready for input to spectral_feature_map().
    """
    dog_weight = float(np.clip(dog_weight, DOG_WEIGHT_MIN, DOG_WEIGHT_MAX))

    primary_norm = _normalize_map(primary_map)
    texture_norm = _normalize_map(texture_map)

    primary_weight = 1.0 - dog_weight
    blended        = (primary_weight * primary_norm) + (dog_weight * texture_norm)

    return blended


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive weight computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_adaptive_dog_weight(hh_energy: float) -> float:
    """
    Compute content-adaptive DoG blend weight from HH subband energy.

    The HH (diagonal detail) subband of the DWT captures high-frequency
    diagonal texture. Its mean energy is a reliable proxy for overall
    image texture density:

        High HH energy  → image has dense texture (fur, foliage, stripes)
                       → DoG should contribute more → higher weight
        Low HH energy   → image is smooth (sky, water, plain walls)
                       → DoG should contribute less → lower weight

    The HH energy is computed by spectral.py and passed here by pipeline.py.
    This function performs a simple linear mapping from energy to weight.

    Parameters
    ----------
    hh_energy : float
        Mean absolute energy of the HH wavelet subband.
        Typically in range [0, 0.1] for normalized input images.

    Returns
    -------
    float
        Adaptive dog_weight in [DOG_WEIGHT_MIN, DOG_WEIGHT_MAX].

    Notes
    -----
    HH_ENERGY_MAX is the empirically observed upper bound for natural
    images. Values above this are clamped — extremely high texture
    images don't get arbitrarily high DoG weights.
    """
    HH_ENERGY_MAX = 0.08

    clamped   = float(np.clip(hh_energy, 0.0, HH_ENERGY_MAX))
    t         = clamped / HH_ENERGY_MAX
    weight    = DOG_WEIGHT_MIN + t * (DOG_WEIGHT_MAX - DOG_WEIGHT_MIN)

    return round(weight, 4)