"""
dch2.core.pipeline
------------------
Hash generation orchestration. Single entry point: generate_dch().

Coordinates all phases in order:
    1. Preprocess     — L+H channel conversion
    2. HH energy      — drives adaptive DoG weight
    3. Feature maps   — Sobel + DoG blend, Laplacian + DoG blend
    4. Spectral maps  — SWT decomposition, spatial + invariant paths
    5. Quantization   — overlapping quadrant local median binarization
    6. Packing        — concatenation and hex conversion
    7. Stats          — structural moment fingerprint

Returns a DCHResult containing bits, hex, stats, and image_path.
"""

import numpy as np

from dch2.core.preprocess import preprocess_image
from dch2.features.filters import (
    compute_edge_map,
    compute_laplacian_map,
    compute_texture_map,
    blend_texture,
    compute_adaptive_dog_weight
)
from dch2.features.spectral import spectral_feature_map, compute_hh_energy
from dch2.encoding.quantizer import quantize
from dch2.encoding.packing import pack_bits, bits_to_hex
from dch2.similarity.stats import compute_structural_stats


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

class DCHResult:
    """
    Result of generate_dch() for a single image.

    Attributes
    ----------
    bits : list
        256-element binary hash vector.
        bits[0:128]   — spatial path   (crop/scale robust)
        bits[128:256] — invariant path (rotation/flip robust)

    hex : str
        64-character hexadecimal hash string.

    stats : np.ndarray
        6-element structural statistical fingerprint.
        [sobel_mean, sobel_var, sobel_skew,
         lap_mean,   lap_var,   lap_skew]

    image_path : str
        Source image path for traceability.
    """

    def __init__(
        self,
        bits:       list,
        hex_hash:   str,
        stats:      np.ndarray,
        image_path: str
    ):
        self.bits       = bits
        self.hex        = hex_hash
        self.stats      = stats
        self.image_path = image_path

    def spatial_bits(self) -> list:
        """128-bit spatial path segment."""
        return self.bits[0:128]

    def invariant_bits(self) -> list:
        """128-bit invariant path segment."""
        return self.bits[128:256]

    def __repr__(self) -> str:
        return (
            f"DCHResult(\n"
            f"  image_path = '{self.image_path}'\n"
            f"  hex        = '{self.hex}'\n"
            f"  bits       = [{self.bits[0]}...{self.bits[-1]}] (256 bits)\n"
            f"  stats      = {np.round(self.stats, 4)}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def generate_dch(image_path: str) -> DCHResult:
    """
    Generate a 256-bit DCH perceptual hash for an image.

    Pipeline
    --------
    Step 1 — Preprocess
        Load image, resize to 256x256, convert to saturation-adaptive
        L+H single channel. Output: float64 array in [0, 1].

    Step 2 — Compute HH energy for adaptive DoG weight
        Run a preliminary SWT on the preprocessed image to measure
        diagonal detail subband energy. High energy = textured image.
        Map energy to dog_weight in [0.05, 0.25].

    Step 3 — Feature extraction
        Compute Sobel edge map and Laplacian structure map.
        Compute DoG texture map once, shared across both paths.
        Blend DoG into each primary map at the adaptive weight.

    Step 4 — Spectral decomposition (two paths per feature map)
        Pass each blended map through spectral_feature_map():
            Spatial map  — SWT magnitude resized to 8x8
            Invariant map — Radial Energy Profile (REP) 8x8

        Four maps total:
            edge_spatial, edge_invariant,
            lap_spatial,  lap_invariant

    Step 5 — Quantization
        Each 8x8 map → overlapping quadrant local median binarization
        → 64-bit vector. Four maps → 4 x 64 = 256 bits total.

    Step 6 — Pack and encode
        Concatenate: spatial_part   = edge_spatial + lap_spatial   (128 bits)
                     invariant_part = edge_invariant + lap_invariant (128 bits)
        combined = spatial_part + invariant_part (256 bits)
        Convert to 64-character hex string.

    Step 7 — Structural stats
        Compute 6-moment statistical fingerprint independently from
        the hash. Stored in result for use by comparator's stats rescue.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    DCHResult
        Contains bits (256), hex (64 chars), stats (6 values),
        and image_path.

    Raises
    ------
    ValueError
        If image cannot be loaded from image_path.
    """

    # ── Step 1: Preprocess ──────────────────────────────────────────────────
    image = preprocess_image(image_path)

    # ── Step 2: Adaptive DoG weight from HH energy ──────────────────────────
    hh_energy  = compute_hh_energy(image)
    dog_weight = compute_adaptive_dog_weight(hh_energy)

    # ── Step 3: Feature extraction ──────────────────────────────────────────
    edge_map    = compute_edge_map(image)
    lap_map     = compute_laplacian_map(image)
    texture_map = compute_texture_map(image)

    # DoG computed once, blended into both primary paths
    edge_enriched = blend_texture(edge_map, texture_map, dog_weight)
    lap_enriched  = blend_texture(lap_map,  texture_map, dog_weight)

    # ── Step 4: Spectral decomposition ──────────────────────────────────────
    edge_spatial, edge_invariant = spectral_feature_map(edge_enriched)
    lap_spatial,  lap_invariant  = spectral_feature_map(lap_enriched)

    # ── Step 5: Quantization ────────────────────────────────────────────────
    h_edge_spatial   = quantize(edge_spatial)    # 64 bits
    h_edge_invariant = quantize(edge_invariant)  # 64 bits
    h_lap_spatial    = quantize(lap_spatial)     # 64 bits
    h_lap_invariant  = quantize(lap_invariant)   # 64 bits

    # ── Step 6: Pack and encode ─────────────────────────────────────────────
    spatial_part   = pack_bits([h_edge_spatial,   h_lap_spatial])    # 128 bits
    invariant_part = pack_bits([h_edge_invariant, h_lap_invariant])  # 128 bits
    combined_bits  = spatial_part + invariant_part                   # 256 bits
    hex_hash       = bits_to_hex(combined_bits)

    # ── Step 7: Structural stats ─────────────────────────────────────────────
    stats = compute_structural_stats(image_path)

    return DCHResult(
        bits       = combined_bits,
        hex_hash   = hex_hash,
        stats      = stats,
        image_path = image_path
    )