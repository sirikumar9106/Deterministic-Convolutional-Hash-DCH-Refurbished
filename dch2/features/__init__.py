"""
dch2.features — feature extraction layer.
    filters.py   — Sobel, Laplacian, DoG, adaptive blending
    spectral.py  — SWT decomposition, spatial map, REP invariant map
"""
from dch2.features.filters import (
    compute_edge_map,
    compute_laplacian_map,
    compute_texture_map,
    blend_texture,
    compute_adaptive_dog_weight
)
from dch2.features.spectral import spectral_feature_map, compute_hh_energy