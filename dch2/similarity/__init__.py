"""
dch2.similarity — three-layer similarity measurement.
    hamming.py   — Hamming distance and similarity score
    coherence.py — bit coherence scoring
    stats.py     — structural statistical moment fingerprinting
"""
from dch2.similarity.hamming import compute_hamming_distance, compute_similarity
from dch2.similarity.coherence import compute_coherence_score
from dch2.similarity.stats import (
    compute_structural_stats,
    compute_stats_similarity,
    apply_stats_rescue
)