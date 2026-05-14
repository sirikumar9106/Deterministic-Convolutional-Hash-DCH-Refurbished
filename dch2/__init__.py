"""
DCH2 — Deterministic Convolutional Hashing v2

A zero-model deterministic perceptual image hashing system.

Usage
-----
    import dch2

    # Generate hash
    result = dch2.generate_dch("image.png")
    print(result.hex)
    print(result.bits)
    print(result.stats)

    # Compare two images
    result = dch2.compare("imageA.png", "imageB.png")
    print(result.verdict)    # SELECT or REJECT
    print(result.is_match()) # True or False
    print(result)            # full diagnostic

    # Batch compare
    results = dch2.compare_batch("base.png", ["t1.png", "t2.png"])
"""

from dch2.core.pipeline import generate_dch
from dch2.core.comparator import compare, compare_batch

__version__ = "2.0.0"
__author__  = "DCH Research"
__all__     = ["generate_dch", "compare", "compare_batch"]