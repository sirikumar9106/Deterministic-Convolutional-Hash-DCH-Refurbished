"""
dch2.core — orchestration layer.
    preprocess.py  — image loading and L+H channel conversion
    pipeline.py    — hash generation, returns DCHResult
    comparator.py  — comparison logic, returns CompareResult
"""
from dch2.core.pipeline import generate_dch
from dch2.core.comparator import compare, compare_batch