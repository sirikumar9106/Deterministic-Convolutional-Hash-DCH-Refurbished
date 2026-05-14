"""
dch2.similarity.hamming
-----------------------
Hamming distance and normalized similarity score between binary hash vectors.
"""

from typing import List


def compute_hamming_distance(hash_a: List[int], hash_b: List[int]) -> int:
    """
    Compute Hamming distance between two binary hash vectors.

    Hamming distance is the number of bit positions where the two
    vectors differ. Lower distance means higher similarity.

    Parameters
    ----------
    hash_a : List[int]
        First binary hash vector.
    hash_b : List[int]
        Second binary hash vector. Must be same length as hash_a.

    Returns
    -------
    int
        Number of differing bit positions. Range [0, len(hash_a)].

    Raises
    ------
    ValueError
        If hash vectors are not the same length.
    """
    if len(hash_a) != len(hash_b):
        raise ValueError(
            f"Hash vectors must be equal length. "
            f"Got {len(hash_a)} and {len(hash_b)}."
        )

    return sum(a != b for a, b in zip(hash_a, hash_b))


def compute_similarity(hash_a: List[int], hash_b: List[int]) -> float:
    """
    Compute normalized similarity score between two binary hash vectors.

    Converts Hamming distance to a similarity score in [0, 1]:
        similarity = 1 - (hamming_distance / hash_length)

    A score of 1.0 means identical hashes.
    A score of 0.0 means every bit differs.
    A score of 0.5 on a random pair is expected by chance.

    Parameters
    ----------
    hash_a : List[int]
        First binary hash vector.
    hash_b : List[int]
        Second binary hash vector.

    Returns
    -------
    float
        Similarity score in [0, 1].
    """
    distance = compute_hamming_distance(hash_a, hash_b)
    return 1.0 - (distance / len(hash_a))