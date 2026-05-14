"""
dch2.similarity.coherence
-------------------------
Bit coherence scoring for match quality assessment.

Answers a different question than Hamming similarity:
    Hamming asks: "How many bits agree?"
    Coherence asks: "Are the agreeing bits clustered or scattered?"

A true perceptual match produces clustered matching bits because
the shared content occupies a contiguous spatial region — the bits
encoding that region form consecutive runs in the hash vector.

A false positive produces scattered matching bits — coincidental
agreements at random positions with no spatial coherence.

Two images can have the same Hamming similarity score but completely
different coherence scores, revealing whether the similarity is
structurally meaningful or coincidental.
"""

from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Minimum run length to be considered a non-trivial cluster
# Runs shorter than this are treated as scattered individual matches
MIN_CLUSTER_LENGTH = 3

# Normalization target for average run length scoring
# A run of this length scores 1.0 on the fragmentation component
AVG_RUN_NORM = 8.0

# Weights for the three coherence components
WEIGHT_LONGEST_RUN  = 0.35
WEIGHT_CLUSTER_RATIO = 0.40
WEIGHT_FRAGMENTATION = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_match_vector(bits_a: List[int], bits_b: List[int]) -> List[int]:
    """
    Build a binary match vector: 1 where bits agree, 0 where they differ.
    """
    return [1 if a == b else 0 for a, b in zip(bits_a, bits_b)]


def _find_runs(match_vector: List[int]) -> List[int]:
    """
    Find all consecutive runs of matching bits (value=1).

    Returns a list of run lengths. A run is a maximal consecutive
    sequence of 1s in the match vector.
    """
    runs        = []
    current_run = 0

    for bit in match_vector:
        if bit == 1:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
                current_run = 0

    if current_run > 0:
        runs.append(current_run)

    return runs


def _compute_components(
    runs:          List[int],
    total_bits:    int,
    total_matching: int
) -> Tuple[float, float, float]:
    """
    Compute the three coherence components from run statistics.

    Returns
    -------
    Tuple (longest_run_score, cluster_ratio, fragmentation_score)
    """
    longest_run       = max(runs)
    num_runs          = len(runs)
    clustered_bits    = sum(r for r in runs if r >= MIN_CLUSTER_LENGTH)

    # Component 1: How long is the longest cluster relative to total bits
    longest_run_score = longest_run / total_bits

    # Component 2: Fraction of matching bits in non-trivial clusters
    cluster_ratio = clustered_bits / total_matching if total_matching > 0 else 0.0

    # Component 3: Average run length — penalizes many short scattered runs
    avg_run_length    = total_matching / num_runs if num_runs > 0 else 0.0
    fragmentation     = min(avg_run_length / AVG_RUN_NORM, 1.0)

    return longest_run_score, cluster_ratio, fragmentation


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def compute_coherence_score(bits_a: List[int], bits_b: List[int]) -> float:
    """
    Compute bit coherence score between two hash vectors.

    Measures whether matching bits form spatial clusters (true match)
    or are scattered randomly (false positive).

    Three components weighted and combined:
        1. Longest run score   (weight 0.35)
           Longest consecutive match run / total bits.
           A long unbroken run strongly indicates true match.

        2. Cluster ratio       (weight 0.40)
           Fraction of matching bits that belong to runs >= 3.
           Distinguishes meaningful clusters from isolated coincidences.

        3. Fragmentation score (weight 0.25)
           Average run length normalized to AVG_RUN_NORM.
           Penalizes many short scattered runs (false positive pattern).

    Parameters
    ----------
    bits_a : List[int]
        First binary hash vector (spatial or invariant segment).
    bits_b : List[int]
        Second binary hash vector, same length as bits_a.

    Returns
    -------
    float
        Coherence score in [0, 1].
        High (> 0.6) — matching bits are clustered, likely true match.
        Low  (< 0.4) — matching bits are scattered, likely false positive.
        Returns 0.0 if no matching bits exist.
    """
    total_bits    = len(bits_a)
    match_vector  = _build_match_vector(bits_a, bits_b)
    total_matching = sum(match_vector)

    if total_matching == 0:
        return 0.0

    runs = _find_runs(match_vector)

    if not runs:
        return 0.0

    longest_run_score, cluster_ratio, fragmentation = _compute_components(
        runs, total_bits, total_matching
    )

    coherence = (
        (WEIGHT_LONGEST_RUN   * longest_run_score) +
        (WEIGHT_CLUSTER_RATIO * cluster_ratio)     +
        (WEIGHT_FRAGMENTATION * fragmentation)
    )

    return min(float(coherence), 1.0)