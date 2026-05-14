# DCH2 — Deterministic Convolutional Hashing v2

A zero-model perceptual image hashing system that detects duplicates, crops, rotations, and mirrors without AI — cutting dataset size before model inference to reduce compute cost and carbon footprint. 4% more accurate than traditional methods, proving classical algorithms still solve problems where even modern AI struggles.

---

## Why DCH Exists

Traditional AI pipelines process every image in a dataset from scratch. DCH acts as a **pre-filter** — it sorts a large dataset into confident matches and confident non-matches first, so the AI model only processes the ambiguous remainder. Less compute, less energy, same accuracy.

**Zero-model constraint:** No weights, no training data, no GPU required. Fully deterministic — same input always produces same output on any hardware.

---

## How It Differs from Existing Methods

| Method | Rotation | Flip | Crop | Color Aware | False Positive Rate |
|--------|----------|------|------|-------------|-------------------|
| aHash  | ❌ | ❌ | Partial | ❌ | High |
| pHash  | ❌ | ❌ | Partial | ❌ | High |
| dHash  | ❌ | ❌ | Partial | ❌ | High |
| wHash  | ❌ | ❌ | Partial | ❌ | High |
| Crypto Hash | ❌ | ❌ | ❌ | ❌ | N/A — exact only |
| **DCH2** | ✅ | ✅ | ✅ | ✅ | ~6% on 20k dataset |

Traditional methods hash raw pixels or DCT frequencies scaled to 8×8. A flipped image produces completely different frequency patterns at that scale. DCH2 extracts **structural feature maps first** (like CNN conv layers) then hashes those — the same edges exist in a flipped image, just repositioned, so the hash survives the transform.

---

## How the Backend Works

### Pipeline Overview

```
Image
  │
  ▼
Preprocessing (core/preprocess.py)
  • Load BGR
  • Resize to 256×256 (INTER_AREA)
  • Saturation-adaptive L+H channel:
      h_weight = H_MIN + (mean_sat / SAT_THRESHOLD) * (H_MAX - H_MIN)
      combined = (1 - h_weight) * L + h_weight * (H * S)
  │
  ├─────────────────────────────────┐
  ▼                                 ▼
Edge Map                      Laplacian Map
|SobelX| + |SobelY|           |∇²f|
  │                                 │
  └──────── DoG Blend ──────────────┘
      dog_weight = f(HH_subband_energy)
      blended = (1 - dog_weight) * primary + dog_weight * DoG
  │
  ▼
Spectral Decomposition (features/spectral.py)
  • Stationary Wavelet Transform (SWT, Haar, level=1)
  • Magnitude map: sqrt(0.33*LH² + 0.33*HL² + 0.33*HH²)
  │
  ├─────────────────────┐
  ▼                     ▼
Spatial Map         Invariant Map
resize to 8×8       Radial Energy Profile (REP)
                    — bins pixels by distance from center
                    — rotation/flip invariant
  │                     │
  ▼                     ▼
Quantization (encoding/quantizer.py)
  • Split 8×8 into 4 overlapping quadrants (25% overlap)
  • Local median threshold per quadrant
  • 16 bits per quadrant → 64 bits per map
  │
  ▼
Pack (encoding/packing.py)
  spatial_part   = edge_spatial   + lap_spatial    (128 bits)
  invariant_part = edge_invariant + lap_invariant  (128 bits)
  combined       = spatial_part   + invariant_part (256 bits)
  hex            = 64-character hex string
  │
  ▼
Structural Stats (similarity/stats.py)
  [sobel_mean, sobel_var, sobel_skew, lap_mean, lap_var, lap_skew]
```

### Four-Layer Decision System (core/comparator.py)

**Layer 1 — Hash Similarity**
```
sim_spatial   = 1 - hamming(bits_a[0:128],   bits_b[0:128])   / 128
sim_invariant = 1 - hamming(bits_a[128:256],  bits_b[128:256]) / 128
```

**Layer 2 — Coherence Adjustment**
```
Fires when: 0.60 ≤ sim_path ≤ 0.70 AND opposing_path < 0.55
Effect:     sim_path = sim_path * (0.7 + 0.3 * coherence_score)
Purpose:    Pushes false positives in ambiguous zone downward
```

Coherence score measures whether matching bits form consecutive runs (true match) or are scattered randomly (false positive). Three components:
- Longest run score (weight 0.35)
- Cluster ratio — fraction of matching bits in runs ≥ 3 (weight 0.40)
- Fragmentation penalty — average run length / 8 (weight 0.25)

**Layer 3 — Statistical Rescue**
```
Fires when: final < threshold AND sim_stats ≥ 0.82
Effect:     final = 0.70 * final + 0.30 * sim_stats
Purpose:    Recovers compounded transforms (mirror + rotation + crop)
            where both hash paths lose signal simultaneously
```

Stats similarity = mean normalized absolute difference across 6 moments:
```
sim_stats = mean( max(0, 1 - |a_i - b_i| / max(|a_i|, |b_i|)) ) for i in 6
```

**Layer 4 — Path Divergence Adjustment**
```
path_divergence = abs(sim_spatial - sim_invariant)

Penalty fires when:  final ≥ threshold AND divergence < 0.08
Effect:              final = final * (0.60 + 0.40 * divergence / 0.08)
Purpose:             False positives score similarly on both paths (gap < 0.06)
                     True matches always have one path dominant (gap > 0.09)

Rescue fires when:   0.55 ≤ final < threshold AND divergence > 0.10
Effect:              final = final * 1.08
Purpose:             Recovers borderline true matches just below threshold
```

**Final verdict:**
```
SELECT if final ≥ 0.60 else REJECT
```

---

## Folder Structure

```
dcHash_Refurbished/
├── dch2/                    ← package — copy this into your project
│   ├── __init__.py          ← public API: generate_dch, compare, compare_batch
│   ├── core/
│   │   ├── preprocess.py    ← image loading and L+H channel
│   │   ├── pipeline.py      ← hash generation
│   │   └── comparator.py    ← comparison and verdict
│   ├── features/
│   │   ├── filters.py       ← Sobel, Laplacian, DoG
│   │   └── spectral.py      ← SWT, spatial map, REP
│   ├── encoding/
│   │   ├── quantizer.py     ← overlapping quadrant binarization
│   │   └── packing.py       ← bit packing and hex conversion
│   └── similarity/
│       ├── hamming.py       ← Hamming distance
│       ├── coherence.py     ← bit coherence scoring
│       └── stats.py         ← structural moment fingerprinting
├── test/                    ← put your query images here
├── data/                    ← put your dataset images here
├── test.py                  ← run this to test
├── requirements.txt
└── .gitignore
```

---

## Setup

```powershell
python -m venv dchenv
dchenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Running Tests

1. Put query images in `test/`
2. Put dataset images in `data/`
3. Run:

```powershell
python test.py
```

Each image in `test/` is compared against every image in `data/`. Output shows all six signals per comparison plus final verdict.

**Output legend:**
- `[SELECT]` — perceptually similar, accepted as match
- `[REJECT]` — perceptually different, not a match
- `★` — stats rescue activated (compounded transform case)
- `◆` — divergence rescue activated (borderline true match)

---

## Using DCH2 as a Library

Copy the `dch2/` folder into your project. Then:

```python
import dch2

# Generate hash for one image
result = dch2.generate_dch("image.png")
print(result.hex)      # 64-char hex hash
print(result.bits)     # 256-bit list
print(result.stats)    # 6-element stats vector

# Compare two images — single API call
result = dch2.compare("imageA.png", "imageB.png")
print(result.verdict)     # SELECT or REJECT
print(result.is_match())  # True or False
print(result.similarity)  # final score 0-1

# Batch compare — base hash generated once, efficient for large datasets
results = dch2.compare_batch(
    "base.png",
    ["img1.png", "img2.png", "img3.png"],
    threshold=0.60
)
for r in results:
    print(f"{r.image_b} → {r.verdict} ({r.similarity:.2%})")
```

### Adjusting the threshold

```python
# More strict — fewer false positives, more false negatives
result = dch2.compare("a.png", "b.png", threshold=0.70)

# More lenient — fewer false negatives, more false positives
result = dch2.compare("a.png", "b.png", threshold=0.55)
```

---

## Dependencies

```
numpy, scipy, opencv-python, Pillow, PyWavelets, Flask, matplotlib, pytest
```

---

## Performance

Tested on 20,000 image dataset:
- Overall accuracy: **94%**
- False negative rate: **0.3%**
- False positive rate: **~6%**
- Improvement over traditional pHash/aHash/whash: **+7%**
- **Note:** These benchmarks were obtained using a limited, friendly dataset and are not representative of all scenarios. I encourage you to stress-test this on larger, real-world datasets and share any optimizations you discover. PRs and discussions are always welcome!!!