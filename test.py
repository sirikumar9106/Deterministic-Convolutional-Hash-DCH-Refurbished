"""
test.py
-------
Batch similarity testing for DCH2.

Compares every image in /test folder against every image in /data folder.
Place this file in the root of dcHash Refurbished/ alongside the
test/ and data/ folders.

Usage
-----
    python test.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import dch2

TEST_FOLDER      = os.path.join(ROOT_DIR, "test")
DATA_FOLDER      = os.path.join(ROOT_DIR, "data")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
THRESHOLD        = 0.60


def get_image_files(folder: str) -> list:
    if not os.path.exists(folder):
        print(f"[ERROR] Folder not found: {folder}")
        sys.exit(1)
    files = []
    for filename in sorted(os.listdir(folder)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(os.path.join(folder, filename))
    if not files:
        print(f"[ERROR] No image files found in: {folder}")
        sys.exit(1)
    return files


def format_bar(similarity: float, width: int = 20) -> str:
    filled = int(similarity * width)
    bar    = "█" * filled + "·" * (width - filled)
    return f"[{bar}]"


def print_separator(char: str = "─", width: int = 110):
    print(char * width)


def run_comparisons(test_files: list, data_files: list) -> list:
    all_results = []
    total_pairs = len(test_files) * len(data_files)
    current     = 0

    print(f"\n  Running {total_pairs} comparisons "
          f"({len(test_files)} test × {len(data_files)} data images)...\n")

    for test_path in test_files:
        test_name = os.path.basename(test_path)

        try:
            dch2.generate_dch(test_path)
        except ValueError as e:
            print(f"  [SKIP] Could not load test image: {test_name} — {e}")
            continue

        for data_path in data_files:
            data_name = os.path.basename(data_path)
            current  += 1

            print(f"  [{current:>4}/{total_pairs}] "
                  f"{test_name} vs {data_name}...", end="\r")

            try:
                result = dch2.compare(test_path, data_path, threshold=THRESHOLD)

                all_results.append({
                    "test_image":          test_name,
                    "data_image":          data_name,
                    "spatial":             result.spatial,
                    "invariant":           result.invariant,
                    "coherence_spatial":   result.coherence_spatial,
                    "coherence_invariant": result.coherence_invariant,
                    "stats":               result.stats_similarity,
                    "similarity":          result.similarity,
                    "verdict":             result.verdict,
                    "stats_rescued":       result.stats_rescued,
                    "divergence_rescued":  result.divergence_rescued,
                })

            except Exception as e:
                print(f"\n  [ERROR] {test_name} vs {data_name}: {e}")
                all_results.append({
                    "test_image":          test_name,
                    "data_image":          data_name,
                    "spatial":             0.0,
                    "invariant":           0.0,
                    "coherence_spatial":   0.0,
                    "coherence_invariant": 0.0,
                    "stats":               0.0,
                    "similarity":          0.0,
                    "verdict":             "ERROR",
                    "stats_rescued":       False,
                    "divergence_rescued":  False,
                })

    print(" " * 80, end="\r")
    return all_results


def print_results_for_test_image(test_name: str, results: list):
    print_separator("═")
    print(f"  TEST IMAGE: {test_name}")
    print_separator("═")

    print(
        f"  {'Data Image':<30} "
        f"{'spa':>6}  {'inv':>6}  "
        f"{'spa_ch':>7}  {'inv_ch':>7}  "
        f"{'stats':>6}  "
        f"{'final':>6}  "
        f"{'':20}  verdict"
    )
    print_separator("─")

    accepted = []
    rejected = []

    for r in results:
        tags = ""
        if r["stats_rescued"]:      tags += "★"
        if r["divergence_rescued"]: tags += "◆"
        verdict_display = f"[{r['verdict']}]{tags}"

        print(
            f"  {r['data_image']:<30} "
            f"{r['spatial']*100:>5.1f}%  "
            f"{r['invariant']*100:>5.1f}%  "
            f"{r['coherence_spatial']*100:>6.1f}%  "
            f"{r['coherence_invariant']*100:>6.1f}%  "
            f"{r['stats']*100:>5.1f}%  "
            f"{r['similarity']*100:>5.1f}%  "
            f"{format_bar(r['similarity']):<22} "
            f"{verdict_display}"
        )

        if r["verdict"] == "SELECT":
            accepted.append(r)
        else:
            rejected.append(r)

    print_separator("─")

    total = len(results)
    print(f"\n  SUMMARY for {test_name}:")
    print(f"    Total comparisons : {total}")
    print(f"    Accepted (SELECT) : {len(accepted)}")
    print(f"    Rejected (REJECT) : {len(rejected)}")

    if accepted:
        print(f"\n  Accepted matches:")
        for r in sorted(accepted, key=lambda x: x["similarity"], reverse=True):
            tags = ""
            if r["stats_rescued"]:      tags += " [STATS RESCUE]"
            if r["divergence_rescued"]: tags += " [DIV RESCUE]"
            print(f"    • {r['data_image']:<30} "
                  f"similarity = {r['similarity']*100:.2f}%{tags}")
    print()


def print_grand_summary(all_results: list, test_files: list):
    print_separator("═")
    print("  GRAND SUMMARY")
    print_separator("═")

    total     = len(all_results)
    accepted  = sum(1 for r in all_results if r["verdict"] == "SELECT")
    rejected  = sum(1 for r in all_results if r["verdict"] == "REJECT")
    errors    = sum(1 for r in all_results if r["verdict"] == "ERROR")
    s_rescued = sum(1 for r in all_results if r["stats_rescued"])
    d_rescued = sum(1 for r in all_results if r["divergence_rescued"])

    print(f"  Test images          : {len(test_files)}")
    print(f"  Total comparisons    : {total}")
    print(f"  Accepted             : {accepted}  ({accepted/total*100:.1f}%)")
    print(f"  Rejected             : {rejected}  ({rejected/total*100:.1f}%)")
    print(f"  Stats rescued    (★) : {s_rescued}")
    print(f"  Divergence rescued(◆): {d_rescued}")
    if errors:
        print(f"  Errors               : {errors}")

    print_separator("═")
    print()


def main():
    print("\n" + "═" * 110)
    print("  DCH2 — Deterministic Convolutional Hashing v2")
    print("  Batch Similarity Test")
    print("═" * 110)
    print(f"  Test folder : {TEST_FOLDER}")
    print(f"  Data folder : {DATA_FOLDER}")
    print(f"  Threshold   : {THRESHOLD}")

    test_files = get_image_files(TEST_FOLDER)
    data_files = get_image_files(DATA_FOLDER)

    print(f"\n  Found {len(test_files)} test image(s)")
    print(f"  Found {len(data_files)} data image(s)")

    all_results = run_comparisons(test_files, data_files)

    for test_path in test_files:
        test_name     = os.path.basename(test_path)
        image_results = [r for r in all_results if r["test_image"] == test_name]
        print_results_for_test_image(test_name, image_results)

    print_grand_summary(all_results, test_files)


if __name__ == "__main__":
    main()