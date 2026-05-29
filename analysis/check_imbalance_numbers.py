
import os
import numpy as np
from pathlib import Path

BASE = Path(os.environ.get("EVAL_DIR", "./runs/A_baseline/eval_noTTA"))
DATASETS = {
    "Megascale (test)": BASE / "targets_megascale_test.npy",
    "S669":             BASE / "targets_s669.npy",
    "S461":             BASE / "targets_s461.npy",     # if exists
    "Ssym (direct)":    BASE / "targets_ssym_direct.npy",  # if exists
}

def report(name, data, thresholds=(0.5, 1.0)):
    print(f"\n=== {name} (n = {len(data):,}) ===")
    print(f"  Range  [{data.min():+.2f}, {data.max():+.2f}]  "
          f"Mean {data.mean():+.3f}  Median {np.median(data):+.3f}  "
          f"Std {data.std():.3f}")
    for thresh in thresholds:
        # Standard biophysics convention: ddG < 0 = stabilizing
        stab   = int(np.sum(data < -thresh))
        destab = int(np.sum(data > +thresh))
        neut   = len(data) - stab - destab
        N      = len(data)
        print(f"  Threshold ±{thresh:.1f} kcal/mol:")
        print(f"    Stabilizing    (ΔΔG < {-thresh:+.1f}): "
              f"{stab:6,}  ({100*stab/N:5.1f}%)")
        print(f"    Neutral        ({-thresh:+.1f} ≤ ΔΔG ≤ {+thresh:+.1f}): "
              f"{neut:6,}  ({100*neut/N:5.1f}%)")
        print(f"    Destabilizing  (ΔΔG > {+thresh:+.1f}): "
              f"{destab:6,}  ({100*destab/N:5.1f}%)")

def main():
    for name, path in DATASETS.items():
        if not path.is_file():
            print(f"\n=== {name} ===  [SKIPPED: file not found at {path}]")
            continue
        data = np.load(path)
        report(name, data)

    print("\n--- Done. Paste this output back to Claude. ---")

if __name__ == "__main__":
    main()
