
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────── #
RUNS_ROOT    = os.environ.get("RUNS_ROOT", "./runs")
RESULTS_ROOT = "$RESULTS_DIR"

CONFIGS = ["A_baseline", "B_bmc", "C_siamese", "D_bmc_sia"]
SEEDS   = [42, 43, 44]


def run_dir(config: str, seed: int) -> Path:
    """Map (config, seed) -> the run directory on disk."""
    if seed == 42:
        return Path(RUNS_ROOT) / config
    return Path(RUNS_ROOT) / f"{config}_s{seed}"


def load_predictions(rd: Path, split: str) -> np.ndarray | None:
    """Load preds_{split}.npy from eval_noTTA folder. Returns None if missing."""
    p = rd / "eval_noTTA" / f"preds_{split}.npy"
    if not p.is_file():
        return None
    return np.load(p)


def compute_metric(preds_direct: np.ndarray,
                   preds_inverse: np.ndarray) -> Dict:
    """
    Return dict of anti-symmetry statistics, including the per-mutation
    violation array for downstream plotting.
    """
    if len(preds_direct) != len(preds_inverse):
        log.warning(f"  length mismatch: direct={len(preds_direct)} "
                    f"inverse={len(preds_inverse)} — taking min")
        n = min(len(preds_direct), len(preds_inverse))
        preds_direct  = preds_direct[:n]
        preds_inverse = preds_inverse[:n]

    # Drop any NaN entries (defensive — shouldn't occur)
    valid = np.isfinite(preds_direct) & np.isfinite(preds_inverse)
    pd_v  = preds_direct[valid]
    pi_v  = preds_inverse[valid]
    if len(pd_v) < 10:
        return {"n_pairs": int(valid.sum()), "error": "too few valid pairs"}

    eps = np.abs(pd_v + pi_v)
    pearson_neg = np.corrcoef(pd_v, -pi_v)[0, 1]

    return {
        "n_pairs":      int(len(pd_v)),
        "mean_eps":     float(eps.mean()),
        "median_eps":   float(np.median(eps)),
        "std_eps":      float(eps.std()),
        "p_below_0.5":  float((eps < 0.5).mean()),
        "p_below_1.0":  float((eps < 1.0).mean()),
        "pearson_neg":  float(pearson_neg),
        "_eps_per_mut": eps.tolist(),       # for plotting; truncated on dump
    }


def main():
    out_root = Path(RESULTS_ROOT) / "antisymmetry"
    out_root.mkdir(parents=True, exist_ok=True)

    log.info(f"Output: {out_root}")
    log.info(f"Configs: {CONFIGS}")
    log.info(f"Seeds:   {SEEDS}")

    per_ckpt_rows = []
    per_ckpt_json = {}

    for cfg in CONFIGS:
        for seed in SEEDS:
            rd  = run_dir(cfg, seed)
            tag = f"{cfg}_s{seed}"
            pd_arr = load_predictions(rd, "ssym_direct")
            pi_arr = load_predictions(rd, "ssym_inverse")

            if pd_arr is None or pi_arr is None:
                log.warning(f"  SKIP {tag}: missing preds at {rd}")
                continue

            res = compute_metric(pd_arr, pi_arr)
            if "error" in res:
                log.warning(f"  {tag}: {res['error']}")
                continue

            log.info(f"  {tag:24s} n={res['n_pairs']:3d} "
                     f"mean_eps={res['mean_eps']:.3f} "
                     f"median={res['median_eps']:.3f} "
                     f"p<0.5={res['p_below_0.5']:.2f} "
                     f"r(-)= {res['pearson_neg']:+.3f}")

            per_ckpt_rows.append({
                "config":      cfg,
                "seed":        seed,
                "tag":         tag,
                "n_pairs":     res["n_pairs"],
                "mean_eps":    res["mean_eps"],
                "median_eps":  res["median_eps"],
                "std_eps":     res["std_eps"],
                "p_below_0.5": res["p_below_0.5"],
                "p_below_1.0": res["p_below_1.0"],
                "pearson_neg": res["pearson_neg"],
            })
            per_ckpt_json[tag] = res

    if not per_ckpt_rows:
        log.error("No checkpoints had usable predictions. Aborting.")
        return

    # Save per-checkpoint
    df_per = pd.DataFrame(per_ckpt_rows)
    per_csv = out_root / "per_checkpoint.csv"
    df_per.to_csv(per_csv, index=False, float_format="%.4f")
    log.info(f"\nSaved: {per_csv}")

    # JSON dump with per-mutation arrays (for plotting / further analysis)
    per_json = out_root / "per_checkpoint.json"
    with open(per_json, "w") as f:
        json.dump(per_ckpt_json, f, indent=2)
    log.info(f"Saved: {per_json}")

    # Aggregate across seeds within each config
    summary_rows = []
    for cfg in CONFIGS:
        sub = df_per[df_per["config"] == cfg]
        if len(sub) == 0:
            continue
        row = {"config": cfg, "n_seeds": len(sub)}
        for col in ["mean_eps", "median_eps", "p_below_0.5",
                    "p_below_1.0", "pearson_neg"]:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"]  = float(sub[col].std()) if len(sub) > 1 else 0.0
        summary_rows.append(row)

    df_sum = pd.DataFrame(summary_rows)
    sum_csv = out_root / "summary_by_config.csv"
    df_sum.to_csv(sum_csv, index=False, float_format="%.4f")
    log.info(f"Saved: {sum_csv}")

    # Pretty-print the summary table
    log.info("\n" + "=" * 78)
    log.info("ANTI-SYMMETRY SUMMARY (across seeds)")
    log.info("=" * 78)
    log.info(f"{'Config':<14s} {'mean ε_sym':>13s} {'median':>10s} "
             f"{'p<0.5':>8s} {'r(-)':>8s}")
    log.info("-" * 78)
    for _, r in df_sum.iterrows():
        log.info(
            f"{r['config']:<14s} "
            f"{r['mean_eps_mean']:>5.3f} ± {r['mean_eps_std']:.3f}   "
            f"{r['median_eps_mean']:>6.3f}    "
            f"{r['p_below_0.5_mean']:>6.2f}   "
            f"{r['pearson_neg_mean']:>+6.3f}"
        )
    log.info("=" * 78)
    log.info("Lower ε_sym = better anti-symmetry.")
    log.info("Higher |r(-)| toward 1 = better anti-symmetry (perfect = 1.0).")


if __name__ == "__main__":
    main()
