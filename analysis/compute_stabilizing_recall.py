
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


RUNS_ROOT    = os.environ.get("RUNS_ROOT", "./runs")
RESULTS_ROOT = "$RESULTS_DIR"

CONFIGS  = ["A_baseline", "B_bmc", "C_siamese", "D_bmc_sia"]
SEEDS    = [42, 43, 44]
DATASETS = [
    "megascale_test", "fireport_hf", "ssym_direct", "ssym_inverse",
    "s669", "s461", "s783", "s8754", "s2648", "s571", "s4346",
]
STAB_THRESHOLD = 0.5      # |ddG| > 0.5 to be called "clearly stabilizing"
TOP_K_PCTS     = [10, 25, 50]


def run_dir(config: str, seed: int) -> Path:
    if seed == 42:
        return Path(RUNS_ROOT) / config
    return Path(RUNS_ROOT) / f"{config}_s{seed}"


def load(rd: Path, split: str):
    p = rd / "eval_noTTA" / f"preds_{split}.npy"
    t = rd / "eval_noTTA" / f"targets_{split}.npy"
    if not (p.is_file() and t.is_file()):
        return None, None
    return np.load(p), np.load(t)


def compute_recall_at_k(preds: np.ndarray, targets: np.ndarray,
                        k_pct: int, stab_threshold: float = STAB_THRESHOLD):
    """
    Sign convention: stabilizing = target > +stab_threshold (SPURS internal).
    Ranking: highest predicted ddG first (most stabilizing first).

    Returns:
        recall    : fraction of true stabilizing mutations in top-k%
        precision : fraction of top-k% predictions that are truly stabilizing
        n_stab    : number of truly stabilizing mutations
        n_top     : number of top-k% predictions
    """
    valid = np.isfinite(preds) & np.isfinite(targets)
    p_v = preds[valid]
    t_v = targets[valid]
    n = len(t_v)
    if n < 20:
        return None

    stab_mask = (t_v > stab_threshold)
    n_stab = int(stab_mask.sum())
    if n_stab == 0:
        # No stabilizing in this dataset (e.g. ssym_inverse is all reversed
        # destabilizing). Return None — caller handles.
        return None

    # Top-k% by predicted ΔΔG (most-stabilizing first)
    k_count = max(1, int(round(k_pct / 100.0 * n)))
    top_idx = np.argsort(-p_v)[:k_count]
    top_mask = np.zeros(n, dtype=bool)
    top_mask[top_idx] = True

    n_in_top_and_stab = int((top_mask & stab_mask).sum())
    recall    = n_in_top_and_stab / n_stab
    precision = n_in_top_and_stab / k_count

    return {
        "recall":   float(recall),
        "precision": float(precision),
        "n_stab":   n_stab,
        "n_top":    k_count,
        "n_total":  n,
        "stab_frac": float(n_stab / n),
    }


def main():
    out_root = Path(RESULTS_ROOT) / "stabilizing_recall"
    out_root.mkdir(parents=True, exist_ok=True)
    log.info(f"Output: {out_root}")

    # ── Sanity check: stabilizing fraction in each test set ──
    log.info("\nGround-truth stabilizing fraction by dataset (sanity check):")
    log.info(f"  (using threshold target > +{STAB_THRESHOLD})")
    sanity_rows = []
    for ds in DATASETS:
        # Read the first available checkpoint's targets (they're identical
        # across configs by construction).
        t = None
        for cfg in CONFIGS:
            for seed in SEEDS:
                _, t_try = load(run_dir(cfg, seed), ds)
                if t_try is not None:
                    t = t_try
                    break
            if t is not None:
                break
        if t is None:
            log.warning(f"  {ds:20s}  NO TARGETS FOUND")
            continue
        valid = np.isfinite(t)
        t_v = t[valid]
        stab = (t_v > STAB_THRESHOLD).sum()
        destab = (t_v < -STAB_THRESHOLD).sum()
        n = len(t_v)
        log.info(f"  {ds:20s}  n={n:5d}  "
                 f"stab={stab:4d} ({100*stab/n:5.1f}%)  "
                 f"destab={destab:4d} ({100*destab/n:5.1f}%)")
        sanity_rows.append({
            "dataset": ds, "n": n,
            "n_stab": int(stab), "n_destab": int(destab),
            "stab_frac": float(stab / n),
        })
    pd.DataFrame(sanity_rows).to_csv(
        out_root / "ground_truth_distribution.csv",
        index=False, float_format="%.4f",
    )
    log.info(f"Saved ground-truth distribution to "
             f"{out_root / 'ground_truth_distribution.csv'}")

    # ── Per-checkpoint recall ──
    rows = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            rd = run_dir(cfg, seed)
            for ds in DATASETS:
                preds, targets = load(rd, ds)
                if preds is None:
                    continue
                for k_pct in TOP_K_PCTS:
                    res = compute_recall_at_k(preds, targets, k_pct)
                    if res is None:
                        continue
                    rows.append({
                        "config":     cfg,
                        "seed":       seed,
                        "dataset":    ds,
                        "k_pct":      k_pct,
                        "recall":     res["recall"],
                        "precision":  res["precision"],
                        "n_stab":     res["n_stab"],
                        "n_top":      res["n_top"],
                        "n_total":    res["n_total"],
                        "stab_frac":  res["stab_frac"],
                    })
    if not rows:
        log.error("No per-checkpoint recall computed. Aborting.")
        return

    df_per = pd.DataFrame(rows)
    per_csv = out_root / "per_checkpoint.csv"
    df_per.to_csv(per_csv, index=False, float_format="%.4f")
    log.info(f"\nSaved: {per_csv}")

    # ── Summarise: per (config, dataset, k_pct), aggregate across seeds ──
    grouped = df_per.groupby(["config", "dataset", "k_pct"])
    summary_rows = []
    for (cfg, ds, k), sub in grouped:
        summary_rows.append({
            "config":     cfg,
            "dataset":    ds,
            "k_pct":      k,
            "recall_mean":    float(sub["recall"].mean()),
            "recall_std":     float(sub["recall"].std()) if len(sub) > 1 else 0.0,
            "precision_mean": float(sub["precision"].mean()),
            "precision_std":  float(sub["precision"].std()) if len(sub) > 1 else 0.0,
            "n_seeds":    len(sub),
        })
    df_sum = pd.DataFrame(summary_rows)
    sum_csv = out_root / "summary_by_config.csv"
    df_sum.to_csv(sum_csv, index=False, float_format="%.4f")
    log.info(f"Saved: {sum_csv}")

    # ── Print headline summary on S669 ──
    log.info("\n" + "=" * 78)
    log.info("STABILIZING-MUTATION RECALL ON S669 (the OOD headline)")
    log.info("=" * 78)
    s669 = df_sum[df_sum["dataset"] == "s669"]
    if len(s669) > 0:
        log.info(f"{'Config':<14s} | "
                 + " | ".join(f"k={k}%  recall (prec)" for k in TOP_K_PCTS))
        log.info("-" * 78)
        for cfg in CONFIGS:
            sub = s669[s669["config"] == cfg]
            if len(sub) == 0:
                continue
            line = f"{cfg:<14s} | "
            parts = []
            for k in TOP_K_PCTS:
                rs = sub[sub["k_pct"] == k]
                if len(rs) > 0:
                    r = rs.iloc[0]
                    parts.append(
                        f"{r['recall_mean']:.2f}±{r['recall_std']:.2f} "
                        f"({r['precision_mean']:.2f})"
                    )
                else:
                    parts.append("—")
            line += " | ".join(parts)
            log.info(line)
    log.info("=" * 78)

    log.info("")
    log.info("Now run:  python build_results_tables.py")


if __name__ == "__main__":
    main()
