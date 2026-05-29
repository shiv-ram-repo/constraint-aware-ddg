
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

CONFIGS = ["A_baseline", "B_bmc", "C_siamese", "D_bmc_sia"]
SEEDS   = [42, 43, 44]


def run_dir(config: str, seed: int) -> Path:
    if seed == 42:
        return Path(RUNS_ROOT) / config
    return Path(RUNS_ROOT) / f"{config}_s{seed}"


def load_pair(rd: Path):
    """Load (preds_direct, preds_inverse, target_direct, target_inverse)
    from a checkpoint's eval_noTTA folder. Returns None if anything missing."""
    e = rd / "eval_noTTA"
    files = [
        e / "preds_ssym_direct.npy",   e / "preds_ssym_inverse.npy",
        e / "targets_ssym_direct.npy", e / "targets_ssym_inverse.npy",
    ]
    if not all(f.is_file() for f in files):
        return None
    return tuple(np.load(f) for f in files)


def main():
    out_root = Path(RESULTS_ROOT) / "antisymmetry_diagnostic"
    out_root.mkdir(parents=True, exist_ok=True)
    log.info(f"Output: {out_root}")

    # ── Per-checkpoint detailed metrics ──
    per_rows = []
    quartile_rows = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            rd = run_dir(cfg, seed)
            data = load_pair(rd)
            if data is None:
                log.warning(f"  SKIP {cfg}_s{seed}")
                continue
            p_dir, p_inv, t_dir, t_inv = data
            n_orig = len(p_dir)
            if not (len(p_inv) == len(t_dir) == len(t_inv) == n_orig):
                log.warning(f"  {cfg}_s{seed}: length mismatch; truncating")
                n_min = min(len(p_dir), len(p_inv), len(t_dir), len(t_inv))
                p_dir, p_inv = p_dir[:n_min], p_inv[:n_min]
                t_dir, t_inv = t_dir[:n_min], t_inv[:n_min]

            valid = (np.isfinite(p_dir) & np.isfinite(p_inv)
                     & np.isfinite(t_dir) & np.isfinite(t_inv))
            p_dir, p_inv = p_dir[valid], p_inv[valid]
            t_dir, t_inv = t_dir[valid], t_inv[valid]
            n = len(p_dir)
            if n < 50:
                continue

            # Anti-symmetry sum and residual
            pred_sum = p_dir + p_inv                          # should be 0
            tgt_sum  = t_dir + t_inv                          # should be 0 if Ssym pairs are clean
            eps_sym  = np.abs(pred_sum)
            mean_bias = pred_sum.mean()
            std_resid = pred_sum.std()
            # "debiased" ε_sym: subtract the mean bias and re-measure
            eps_debiased = np.abs(pred_sum - mean_bias)

            # Also: ground-truth-aligned residual. The ideal model would
            # predict f_fwd + f_rev = t_fwd + t_rev (which should be ~0
            # but might have small experimental noise).
            eps_vs_truth = np.abs(pred_sum - tgt_sum)

            per_rows.append({
                "config":         cfg,
                "seed":           seed,
                "n":              n,
                "mean_eps":       float(eps_sym.mean()),
                "median_eps":     float(np.median(eps_sym)),
                "mean_bias":      float(mean_bias),       # H1 — systematic bias
                "std_resid":      float(std_resid),       # H2 — noise
                "mean_eps_debiased": float(eps_debiased.mean()),  # ε_sym if we knew bias
                "mean_eps_vs_truth": float(eps_vs_truth.mean()),  # against ground-truth sum
                "tgt_sum_mean":   float(tgt_sum.mean()),  # check: should be near 0
                "tgt_sum_std":    float(tgt_sum.std()),
            })

            # ── Stratified by magnitude quartile of target ──
            mag = np.abs(t_dir)                            # the forward direction's magnitude
            q = np.quantile(mag, [0.25, 0.5, 0.75])
            for qi, (lo, hi, label) in enumerate([
                (-np.inf, q[0],  "Q1_easy"),
                (q[0],    q[1],  "Q2"),
                (q[1],    q[2],  "Q3"),
                (q[2],    np.inf, "Q4_hard"),
            ]):
                mask = (mag >= lo) & (mag < hi)
                if mask.sum() < 5:
                    continue
                quartile_rows.append({
                    "config":  cfg,
                    "seed":    seed,
                    "quartile": label,
                    "n_in_quartile": int(mask.sum()),
                    "mag_range":     f"[{lo if np.isfinite(lo) else 0:.2f}, {hi if np.isfinite(hi) else 99:.2f})",
                    "mean_eps":     float(eps_sym[mask].mean()),
                    "median_eps":   float(np.median(eps_sym[mask])),
                })

    if not per_rows:
        log.error("No diagnostic data computed; aborting.")
        return

    df_per = pd.DataFrame(per_rows)
    df_per.to_csv(out_root / "per_checkpoint.csv",
                  index=False, float_format="%.4f")
    log.info(f"\nSaved: {out_root / 'per_checkpoint.csv'}")

    df_q = pd.DataFrame(quartile_rows)
    df_q.to_csv(out_root / "by_magnitude_quartile.csv",
                index=False, float_format="%.4f")
    log.info(f"Saved: {out_root / 'by_magnitude_quartile.csv'}")

    # ── Aggregate across seeds within each config ──
    log.info("\n" + "=" * 80)
    log.info("CONFIG-LEVEL DIAGNOSTIC (mean across 3 seeds)")
    log.info("=" * 80)
    log.info(f"{'config':<14s} {'ε_sym':>9s} {'bias':>9s} {'σ_resid':>9s} "
             f"{'ε_debias':>10s} {'ε_vs_tgt':>10s}")
    log.info("-" * 80)
    summary_rows = []
    for cfg in CONFIGS:
        sub = df_per[df_per["config"] == cfg]
        if len(sub) == 0:
            continue
        row = {
            "config": cfg,
            "n_seeds": len(sub),
            "mean_eps_mean":    float(sub["mean_eps"].mean()),
            "mean_eps_sd":      float(sub["mean_eps"].std()) if len(sub) > 1 else 0.0,
            "mean_bias_mean":   float(sub["mean_bias"].mean()),
            "mean_bias_sd":     float(sub["mean_bias"].std()) if len(sub) > 1 else 0.0,
            "std_resid_mean":   float(sub["std_resid"].mean()),
            "eps_debiased_mean": float(sub["mean_eps_debiased"].mean()),
            "eps_vs_truth_mean": float(sub["mean_eps_vs_truth"].mean()),
        }
        summary_rows.append(row)
        log.info(
            f"{cfg:<14s} "
            f"{row['mean_eps_mean']:>5.3f}±{row['mean_eps_sd']:.3f} "
            f"{row['mean_bias_mean']:>+6.3f}    "
            f"{row['std_resid_mean']:>7.3f}    "
            f"{row['eps_debiased_mean']:>7.3f}    "
            f"{row['eps_vs_truth_mean']:>7.3f}"
        )

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(out_root / "summary_by_config.csv",
                  index=False, float_format="%.4f")
    log.info(f"\nSaved: {out_root / 'summary_by_config.csv'}")

    # ── Print quartile table ──
    log.info("\n" + "=" * 80)
    log.info("ε_sym BY GROUND-TRUTH MAGNITUDE QUARTILE")
    log.info("=" * 80)
    log.info(f"{'config':<14s} {'Q1_easy':>10s} {'Q2':>10s} {'Q3':>10s} {'Q4_hard':>10s}")
    log.info("-" * 80)
    for cfg in CONFIGS:
        line = f"{cfg:<14s}"
        for qlabel in ["Q1_easy", "Q2", "Q3", "Q4_hard"]:
            sub = df_q[(df_q["config"] == cfg) & (df_q["quartile"] == qlabel)]
            if len(sub) == 0:
                line += "        — "
            else:
                m = sub["mean_eps"].mean()
                s = sub["mean_eps"].std() if len(sub) > 1 else 0.0
                line += f"   {m:.2f}±{s:.2f}"
        log.info(line)

    # ── Write the diagnostic report ──
    md = []
    md.append("# Anti-symmetry diagnostic report\n")
    md.append("## Headline (across seeds)\n")
    md.append(df_sum.to_markdown(index=False, floatfmt=".3f"))
    md.append("\n\n## Interpretation\n")
    md.append("Key columns:")
    md.append("- **mean_eps**: the raw ε_sym we reported in summary.md")
    md.append("- **mean_bias**: average value of (f_fwd + f_rev). If 0, model is "
              "unbiased. If non-zero, model has a systematic additive bias that "
              "inflates ε_sym independent of consistency.")
    md.append("- **std_resid**: standard deviation of (f_fwd + f_rev). This is "
              "the part of ε_sym that comes from per-mutation inconsistency, "
              "after removing the systematic bias.")
    md.append("- **eps_debiased**: ε_sym we'd see if we subtracted the bias. "
              "This measures pure inconsistency.")
    md.append("- **eps_vs_truth**: |pred_fwd + pred_rev − (target_fwd + target_rev)|. "
              "How far the pred-sum is from the target-sum (which should be ~0 "
              "if Ssym pairs are clean).")
    md.append("\n## Quartile-stratified ε_sym\n")
    md.append(df_q.groupby(["config", "quartile"])["mean_eps"]
              .mean().unstack()
              .to_markdown(floatfmt=".3f"))
    md.append("\n\n## Conclusion\n")
    md.append("Compare `eps_debiased` across configs. If B and D are closer to A "
              "on the debiased metric, then BMC's effect on ε_sym is a "
              "systematic prediction-range expansion, not a consistency failure. "
              "If `eps_debiased` is still worst for B/D, then siamese genuinely "
              "didn't help consistency.")

    (out_root / "diagnostic_report.md").write_text("\n".join(md))
    log.info(f"Saved: {out_root / 'diagnostic_report.md'}")


if __name__ == "__main__":
    main()
