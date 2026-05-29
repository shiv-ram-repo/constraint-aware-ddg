
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

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

# Display order for the main table. ssym_direct/inverse and s669/s461 first
# (the headline OOD benchmarks). megascale_test second (in-distribution).
BENCHMARK_ORDER = [
    "s669", "s461", "s571",
    "ssym_direct", "ssym_inverse",
    "megascale_test", "fireport_hf",
    "s783", "s8754", "s2648", "s4346",
]

CONFIG_LABELS = {
    "A_baseline": "A: baseline (Huber)",
    "B_bmc":      "B: + BMC",
    "C_siamese":  "C: + siamese",
    "D_bmc_sia":  "D: + BMC + siamese",
}


def run_dir(config: str, seed: int) -> Path:
    if seed == 42:
        return Path(RUNS_ROOT) / config
    return Path(RUNS_ROOT) / f"{config}_s{seed}"


def load_test_metrics(rd: Path) -> pd.DataFrame | None:
    p = rd / "eval_noTTA" / "test_metrics.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p)


# ────────────────────────────────────────────────────────────────────────── #
# Aggregate test metrics across seeds
# ────────────────────────────────────────────────────────────────────────── #
def aggregate_test_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        df_long : one row per (config, seed, dataset) with all metrics
        df_wide : mean±std formatted strings, dataset rows × config columns,
                  for the main Spearman table
    """
    rows = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            rd = run_dir(cfg, seed)
            tm = load_test_metrics(rd)
            if tm is None:
                log.warning(f"  MISSING test_metrics.csv: {rd}")
                continue
            for _, r in tm.iterrows():
                rows.append({
                    "config":   cfg,
                    "seed":     seed,
                    "dataset":  r["dataset"],
                    "n":        int(r["n"]),
                    "spearman": float(r["spearman"]),
                    "pearson":  float(r["pearson"]),
                    "rmse":     float(r["rmse"]),
                    "mae":      float(r["mae"]),
                    "r2":       float(r.get("r2", np.nan)),
                })
    if not rows:
        raise RuntimeError("No test metrics found. Did you run the seed sweep?")
    df_long = pd.DataFrame(rows)
    return df_long


def format_mean_std(vals: List[float], dec: int = 3) -> str:
    if len(vals) == 0:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.{dec}f}"
    m = np.mean(vals); s = np.std(vals, ddof=0)
    return f"{m:.{dec}f}±{s:.{dec}f}"


def build_main_table(df_long: pd.DataFrame, metric: str = "spearman",
                     decimals: int = 3) -> pd.DataFrame:
    """Dataset rows × config columns, mean±std strings."""
    out_rows = []
    for ds in BENCHMARK_ORDER:
        sub_ds = df_long[df_long["dataset"] == ds]
        if len(sub_ds) == 0:
            continue
        n_val = int(sub_ds["n"].iloc[0])
        row = {"benchmark": ds, "n": n_val}
        for cfg in CONFIGS:
            vals = sub_ds[sub_ds["config"] == cfg][metric].tolist()
            row[CONFIG_LABELS[cfg]] = format_mean_std(vals, decimals)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def build_delta_vs_baseline(df_long: pd.DataFrame,
                            metric: str = "spearman") -> pd.DataFrame:
    """For each dataset, gain in Spearman of B/C/D over A_baseline."""
    rows = []
    for ds in BENCHMARK_ORDER:
        sub = df_long[df_long["dataset"] == ds]
        if len(sub) == 0:
            continue
        a_vals = sub[sub["config"] == "A_baseline"][metric].tolist()
        if not a_vals:
            continue
        a_mean = float(np.mean(a_vals))
        row = {"benchmark": ds, "A_mean": a_mean}
        for cfg in ["B_bmc", "C_siamese", "D_bmc_sia"]:
            cfg_vals = sub[sub["config"] == cfg][metric].tolist()
            if not cfg_vals:
                row[f"Δ_{cfg}"] = np.nan
                continue
            row[f"Δ_{cfg}"] = float(np.mean(cfg_vals) - a_mean)
        rows.append(row)
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────── #
# LaTeX table builders
# ────────────────────────────────────────────────────────────────────────── #
def df_to_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Minimal booktabs LaTeX table writer that doesn't choke on the ± char."""
    cols = list(df.columns)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    col_spec = "l" + "c" * (len(cols) - 1)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    # Header
    header = " & ".join(str(c).replace("_", r"\_") for c in cols) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                s = f"{v:.3f}"
            else:
                s = str(v).replace("_", r"\_").replace("±", r"$\pm$")
            cells.append(s)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────── #
# Master summary.md
# ────────────────────────────────────────────────────────────────────────── #
def write_summary_md(out_path: Path, df_main: pd.DataFrame,
                     df_delta: pd.DataFrame, df_antisym: pd.DataFrame | None,
                     df_stab: pd.DataFrame | None):
    md = []
    md.append("# Final Results Summary")
    md.append("")
    md.append("Generated by `build_results_tables.py`.")
    md.append("")
    md.append("## Headline (Spearman across 3 seeds)")
    md.append("")
    md.append(df_main.to_markdown(index=False))
    md.append("")
    md.append("## Gain over A_baseline (mean Δ Spearman, averaged over seeds)")
    md.append("")
    md.append(df_delta.to_markdown(index=False, floatfmt=".3f"))
    md.append("")
    if df_antisym is not None and len(df_antisym):
        md.append("## Anti-symmetry violation ε_sym on Ssym")
        md.append("")
        md.append(df_antisym.to_markdown(index=False, floatfmt=".3f"))
        md.append("")
        md.append("Lower ε_sym = better anti-symmetry. "
                  "Perfect anti-symmetry would give ε_sym = 0 "
                  "and pearson_neg = +1.0.")
        md.append("")
    if df_stab is not None and len(df_stab):
        md.append("## Stabilizing-mutation recall on S669 (top-k% recall)")
        md.append("")
        md.append(df_stab.to_markdown(index=False, floatfmt=".3f"))
        md.append("")
    md.append("---")
    md.append("")
    md.append("Files in this directory:")
    md.append("- `main_table.csv` / `main_table.tex` — Spearman headline")
    md.append("- `full_metrics.csv` — long-form: every (config, seed, dataset) "
              "with all four metrics")
    md.append("- `delta_vs_baseline.csv` — gain over A_baseline")
    md.append("- `antisymmetry/` — anti-symmetry analysis")
    md.append("- `stabilizing_recall/` — stabilizing-mutation recall analysis")
    md.append("")
    out_path.write_text("\n".join(md))
    log.info(f"Wrote master summary: {out_path}")


# ────────────────────────────────────────────────────────────────────────── #
def main():
    out_root = Path(RESULTS_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("=" * 78)
    log.info("Step 1: aggregate test metrics across all (config × seed × dataset)")
    log.info("=" * 78)
    df_long = aggregate_test_metrics()
    full_csv = out_root / "full_metrics.csv"
    df_long.to_csv(full_csv, index=False, float_format="%.4f")
    log.info(f"Saved: {full_csv}  ({len(df_long)} rows)")

    log.info("\nMain table (Spearman mean±std, dataset × config):")
    df_main = build_main_table(df_long, metric="spearman", decimals=3)
    print(df_main.to_string(index=False))
    df_main.to_csv(out_root / "main_table.csv", index=False)
    (out_root / "main_table.tex").write_text(df_to_latex(
        df_main,
        caption="Spearman correlation across 11 benchmarks. "
                "Mean $\\pm$ std over 3 random seeds (42, 43, 44).",
        label="tab:main",
    ))

    log.info("\nΔ vs A_baseline (mean across seeds):")
    df_delta = build_delta_vs_baseline(df_long, metric="spearman")
    print(df_delta.to_string(index=False))
    df_delta.to_csv(out_root / "delta_vs_baseline.csv",
                    index=False, float_format="%.4f")

    # Pearson and RMSE tables too (for the supplement)
    df_pear  = build_main_table(df_long, metric="pearson", decimals=3)
    df_pear.to_csv(out_root / "main_table_pearson.csv", index=False)
    df_rmse  = build_main_table(df_long, metric="rmse", decimals=3)
    df_rmse.to_csv(out_root / "main_table_rmse.csv", index=False)

    # ── Anti-symmetry table (if available) ──
    df_antisym = None
    antisym_csv = out_root / "antisymmetry" / "summary_by_config.csv"
    if antisym_csv.is_file():
        log.info("\n" + "=" * 78)
        log.info("Anti-symmetry table:")
        log.info("=" * 78)
        df_antisym_full = pd.read_csv(antisym_csv)
        # Pretty short version for summary.md
        df_antisym = pd.DataFrame({
            "config":      df_antisym_full["config"],
            "mean_eps":    df_antisym_full["mean_eps_mean"],
            "mean_eps_sd": df_antisym_full["mean_eps_std"],
            "p_below_0.5": df_antisym_full["p_below_0.5_mean"],
            "pearson_neg": df_antisym_full["pearson_neg_mean"],
        })
        print(df_antisym.to_string(index=False))
        (out_root / "antisymmetry_table.tex").write_text(df_to_latex(
            df_antisym,
            caption="Anti-symmetry violation $\\epsilon_{\\mathrm{sym}}$ "
                    "on Ssym (mean $\\pm$ std over 3 seeds).",
            label="tab:antisym",
        ))
    else:
        log.warning(f"  Skipping anti-symmetry table — "
                    f"{antisym_csv} not found. Run compute_antisymmetry.py first.")

    # ── Stabilizing recall table (if available) ──
    df_stab = None
    stab_csv = out_root / "stabilizing_recall" / "summary_by_config.csv"
    if stab_csv.is_file():
        log.info("\n" + "=" * 78)
        log.info("Stabilizing-recall on S669:")
        log.info("=" * 78)
        df_stab_full = pd.read_csv(stab_csv)
        s669 = df_stab_full[df_stab_full["dataset"] == "s669"]
        rows = []
        for cfg in CONFIGS:
            row = {"config": cfg}
            for k in [10, 25, 50]:
                rs = s669[(s669["config"] == cfg) & (s669["k_pct"] == k)]
                if len(rs) > 0:
                    r = rs.iloc[0]
                    row[f"recall@{k}%"] = float(r["recall_mean"])
                    row[f"recall@{k}%_sd"] = float(r["recall_std"])
            rows.append(row)
        df_stab = pd.DataFrame(rows)
        print(df_stab.to_string(index=False))
        (out_root / "stabilizing_recall_s669.tex").write_text(df_to_latex(
            df_stab,
            caption="Stabilizing-mutation recall at top-$k\\%$ on S669 "
                    "(mean $\\pm$ std over 3 seeds, threshold "
                    "$|\\Delta\\Delta G| > 0.5$).",
            label="tab:stab",
        ))
    else:
        log.warning(f"  Skipping stabilizing recall table — "
                    f"{stab_csv} not found.")

    # ── Master summary.md ──
    write_summary_md(
        out_root / "summary.md",
        df_main, df_delta, df_antisym, df_stab,
    )

    log.info("\n" + "=" * 78)
    log.info("ALL RESULTS COMPILED")
    log.info("=" * 78)
    log.info(f"Master summary:  {out_root / 'summary.md'}")
    log.info(f"All artefacts in: {out_root}")
    log.info("")
    log.info("Send back summary.md and we'll start writing the paper.")


if __name__ == "__main__":
    main()
