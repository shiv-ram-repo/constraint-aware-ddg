"""
make_figures.py
================
Generates Figure 1 (σ-sweep + bias bar chart) and Figure 2 (stabilizing
recall + magnitude-stratified anti-symmetry) for the Bioinformatics
manuscript.

Reads from:
    $RESULTS_DIR/summary.md  (or the equivalent CSVs)
    $RESULTS_DIR/antisymmetry_diagnostic/summary_by_config.csv
    $RESULTS_DIR/antisymmetry_diagnostic/by_magnitude_quartile.csv
    $RESULTS_DIR/stabilizing_recall/  (per-config tables)
    $RESULTS_DIR/path_a_and_b/report.md  (for σ sweep)

Outputs (in $FIGS_DIR/):
    fig1.pdf        Two-panel: σ sweep + mean-bias bar chart
    fig1.png        300dpi PNG version
    fig2.pdf        Two-panel: stabilizing recall + magnitude stratification
    fig2.png        300dpi PNG version

Style:
    - Okabe-Ito colorblind-safe palette
    - Serif axis labels (matches LaTeX paper font)
    - 7" × 3.5" each figure (full text width when included as
      \\includegraphics[width=\\textwidth]{fig1.pdf})
    - 300 dpi for raster; vector PDF preferred for LaTeX

Run:
    cd .
    python make_figures.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────── #
# Style: Okabe-Ito colorblind-safe palette
# ────────────────────────────────────────────────────────────────────────── #
OKABE_ITO = {
    "black":         "#000000",
    "orange":        "#E69F00",
    "sky_blue":      "#56B4E9",
    "bluish_green":  "#009E73",
    "yellow":        "#F0E442",
    "blue":          "#0072B2",
    "vermillion":    "#D55E00",
    "reddish_purple":"#CC79A7",
}

# Config-to-color mapping (kept consistent across both figures)
COLOR_BY_CONFIG = {
    "A_baseline":  OKABE_ITO["black"],
    "B_bmc":       OKABE_ITO["sky_blue"],
    "C_siamese":   OKABE_ITO["orange"],
    "D_bmc_sia":   OKABE_ITO["bluish_green"],
    "E_full":      OKABE_ITO["vermillion"],
}
LABEL_BY_CONFIG = {
    "A_baseline":  "A: baseline",
    "B_bmc":       "B: +BMC",
    "C_siamese":   "C: +Siamese",
    "D_bmc_sia":   "D: BMC+Siamese",
    "E_full":      "E: full",
}

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.labelsize":   10,
    "axes.titlesize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth":   1.5,
    "lines.markersize":  6,
})


# ────────────────────────────────────────────────────────────────────────── #
# Hard-coded data from your locked results (sources cited in inline comments)
#   These are the actual numbers from your summary.md and diagnostic_report.md.
#   If you'd rather have these read from CSV at runtime, see the "READ_FROM_CSV"
#   variant in the function bodies below.
# ────────────────────────────────────────────────────────────────────────── #
SIGMA_SWEEP = {
    # σ_OOD value → (S669_mean, S669_std, S461_mean, S461_std)
    # σ=0.00 is D_bmc_sia 3-seed; σ=0.20 is E (3-seed); others seed-42 only
    0.00: (0.517, 0.007, 0.683, 0.005),
    0.10: (0.531, None,  0.696, None),
    0.20: (0.540, 0.002, 0.711, 0.006),
    0.50: (0.520, None,  0.678, None),
}

# From results/antisymmetry_diagnostic/summary_by_config.csv
ANTISYM_BIAS = {
    "A_baseline": (+0.335, 0.013),
    "B_bmc":      (-0.393, 0.153),
    "C_siamese":  (+0.396, 0.025),
    "D_bmc_sia":  (-0.286, 0.114),
}

# From results/stabilizing_recall/  (mean ± std over 3 seeds)
RECALL_DATA = {
    # config → {k% → (mean, std)}
    "A_baseline": {10: (0.148, 0.002), 25: (0.358, 0.002), 50: (0.659, 0.013)},
    "B_bmc":      {10: (0.154, 0.003), 25: (0.359, 0.004), 50: (0.678, 0.012)},
    "C_siamese":  {10: (0.154, 0.003), 25: (0.362, 0.003), 50: (0.662, 0.013)},
    "D_bmc_sia":  {10: (0.150, 0.004), 25: (0.360, 0.009), 50: (0.685, 0.008)},
}

# From results/antisymmetry_diagnostic/by_magnitude_quartile.csv (mean over seeds)
MAGNITUDE_DATA = {
    "A_baseline": {"Q1": 0.26, "Q2": 0.30, "Q3": 0.39, "Q4": 0.50},
    "B_bmc":      {"Q1": 0.63, "Q2": 0.58, "Q3": 0.55, "Q4": 0.73},
    "C_siamese":  {"Q1": 0.29, "Q2": 0.35, "Q3": 0.44, "Q4": 0.58},
    "D_bmc_sia":  {"Q1": 0.59, "Q2": 0.52, "Q3": 0.47, "Q4": 0.62},
}


# ────────────────────────────────────────────────────────────────────────── #
# Figure 1 — σ sweep + mean-bias bar chart
# ────────────────────────────────────────────────────────────────────────── #
def make_figure_1(out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))

    # ─── Panel A: σ-sweep ───────────────────────────────────────────────
    ax = axes[0]
    sigmas = sorted(SIGMA_SWEEP.keys())
    s669_m  = [SIGMA_SWEEP[s][0] for s in sigmas]
    s669_e  = [SIGMA_SWEEP[s][1] for s in sigmas]
    s461_m  = [SIGMA_SWEEP[s][2] for s in sigmas]
    s461_e  = [SIGMA_SWEEP[s][3] for s in sigmas]

    # S669 line. Error bars only on the points we have 3-seed data for.
    err_s669 = [e if e is not None else 0 for e in s669_e]
    ax.errorbar(
        sigmas, s669_m, yerr=err_s669,
        color=OKABE_ITO["vermillion"], marker="o",
        capsize=3, capthick=0.8, elinewidth=0.8,
        label="S669",
    )
    err_s461 = [e if e is not None else 0 for e in s461_e]
    ax.errorbar(
        sigmas, s461_m, yerr=err_s461,
        color=OKABE_ITO["blue"], marker="s", linestyle="--",
        capsize=3, capthick=0.8, elinewidth=0.8,
        label="S461",
    )

    # Mark the headline point (σ=0.20)
    ax.axvline(0.20, color="grey", linewidth=0.5, linestyle=":", zorder=0)
    ax.annotate(
        "headline\n(config E)",
        xy=(0.20, 0.540), xytext=(0.32, 0.560),
        fontsize=8, color="grey",
        arrowprops=dict(arrowstyle="-", color="grey", linewidth=0.5),
    )

    ax.set_xlabel(r"OOD-margin noise scale $\sigma_{\mathrm{OOD}}$")
    ax.set_ylabel("Spearman correlation")
    ax.set_xticks([0.00, 0.10, 0.20, 0.50])
    ax.set_xticklabels(["0.00\n(D)", "0.10", "0.20\n(E)", "0.50"])
    ax.set_ylim(0.47, 0.74)
    ax.legend(loc="lower left", frameon=False)
    ax.text(-0.16, 1.02, "A", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

    # ─── Panel B: mean-bias bar chart ────────────────────────────────────
    ax = axes[1]
    configs = ["A_baseline", "B_bmc", "C_siamese", "D_bmc_sia"]
    means   = [ANTISYM_BIAS[c][0] for c in configs]
    stds    = [ANTISYM_BIAS[c][1] for c in configs]
    colors  = [COLOR_BY_CONFIG[c] for c in configs]
    labels  = [LABEL_BY_CONFIG[c] for c in configs]

    x_pos = np.arange(len(configs))
    bars = ax.bar(
        x_pos, means, yerr=stds,
        color=colors, alpha=0.85, edgecolor="black", linewidth=0.6,
        capsize=4, error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )

    # Zero line — visually anchors the "should be near 0" claim
    ax.axhline(0, color="black", linewidth=0.8, zorder=0)

    # Annotate sign-flip — A,C are positive; B,D negative
    for i, m in enumerate(means):
        offset = 0.04 if m >= 0 else -0.08
        ax.text(i, m + offset, f"{m:+.2f}",
                ha="center", va="bottom" if m >= 0 else "top",
                fontsize=8.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(r"Mean signed bias $\langle f_{\rightarrow} + f_{\leftarrow}\rangle$ (kcal/mol)")
    ax.set_ylim(-0.65, 0.65)
    ax.text(-0.16, 1.02, "B", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

    fig.tight_layout(w_pad=2.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig1.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out_dir/'fig1.pdf'} and {out_dir/'fig1.png'}")


# ────────────────────────────────────────────────────────────────────────── #
# Figure 2 — stabilizing recall + magnitude-stratified anti-symmetry
# ────────────────────────────────────────────────────────────────────────── #
def make_figure_2(out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))

    # ─── Panel A: Stabilizing recall — grouped bar chart ────────────────
    ax = axes[0]
    configs    = ["A_baseline", "B_bmc", "C_siamese", "D_bmc_sia"]
    k_values   = [10, 25, 50]
    x          = np.arange(len(k_values))
    bar_width  = 0.20

    for i, cfg in enumerate(configs):
        means = [RECALL_DATA[cfg][k][0] for k in k_values]
        stds  = [RECALL_DATA[cfg][k][1] for k in k_values]
        offset = (i - 1.5) * bar_width  # center the group of 4 around each k
        ax.bar(
            x + offset, means, bar_width,
            yerr=stds, capsize=2,
            color=COLOR_BY_CONFIG[cfg], alpha=0.85,
            edgecolor="black", linewidth=0.4,
            error_kw={"elinewidth": 0.6, "capthick": 0.6},
            label=LABEL_BY_CONFIG[cfg],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"top-{k}%" for k in k_values])
    ax.set_ylabel("Stabilizing recall")
    ax.set_xlabel("Threshold (% of S669)")
    ax.set_ylim(0, 0.78)
    ax.legend(loc="upper left", frameon=False, ncol=2,
              columnspacing=1.0, handlelength=1.2, handletextpad=0.4)
    ax.text(-0.16, 1.02, "A", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

    # ─── Panel B: Magnitude-stratified anti-symmetry — grouped bars ─────
    ax = axes[1]
    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    x         = np.arange(len(quartiles))
    bar_width = 0.20

    for i, cfg in enumerate(configs):
        vals = [MAGNITUDE_DATA[cfg][q] for q in quartiles]
        offset = (i - 1.5) * bar_width
        ax.bar(
            x + offset, vals, bar_width,
            color=COLOR_BY_CONFIG[cfg], alpha=0.85,
            edgecolor="black", linewidth=0.4,
            label=LABEL_BY_CONFIG[cfg],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Q1\n(easy)", "Q2", "Q3", "Q4\n(hard)"])
    ax.set_ylabel(r"$\bar{\varepsilon}_{\mathrm{sym}}$ (kcal/mol)")
    ax.set_xlabel(r"Mutation $|\Delta\Delta G|$ quartile")
    ax.set_ylim(0, 0.82)
    ax.legend(loc="upper left", frameon=False, ncol=2,
              columnspacing=1.0, handlelength=1.2, handletextpad=0.4)
    ax.text(-0.16, 1.02, "B", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

    fig.tight_layout(w_pad=2.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig2.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out_dir/'fig2.pdf'} and {out_dir/'fig2.png'}")


# ────────────────────────────────────────────────────────────────────────── #
# Main
# ────────────────────────────────────────────────────────────────────────── #
def main():
    out_dir = Path(os.environ.get("FIGS_DIR", "./figures/out"))
    # Alternative: relative path if you're not on the backup machine
    # out_dir = Path("./figs")

    log.info("Generating Figure 1 (σ sweep + mean-bias) ...")
    make_figure_1(out_dir)

    log.info("Generating Figure 2 (recall + magnitude stratification) ...")
    make_figure_2(out_dir)

    log.info("Done. Use the PDF versions in LaTeX:")
    log.info(r"  \includegraphics[width=\columnwidth]{fig1.pdf}")
    log.info(r"  \includegraphics[width=\columnwidth]{fig2.pdf}")


if __name__ == "__main__":
    main()
