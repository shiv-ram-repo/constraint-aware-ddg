import os
"""
make_imbalance_figure.py
=========================
Generates a supplementary two-panel figure documenting the label-space
imbalance that motivates the BMC loss.

  Panel A: Smoothed density of true ΔΔG across Megascale-test and S669
  Panel B: Stacked-bar three-class breakdown (stabilizing / neutral /
           destabilizing) at the conventional ±0.5 kcal/mol threshold

Key differences from the earlier version:
  - Threshold lowered from ±1.0 to ±0.5 kcal/mol to match the rest of
    the paper (Methods, stabilizing recall, JanusDDG conventions)
  - Adds the two additional OOD benchmarks (S461, Ssym-direct) so the
    figure documents imbalance across the full primary benchmark set
  - Uses standard biophysics convention: ΔΔG < 0 = stabilizing,
    ΔΔG > 0 = destabilizing (matches Megascale/FoldX/Rosetta/SPURS)

Designed for Bioinformatics two-column layout: render at column width
or 1.5× column width depending on whether you place it in main text
or supplementary.

Run:
    cd .
    python make_imbalance_figure.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────── #
# Style: Okabe-Ito colorblind-safe palette (consistent with Fig 1 + Fig 2)
# ────────────────────────────────────────────────────────────────────── #
OKABE_ITO = {
    "stabilizing":    "#009E73",   # bluish green
    "neutral":        "#BDBDBD",   # neutral grey
    "destabilizing":  "#D55E00",   # vermillion
    "megascale_line": "#000000",   # black
    "s669_line":      "#0072B2",   # blue
    "s461_line":      "#CC79A7",   # reddish purple
    "ssym_line":      "#E69F00",   # orange
}

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.labelsize":   10,
    "axes.titlesize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
})

# Threshold for stabilizing/neutral/destabilizing classification.
# 0.5 kcal/mol is the convention used throughout the manuscript and is
# approximately the experimental noise floor of ΔΔG measurements.
THRESH = 0.5

BASE_DIR = Path(os.environ.get("EVAL_DIR", "./runs/A_baseline/eval_noTTA"))
DATASETS = [
    ("Megascale-test", BASE_DIR / "targets_megascale_test.npy"),
    ("S669",            BASE_DIR / "targets_s669.npy"),
    ("S461",            BASE_DIR / "targets_s461.npy"),
    ("Ssym (direct)",   BASE_DIR / "targets_ssym_direct.npy"),
]


def compute_stats(data: np.ndarray, threshold: float = THRESH):
    """Standard biophysics convention: ΔΔG < 0 = stabilizing."""
    n = len(data)
    n_stab   = int(np.sum(data < -threshold))
    n_destab = int(np.sum(data >  +threshold))
    n_neut   = n - n_stab - n_destab
    pct_stab   = 100.0 * n_stab   / n
    pct_destab = 100.0 * n_destab / n
    pct_neut   = 100.0 - pct_stab - pct_destab   # rounds to exactly 100
    return {
        "n": n,
        "stab": n_stab, "neut": n_neut, "destab": n_destab,
        "pct_stab": pct_stab, "pct_neut": pct_neut, "pct_destab": pct_destab,
    }


def make_figure(out_dir: Path):
    # Load all datasets
    raw = {}
    for name, path in DATASETS:
        if not path.is_file():
            log.warning(f"missing: {path}")
            continue
        raw[name] = np.load(path)
        s = compute_stats(raw[name])
        log.info(f"{name}: n={s['n']:,}, "
                 f"stab={s['pct_stab']:.1f}%, "
                 f"neut={s['pct_neut']:.1f}%, "
                 f"destab={s['pct_destab']:.1f}%")

    if "Megascale-test" not in raw:
        log.error("Megascale-test target file required; aborting.")
        return

    fig, (ax_kde, ax_bar) = plt.subplots(
        1, 2, figsize=(9.5, 3.8),
        gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.30},
    )

    # ─── Panel A: density curves ────────────────────────────────────────
    x_grid = np.linspace(-4.0, 6.0, 600)
    line_colors = {
        "Megascale-test": OKABE_ITO["megascale_line"],
        "S669":            OKABE_ITO["s669_line"],
        "S461":            OKABE_ITO["s461_line"],
        "Ssym (direct)":   OKABE_ITO["ssym_line"],
    }
    line_styles = {
        "Megascale-test": "-",
        "S669":            "--",
        "S461":            "-.",
        "Ssym (direct)":   ":",
    }

    # Compute the max density across all curves (for shading band heights)
    y_curves = {}
    for name, data in raw.items():
        clipped = np.clip(data, -4.0, 6.0)
        kde = gaussian_kde(clipped, bw_method=0.18)
        y_curves[name] = kde(x_grid)

    y_max_overall = max(y.max() for y in y_curves.values())

    # Shade the stabilizing and destabilizing regions
    ax_kde.axvspan(-4.0, -THRESH,
                    color=OKABE_ITO["stabilizing"], alpha=0.10, zorder=0)
    ax_kde.axvspan(+THRESH, 6.0,
                    color=OKABE_ITO["destabilizing"], alpha=0.10, zorder=0)
    ax_kde.axvline(0, color="grey", linewidth=0.5, linestyle=":", zorder=1)
    ax_kde.axvline(-THRESH, color="grey", linewidth=0.5, linestyle=":", zorder=1)
    ax_kde.axvline(+THRESH, color="grey", linewidth=0.5, linestyle=":", zorder=1)

    # Plot density curves
    for name in ["Megascale-test", "S669", "S461", "Ssym (direct)"]:
        if name not in y_curves:
            continue
        ax_kde.plot(x_grid, y_curves[name],
                    color=line_colors[name],
                    linestyle=line_styles[name],
                    linewidth=1.5,
                    label=name)

    ax_kde.set_xlabel(r"True $\Delta\Delta G$ (kcal/mol)")
    ax_kde.set_ylabel("Density")
    ax_kde.set_xlim(-4.0, 6.0)
    ax_kde.set_ylim(0, y_max_overall * 1.15)
    ax_kde.legend(loc="upper right", frameon=False)
    ax_kde.text(-0.10, 1.05, "A", transform=ax_kde.transAxes,
                fontsize=12, fontweight="bold", va="top")

    # Region labels — placed near the BOTTOM of the panel so they cannot
    # clash with the legend at top-right or with rising density curves
    y_lab = y_max_overall * 0.04
    ax_kde.text(-2.7, y_lab, "stabilizing",
                color=OKABE_ITO["stabilizing"], ha="center",
                fontsize=8.5, fontweight="bold")
    ax_kde.text(0.0, y_lab, "neutral",
                color="grey", ha="center",
                fontsize=8.5, fontweight="bold")
    ax_kde.text(3.5, y_lab, "destabilizing",
                color=OKABE_ITO["destabilizing"], ha="center",
                fontsize=8.5, fontweight="bold")

    # ─── Panel B: stacked-bar three-class breakdown ─────────────────────
    bar_names = [name for name, _ in DATASETS if name in raw]
    stats     = [compute_stats(raw[name]) for name in bar_names]

    y_positions = np.arange(len(bar_names))
    bar_height  = 0.55

    pct_stab   = [s["pct_stab"]   for s in stats]
    pct_neut   = [s["pct_neut"]   for s in stats]
    pct_destab = [s["pct_destab"] for s in stats]
    left_neut    = pct_stab
    left_destab  = [a + b for a, b in zip(pct_stab, pct_neut)]

    ax_bar.barh(y_positions, pct_stab, bar_height,
                color=OKABE_ITO["stabilizing"], edgecolor="black",
                linewidth=0.5, label=f"stabilizing (ΔΔG < −{THRESH:.1f})")
    ax_bar.barh(y_positions, pct_neut, bar_height, left=left_neut,
                color=OKABE_ITO["neutral"], edgecolor="black",
                linewidth=0.5, label=f"neutral (|ΔΔG| ≤ {THRESH:.1f})")
    ax_bar.barh(y_positions, pct_destab, bar_height, left=left_destab,
                color=OKABE_ITO["destabilizing"], edgecolor="black",
                linewidth=0.5, label=f"destabilizing (ΔΔG > +{THRESH:.1f})")

    # Inline labels: stabilizing % inside the bar if wide enough (≥6%),
    # otherwise as a callout just outside the right edge of the stabilizing
    # band (still inside the plot area, but on the neutral grey band so it
    # remains readable).
    for i, s in enumerate(stats):
        if s["pct_stab"] >= 6.0:
            # In-bar white label
            ax_bar.text(s["pct_stab"]/2, y_positions[i],
                        f"{s['pct_stab']:.1f}%",
                        ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="bold")
        else:
            # Callout just to the right of the stabilizing band
            ax_bar.text(s["pct_stab"] + 1.0, y_positions[i],
                        f"{s['pct_stab']:.1f}%",
                        ha="left", va="center", fontsize=8.5,
                        color=OKABE_ITO["stabilizing"], fontweight="bold")

        # Neutral — center of neutral band
        ax_bar.text(s["pct_stab"] + s["pct_neut"]/2, y_positions[i],
                    f"{s['pct_neut']:.1f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="black")

        # Destabilizing — center of destabilizing band
        ax_bar.text(s["pct_stab"] + s["pct_neut"] + s["pct_destab"]/2,
                    y_positions[i],
                    f"{s['pct_destab']:.1f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold")

    ax_bar.set_yticks(y_positions)
    ax_bar.set_yticklabels(
        [f"{name}\n(n={s['n']:,})" for name, s in zip(bar_names, stats)],
        fontsize=8.5)
    ax_bar.set_xlabel("Proportion of dataset (%)")
    ax_bar.set_xlim(0, 100)
    ax_bar.set_ylim(-0.5, len(bar_names) - 0.5)
    ax_bar.invert_yaxis()      # Megascale on top
    ax_bar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=3, frameon=False, fontsize=7.5)
    ax_bar.text(-0.10, 1.05, "B", transform=ax_bar.transAxes,
                fontsize=12, fontweight="bold", va="top")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_imbalance.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_imbalance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out_dir/'fig_imbalance.pdf'} and "
             f"{out_dir/'fig_imbalance.png'}")


def main():
    out_dir = Path(os.environ.get("FIGS_DIR", "./figures/out"))
    make_figure(out_dir)


if __name__ == "__main__":
    main()
