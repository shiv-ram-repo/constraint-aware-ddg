# Constraint-Aware Optimization for Robust Protein Stability Prediction
Implementation accompanying the manuscript
"Constraint-Aware Optimization for Robust Protein Stability Prediction".

The work applies three loss-level interventions to multimodal $\Delta\Delta G$ predictor (ESM-2 + ProteinMPNN), without
architectural changes:

1. **Balanced Mean Squared Error** (Ren et al., 2022) for
   distribution-aware regression on the imbalanced Megascale label
   distribution.
2. **Siamese anti-symmetric regularization** that penalizes the
   squared forward-reverse prediction sum.
3. **OOD-margin consistency loss** : a novel input-noise consistency
   regularizer applied to the per-position feature representation.

The combined three-loss objective (configuration E in the paper)
improves Spearman correlation on the S669 OOD benchmark from
$0.486 \to 0.540$ ($\sigma = 0.002$ across three random seeds) and on
S461 from $0.653 \to 0.711$, while requiring no architectural
modification of the underlying SPURS backbone.

## Repository layout

```
src/                        Core model and loss code
  ddg_data/              Datasets, featurizers, MMseqs2 splits
  ddg_models/            SPURS backbone (ESM-2 + ProteinMPNN adapter + MLP head)
  ddg_train/             Training loop, metrics, evaluation harness
  custom_losses.py            BMC, Siamese, OOD-margin, BCAS loss modules
  imbalanced_losses.py        Standalone Balanced MSE implementation

scripts/                    Training and evaluation entrypoints
  train.py                    Full training run (all loss flags)
  evaluate.py                 Multi-benchmark evaluation (auto-detects checkpoint heads)
  tta_inference.py            Test-time augmentation evaluation
  run_new_loss_ablation.py    Five-configuration A->E ablation sweep
  run_seed_variance.py        Three-seed verification of the headline configuration
  run_finalize.py             Final sweep over OOD-margin noise scale sigma

analysis/                   Numerical analyses that produce paper tables
  compute_antisymmetry.py        Ssym forward-reverse decomposition (Table 3)
  diagnose_antisymmetry.py       Mean-bias / residual sigma decomposition (Fig 1B)
  compute_stabilizing_recall.py  Top-k% recall of stabilizing mutations on S669 (Fig 2A)
  build_results_tables.py        Aggregates per-seed metrics into manuscript tables
  check_imbalance_numbers.py     Verifies the label-imbalance numbers in the BMC paragraph

figures/                    Figure generation
  make_figures.py             Figure 1 (sigma sweep + bias bar chart), Figure 2 (recall + magnitude)
  make_imbalance_figure.py    Supplementary Figure S1 (label-imbalance density + stacked bar)

supplementary/              Supplementary text and captions

data/                       Dataset preparation instructions (no raw data committed)

docs/                       Extended documentation
```

## Quick start

### 1. Install

```bash
git clone https://github.com/<user>/constraint-aware-ddg
cd constraint-aware-ddg
pip install -r requirements.txt
pip install -e src/ddg_data -e src/ddg_models -e src/ddg_train
```

### 2. Configure paths

The scripts read four environment variables, each with a sensible
repo-relative default. Override these if your data and runs live
elsewhere:

```bash
export DDG_DATA_ROOT="./data"             # benchmark datasets
export MEGASCALE_DIR="./data/megascale"    # Megascale CSV directory
export MPNN_CKPT="./checkpoints/v_48_020.pt"  # ProteinMPNN weights
export RUNS_ROOT="./runs"                  # where training outputs are written
```

### 3. Get the data and pretrained encoders

- **Megascale (Tsuboyama 2023)**: download
  `Tsuboyama2023_Dataset2_Dataset3_20230416.csv` from the published
  data release and place under `data/megascale/`. See `data/README.md`
  for the MMseqs2 30%-identity filter we apply.
- **ProteinMPNN weights**: download `v_48_020.pt` from the
  ProteinMPNN repository and place under `checkpoints/v_48_020.pt`.
- **ESM-2 (650M)**: loaded automatically by the model from
  HuggingFace Hub on first run.
- **OOD benchmarks (S669, S461, Ssym, FireProt-HF, S2648, S4346,
  S571, S783, S8754)**: scripts to assemble each split are in
  `data/`.

### 4. Reproduce the headline configuration (config E)

```bash
# train one seed of configuration E (full three-loss objective)
python scripts/train.py \
    --config E \
    --use_bcas false \
    --use_bmc true \
    --use_sym true \
    --use_ood_margin true \
    --ood_margin_sigma 0.20 \
    --lambda_sym 0.5 \
    --lambda_ood 0.5 \
    --seed 42 \
    --output_dir runs/E_full_s42
```

Repeat with `--seed 43` and `--seed 44` for the three-seed estimate.

### 5. Evaluate on all benchmarks

```bash
python scripts/evaluate.py \
    --checkpoint runs/E_full_s42/best.pt \
    --benchmarks all \
    --output runs/E_full_s42/eval/
```

### 6. Generate figures

```bash
python figures/make_figures.py
python figures/make_imbalance_figure.py
```

Outputs land in `figures/out/`.

## Reproducing the paper from scratch

```bash
# Trains all five configurations A->E for three seeds each = 15 runs.
# Approximate wall time: ~6 hours on one NVIDIA RTX A6000.
bash scripts/run_all.sh
```

Then run the analysis scripts in `analysis/` against the resulting
checkpoint directories to regenerate every numerical claim in the
manuscript.

## Citation

A preprint will be added once publicly available.
For now, please cite the repository and contact the authors if a formal citation is required.

Our work is inspired from SPURS (Li and Luo, *Nature Communications* 2025);
if you compare against their architecture, please cite their paper as
well.

## License

MIT, with the caveat that ESM-2 and ProteinMPNN have their own
licenses that govern the weights themselves (not redistributed here).

## Contact

A Shivram --- `p20230075@hyderabad.bits-pilani.ac.in`
Sourav Chowdhury --- `sourav@hyderabad.bits-pilani.ac.in`
