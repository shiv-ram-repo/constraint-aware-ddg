# Datasets

Raw datasets are not redistributed in this repository — they are
downloaded from the original publications and processed locally.

## Megascale (training)

Source: Tsuboyama et al., *Nature* 2023.

```bash
# 1. Download Tsuboyama2023_Dataset2_Dataset3_20230416.csv from the
#    published data release and place at:
#    data/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv

# 2. Apply MMseqs2 30%-identity redundancy filter
bash data/build_megascale_splits.sh

# Result:
#   data/megascale/train.csv   (212 wild-type proteins, ~190k mutations)
#   data/megascale/val.csv     (29 proteins)
#   data/megascale/test.csv    (28 proteins, 28172 mutations)
```

## Out-of-distribution benchmarks

| Benchmark | n      | Source                                  |
|-----------|--------|-----------------------------------------|
| S669      | 669    | Pancotti et al. 2022 (Briefings Bioinf) |
| S461      | 461    | Hernández-Alías et al. 2024             |
| Ssym      | 342×2  | Pucci et al. 2018; Benevenuta et al. 2021 |
| FireProt-HF | 2578 | Stourac et al. 2021                     |
| S783      | 783    | Pancotti et al. 2022                    |
| S8754     | 8228   | Pancotti et al. 2022                    |
| S2648     | 2648   | Dehouck et al. 2009                     |
| S4346     | 3969   | Capriotti et al. 2008                   |
| S571      | 571    | Pucci et al. 2017 (delta T_m)           |

Each benchmark has its own assembly script:

```bash
bash data/build_s669.sh
bash data/build_s461.sh
bash data/build_ssym.sh
# ... etc
```

## ProteinMPNN weights

```bash
mkdir -p checkpoints
curl -L -o checkpoints/v_48_020.pt \
    https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_020.pt
```

## ESM-2

Loaded automatically from HuggingFace Hub on first run
(`facebook/esm2_t33_650M_UR50D`, approximately 2.5 GB).
