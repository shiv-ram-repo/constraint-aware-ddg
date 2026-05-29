#!/usr/bin/env bash
# Full reproduction sweep: trains all 5 configurations (A-E) for 3 seeds each.
#
# Total runtime: roughly 6 hours on one NVIDIA RTX A6000.
# Each individual run is ~15-25 minutes including evaluation across all
# eleven benchmarks.

set -euo pipefail

ROOT=${ROOT:-runs}
mkdir -p "$ROOT"

# Hyperparameters used in the paper
LR=1e-4
WD=1e-2
GRAD_CLIP=1.0
EPOCHS=200
PATIENCE=20

# Common arguments
COMMON=(
    --lr "$LR"
    --weight_decay "$WD"
    --grad_clip "$GRAD_CLIP"
    --max_epochs "$EPOCHS"
    --patience "$PATIENCE"
)

# Loss-component flags per configuration
declare -A USE_BMC=(  [A]=false [B]=true  [C]=false [D]=true  [E]=true  )
declare -A USE_SYM=(  [A]=false [B]=false [C]=true  [D]=true  [E]=true  )
declare -A USE_OOD=(  [A]=false [B]=false [C]=false [D]=false [E]=true  )

for CONFIG in A B C D E; do
    for SEED in 42 43 44; do
        OUT="$ROOT/${CONFIG}_seed${SEED}"

        if [[ -f "$OUT/best.pt" ]]; then
            echo "[skip] $OUT/best.pt already exists"
            continue
        fi

        echo "[run]  config=$CONFIG seed=$SEED -> $OUT"
        python scripts/train.py \
            "${COMMON[@]}" \
            --config "$CONFIG" \
            --seed "$SEED" \
            --use_bmc "${USE_BMC[$CONFIG]}" \
            --use_sym "${USE_SYM[$CONFIG]}" \
            --use_ood_margin "${USE_OOD[$CONFIG]}" \
            --ood_margin_sigma 0.20 \
            --lambda_sym 0.5 \
            --lambda_ood 0.5 \
            --output_dir "$OUT"

        echo "[eval] $OUT"
        python scripts/evaluate.py \
            --checkpoint "$OUT/best.pt" \
            --benchmarks all \
            --output "$OUT/eval_noTTA/"
    done
done

echo "Sweep done. Run scripts/run_finalize.py to compile per-seed summary tables."
