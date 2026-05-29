import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

from ddg_train import TrainConfig, train, evaluate_all_benchmarks

DATA_ROOT = os.environ.get("DDG_DATA_ROOT", "./data")
MPNN_CKPT = os.environ.get("MPNN_CKPT", "./checkpoints/v_48_020.pt")
RUNS_ROOT = os.environ.get("RUNS_ROOT", "./runs")

common = dict(
    data_root        = DATA_ROOT,
    proteinmpnn_ckpt = MPNN_CKPT,
    max_epochs       = 80,
    early_stop_pat   = 20,
    adapter_layer    = -3,
    tune_mpnn        = True,
)

# Each entry: (config_name, hyperparam overrides)
configs = {
    "A_baseline": dict(loss_type="huber", use_siamese=False,
                       use_k50_head=False, use_move_b_features=False),
    "B_bmc":      dict(loss_type="bmc",   use_siamese=False,
                       use_k50_head=False, use_move_b_features=False),
    "C_siamese":  dict(loss_type="huber", use_siamese=True, siamese_weight=0.5,
                       use_k50_head=False, use_move_b_features=False),
}

seeds_to_run = [43, 44]

# Pre-flight: check that the SEED-42 checkpoints exist (so we don't
# silently retrain something that was already done).
log.info("=" * 80)
log.info("Pre-flight check: existing seed-42 checkpoints")
log.info("=" * 80)
for name in configs.keys():
    s42_path = os.path.join(RUNS_ROOT, name, "best.pt")
    if os.path.isfile(s42_path):
        log.info(f"  OK: {name}/best.pt exists (seed 42 — keeping as-is)")
    else:
        log.warning(f"  MISSING: {name}/best.pt — seed 42 will need re-training too")

# Check D already has all 3 seeds
for s in [42, 43, 44]:
    d_path = os.path.join(RUNS_ROOT,
                          f"D_bmc_sia{'' if s == 42 else f'_s{s}'}",
                          "best.pt")
    if os.path.isfile(d_path):
        log.info(f"  OK: D_bmc_sia seed {s} exists")
    else:
        log.warning(f"  MISSING: D_bmc_sia seed {s} — table will be incomplete")

log.info("")
log.info("Configs to train (6 total runs):")
for name in configs:
    for seed in seeds_to_run:
        log.info(f"  {name}_s{seed}")
log.info("")

# Run sequentially
total_start = time.time()
results_log = []

for name, overrides in configs.items():
    for seed in seeds_to_run:
        run_name = f"{name}_s{seed}"
        out      = os.path.join(RUNS_ROOT, run_name)

        # Skip if checkpoint already exists (resumability across reboots)
        ckpt_path = os.path.join(out, "best.pt")
        if os.path.isfile(ckpt_path):
            log.info(f"\nSKIP: {run_name} already trained at {ckpt_path}")
            # Still re-eval if metrics CSV missing
            metrics_path = os.path.join(out, "eval_noTTA", "test_metrics.csv")
            if os.path.isfile(metrics_path):
                log.info(f"      eval already done at {metrics_path}")
                continue
        else:
            log.info("\n" + "=" * 80)
            log.info(f"TRAINING: {run_name}")
            log.info("=" * 80)
            run_start = time.time()
            cfg = TrainConfig(
                output_dir = out,
                seed       = seed,
                **common,
                **overrides,
            )
            ckpt_path = train(cfg)
            log.info(f"Training time for {run_name}: "
                     f"{(time.time() - run_start) / 60:.1f} min")

        log.info("-" * 80)
        log.info(f"EVAL: {run_name}")
        log.info("-" * 80)
        eval_start = time.time()
        df = evaluate_all_benchmarks(
            data_root        = common["data_root"],
            ckpt_path        = ckpt_path,
            output_dir       = os.path.join(out, "eval_noTTA"),
            proteinmpnn_ckpt = common["proteinmpnn_ckpt"],
            adapter_layer    = common["adapter_layer"],
        )
        log.info(f"Eval time for {run_name}: "
                 f"{(time.time() - eval_start) / 60:.1f} min")
        log.info(f"Results:\n{df.to_string()}")
        results_log.append((run_name, df))

log.info("\n" + "=" * 80)
log.info(f"SEED-VARIANCE SUITE COMPLETE")
log.info(f"Total wall-clock: {(time.time() - total_start) / 60:.1f} min")
log.info("=" * 80)
log.info("")
log.info("Now run:")
log.info("  python compute_antisymmetry.py")
log.info("  python compute_stabilizing_recall.py")
log.info("  python build_results_tables.py")
