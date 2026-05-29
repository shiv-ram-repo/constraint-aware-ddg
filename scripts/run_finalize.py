import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from ddg_train import TrainConfig, train, evaluate_all_benchmarks

DATA_ROOT  = os.environ.get("DDG_DATA_ROOT", "./data")
MPNN_CKPT  = os.environ.get("MPNN_CKPT", "./checkpoints/v_48_020.pt")
RUNS_ROOT  = os.environ.get("RUNS_ROOT", "./runs")

common = dict(
    data_root        = DATA_ROOT,
    proteinmpnn_ckpt = MPNN_CKPT,
    max_epochs       = 80,
    adapter_layer    = -3,
    tune_mpnn        = True,
)


# ────────────────────────────────────────────────────────────── #
# Job 1: D_bmc_sia seed variance
# ────────────────────────────────────────────────────────────── #
# D was originally trained with seed=42 — DO NOT re-run it; we already
# have that checkpoint at runs/D_bmc_sia/. Train two more seeds.
job1 = {
    "D_bmc_sia_s43": dict(
        loss_type="bmc", use_siamese=True, use_thermo_head=False,
        seed=43, early_stop_pat=20,
    ),
    "D_bmc_sia_s44": dict(
        loss_type="bmc", use_siamese=True, use_thermo_head=False,
        seed=44, early_stop_pat=20,
    ),
}


# ────────────────────────────────────────────────────────────── #
# Job 2: G_thermo_bmc_sia with longer patience
# ────────────────────────────────────────────────────────────── #
# The original G stopped at epoch 29 (best at epoch 9). BMC loss anneals
# slower than Huber because of its cross-entropy scale. Give it more
# patience and see if S669/S571 improve beyond the 0.514/0.463 plateau.
job2 = {
    "G_thermo_bmc_sia_pat40": dict(
        loss_type="bmc", use_siamese=True, use_thermo_head=True,
        seed=42, early_stop_pat=40, max_epochs=120,
    ),
}


all_runs = {**job1, **job2}

for name, overrides in all_runs.items():
    print("=" * 80)
    print(f"RUN: {name}")
    print("=" * 80)
    out = os.path.join(RUNS_ROOT, name)
    cfg = TrainConfig(output_dir=out, **{k: v for k, v in common.items() if k not in overrides}, **overrides)
    ckpt = train(cfg)

    print("-" * 80)
    print(f"EVAL: {name}")
    print("-" * 80)
    df = evaluate_all_benchmarks(
        data_root        = common["data_root"],
        ckpt_path        = ckpt,
        output_dir       = os.path.join(out, "eval_noTTA"),
        proteinmpnn_ckpt = common["proteinmpnn_ckpt"],
        adapter_layer    = common["adapter_layer"],
    )
    print(df.to_string())

print("\n" + "=" * 80)
print("BOTH JOBS COMPLETE")
print("=" * 80)
print("Next step: send me the seed-variance and long-patience numbers.")
