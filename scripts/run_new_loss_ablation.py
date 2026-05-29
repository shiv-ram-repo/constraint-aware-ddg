import logging
import os
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
    seed             = 42,
)

runs = {
    # E: BMC + BCAS (replaces siamese)
    "E_bmc_bcas": dict(
        loss_type           = "bmc",
        use_siamese         = False,
        use_bcas            = True,
        bcas_alpha          = 1.0,
        bcas_beta           = 0.5,
        use_ood_margin      = False,
    ),
    # F: BMC + siamese + OOD-margin (additive on top of D_bmc_sia)
    "F_bmc_sia_margin": dict(
        loss_type           = "bmc",
        use_siamese         = True,
        siamese_weight      = 0.5,
        use_bcas            = False,
        use_ood_margin      = True,
        ood_margin_sigma    = 0.1,
        ood_margin_weight   = 0.5,
        ood_margin_samples  = 1,
    ),
    # G: BMC + BCAS + OOD-margin (full combo)
    "G_bmc_bcas_margin": dict(
        loss_type           = "bmc",
        use_siamese         = False,
        use_bcas            = True,
        bcas_alpha          = 1.0,
        bcas_beta           = 0.5,
        use_ood_margin      = True,
        ood_margin_sigma    = 0.1,
        ood_margin_weight   = 0.5,
        ood_margin_samples  = 1,
    ),
}

t0 = time.time()
for name, overrides in runs.items():
    log.info("=" * 80)
    log.info(f"RUN: {name}")
    log.info("=" * 80)
    out = os.path.join(RUNS_ROOT, name)
    cfg = TrainConfig(output_dir=out, **common, **overrides)

    # Skip-if-exists for resumability
    ckpt_path = os.path.join(out, "best.pt")
    metrics_path = os.path.join(out, "eval_noTTA", "test_metrics.csv")
    if os.path.isfile(ckpt_path) and os.path.isfile(metrics_path):
        log.info(f"SKIP: {name} already complete")
        continue

    if not os.path.isfile(ckpt_path):
        ckpt = train(cfg)
    else:
        ckpt = ckpt_path

    log.info("-" * 80)
    log.info(f"EVAL: {name}")
    log.info("-" * 80)
    df = evaluate_all_benchmarks(
        data_root        = common["data_root"],
        ckpt_path        = ckpt,
        output_dir       = os.path.join(out, "eval_noTTA"),
        proteinmpnn_ckpt = common["proteinmpnn_ckpt"],
        adapter_layer    = common["adapter_layer"],
    )
    log.info(df.to_string())

log.info("=" * 80)
log.info(f"NEW-LOSS ABLATION COMPLETE  ({(time.time()-t0)/60:.1f} min)")
log.info("=" * 80)
log.info("")
log.info("Headline comparison vs D_bmc_sia (seed 42 baseline):")
log.info("  D_bmc_sia:    S669=0.524  S461=0.683  ssym_direct=0.716  ssym_inverse=0.608")
log.info("")
log.info("Decision rule: if E, F, or G beats D_bmc_sia by ≥+0.005 on S669")
log.info("OR ≥+0.01 on Ssym-inverse, run seeds 43 and 44 for variance.")
log.info("Otherwise, report as negative results in the paper.")
