
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddg_data import (
    Alphabet,
    Featurizer,
    MegaScaleDataset,
)
from ddg_models import (
    MultimodalDDG, MultimodalDDGConfig,
    L1Loss,
)

from .evaluate import evaluate
from .metrics  import compute_metrics

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # Data
    data_root:        str = "/data/spurs_root"

    # Output
    output_dir:       str = "runs/spurs_repro_v1"
    ckpt_filename:    str = "best.pt"

    # ProteinMPNN checkpoint (optional but recommended for reproducing MultimodalDDG)
    proteinmpnn_ckpt: str = ""

    # Schedule
    max_epochs:       int = 200
    early_stop_pat:   int = 30
    lr:               float = 1e-4
    weight_decay:     float = 1e-2
    val_every:        int = 1
    grad_accum:       int = 1
    seed:             int = 42

    # ESM
    esm_name:         str = "esm2_t33_650M_UR50D"
    adapter_layer:    int = -3   # equivalent to layer 31 of 33 (= layer index 30)

    # ProteinMPNN tune flag
    tune_mpnn:        bool = True


def _move_batch(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def build_dataloaders(cfg: TrainConfig):
    alphabet   = Alphabet(name="esm")
    featurizer = Featurizer(alphabet)

    train_ds = MegaScaleDataset(data_root=cfg.data_root, split="train")
    val_ds   = MegaScaleDataset(data_root=cfg.data_root, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=featurizer,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=featurizer,
    )
    return train_loader, val_loader


def build_model(cfg: TrainConfig) -> MultimodalDDG:
    spurs_cfg = MultimodalDDGConfig()
    # Set the adapter location (MultimodalDDG uses layer 31 of 33 → index -3)
    spurs_cfg.adapter_layer_indices = [cfg.adapter_layer]
    spurs_cfg.name                  = cfg.esm_name
    spurs_cfg.proteinmpnn_ckpt      = cfg.proteinmpnn_ckpt

    # Make ProteinMPNN trainable (MultimodalDDG does this)
    spurs_cfg.encoder.tune = cfg.tune_mpnn

    # MLP defaults — input_dim is 128 (encoder feats), MLP class adds +1280 internally
    spurs_cfg.mlp.input_dim  = 128
    spurs_cfg.mlp.hidden_dim = 512
    spurs_cfg.mlp.output_dim = 21
    spurs_cfg.mlp.dropout    = 0.1

    return MultimodalDDG(spurs_cfg)


def train_one_epoch(model, loss_fn, loader, optimizer, device, epoch, grad_accum=1):
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    n            = 0

    loop = tqdm(loader, desc=f"Train Ep{epoch:03d}", leave=False)
    for step, batch in enumerate(loop):
        batch    = _move_batch(batch, device)
        ddg_pred = model(batch).view(-1)
        ddg_true = batch["ddG"].view(-1)

        # Drop sentinel values
        valid = (ddg_pred.abs() < 9000) & torch.isfinite(ddg_true)
        if valid.sum() == 0:
            continue
        loss, _ = loss_fn(ddg_pred[valid], ddg_true[valid])
        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += float(loss.item())
        n            += 1
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(n, 1)


def train(cfg: Optional[TrainConfig] = None):
    if cfg is None:
        cfg = TrainConfig()

    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Config: {asdict(cfg)}")

    train_loader, val_loader = build_dataloaders(cfg)
    log.info(f"Train: {len(train_loader)}  Val: {len(val_loader)}")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    loss_fn   = L1Loss()
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    best_sp     = -1.0
    best_epoch  = 0
    no_improve  = 0
    logs        = []
    ckpt_path   = os.path.join(cfg.output_dir, cfg.ckpt_filename)
    log_path    = os.path.join(cfg.output_dir, "training_log.csv")

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(
            model, loss_fn, train_loader, optimizer, device, epoch,
            grad_accum=cfg.grad_accum,
        )

        if epoch % cfg.val_every == 0:
            preds, targets = evaluate(model, val_loader, device, name="val")
            val_m = compute_metrics(preds, targets)
        else:
            val_m = {"spearman": float("nan"), "pearson": float("nan"),
                     "rmse": float("nan"), "mae": float("nan"),
                     "r2": float("nan"), "n": 0}

        flag = ""
        if not np.isnan(val_m["spearman"]) and val_m["spearman"] > best_sp:
            best_sp     = val_m["spearman"]
            best_epoch  = epoch
            no_improve  = 0
            torch.save(model.state_dict(), ckpt_path)
            flag = "  ← BEST"
        else:
            no_improve += 1

        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_m.items()}}
        logs.append(row)
        pd.DataFrame(logs).to_csv(log_path, index=False)

        log.info(
            f"Ep {epoch:03d} | train={train_loss:.4f} | "
            f"val ρ={val_m['spearman']:.4f}  r={val_m['pearson']:.4f}  "
            f"rmse={val_m['rmse']:.4f}{flag}"
        )

        if no_improve >= cfg.early_stop_pat:
            log.info(f"Early stopping at epoch {epoch}")
            break

    log.info(f"Best val Spearman: {best_sp:.4f} at epoch {best_epoch}")
    log.info(f"Best checkpoint  : {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    train()
