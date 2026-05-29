
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddg_data    import Alphabet, Featurizer, MegaScaleTestDatasets
from ddg_models  import MultimodalDDG, MultimodalDDGConfig

from .metrics import compute_metrics

log = logging.getLogger(__name__)


def _move_batch(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


@torch.no_grad()
def evaluate(
    model, loader, device, name: str = ""
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run `model` on `loader` and return concatenated (preds, targets) numpy arrays.
    """
    model.eval()
    preds, targets = [], []
    for batch in tqdm(loader, desc=f"Eval {name}", leave=False):
        batch    = _move_batch(batch, device)
        ddg_pred = model(batch).view(-1).cpu().numpy()
        ddg_true = batch["ddG"].view(-1).cpu().numpy()
        preds.append(ddg_pred)
        targets.append(ddg_true)
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(targets)


def evaluate_all_benchmarks(
    data_root: str,
    ckpt_path: str,
    output_dir: str,
    *,
    proteinmpnn_ckpt: str = "",
    esm_name: str         = "esm2_t33_650M_UR50D",
    adapter_layer: int    = -3,
    tune_mpnn: bool       = True,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a trained SPURS checkpoint and evaluate on every test benchmark.

    Writes per-dataset predictions to {output_dir}/preds_<name>.npy and
    a summary table to {output_dir}/test_metrics.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Rebuild the model and load weights ──
    cfg = MultimodalDDGConfig()
    cfg.adapter_layer_indices = [adapter_layer]
    cfg.name                  = esm_name
    cfg.proteinmpnn_ckpt      = proteinmpnn_ckpt
    cfg.encoder.tune          = tune_mpnn
    cfg.mlp.input_dim         = 128
    cfg.mlp.hidden_dim        = 512
    cfg.mlp.output_dim        = 21
    cfg.mlp.dropout           = 0.1
    model = MultimodalDDG(cfg).to(device)

    state = torch.load(ckpt_path, map_location=device)
    # Allow loading from a Lightning ckpt or a plain state_dict
    if "state_dict" in state:
        state = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    # ── Test datasets ──
    alphabet   = Alphabet(name="esm")
    featurizer = Featurizer(alphabet)
    collection = MegaScaleTestDatasets(data_root=data_root)

    results = []
    for name, dataset in collection.iter_named():
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=2,
            collate_fn=featurizer,
        )
        preds, targets = evaluate(model, loader, device, name=name)
        m = compute_metrics(preds, targets)
        m["dataset"] = name
        results.append(m)
        log.info(
            f"{name:18s} (n={m['n']:6d}): "
            f"ρ={m['spearman']:.4f}  r={m['pearson']:.4f}  "
            f"RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}"
        )

        np.save(os.path.join(output_dir, f"preds_{name}.npy"),  preds)
        np.save(os.path.join(output_dir, f"targets_{name}.npy"), targets)

    df = pd.DataFrame(results)[
        ["dataset", "n", "spearman", "pearson", "r2", "rmse", "mae"]
    ]
    csv_path = os.path.join(output_dir, "test_metrics.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    log.info(f"Saved test metrics to {csv_path}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  required=True)
    p.add_argument("--ckpt",       required=True)
    p.add_argument("--out",        required=True)
    p.add_argument("--mpnn_ckpt",  default="")
    p.add_argument("--esm_name",   default="esm2_t33_650M_UR50D")
    args = p.parse_args()
    evaluate_all_benchmarks(
        data_root=args.data_root, ckpt_path=args.ckpt,
        output_dir=args.out, proteinmpnn_ckpt=args.mpnn_ckpt,
        esm_name=args.esm_name,
    )
