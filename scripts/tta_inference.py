from __future__ import annotations

from typing import Callable

import torch


def perturb_backbone(
    coords: torch.Tensor,
    sigma: float = 0.1,
    keep_first: bool = True,
) -> torch.Tensor:
    """
    Add isotropic Gaussian noise to backbone coordinates.

    Args:
        coords:     (B, L, 4, 3) backbone atom coordinates (N, Cα, C, O)
                    in the SPURS convention.
        sigma:      noise stddev in angstroms. Typical: 0.05 to 0.2.
        keep_first: leave the first residue fixed so the global frame
                    doesn't drift.

    Returns:
        Perturbed coordinates, same shape.
    """
    noise = torch.randn_like(coords) * sigma
    if keep_first:
        noise[:, 0] = 0.0
    return coords + noise


@torch.no_grad()
def tta_predict(
    forward_fn: Callable,
    batch: dict,
    n_aug: int = 8,
    sigma: float = 0.1,
    coord_key: str = "X",
) -> torch.Tensor:
    """
    Run `n_aug` perturbed forward passes and return the mean prediction.

    Args:
        forward_fn:  callable that takes a batch dict and returns predictions
                     of shape (K,). Should already be in eval mode.
        batch:       batch dict containing backbone coordinates under
                     `coord_key` (default 'X', SPURS convention).
        n_aug:       number of stochastic forward passes. 1 = no TTA.
        sigma:       perturbation magnitude in angstroms.
        coord_key:   which key in the batch holds (B, L, 4, 3) coords.

    Returns:
        Mean predictions over the n_aug augmented passes, shape (K,).
    """
    if n_aug <= 1:
        return forward_fn(batch)

    coords_clean = batch[coord_key]
    preds = []
    for _ in range(n_aug):
        batch[coord_key] = perturb_backbone(coords_clean, sigma=sigma)
        preds.append(forward_fn(batch))
    batch[coord_key] = coords_clean  # restore

    return torch.stack(preds, dim=0).mean(dim=0)
