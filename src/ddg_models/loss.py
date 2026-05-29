
from __future__ import annotations

import torch
import torch.nn as nn


class L1Loss(nn.L1Loss):
    def forward(self, pred_value: torch.Tensor, y: torch.Tensor):
        loss_avg = super().forward(pred_value, y)
        return loss_avg, {
            "loss_sum":   loss_avg.detach(),
            "pred_value": pred_value.detach(),
            "y":          y.detach(),
        }


class MSELoss(nn.MSELoss):
    def forward(self, pred_value: torch.Tensor, y: torch.Tensor):
        loss_avg = super().forward(pred_value, y)
        return loss_avg, {
            "loss_sum":   loss_avg.detach(),
            "pred_value": pred_value.detach(),
            "y":          y.detach(),
        }
