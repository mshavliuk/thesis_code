import torch
from torch import nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        loss = (pred - target)[mask].pow(2).mean()
        return loss
