"""Shared fixtures. Everything here is CPU-only and offline: no downloads, no checkpoints."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffuselab.models import build_unet
from diffuselab.schedules import make_schedule


class OracleModel(torch.nn.Module):
    """A perfect epsilon-predictor for a dataset that is a single known image c.

    eps(x_t, t) = (x_t - sqrt(abar) c) / sqrt(1 - abar), so x0_hat is exactly c at every step.
    """

    def __init__(self, c: torch.Tensor, kind: str = "linear", T: int = 1000):
        super().__init__()
        self.c = c
        self.sch = make_schedule(kind, T)
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(in_channels=c.shape[0], sample_size=c.shape[-1])

    def forward(self, x, t):
        ab = float(self.sch.alpha_bar[int(t)])
        eps = (x - np.sqrt(ab) * self.c) / np.sqrt(1 - ab)
        return SimpleNamespace(sample=eps)


@pytest.fixture
def oracle():
    c = torch.linspace(-0.8, 0.8, 1 * 8 * 8).reshape(1, 8, 8)
    return OracleModel(c), c


@pytest.fixture
def tiny_unet():
    torch.manual_seed(0)
    return build_unet(channels=1, size=16, widths=(32, 64), layers=1).eval()
