"""Loading models and choosing a device."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from diffusers import UNet2DModel

PRETRAINED = {"cifar10": "google/ddpm-cifar10-32"}
CHECKPOINT_DIR = Path(os.environ.get("DIFFUSELAB_CKPT", "checkpoints"))


def get_device(prefer: str | None = None) -> torch.device:
    prefer = prefer or os.environ.get("DIFFUSELAB_DEVICE", "auto")
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


def build_unet(channels: int = 1, size: int = 32, widths=(64, 128, 256), layers: int = 2) -> UNet2DModel:
    """The small UNet used for from-scratch training (attention at the lowest resolution)."""
    n = len(widths)
    down = ["DownBlock2D"] * (n - 1) + ["AttnDownBlock2D"]
    up = ["AttnUpBlock2D"] + ["UpBlock2D"] * (n - 1)
    return UNet2DModel(
        sample_size=size, in_channels=channels, out_channels=channels, layers_per_block=layers,
        block_out_channels=tuple(widths), down_block_types=tuple(down), up_block_types=tuple(up),
        attention_head_dim=32,
    )  # fmt: skip


def load_pretrained(name: str = "cifar10", device=None) -> UNet2DModel:
    return UNet2DModel.from_pretrained(PRETRAINED[name]).to(device or get_device()).eval()


def load_checkpoint(path: str | Path, device=None):
    """Return (unet, info) from a checkpoint written by diffuselab.train."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_unet(**ck["unet_cfg"])
    net.load_state_dict(ck["ema"])
    return net.to(device or get_device()).eval(), ck.get("info", {})


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
