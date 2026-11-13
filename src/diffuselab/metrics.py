"""Sample-quality metrics: FID and KID with torch-fidelity (Inception-v3 pool features).

torch-fidelity and torchvision are optional (``pip install -e ".[metrics]"``) and imported only inside the functions
that need them. Generated and real images are 32x32 greyscale in [-1, 1]; for the metric they are rounded to uint8,
replicated to three channels and bilinearly resized to 299x299 by the library. Numbers from this module are comparable
with each other inside this repo; they are not comparable with FIDs in papers (different data, a tiny greyscale
domain, a different feature extractor weights file).
"""

from __future__ import annotations

import numpy as np
import torch

from .sampler import initial_noise
from .schedulers import make_scheduler, set_steps


def to_uint8_rgb(x: np.ndarray) -> torch.Tensor:
    """(N, 1, H, W) float in [-1, 1] -> (N, 3, H, W) uint8."""
    q = np.round((np.clip(x, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
    return torch.from_numpy(np.repeat(q, 3, axis=1))


@torch.no_grad()
def generate(
    net, kind: str, scheduler: str, steps: int, n: int, seed: int, batch: int = 500, size: int = 32
) -> np.ndarray:
    """n samples (N, 1, 32, 32) in [-1, 1] from a trained model, entirely on the model's device.

    Each batch has its own seeded starting noise (seed, batch index), so the result does not depend on the batch size
    chosen only through the split points; the same (seed, n, batch) always gives the same samples."""
    dev = next(net.parameters()).device
    euler = scheduler == "euler"
    out = []
    for b, start in enumerate(range(0, n, batch)):
        m = min(batch, n - start)
        sched = make_scheduler(scheduler, kind)
        set_steps(sched, scheduler, steps)
        x = initial_noise((m, 1, size, size), seed * 100_003 + b)
        if euler:
            x = x * (float(sched.sigmas[0]) ** 2 + 1) ** 0.5
        x = x.to(dev)
        gen = torch.Generator("cpu").manual_seed(seed * 100_003 + b + 1)
        for t in sched.timesteps:
            tt = float(t) if euler else int(t)
            x_in = sched.scale_model_input(x, t) if euler else x
            eps = net(x_in, tt).sample.float()
            kw = {"generator": gen} if scheduler == "ddpm" else {}
            x = sched.step(eps, tt, x, **kw).prev_sample
        out.append(x.clamp(-1, 1).cpu().numpy())
    return np.concatenate(out)


class _Images(torch.utils.data.Dataset):
    def __init__(self, u8: torch.Tensor) -> None:
        self.u8 = u8

    def __len__(self) -> int:
        return len(self.u8)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.u8[i]


def fid_kid(
    generated: torch.Tensor, real: torch.Tensor, cuda: bool = True, batch_size: int = 250, kid_subset_size: int = 1000,
    real_cache_name: str | None = None,
) -> dict:
    """FID and KID (mean, std over subsets) between two uint8 (N, 3, H, W) tensors, via torch-fidelity."""
    try:
        import torch_fidelity
    except ImportError as e:  # pragma: no cover
        raise SystemExit('FID/KID need the optional extra: pip install -e ".[metrics]"') from e
    m = torch_fidelity.calculate_metrics(
        input1=_Images(generated), input2=_Images(real), cuda=cuda and torch.cuda.is_available(), fid=True, kid=True,
        kid_subset_size=kid_subset_size, batch_size=batch_size, verbose=False, samples_find_deep=False,
        input2_cache_name=real_cache_name, cache=real_cache_name is not None,
    )  # fmt: skip
    return dict(
        fid=float(m["frechet_inception_distance"]),
        kid_mean=float(m["kernel_inception_distance_mean"]),
        kid_std=float(m["kernel_inception_distance_std"]),
    )
