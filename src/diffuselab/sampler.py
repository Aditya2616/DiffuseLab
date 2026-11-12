"""The instrumented sampler: one loop, any Diffusers scheduler, every internal quantity recorded."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch

from .recorder import Trajectory
from .schedulers import make_scheduler
from .schedules import make_schedule, predict_x0


def initial_noise(shape, seed: int) -> torch.Tensor:
    """x_T for a seed. Drawn on the CPU so every scheduler and device starts from identical noise."""
    g = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(shape, generator=g)


@torch.no_grad()
def sample(
    unet,
    scheduler: str = "ddim",
    steps: int = 50,
    seed: int = 0,
    kind: str = "linear",
    T: int = 1000,
    channels: int | None = None,
    size: int | None = None,
    model_name: str = "model",
    record_every: int = 1,
    on_frame: Callable[[int, dict], None] | None = None,
) -> Trajectory:
    """Run the reverse process and record it.

    record_every keeps every k-th step (plus the first and the last frame), so a 1000-step DDPM run
    can be stored compactly. on_frame(k, frame) is called for each kept frame as it is produced.
    """
    device = next(unet.parameters()).device
    channels = channels or unet.config.in_channels
    size = size or unet.config.sample_size
    sch = make_schedule(kind, T)
    sched = make_scheduler(scheduler, kind, T)
    sched.set_timesteps(steps)
    euler = scheduler == "euler"
    # Euler works on the variance-exploding variable x_ve = x_t / sqrt(abar); the network still sees x_t.
    ts = [float(t) for t in sched.timesteps] if euler else [int(t) for t in sched.timesteps]
    n = len(ts)

    x = initial_noise((1, channels, size, size), seed).to(device)
    if euler:
        x = x * (float(sched.sigmas[0]) ** 2 + 1) ** 0.5  # so the network sees exactly the seeded x_T
    gen = torch.Generator("cpu").manual_seed(seed + 1)  # only DDPM consumes this

    keep = {k for k in range(n) if k % record_every == 0 or k == n - 1}
    rows: dict[str, list] = {"t": [], "abar": [], "beta": [], "snr": [], "x_t": [], "eps": [], "x0": []}

    def push(frame: dict) -> None:
        for key, val in frame.items():
            rows[key].append(val)
        if on_frame:
            on_frame(len(rows["t"]) - 1, frame)

    t0 = time.perf_counter()
    last_eps = None
    for k, t in enumerate(ts):
        if euler:
            sigma = float(sched.sigmas[k])
            ab = 1.0 / (1.0 + sigma**2)  # Karras sigmas fall between integer timesteps, so abar comes from sigma
            x_in = sched.scale_model_input(x, t)  # x / sqrt(sigma^2 + 1)
            t_rec = min(max(int(round(t)), 0), T - 1)
        else:
            ab, x_in, t_rec = float(sch.alpha_bar[t]), x, t
        eps = unet(x_in, t).sample.float()
        if k in keep:
            push(dict(
                t=t_rec, abar=ab, beta=float(sch.betas[t_rec]), snr=ab / (1 - ab),
                x_t=x_in[0].cpu().numpy(), eps=eps[0].cpu().numpy(),
                x0=predict_x0(x_in, eps, ab)[0].cpu().numpy(),
            ))  # fmt: skip
        last_eps = eps
        kwargs = {"generator": gen} if scheduler == "ddpm" else {}
        x = sched.step(eps.cpu(), t, x.cpu(), **kwargs).prev_sample.to(device)
    final = x[0].cpu().numpy()
    push(dict(t=-1, abar=1.0, beta=0.0, snr=float("inf"), x_t=final,
              eps=last_eps[0].cpu().numpy(), x0=final))  # fmt: skip
    elapsed = time.perf_counter() - t0

    f32 = ("x_t", "eps", "x0")
    arr = {k: np.asarray(v, dtype=np.float32 if k in f32 else np.float64) for k, v in rows.items()}
    return Trajectory(
        model=model_name, scheduler=scheduler, kind=kind, T=T, seed=seed,
        t=arr["t"].astype(np.int64), abar=arr["abar"], beta=arr["beta"], snr=arr["snr"],
        x_t=arr["x_t"], eps=arr["eps"], x0=arr["x0"],
        meta=dict(steps=n, record_every=record_every, seconds=round(elapsed, 3), device=str(device)),
    )  # fmt: skip
