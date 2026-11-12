"""Train a small epsilon-prediction DDPM from scratch on Fashion-MNIST.

The loop is the plain DDPM objective: pick t uniformly, noise x0 with the closed-form forward process,
ask the UNet for the noise, minimise MSE. An EMA copy of the weights is what gets sampled.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import HOLDOUT, load_fashion
from .models import build_unet, count_params, get_device
from .sampler import sample
from .schedules import make_schedule

PRESETS = {
    # full: what the published results use (GPU, about 15 minutes per schedule)
    "full": dict(widths=(64, 128, 256), epochs=30, batch=128),
    # tiny: a CPU-friendly run that finishes in a few minutes and still draws recognisable shapes
    "tiny": dict(widths=(32, 64, 128), epochs=4, batch=128),
}


def train(
    kind: str = "linear",
    preset: str = "full",
    epochs: int | None = None,
    batch: int | None = None,
    lr: float = 2e-4,
    ema_decay: float = 0.999,
    seed: int = 0,
    device: str | None = None,
    out_dir: str | Path = "checkpoints",
    max_steps: int | None = None,
    data_dir: str | Path = "data",
    log=print,
) -> dict:
    cfg = PRESETS[preset]
    epochs = epochs or cfg["epochs"]
    batch = batch or cfg["batch"]
    dev = get_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_all, _ = load_fashion(Path(data_dir))
    x_train = torch.from_numpy(x_all[:-HOLDOUT]).to(dev)
    n = len(x_train)
    sch = make_schedule(kind, 1000)
    abar = torch.tensor(sch.alpha_bar, dtype=torch.float32, device=dev)

    unet_cfg = dict(channels=1, size=32, widths=tuple(cfg["widths"]), layers=2)
    net = build_unet(**unet_cfg).to(dev)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    use_amp = dev.type == "cuda"
    log(f"[train] kind={kind} preset={preset} device={dev} params={count_params(net):,} "
        f"epochs={epochs} batch={batch} train_images={n}")  # fmt: skip

    losses: list[float] = []  # mean loss per 50 steps
    epoch_loss: list[float] = []
    step, acc, acc_n = 0, 0.0, 0
    t_start = time.perf_counter()
    done = False
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        run, run_n = 0.0, 0
        for i in range(0, n - batch + 1, batch):
            x0 = x_train[perm[i : i + batch]]
            t = torch.randint(0, 1000, (batch,), device=dev)
            eps = torch.randn_like(x0)
            ab = abar[t].view(-1, 1, 1, 1)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                pred = net(xt, t).sample
            loss = F.mse_loss(pred.float(), eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                d = min(ema_decay, (1 + step) / (10 + step))
                for pe, pn in zip(ema.parameters(), net.parameters(), strict=True):
                    pe.lerp_(pn, 1 - d)
            step += 1
            lv = loss.item() if step % 10 == 0 else None
            if lv is not None:
                acc += lv
                acc_n += 1
                run += lv
                run_n += 1
            if step % 50 == 0 and acc_n:
                losses.append(acc / acc_n)
                acc, acc_n = 0.0, 0
            if max_steps and step >= max_steps:
                done = True
                break
        epoch_loss.append(run / max(run_n, 1))
        log(f"[train] epoch {ep + 1}/{epochs} loss={epoch_loss[-1]:.4f} elapsed={time.perf_counter() - t_start:.0f}s")
        if done:
            break
    seconds = time.perf_counter() - t_start

    info = dict(
        kind=kind, preset=preset, epochs_run=len(epoch_loss), steps=step, batch=batch, lr=lr,
        ema_decay=ema_decay, seed=seed, seconds=round(seconds, 1), device=str(dev),
        gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
        params=count_params(net), torch=torch.__version__, train_images=n,
        loss_every_50_steps=[round(v, 5) for v in losses], loss_per_epoch=[round(v, 5) for v in epoch_loss],
        final_epoch_loss=round(epoch_loss[-1], 5),
    )  # fmt: skip
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"fmnist_{kind}_{preset}.pt"
    torch.save(dict(ema=ema.state_dict(), unet_cfg=unet_cfg, info=info), path)
    (out / f"fmnist_{kind}_{preset}.json").write_text(json.dumps(info, indent=1))
    log(f"[train] saved {path} in {seconds:.0f}s")
    return info


def sample_grid(path: str | Path, kind: str, n: int = 64, steps: int = 50, seed: int = 123) -> np.ndarray:
    """Draw n DDIM samples from a checkpoint; returns (n,1,32,32) in [-1,1]. Used for grids and checks."""
    from .models import load_checkpoint

    net, _ = load_checkpoint(path)
    dev = next(net.parameters()).device
    from .sampler import initial_noise
    from .schedulers import make_scheduler

    sched = make_scheduler("ddim", kind)
    sched.set_timesteps(steps)
    x = initial_noise((n, 1, 32, 32), seed).to(dev)
    with torch.no_grad():
        for t in sched.timesteps:
            eps = net(x, int(t)).sample.float()
            x = sched.step(eps.cpu(), int(t), x.cpu()).prev_sample.to(dev)
    return x.clamp(-1, 1).cpu().numpy()


__all__ = ["PRESETS", "sample", "sample_grid", "train"]
