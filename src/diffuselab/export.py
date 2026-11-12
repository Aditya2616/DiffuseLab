"""Turn recorded trajectories into the compact files the static demo replays.

A run becomes one WebP sprite sheet plus a small JSON file. The sheet has one column per recorded frame
and three rows: x_t, predicted noise, predicted x0. Pixels are 8-bit for display only (x_t and x0 clipped
to [-1, 1], eps clipped to [-3, 3]); all maths in the UI uses the schedule numbers in the JSON, not the
sprite. Frame 0 is the starting noise and the last column is the finished sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .recorder import Trajectory

EPS_RANGE = 3.0
WEBP_QUALITY = 88


def quantise(a: np.ndarray, rng: float = 1.0) -> np.ndarray:
    """float -> uint8, clip(a, -rng, rng) mapped to 0..255."""
    return np.round((np.clip(a, -rng, rng) / rng + 1.0) * 127.5).astype(np.uint8)


def dequantise(q: np.ndarray, rng: float = 1.0) -> np.ndarray:
    return (q.astype(np.float32) / 127.5 - 1.0) * rng


def _to_rgb(frames_u8: np.ndarray) -> np.ndarray:
    """(n,C,H,W) uint8 -> (n,H,W,3), replicating grey."""
    f = frames_u8.transpose(0, 2, 3, 1)
    return np.repeat(f, 3, axis=3) if f.shape[-1] == 1 else f


def sprite(traj: Trajectory) -> Image.Image:
    rows = [quantise(traj.x_t), quantise(traj.eps, EPS_RANGE), quantise(traj.x0)]
    sheet = [np.concatenate(list(_to_rgb(r)), axis=1) for r in rows]  # each (H, n*W, 3)
    return Image.fromarray(np.concatenate(sheet, axis=0), "RGB")


def run_id(traj: Trajectory, model_key: str) -> str:
    return f"{model_key}-{traj.scheduler}-{traj.kind}-s{traj.seed}"


def _num(x: float, nd: int = 6):
    return None if not np.isfinite(x) else round(float(x), nd)


def write_run(traj: Trajectory, out_dir: str | Path, model_key: str, quality: int = WEBP_QUALITY) -> dict:
    out = Path(out_dir) / "runs"
    out.mkdir(parents=True, exist_ok=True)
    rid = run_id(traj, model_key)
    c, h, w = traj.x_t.shape[1:]
    sprite(traj).save(out / f"{rid}.webp", "WEBP", quality=quality, method=6)
    meta = dict(
        id=rid, model=model_key, scheduler=traj.scheduler, kind=traj.kind, T=traj.T, seed=traj.seed,
        channels=c, height=h, width=w, frames=traj.n_frames, steps=traj.meta.get("steps"),
        record_every=traj.meta.get("record_every"), seconds=traj.meta.get("seconds"),
        device=traj.meta.get("device"), eps_range=EPS_RANGE,
        t=traj.t.tolist(), abar=[_num(v, 8) for v in traj.abar], beta=[_num(v, 8) for v in traj.beta],
        snr=[_num(v, 6) for v in traj.snr],
    )  # fmt: skip
    (out / f"{rid}.json").write_text(json.dumps(meta, separators=(",", ":")))
    size = (out / f"{rid}.webp").stat().st_size + (out / f"{rid}.json").stat().st_size
    return dict(id=rid, model=model_key, scheduler=traj.scheduler, kind=traj.kind, seed=traj.seed,
                steps=traj.meta.get("steps"), frames=traj.n_frames, bytes=size)  # fmt: skip


def save_png_grid(traj: Trajectory, path: str) -> None:
    sprite(traj).save(path)


def image_grid(imgs: np.ndarray, cols: int = 8) -> Image.Image:
    """(n,C,H,W) in [-1,1] -> a tiled PNG-ready image."""
    q = _to_rgb(quantise(imgs))
    n, h, w, _ = q.shape
    rows = -(-n // cols)
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = q[i]
    return Image.fromarray(canvas)


CIFAR_SEEDS = (3, 5, 11)
FASHION_SEEDS = (1, 2, 3)
RUN_SPECS = [("ddpm", 1000, 20), ("ddim", 50, 1), ("dpmpp", 20, 1)]  # (scheduler, steps, record_every)


def record_demo(out_dir: str | Path = "docs/data", which: str = "all", ckpt_dir: str | Path = "checkpoints",
                preset: str = "full") -> list[dict]:
    """Regenerate every recorded run the static demo ships. Needs the pretrained model and trained checkpoints."""
    from .models import load_checkpoint, load_pretrained
    from .sampler import sample

    out = Path(out_dir)
    index: list[dict] = []
    if which in ("all", "cifar10"):
        net = load_pretrained("cifar10")
        for seed in CIFAR_SEEDS:
            for sched, steps, every in RUN_SPECS:
                tr = sample(net, sched, steps, seed, "linear", model_name="cifar10", record_every=every)
                index.append(write_run(tr, out, "cifar10"))
                print("recorded", index[-1]["id"], index[-1]["bytes"] // 1024, "KB")
    if which in ("all", "fashion"):
        for kind in ("linear", "cosine"):
            ck = Path(ckpt_dir) / f"fmnist_{kind}_{preset}.pt"
            if not ck.exists():
                print("skipping", ck, "(not trained)")
                continue
            net, _ = load_checkpoint(ck)
            for seed in FASHION_SEEDS:
                for sched, steps, every in RUN_SPECS:
                    tr = sample(net, sched, steps, seed, kind, model_name="fashion", record_every=every)
                    index.append(write_run(tr, out, f"fashion-{kind}"))
                    print("recorded", index[-1]["id"], index[-1]["bytes"] // 1024, "KB")
    if which == "all":
        (out / "index.json").write_text(json.dumps(index, indent=1))
    return index
