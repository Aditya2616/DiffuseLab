"""Collect the trained checkpoints into docs/data: loss curves, sample grids, the memorisation check, real images.

    python scripts/build_training_report.py [--preset full]

Memorisation check (nearest neighbour): draw 256 samples from each model, find each sample's nearest neighbour
(L2 over the 1024 pixels, values in [-1, 1]) among the 55,000 training images, and compare with the same distance
for 256 real images that were held out of training. If the model copied training images, the generated distances would
sit far below the held-out distances. This is a sanity check, not a proof either way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffuselab.data import CLASSES, HOLDOUT, load_fashion
from diffuselab.export import image_grid, quantise
from diffuselab.train import sample_grid

ROOT = Path(__file__).resolve().parents[1]
GRID_SEED = 123
SAMPLERS = [("ddim", 50), ("dpmpp", 20), ("euler", 20), ("ddpm", 1000)]


def nn_distances(queries: np.ndarray, train: torch.Tensor) -> np.ndarray:
    q = torch.from_numpy(queries.reshape(len(queries), -1)).float()
    d = torch.cdist(q, train)  # (nq, ntrain)
    return d.min(1).values.numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "data"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    x_all, labels = load_fashion(ROOT / "data")
    train = torch.from_numpy(x_all[:-HOLDOUT].reshape(len(x_all) - HOLDOUT, -1)).float()
    held = x_all[-HOLDOUT:]
    held_nn = nn_distances(held[:256], train)

    # eight real held-out images for the forward-process demo (one per class where possible)
    picks, seen = [], set()
    for i in range(HOLDOUT):
        c = int(labels[-HOLDOUT + i])
        if c not in seen and len(picks) < 8:
            seen.add(c)
            picks.append(i)
    strip = np.concatenate([quantise(held[i, 0]) for i in picks], axis=1)
    Image.fromarray(strip, "L").convert("RGB").save(out / "fashion_real.webp", "WEBP", lossless=True)

    report = dict(
        runs={}, nn_check={}, grid_seed=GRID_SEED, holdout=HOLDOUT, real_classes=[CLASSES[int(labels[-HOLDOUT + i])] for i in picks]
    )
    for kind in ("linear", "cosine"):
        ck = Path(a.ckpt) / f"fmnist_{kind}_{a.preset}.pt"
        if not ck.exists():
            print("missing", ck)
            continue
        info = json.loads(ck.with_suffix(".json").read_text())
        report["runs"][kind] = info
        samples = sample_grid(ck, kind, n=256, steps=50, seed=GRID_SEED)
        image_grid(samples[:64], cols=8).save(out / f"grid_{kind}.png", optimize=True)
        gen_nn = nn_distances(samples, train)
        report["nn_check"][kind] = dict(
            n=256,
            generated_to_train_median=float(np.median(gen_nn)),
            generated_to_train_min=float(gen_nn.min()),
            heldout_to_train_median=float(np.median(held_nn)),
            heldout_to_train_min=float(held_nn.min()),
            generated_below_heldout_p1=float((gen_nn < np.percentile(held_nn, 1)).mean()),
        )
        print(kind, report["nn_check"][kind])
        report.setdefault("samplers", {})[kind] = {}
        for sched, steps in SAMPLERS:
            smp = samples if (sched, steps) == ("ddim", 50) else sample_grid(ck, kind, n=256, steps=steps, seed=GRID_SEED, scheduler=sched)
            image_grid(smp[:64], cols=8).save(out / f"grid_{kind}_{sched}{steps}.png", optimize=True)
            d = nn_distances(smp, train)
            report["samplers"][kind][f"{sched}-{steps}"] = dict(
                finite=bool(np.isfinite(smp).all()), pixel_std=float(smp.std()),
                generated_to_train_median=float(np.median(d)),
                mean_abs_pixel_gap_to_real=float(abs(smp.mean() - held.mean())),
            )
            print(kind, sched, steps, report["samplers"][kind][f"{sched}-{steps}"])
    report["notes"] = [
        "Loss is the mean epsilon-prediction MSE over random t, so it falls fast and then plateaus; it says little about sample quality.",
        "NN distance is L2 over 32x32 pixels in [-1, 1] to the closest of the 55,000 training images, median of 256 samples versus 256 real held-out images.",
        "Sample quality is judged by eye from the grids below and by FID and KID (scripts/eval_fid.py), which are comparable inside this repo only.",
    ]
    (out / "training.json").write_text(json.dumps(report, indent=1))
    print("wrote", out / "training.json")


if __name__ == "__main__":
    main()
