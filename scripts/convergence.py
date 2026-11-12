"""How close do the fast samplers get to the many-step answer?

The pretrained CIFAR-10 model, the same starting noise for every sampler, and a 1000-step DDIM run as the reference
(DDIM is deterministic, so it is an ODE solution that more steps approach). Reports the RMSE between each sampler's
final image and the reference, in [-1, 1] pixel units, averaged over a batch of seeds.

    python scripts/convergence.py [--seeds 16] [--out docs/data/convergence.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from diffuselab.models import get_device, load_pretrained
from diffuselab.sampler import initial_noise
from diffuselab.schedulers import make_scheduler


@torch.no_grad()
def run(net, scheduler: str, steps: int, x_T: torch.Tensor) -> torch.Tensor:
    dev = next(net.parameters()).device
    sched = make_scheduler(scheduler, "linear")
    sched.set_timesteps(steps)
    x = x_T.to(dev)
    for t in sched.timesteps:
        eps = net(x, int(t)).sample.float()
        x = sched.step(eps.cpu(), int(t), x.cpu()).prev_sample.to(dev)
    return x.cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--out", default="docs/data/convergence.json")
    a = ap.parse_args()
    net = load_pretrained("cifar10", get_device())
    x_T = initial_noise((a.seeds, 3, 32, 32), 0)
    ref = run(net, "ddim", 1000, x_T)
    rows = []
    for steps in (5, 10, 20, 50):
        row = {"steps": steps}
        for s in ("ddim", "dpmpp"):
            row[s] = float(((run(net, s, steps, x_T) - ref) ** 2).mean(dim=(1, 2, 3)).sqrt().mean())
        rows.append(row)
        print(row)
    Path(a.out).write_text(
        json.dumps(dict(n_seeds=a.seeds, reference="DDIM, 1000 steps", model="google/ddpm-cifar10-32", rows=rows), indent=1)
    )


if __name__ == "__main__":
    main()
