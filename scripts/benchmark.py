"""Time one 32x32 sample per sampler on the pretrained CIFAR-10 model (batch 1, recording off).

    python scripts/benchmark.py [--device cuda] [--out docs/data/benchmark.json]

Wall-clock numbers depend on whatever else the machine is doing; the JSON records the device and torch build.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from diffuselab.models import get_device, load_pretrained
from diffuselab.sampler import sample

CASES = [("ddpm", 1000), ("ddim", 50), ("dpmpp", 20), ("ddim", 20)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="docs/data/benchmark.json")
    a = ap.parse_args()
    dev = get_device(a.device)
    net = load_pretrained("cifar10", dev)
    sample(net, "ddim", 5, 0, record_every=100)  # warm-up
    rows = []
    for sched, steps in CASES:
        times = []
        for r in range(a.repeats):
            t0 = time.perf_counter()
            sample(net, sched, steps, r, record_every=max(1, steps))
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        rows.append(dict(scheduler=sched, steps=steps, seconds_median=sorted(times)[len(times) // 2], seconds_all=times))
        print(rows[-1])
    Path(a.out).write_text(json.dumps(dict(device=str(dev), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None, rows=rows), indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
