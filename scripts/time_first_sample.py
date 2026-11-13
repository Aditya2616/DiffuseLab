"""Time the quickstart's first command on the CPU: one pretrained CIFAR-10 sample with the fast sampler.

    python scripts/time_first_sample.py --threads 3 --load-note "..."

Runs `diffuselab sample --device cpu --scheduler dpmpp --steps 20` as a subprocess, repeated, and records the wall time of
the whole command (interpreter start, model load, sampling, writing files) in docs/data/first_sample.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--load-note", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "docs" / "data" / "first_sample.json"))
    a = ap.parse_args()
    env = {**os.environ, "OMP_NUM_THREADS": str(a.threads), "CUDA_VISIBLE_DEVICES": ""}
    cmd = [sys.executable, "-m", "diffuselab", "sample", "--device", "cpu", "--scheduler", "dpmpp", "--steps", "20",
           "--out", str(ROOT / "runs" / "first_sample.npz")]  # fmt: skip
    times = []
    for _ in range(a.repeats):
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, env=env, cwd=ROOT, capture_output=True)
        times.append(round(time.perf_counter() - t0, 1))
        print(times[-1], flush=True)
    rec = dict(
        command="diffuselab sample --device cpu --scheduler dpmpp --steps 20", model="google/ddpm-cifar10-32",
        what="wall time of the whole command, model already downloaded, one 32x32 sample", device="cpu",
        threads=a.threads, load_note=a.load_note, repeats=a.repeats, seconds_all=times, seconds_median=sorted(times)[len(times) // 2],
    )  # fmt: skip
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(rec)


if __name__ == "__main__":
    main()
