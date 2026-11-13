"""FID and KID for the two trained models, with spread over sampling seeds and a real-vs-real floor.

    python scripts/eval_fid.py [--seeds 3] [--n 10000] [--ddpm-seeds 1]

For every schedule (linear, cosine) and sampler (DDIM 50, DPM-Solver++ 20, Euler-Karras 20; DDPM 1000 for the first
``--ddpm-seeds`` seeds) it draws ``--n`` samples per sampling seed and compares them with the 10,000 Fashion-MNIST
test images. The floor is the same metric between ``--n`` real training images and the test images (a different random
subset per seed): a perfect generator cannot score below it. Results are written to docs/data/fid.json after every
configuration, and finished configurations are skipped on a re-run.

Metric: torch-fidelity Inception-v3 pool features, FID and KID (subsets of 1,000). Images are 32x32 greyscale, rounded to
uint8, replicated to RGB and resized to 299x299 by the library. Read the numbers as comparisons inside this repo only.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch

from diffuselab.data import HOLDOUT, load_fashion
from diffuselab.metrics import fid_kid, generate, to_uint8_rgb
from diffuselab.models import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data" / "fid.json"
SAMPLERS = [("ddim", 50), ("dpmpp", 20), ("euler", 20)]
SEED_BASE = 1000
CACHE = "fashion_mnist_test_u8_n10000"  # torch-fidelity caches the real images' Inception features under this name


def summarise(runs: list[dict]) -> dict:
    f = [r["fid"] for r in runs]
    k = [r["kid_mean"] for r in runs]
    return dict(
        runs=runs,
        fid_mean=st.fmean(f), fid_std=st.stdev(f) if len(f) > 1 else None,
        kid_mean=st.fmean(k), kid_std=st.stdev(k) if len(k) > 1 else None,
    )  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--ddpm-seeds", type=int, default=1, help="sampling seeds to run DDPM-1000 for (0 skips it)")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints"))
    ap.add_argument("--preset", default="full")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    import torch_fidelity

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test, _ = load_fashion(ROOT / "data", split="test")
    train, _ = load_fashion(ROOT / "data")
    train = train[:-HOLDOUT]
    test_u8 = to_uint8_rgb(test)
    report = json.loads(out.read_text()) if out.exists() else {}
    report["meta"] = dict(
        n_generated=a.n, n_real=len(test), real_set="Fashion-MNIST test split", kid_subset_size=1000,
        sampling_seeds=[SEED_BASE + i for i in range(a.seeds)], device=str(dev),
        gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None, torch=torch.__version__,
        torch_fidelity=torch_fidelity.__version__, feature_extractor="Inception-v3 (torch-fidelity default weights), pool features",
    )  # fmt: skip
    save = lambda: out.write_text(json.dumps(report, indent=1))  # noqa: E731

    if "floor" not in report:
        runs = []
        for i in range(a.seeds):
            idx = np.random.default_rng(SEED_BASE + i).choice(len(train), a.n, replace=False)
            runs.append(dict(seed=SEED_BASE + i, **fid_kid(to_uint8_rgb(train[idx]), test_u8, cuda=dev.type == "cuda", real_cache_name=CACHE)))
            print("floor", runs[-1], flush=True)
        report["floor"] = summarise(runs)
        save()

    report.setdefault("results", {})
    for kind in ("linear", "cosine"):
        net, info = load_checkpoint(Path(a.ckpt) / f"fmnist_{kind}_{a.preset}.pt", dev)
        report.setdefault("models", {})[kind] = dict(steps=info["steps"], seconds=info["seconds"], final_epoch_loss=info["final_epoch_loss"])
        cells = [(s, n, a.seeds) for s, n in SAMPLERS] + ([("ddpm", 1000, a.ddpm_seeds)] if a.ddpm_seeds else [])
        for sched, steps, n_seeds in cells:
            key = f"{sched}-{steps}"
            if key in report["results"].get(kind, {}):
                continue
            runs = []
            for i in range(n_seeds):
                t0 = time.perf_counter()
                gen = generate(net, kind, sched, steps, a.n, SEED_BASE + i)
                t_gen = time.perf_counter() - t0
                runs.append(dict(seed=SEED_BASE + i, seconds_to_sample=round(t_gen, 1), **fid_kid(to_uint8_rgb(gen), test_u8, cuda=dev.type == "cuda", real_cache_name=CACHE)))
                print(kind, key, runs[-1], flush=True)
            report["results"].setdefault(kind, {})[key] = summarise(runs)
            save()
        del net
        torch.cuda.empty_cache()
    print("wrote", out)


if __name__ == "__main__":
    main()
