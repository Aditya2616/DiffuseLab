"""Record what the CPU preset produced: timing from its checkpoint, plus a grid of samples.

    diffuselab train --schedule linear --device cpu      # the tiny preset, about ten minutes on 3 threads
    python scripts/cpu_preset_report.py --threads 3 --load-note "..."

Writes docs/data/cpu_timing.json and docs/data/grid_cpu_preset.png. The load note is what you observed while the
training ran (the machine is rarely idle), passed in so the README can state it next to the time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from diffuselab.export import image_grid
from diffuselab.train import sample_grid

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "fmnist_linear_tiny.pt"))
    ap.add_argument("--threads", type=int, required=True, help="torch threads the training used")
    ap.add_argument("--load-note", required=True, help="what else was running while it trained")
    ap.add_argument("--out", default=str(ROOT / "docs" / "data"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ck = Path(a.ckpt)
    info = json.loads(ck.with_suffix(".json").read_text())
    assert info["device"] == "cpu", "this report is for a CPU run"
    out = Path(a.out)
    samples = sample_grid(ck, info["kind"], n=64, steps=20, seed=123, scheduler="dpmpp")
    grid = image_grid(samples, cols=8)
    grid.resize((grid.width * 3, grid.height * 3), Image.NEAREST).save(out / "grid_cpu_preset.png", optimize=True)  # 16x16 tiles, shown 3x
    rec = dict(
        what="The tiny CPU preset, trained to completion, with 64 DPM-Solver++ 20-step samples (seed 123, not cherry-picked).",
        command=f"diffuselab train --schedule {info['kind']} --device cpu",
        preset=info["preset"], device=info["device"], torch=info["torch"], threads=a.threads, load_note=a.load_note,
        params=info["params"], steps=info["steps"], epochs=info["epochs_run"], batch=info["batch"],
        seconds=info["seconds"], final_epoch_loss=info["final_epoch_loss"], grid="grid_cpu_preset.png",
    )  # fmt: skip
    (out / "cpu_timing.json").write_text(json.dumps(rec, indent=1))
    print(rec)


if __name__ == "__main__":
    main()
