"""Command line: diffuselab {verify-gpu,download,train,sample,record,serve}."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _verify_cuda(_args) -> int:
    import torch

    from .models import get_device

    info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "device": str(get_device())}
    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, device="cuda")
        info.update(gpu=torch.cuda.get_device_name(0), arch_list=torch.cuda.get_arch_list(),
                    kernel_ok=bool(torch.isfinite((x @ x).sum()).item()))  # fmt: skip
    print(json.dumps(info, indent=1))
    return 0


def _download(_args) -> int:
    from .data import fetch
    from .models import PRETRAINED, load_pretrained

    fetch()
    for name in PRETRAINED:
        load_pretrained(name, device="cpu")
    print("models and data ready")
    return 0


def _train(a) -> int:
    from .train import train

    train(kind=a.schedule, preset=a.preset, epochs=a.epochs, device=a.device, max_steps=a.max_steps)
    return 0


def _sample(a) -> int:
    from .export import save_png_grid
    from .models import get_device, load_checkpoint, load_pretrained
    from .sampler import sample

    dev = get_device(a.device)
    if a.model == "cifar10":
        net, ch = load_pretrained("cifar10", dev), 3
    else:
        net, _ = load_checkpoint(a.model, dev)
        ch = 1
    tr = sample(net, a.scheduler, a.steps, a.seed, a.schedule, channels=ch, record_every=max(1, a.steps // 10))
    tr.save(a.out)
    save_png_grid(tr, str(a.out).replace(".npz", ".png"))
    print(f"saved {a.out} ({tr.n_frames} frames, {tr.meta['seconds']}s on {tr.meta['device']})")
    return 0


def _record(a) -> int:
    from .export import record_demo

    record_demo(a.out, which=a.which)
    return 0


def _serve(a) -> int:
    import uvicorn

    uvicorn.run("diffuselab.server:app", host=a.host, port=a.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="diffuselab", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify-gpu", help="report the torch build and run a real GPU kernel").set_defaults(fn=_verify_cuda)
    sub.add_parser("download", help="fetch the pretrained model and Fashion-MNIST").set_defaults(fn=_download)

    t = sub.add_parser("train", help="train a small DDPM from scratch on Fashion-MNIST")
    t.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    t.add_argument("--preset", choices=["full", "tiny"], default="tiny")
    t.add_argument("--epochs", type=int)
    t.add_argument("--device")
    t.add_argument("--max-steps", type=int)
    t.set_defaults(fn=_train)

    s = sub.add_parser("sample", help="record one reverse run")
    s.add_argument("--model", default="cifar10", help="'cifar10' or a path to a checkpoint")
    s.add_argument("--scheduler", choices=["ddpm", "ddim", "dpmpp", "euler"], default="dpmpp")
    s.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    s.add_argument("--steps", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", help="cpu or cuda (default: cuda if available)")
    s.add_argument("--out", default="runs/sample.npz")
    s.set_defaults(fn=_sample)

    r = sub.add_parser("record", help="regenerate the static demo data under docs/data")
    r.add_argument("--out", default="docs/data")
    r.add_argument("--which", choices=["all", "cifar10", "fashion"], default="all")
    r.set_defaults(fn=_record)

    v = sub.add_parser("serve", help="run the web app with live sampling")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=int(os.environ.get("DIFFUSELAB_PORT", "8105")))
    v.set_defaults(fn=_serve)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
