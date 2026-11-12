"""FastAPI app: serves the web UI from docs/ and runs fresh instrumented samples, streamed frame by frame."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .export import EPS_RANGE, quantise
from .models import CHECKPOINT_DIR, get_device, load_checkpoint, load_pretrained
from .sampler import sample
from .schedules import KINDS, make_schedule

ModelKey = Literal["cifar10", "fashion-linear", "fashion-cosine"]
Loader = Callable[[str], tuple[torch.nn.Module, int, str]]  # key -> (unet, channels, schedule kind)


def _checkpoint(kind: str) -> Path | None:
    for preset in ("full", "tiny"):
        p = CHECKPOINT_DIR / f"fmnist_{kind}_{preset}.pt"
        if p.exists():
            return p
    return None


def default_loader(key: str):
    dev = get_device()
    if key == "cifar10":
        return load_pretrained("cifar10", dev), 3, "linear"
    kind = key.split("-", 1)[1]
    ck = _checkpoint(kind)
    if ck is None:
        raise FileNotFoundError(f"no trained checkpoint for {key}; run `diffuselab train --schedule {kind}`")
    net, _ = load_checkpoint(ck, dev)
    return net, 1, kind


class RunRequest(BaseModel):
    model: ModelKey = "cifar10"
    scheduler: Literal["ddpm", "ddim", "dpmpp"] = "ddim"
    steps: int = Field(50, ge=1, le=1000)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    record_every: int = Field(1, ge=1, le=1000)


class _Run:
    def __init__(self) -> None:
        self.meta: dict | None = None
        self.frames: list[dict] = []
        self.done = False
        self.error: str | None = None


def pack_frame(k: int, f: dict) -> dict:
    """One frame as JSON: scalars plus base64 of three uint8 panes (x_t, eps, x0), each C*H*W, CHW order."""
    raw = np.concatenate([quantise(f["x_t"]).ravel(), quantise(f["eps"], EPS_RANGE).ravel(), quantise(f["x0"]).ravel()])
    snr = f["snr"] if np.isfinite(f["snr"]) else None
    return dict(k=k, t=int(f["t"]), abar=f["abar"], beta=f["beta"], snr=snr, data=base64.b64encode(raw.tobytes()).decode())


def create_app(loader: Loader = default_loader, docs_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="DiffuseLab", version=__version__)
    runs: dict[str, _Run] = {}
    models: dict[str, tuple] = {}
    gpu_lock = threading.Lock()

    def get_model(key: str):
        if key not in models:
            models[key] = loader(key)
        return models[key]

    @app.get("/api/health")
    def health() -> dict:
        return dict(status="ok", version=__version__, device=str(get_device()), cuda=torch.cuda.is_available(),
                    torch=torch.__version__)  # fmt: skip

    @app.get("/api/models")
    def list_models() -> list[dict]:
        out = [dict(key="cifar10", label="CIFAR-10 pretrained (google/ddpm-cifar10-32)", channels=3, kind="linear")]
        for kind in KINDS:
            if _checkpoint(kind):
                out.append(dict(key=f"fashion-{kind}", label=f"Fashion-MNIST, trained here ({kind})",
                                channels=1, kind=kind))  # fmt: skip
        return out

    @app.get("/api/schedules")
    def schedules(kind: Literal["linear", "cosine"] = "linear", T: int = Query(1000, ge=10, le=4000)) -> dict:
        s = make_schedule(kind, T)
        return dict(kind=kind, T=T, beta=s.betas.tolist(), alpha_bar=s.alpha_bar.tolist(), log_snr=s.log_snr.tolist())

    @app.post("/api/runs", status_code=202)
    def start_run(req: RunRequest) -> dict:
        try:
            unet, channels, kind = get_model(req.model)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        if len(runs) > 32:
            runs.pop(next(iter(runs)))
        rid = uuid.uuid4().hex[:12]
        run = _Run()
        runs[rid] = run
        run.meta = dict(model=req.model, scheduler=req.scheduler, kind=kind, steps=req.steps, seed=req.seed,
                        channels=channels, height=unet.config.sample_size, width=unet.config.sample_size,
                        eps_range=EPS_RANGE, T=1000)  # fmt: skip

        def work() -> None:
            try:
                with gpu_lock:
                    sample(
                        unet, req.scheduler, req.steps, req.seed, kind, channels=channels, model_name=req.model,
                        record_every=req.record_every, on_frame=lambda k, f: run.frames.append(pack_frame(k, f)),
                    )  # fmt: skip
            except Exception as e:  # noqa: BLE001 - surfaced to the client as an error event
                run.error = f"{type(e).__name__}: {e}"
            finally:
                run.done = True

        threading.Thread(target=work, daemon=True).start()
        return dict(run_id=rid)

    @app.get("/api/runs/{rid}/stream")
    async def stream(rid: str):
        run = runs.get(rid)
        if run is None:
            raise HTTPException(404, "unknown run")

        async def gen():
            yield f"event: meta\ndata: {json.dumps(run.meta)}\n\n"
            sent = 0
            while True:
                while sent < len(run.frames):
                    yield f"event: frame\ndata: {json.dumps(run.frames[sent])}\n\n"
                    sent += 1
                if run.done and sent >= len(run.frames):
                    break
                await asyncio.sleep(0.02)
            if run.error:
                yield f"event: error\ndata: {json.dumps({'message': run.error})}\n\n"
            else:
                yield "event: done\ndata: {}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    docs = Path(docs_dir or os.environ.get("DIFFUSELAB_DOCS") or Path(__file__).resolve().parents[2] / "docs")
    if docs.exists():
        app.mount("/", StaticFiles(directory=docs, html=True), name="web")
    return app


app = create_app()
