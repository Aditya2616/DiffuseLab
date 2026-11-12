"""A Trajectory is everything an instrumented sampling run saw, step by step."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Trajectory:
    """Frames 0..n. Frame k < n is the state the model saw at step k (x_t, its eps, its x0_hat).
    Frame n is the finished sample (t = -1 by convention, abar = 1, x0 = x_t, eps = last eps)."""

    model: str
    scheduler: str
    kind: str
    T: int
    seed: int
    t: np.ndarray  # (n+1,) int, model timestep at each frame; last is -1 (done)
    abar: np.ndarray  # (n+1,)
    beta: np.ndarray  # (n+1,)
    snr: np.ndarray  # (n+1,) abar / (1 - abar); inf at the final frame
    x_t: np.ndarray  # (n+1, C, H, W) float32
    eps: np.ndarray  # (n+1, C, H, W) float32
    x0: np.ndarray  # (n+1, C, H, W) float32 (unclipped)
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.t)

    @property
    def final(self) -> np.ndarray:
        return self.x_t[-1]

    def select(self, idx) -> Trajectory:
        idx = np.asarray(idx)
        return Trajectory(
            self.model, self.scheduler, self.kind, self.T, self.seed,
            self.t[idx], self.abar[idx], self.beta[idx], self.snr[idx],
            self.x_t[idx], self.eps[idx], self.x0[idx], dict(self.meta),
        )  # fmt: skip

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        head = dict(model=self.model, scheduler=self.scheduler, kind=self.kind, T=self.T,
                    seed=self.seed, meta=self.meta)  # fmt: skip
        np.savez_compressed(
            path, head=np.array(json.dumps(head)), t=self.t, abar=self.abar, beta=self.beta,
            snr=self.snr, x_t=self.x_t, eps=self.eps, x0=self.x0,
        )  # fmt: skip

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        with np.load(path, allow_pickle=False) as z:
            head = json.loads(str(z["head"]))
            return cls(
                head["model"], head["scheduler"], head["kind"], head["T"], head["seed"],
                z["t"], z["abar"], z["beta"], z["snr"], z["x_t"], z["eps"], z["x0"], head["meta"],
            )  # fmt: skip
