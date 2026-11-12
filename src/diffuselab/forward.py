"""The forward (noising) process, recorded in the same format as a reverse run."""

from __future__ import annotations

import numpy as np

from .recorder import Trajectory
from .schedules import make_schedule, q_sample


def noise_trajectory(x0: np.ndarray, kind: str = "linear", T: int = 1000, n_frames: int = 51, seed: int = 0):
    """Noise x0 (C,H,W in [-1,1]) with one fixed eps, at n_frames timesteps from 0 to T-1."""
    sch = make_schedule(kind, T)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(x0.shape)
    ts = np.unique(np.linspace(0, T - 1, n_frames).round().astype(int))
    xt = np.stack([q_sample(x0, int(t), eps, sch) for t in ts]).astype(np.float32)
    ab = sch.alpha_bar[ts]
    n = len(ts)
    return Trajectory(
        "forward", "q", kind, T, seed, ts.astype(np.int64), ab, sch.betas[ts], ab / (1 - ab), xt,
        np.repeat(eps[None].astype(np.float32), n, 0), np.repeat(x0[None].astype(np.float32), n, 0),
        {"note": "eps and x0 are the true values, not predictions"},
    )  # fmt: skip
