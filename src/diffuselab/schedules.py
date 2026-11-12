"""Noise schedules and the quantities derived from them, in float64 numpy.

These are written out from the formulas in the DDPM and Improved DDPM papers rather than
read from Diffusers, so the browser implementation (docs/js/math.js) can be checked against
the same numbers. The tests check them against Diffusers' schedulers as well.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

KINDS = ("linear", "cosine")
# Name Diffusers uses for each kind.
DIFFUSERS_NAME = {"linear": "linear", "cosine": "squaredcos_cap_v2"}


def make_betas(kind: str, T: int = 1000) -> np.ndarray:
    """Return betas[0..T-1].

    linear: beta from 1e-4 to 0.02 over T steps (the Diffusers convention; the DDPM paper uses T = 1000).
    cosine: Nichol and Dhariwal, s = 0.008, each beta clipped at 0.999.
    """
    if kind == "linear":
        return np.linspace(1e-4, 0.02, T, dtype=np.float64)
    if kind == "cosine":
        s = 0.008

        def f(u: float) -> float:
            return math.cos((u + s) / (1 + s) * math.pi / 2) ** 2

        return np.array([min(1.0 - f((i + 1) / T) / f(i / T), 0.999) for i in range(T)], dtype=np.float64)
    raise ValueError(f"unknown schedule kind {kind!r}; expected one of {KINDS}")


@dataclass(frozen=True)
class Schedule:
    kind: str
    T: int
    betas: np.ndarray
    alphas: np.ndarray
    alpha_bar: np.ndarray
    snr: np.ndarray
    log_snr: np.ndarray

    def abar_at(self, t: int) -> float:
        """alpha_bar at timestep t; t < 0 means 'after the last step' and gives 1."""
        return 1.0 if t < 0 else float(self.alpha_bar[t])


def make_schedule(kind: str, T: int = 1000) -> Schedule:
    betas = make_betas(kind, T)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    snr = abar / (1.0 - abar)
    return Schedule(kind, T, betas, alphas, abar, snr, np.log(snr))


def q_sample(x0: np.ndarray, t: int, eps: np.ndarray, sch: Schedule) -> np.ndarray:
    """Closed-form forward process: sqrt(abar_t) x0 + sqrt(1 - abar_t) eps."""
    ab = sch.alpha_bar[t]
    return math.sqrt(ab) * x0 + math.sqrt(1.0 - ab) * eps


def predict_x0(x_t, eps, abar: float):
    """x0_hat = (x_t - sqrt(1-abar) eps) / sqrt(abar). Works on numpy arrays and torch tensors."""
    return (x_t - math.sqrt(1.0 - abar) * eps) / math.sqrt(abar)
