"""Reference implementation of the 2-D toy diffusion that runs live in the browser (docs/js/toy.js).

For a finite dataset {x_i} the best possible noise predictor is known in closed form. Under the forward
process x_t = sqrt(abar) x0 + sqrt(1 - abar) eps, the posterior over which data point produced x_t is a
softmax of Gaussian likelihoods, so

    x0_hat(x_t, t) = sum_i w_i x_i,    w_i = softmax_i( -|x_t - sqrt(abar) x_i|^2 / (2 (1 - abar)) )
    eps_hat       = (x_t - sqrt(abar) x0_hat) / sqrt(1 - abar)

No network is trained, nothing is approximated, and the DDPM and DDIM updates below are the same
formulas the image sampler uses. The browser version must match this file to float tolerance
(tests/test_toy2d.py writes the vectors, web tests read them).
"""

from __future__ import annotations

import math

import numpy as np

from .schedules import Schedule, make_schedule


def moons(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = n // 2
    a = rng.uniform(0, math.pi, k)
    b = rng.uniform(0, math.pi, n - k)
    top = np.stack([np.cos(a), np.sin(a)], 1)
    bot = np.stack([1 - np.cos(b), 0.5 - np.sin(b)], 1)
    pts = np.concatenate([top, bot]) + rng.normal(0, 0.04, (n, 2))
    return (pts - pts.mean(0)) / pts.std()


def ring8(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 8, n)
    ang = c * math.pi / 4
    pts = np.stack([np.cos(ang), np.sin(ang)], 1) * 1.6 + rng.normal(0, 0.12, (n, 2))
    return pts / pts.std()


def spiral(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.15, 1.0, n)
    ang = r * 3.5 * math.pi
    pts = np.stack([r * np.cos(ang), r * np.sin(ang)], 1) + rng.normal(0, 0.02, (n, 2))
    return pts / pts.std()


DATASETS = {"moons": moons, "ring8": ring8, "spiral": spiral}


def ideal_denoiser(x_t: np.ndarray, t: int, data: np.ndarray, sch: Schedule):
    """Exact posterior mean of x0 given x_t, and the matching noise prediction. x_t: (M,2), data: (N,2)."""
    ab = sch.alpha_bar[t]
    d2 = ((x_t[:, None, :] - math.sqrt(ab) * data[None, :, :]) ** 2).sum(-1)  # (M,N)
    logits = -d2 / (2.0 * (1.0 - ab))
    logits -= logits.max(1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(1, keepdims=True)
    x0 = w @ data
    eps = (x_t - math.sqrt(ab) * x0) / math.sqrt(1.0 - ab)
    return x0, eps


def ddpm_step(x_t, t, eps, sch: Schedule, z):
    """One ancestral step t -> t-1. z is standard normal noise (ignored at t = 0)."""
    beta, alpha, ab = sch.betas[t], sch.alphas[t], sch.alpha_bar[t]
    mean = (x_t - beta / math.sqrt(1.0 - ab) * eps) / math.sqrt(alpha)
    if t == 0:
        return mean
    ab_prev = sch.alpha_bar[t - 1]
    var = beta * (1.0 - ab_prev) / (1.0 - ab)
    return mean + math.sqrt(var) * z


def ddim_step(x_t, t, t_prev, eps, sch: Schedule, eta: float = 0.0, z=None):
    """DDIM step from t to t_prev (t_prev < 0 means the clean sample).

    eta = 0 is deterministic DDIM; eta = 1 gives the DDPM-like stochastic step on the same timestep subsequence
    (Song et al. 2020, eq. 16). z is standard normal noise, only used when eta > 0.
    """
    ab = sch.alpha_bar[t]
    ab_prev = sch.abar_at(t_prev)
    x0 = (x_t - math.sqrt(1.0 - ab) * eps) / math.sqrt(ab)
    sigma = eta * math.sqrt((1.0 - ab_prev) / (1.0 - ab)) * math.sqrt(1.0 - ab / ab_prev)
    out = math.sqrt(ab_prev) * x0 + math.sqrt(max(1.0 - ab_prev - sigma**2, 0.0)) * eps
    return out + sigma * z if (eta > 0 and z is not None) else out


def trailing_timesteps(T: int, steps: int) -> list[int]:
    """The 'trailing' spacing used by the Diffusers schedulers in this project: T-1 down in equal strides."""
    stride = T / steps
    return [int(round(T - i * stride)) - 1 for i in range(steps)]


def sample_toy(data, kind="linear", steps=50, sampler="ddim", n=500, seed=0, T=1000, eta=0.0):
    """Run the reverse process on n points. Returns the list of states, one per step plus the end."""
    sch = make_schedule(kind, T)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    ts = list(range(T - 1, -1, -1)) if sampler == "ddpm" and steps >= T else trailing_timesteps(T, steps)
    states = [x.copy()]
    for k, t in enumerate(ts):
        _, eps = ideal_denoiser(x, t, data, sch)
        if sampler == "ddpm":
            x = ddpm_step(x, t, eps, sch, rng.standard_normal(x.shape))
        else:
            x = ddim_step(x, t, ts[k + 1] if k + 1 < len(ts) else -1, eps, sch, eta, rng.standard_normal(x.shape))
        states.append(x.copy())
    return states
