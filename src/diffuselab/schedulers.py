"""The sampler registry: Diffusers schedulers configured for a given noise schedule."""

from __future__ import annotations

from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler

from .schedules import DIFFUSERS_NAME

SCHEDULERS = {
    "ddpm": "DDPM ancestral sampler (stochastic, Ho et al. 2020)",
    "ddim": "DDIM, eta = 0 (deterministic, Song et al. 2020)",
    "dpmpp": "DPM-Solver++ (2M), 2nd-order multistep (Lu et al. 2022)",
    "euler": "Euler on Karras sigmas (Karras et al. 2022), first-order",
}


def make_scheduler(name: str, kind: str = "linear", T: int = 1000):
    """Build a Diffusers scheduler. x0 clipping is disabled everywhere so that the recorded
    x0 prediction is the raw formula and not a clipped variant."""
    # The cosine schedule ends at alpha-bar ~ 4e-9, so x0_hat = (x_t - sqrt(1 - abar) eps) / sqrt(abar) multiplies
    # any error in eps by ~15,000 at t = 999. "trailing" spacing starts the short samplers exactly there and the
    # samples come out as noise; "leading" starts one stride earlier and is clean. Linear is fine either way.
    spacing = "leading" if kind == "cosine" else "trailing"
    common = dict(num_train_timesteps=T, beta_schedule=DIFFUSERS_NAME[kind], prediction_type="epsilon")
    if name == "ddpm":
        return DDPMScheduler(clip_sample=False, timestep_spacing="trailing", **common)
    if name == "ddim":
        return DDIMScheduler(clip_sample=False, set_alpha_to_one=True, timestep_spacing=spacing, **common)
    if name == "dpmpp":
        return DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++", solver_order=2, timestep_spacing=spacing, **common
        )
    if name == "euler":
        return EulerDiscreteScheduler(**common)
    raise ValueError(f"unknown scheduler {name!r}; expected one of {sorted(SCHEDULERS)}")


SIGMA_MAX = 80.0  # the EDM paper's upper end; matters only for the cosine schedule, whose own maximum is ~15,000


def karras_sigmas(alphas_cumprod, steps: int, rho: float = 7.0, sigma_max: float = SIGMA_MAX) -> list[float]:
    """Karras et al. 2022 sigma spacing between the schedule's smallest sigma and min(its largest, sigma_max)."""
    import numpy as np

    sig = ((1.0 - np.asarray(alphas_cumprod, dtype=np.float64)) / np.asarray(alphas_cumprod, dtype=np.float64)) ** 0.5
    hi, lo = min(float(sig.max()), sigma_max), float(sig.min())
    ramp = np.linspace(0.0, 1.0, steps)
    return ((hi ** (1 / rho)) + ramp * ((lo ** (1 / rho)) - (hi ** (1 / rho)))) ** rho


def set_steps(sched, name: str, steps: int) -> None:
    """set_timesteps for any registry scheduler (Euler gets Karras sigmas, capped at SIGMA_MAX)."""
    if name == "euler":
        sched.set_timesteps(sigmas=[*karras_sigmas(sched.alphas_cumprod.numpy(), steps).tolist(), 0.0])
    else:
        sched.set_timesteps(steps)
