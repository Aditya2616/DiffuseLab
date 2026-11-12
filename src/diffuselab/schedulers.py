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
    common = dict(num_train_timesteps=T, beta_schedule=DIFFUSERS_NAME[kind], prediction_type="epsilon")
    if name == "ddpm":
        return DDPMScheduler(clip_sample=False, timestep_spacing="trailing", **common)
    if name == "ddim":
        return DDIMScheduler(clip_sample=False, set_alpha_to_one=True, timestep_spacing="trailing", **common)
    if name == "dpmpp":
        return DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++", solver_order=2, timestep_spacing="trailing", **common
        )
    if name == "euler":
        return EulerDiscreteScheduler(use_karras_sigmas=True, **common)
    raise ValueError(f"unknown scheduler {name!r}; expected one of {sorted(SCHEDULERS)}")
