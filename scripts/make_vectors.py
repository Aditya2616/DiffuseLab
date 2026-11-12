"""Write tests/vectors/toy2d.json: reference numbers from the Python code that the browser maths must match."""

import json
from pathlib import Path

import numpy as np

from diffuselab.schedules import make_schedule
from diffuselab.toy2d import ddim_step, ddpm_step, ideal_denoiser, trailing_timesteps

ROOT = Path(__file__).resolve().parents[1]
TS = [0, 1, 2, 10, 49, 100, 250, 499, 500, 750, 900, 998, 999]


def main() -> None:
    out: dict = {"schedule": {}}
    for kind in ("linear", "cosine"):
        s = make_schedule(kind)
        out["schedule"][kind] = dict(
            t=TS, beta=[float(s.betas[t]) for t in TS], abar=[float(s.alpha_bar[t]) for t in TS],
            log_snr=[float(s.log_snr[t]) for t in TS],
        )  # fmt: skip
    rng = np.random.default_rng(42)
    data = rng.normal(size=(12, 2))
    x_t = rng.normal(size=(6, 2))
    sch = make_schedule("cosine")
    t = 612
    x0, eps = ideal_denoiser(x_t, t, data, sch)
    out["denoiser"] = dict(data=data.tolist(), x_t=x_t.tolist(), t=t, kind="cosine", x0=x0.tolist(), eps=eps.tolist())
    z = rng.normal(size=(6, 2))
    out["steps"] = dict(
        kind="cosine", t=t, eps=eps.tolist(), z=z.tolist(),
        ddpm=ddpm_step(x_t, t, eps, sch, z).tolist(),
        ddim_to_550=ddim_step(x_t, t, 550, eps, sch).tolist(),
        ddim_to_clean=ddim_step(x_t, t, -1, eps, sch).tolist(),
    )  # fmt: skip
    out["trailing"] = {str(n): trailing_timesteps(1000, n) for n in (5, 10, 20, 50)}
    dest = ROOT / "tests" / "vectors" / "toy2d.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out))
    print("wrote", dest)


if __name__ == "__main__":
    main()
