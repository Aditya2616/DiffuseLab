# DiffuseLab design notes

## What it is

An instrumented diffusion sampler plus an explainer UI. It runs real diffusion models and records, at every step, the noisy sample x_t, the predicted noise eps, the predicted clean image x0, and the schedule quantities (beta, alpha-bar, SNR). The UI lets you scrub t from 1000 to 0 and watch the three images side by side with the schedule curve and a playhead underneath.

## Models

- **Pretrained:** `google/ddpm-cifar10-32` (Apache-2.0), downloaded by script.
- **From scratch:** a `UNet2DModel` trained on Fashion-MNIST (MIT) padded to 32x32, once with a linear and once with a cosine schedule, same budget. Training script, loss curves, sample grids, a nearest-neighbour memorisation check and FID/KID (`scripts/eval_fid.py`) are in the repo.
- **In the browser:** a 2-D toy diffusion with the exact analytic denoiser.

## Package layout

```
src/diffuselab/
  schedules.py   betas, alphas, abar, SNR for linear/cosine (own formulas, tested against diffusers)
  recorder.py    Trajectory: per-step x_t, eps, x0, t and schedule row; npz save/load
  sampler.py     the instrumented loop: UNet + any scheduler, same seed gives the same x_T
  schedulers.py  registry: ddpm, ddim, dpmpp (DPM-Solver++ 2M), euler (Karras sigmas)
  forward.py     forward noising q(x_t | x_0)
  models.py      pretrained / checkpoint loading, device selection
  train.py       from-scratch training (presets: full on GPU, tiny on CPU)
  metrics.py     FID/KID helpers (torch-fidelity imported lazily)
  data.py        Fashion-MNIST fetch and parse
  toy2d.py       reference 2-D toy: analytic denoiser, sampler, test vectors
  export.py      trajectories to sprite-sheet WebP + JSON for docs/
  server.py      FastAPI app with SSE streaming
  cli.py         diffuselab {verify-gpu, download, train, sample, record, serve}
docs/            the static site (also what the server serves)
```

## Data flow

1. `sampler.sample(model, scheduler, steps, seed, kind)` builds x_T from the seed, loops over timesteps, calls the UNet once per step, calls `scheduler.step`, and appends to a `Trajectory`.
2. `export` writes each run as one lossy 8-bit WebP sprite sheet (rows x_t, eps, x0; one column per recorded step) plus a JSON file with timesteps and schedule arrays.
3. `POST /api/runs` starts a run, `GET /api/runs/{id}/stream` streams frames over SSE, `GET /api/health`, `/api/models`, `/api/schedules`.
4. The front end loads either the live stream or a recorded run; one renderer draws both.

## UI

Dark, mono labels, lime accent, halftone rendering of the panes. At 390 px the panes stack and the time slider is a large touch target. Sections: scrubber, schedules, sampler comparison, forward process, 2-D lab, math panel, training page. A badge says "recorded run" in the static demo and "live" on the local server. Animation shows real state only, pauses off screen, and respects `prefers-reduced-motion`.

## The 2-D toy is exact

For an N-point empirical distribution the ideal denoiser is `x0_hat(x_t, t) = sum_i w_i x_i` with `w_i = softmax_i(-||x_t - sqrt(abar) x_i||^2 / (2 (1 - abar)))`, and `eps_hat = (x_t - sqrt(abar) x0_hat) / sqrt(1 - abar)`. Python and JS are tested against shared JSON vectors, and the sampler steps use the same DDPM/DDIM formulas as the image sampler.

## Tests

Fast and offline: schedules against diffusers, forward identity, sampler with an oracle noise predictor (recovers the target, shapes, determinism, solver state does not leak), recorder round trip, export manifest, toy denoiser and vectors, server lifecycle with a tiny model, IDX parser, training smoke run. A Node test checks the JS maths against the same vectors. Slow tests (marker `slow`) need more time. UI is checked with `scripts/shots.py` (headless Chromium, 1440x900 and 390x844).
