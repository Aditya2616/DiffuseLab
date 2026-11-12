# DiffuseLab SPEC

Repo `DiffuseLab`, local port 8105, MIT, copyright Aditya Jain. Stack: Python, PyTorch, HF Diffusers.

## 1. What it is

An instrumented diffusion sampler plus an explainer UI. It runs real diffusion models and records, at every step, the noisy sample x_t, the predicted noise eps, the predicted clean image x0, and the schedule quantities (beta, alpha, alpha-bar, SNR, log-SNR). The UI lets you scrub t from 1000 to 0 and watch all three images side by side with the schedule curves and a playhead underneath. It is the real version of the portfolio scene (a halftone image emerging from noise while t runs 1000 to 0, schedule curve below).

## 2. Models and tracks

- **Track A (pretrained):** `google/ddpm-cifar10-32` (Apache-2.0, 32x32, T=1000, linear). Downloaded by script.
- **Track B (from scratch):** `UNet2DModel` ~10M params trained here on Fashion-MNIST (MIT) padded to 32x32, once with a linear schedule and once with a cosine schedule, same budget. Training script, loss curves, EMA, sample grids, nearest-neighbour memorisation check, all in the repo and reproducible with one command.
- **Track C (optional, decide after A and B):** a latent text-to-image model via the Diffusers callback API, only if it fits the download budget and adds to the explanation (decoded intermediate x0 previews). Not committed to.
- **Toy track (browser):** 2-D diffusion in JS, exact (section 6).

## 3. Package layout

```
DiffuseLab/
  pyproject.toml  requirements.txt  .env.example  LICENSE  README.md
  RESEARCH.md  SPEC.md
  src/diffuselab/
    schedules.py     # betas, alphas, abar, SNR for linear/cosine (own formulas, tested vs diffusers)
    recorder.py      # Trajectory dataclass: per-step x_t, eps, x0, t, schedule row; save/load npz
    sampler.py       # instrumented loop: UNet + any scheduler, same seed => same x_T
    forward.py       # forward noising q(x_t|x_0) trajectory
    schedulers.py    # registry: ddpm, ddim, dpmpp2m, euler(karras)
    models.py        # load pretrained / load trained checkpoint, device selection (CUDA/CPU)
    train.py         # from-scratch training (CLI: diffuselab train)
    data.py          # Fashion-MNIST fetch + parse (numpy, checksums)
    toy2d.py         # reference 2-D toy: analytic denoiser, sampler, test vectors
    export.py        # trajectories -> sprite-sheet WebP + JSON for docs/
    server.py        # FastAPI app + SSE streaming
    cli.py           # diffuselab {download,train,sample,record,export,serve,verify-gpu}
  web/ (source)  ->  docs/ (built static demo, committed)
  tests/ scripts/ (record_demo.py, shots.py Playwright) .github/workflows/ci.yml
```

## 4. Data flow

1. `sampler.sample(model, scheduler_name, steps, seed, schedule)` builds x_T from the seed, loops over timesteps, calls the UNet once per step, calls `scheduler.step`, and appends to a `Trajectory`: `t`, `x_t`, `eps`, `x0_hat`, plus `abar_t`, `beta_t`, `snr_t`.
2. `forward.noise_trajectory(x0, ts, seed)` gives the forward process with the same quantities (eps is the true noise).
3. `export` writes, per run, a sprite sheet (rows: x_t, eps, x0; columns: recorded steps) as lossless WebP plus a JSON manifest (timesteps, schedule arrays, scheduler, seed, model id, git commit, wall time).
4. Server `POST /api/runs` starts a run on the GPU (CPU fallback), `GET /api/runs/{id}/stream` streams frames (SSE), `GET /api/health`, `GET /api/models`, `GET /api/schedules?kind=cosine&T=1000`.
5. Front end loads either the live stream or the recorded demo manifest; one renderer for both.

## 5. UI concept

Dark, mono labels, lime accent, halftone rendering of the three panes (a canvas shader reading real pixel values, same dot grid as the site scene). Phone first: panes stack vertically at 390 px, scrubber is a large touch slider.

1. **Hero/scrubber:** three panes (x_t, predicted noise, predicted x0), one big timeline slider t = 1000 to 0 with play, speed, step-by-step. Playhead is drawn on the schedule curves below. Frame cache in memory; scrubbing is pure canvas blits.
2. **Schedule plot:** beta, alpha-bar, SNR (log) curves for linear vs cosine, overlay of the scheduler's chosen timesteps as ticks, playhead synced with the scrubber.
3. **Scheduler comparison:** same seed, DDPM 1000 / DDIM 50 / DPM-Solver++ 20 side by side, x0 prediction per step, final images, plus step count and wall time (measured).
4. **Forward process:** pick a training image, scrub it into noise, show the closed-form `sqrt(abar) x0 + sqrt(1-abar) eps`.
5. **Toy lab (2-D, in browser):** choose distribution, scrub forward noising of a point cloud, run the reverse process with the analytic denoiser (exact) or a small MLP trained live, optional vector field; schedule linear/cosine and step count adjustable.
6. **Math panel:** formulas rendered, each linked to the quantity visible on screen (hover or tap highlights it).
7. **Training page:** loss curves, EMA samples by epoch (recorded), nearest-neighbour check.
8. Mode banner: "Recorded run" in static demo, "Live" on the local server. Footer: Built by Aditya Jain with links.

Motion: animation shows real state only; canvas at capped DPR; pauses off screen; `prefers-reduced-motion` disables autoplay and easing.

## 6. The 2-D toy has to be exact

- Forward: closed form. Ideal denoiser for an N-point empirical distribution: `x0_hat(x_t, t) = sum_i w_i x_i`, `w_i = softmax_i(-||x_t - sqrt(abar) x_i||^2 / (2 (1-abar)))`; `eps_hat = (x_t - sqrt(abar) x0_hat)/sqrt(1-abar)`.
- Python `toy2d.py` and JS implementations are tested against shared JSON vectors (tolerance 1e-5 in float32, 1e-9 in float64 reference).
- Sampler steps use the same DDPM/DDIM update formulas as the image sampler, so the toy teaches the exact algorithm.

## 7. Test plan (fast, offline by default)

- `schedules`: linear and cosine match `diffusers` DDPMScheduler `betas`/`alphas_cumprod` to 1e-7; alpha-bar monotone decreasing, in (0,1); cosine clipping at 0.999.
- Forward identity: `x_t` from the recorder equals the closed form; x0 recovered from true eps equals the original exactly (float tolerance).
- Sampler with a tiny random UNet and a mocked "oracle" eps-model: oracle recovers x0 in one step; trajectory shapes, step counts, determinism for a given seed, different seeds differ.
- DDIM eta=0 determinism; DDPM stochasticity governed by seed; multistep solver state reset between runs.
- Recorder save/load round trip; export manifest schema; WebP sprite decode equals source within quantisation.
- `toy2d` analytic denoiser: single-point dataset returns that point; test vectors vs JS (run with Node in CI).
- Server: health, schedules, run lifecycle with the tiny model, SSE frame count; validation errors (422).
- Data: IDX parser on a synthetic file; checksum failure raises.
- Slow marker: pretrained checkpoint sampling sanity (non-NaN, range), short training smoke run that reduces loss, GPU availability check.
- Playwright scripts (not in CI): desktop 1440x900 and phone 390x844 screenshots, console-error check, static `docs/` served over plain HTTP.
- `ruff check` clean; GitHub Actions runs ruff + fast tests on CPU.

## 8. README results (all measured, each with a reproducing command)

Training wall time and final loss for both schedules, sampling throughput per scheduler on GPU and CPU, steps-vs-quality for DDIM/DPM-Solver++ (an objective metric only if I can compute it honestly; otherwise visual grids), size of the static demo, and the GPU/torch versions used. Limitations stated: small-scale models, 32x32, no state-of-the-art claims, optional track C status.

## 9. Definition of done

Fresh clone plus quickstart works with a new venv; fast tests and ruff pass; Track A and B record and replay; UI verified by screenshots at both sizes with no console errors; `docs/` demo works from a static server and stays under about 8 MB; no model weights committed (download scripts only); no AI mentions anywhere; clean git history under Aditya2616; final report with measured results and anything cut.

## 10. Build order

1. venv + CUDA torch + `verify-gpu`; 2. schedules, recorder, sampler, tests; 3. Track A recording; 4. training script and runs (GPU, background) while the UI is built; 5. export, UI scrubber and schedule plot; 6. scheduler comparison, forward process, toy lab; 7. server and live mode; 8. README, CI, screenshots, clean-clone test.
