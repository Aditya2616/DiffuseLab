# DiffuseLab research notes

Phase 1 findings. Items marked **verified here** were checked on this machine on 2026-11-12; items marked **to verify** are from papers or memory and must be confirmed in the build phase before they appear in the README.

## 1. Environment checks

| Check | Result |
|---|---|
| GPU | RTX 5060 Ti 16 GB (Blackwell, compute capability 12.0, sm_120), driver 621.14, CUDA UMD 13.5. **Verified here** |
| Global torch | 2.11.0+cpu (cannot use the GPU). **Verified here** |
| Python | 3.13.15. **Verified here** |
| `diffusers` | not installed globally; latest on PyPI is 0.43.0. **Verified here** |
| `torchvision`, `accelerate` | not installed globally. **Verified here** |
| CUDA wheels | The PyTorch index `download.pytorch.org/whl/cu128` lists torch 2.11.0+cu128 (and 2.7.0 to 2.10.0); `cu130` lists up to 2.14.1+cu130. **Verified here** (index listing only; whether the Windows cp313 wheel runs on sm_120 is confirmed in the build phase with `torch.cuda.get_arch_list()` and a real kernel). |
| Free disk | about 52 GB on C: (shared with eight other builds) |

Blackwell support: community reports and PyTorch forum threads say wheels built with CUDA 12.8 or newer include sm_120 (since torch 2.7.0). Older "sm_120 not supported" threads predate that. I could not find an official release note through search, so the runtime check is the authority.

Decision: venv with `--system-site-packages`, then `pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128` into the venv only. That keeps the same torch minor version as the global CPU build, so nothing else shared breaks, and the venv copy shadows the global one. Fallback: cu130 build, then CPU with a smaller model. Expected download about 2.5 to 3 GB, within the 8 GB budget. Install `torchvision` from the same index only if needed (I plan to avoid it; see datasets).

## 2. The maths (what the app has to get exactly right)

- **DDPM** (Ho, Jain, Abbeel 2020, arXiv 2006.11239). Forward: `q(x_t | x_0) = N(sqrt(abar_t) x_0, (1-abar_t) I)`, with `alpha_t = 1 - beta_t`, `abar_t = prod alpha_s`. The network predicts the noise `eps_theta(x_t, t)`. Predicted clean image: `x0_hat = (x_t - sqrt(1-abar_t) eps) / sqrt(abar_t)`. Ancestral step: posterior mean `mu = (1/sqrt(alpha_t)) (x_t - beta_t/sqrt(1-abar_t) eps)`, variance `beta_tilde_t = beta_t (1-abar_{t-1})/(1-abar_t)`. Linear schedule beta from 1e-4 to 0.02 over T=1000.
- **Improved DDPM** (Nichol and Dhariwal 2021, arXiv 2102.09672). Cosine schedule `abar_t = f(t)/f(0)`, `f(t) = cos^2(((t/T + s)/(1+s)) pi/2)`, s = 0.008, beta clipped at 0.999. Gives a gentler SNR decay than linear, which wastes fewer steps in almost-pure noise. Also learned variances (not needed here).
- **DDIM** (Song, Meng, Ermon 2020, arXiv 2010.02502). Deterministic (eta = 0) non-Markovian sampler: `x_{t_prev} = sqrt(abar_prev) x0_hat + sqrt(1-abar_prev-sigma^2) eps + sigma z`. Works on a subsequence of timesteps, so 20 to 50 steps instead of 1000. Same trained network.
- **DPM-Solver / DPM-Solver++** (Lu et al. 2022, arXiv 2206.00927, 2211.01095). High-order ODE solvers in the log-SNR variable `lambda = log(alpha/sigma)`; 2nd-order multistep gives good samples in 10 to 20 steps. Diffusers: `DPMSolverMultistepScheduler`.
- **EDM** (Karras et al. 2022, arXiv 2206.00364). Reparameterises with sigma, preconditioning, Heun 2nd-order sampler, and the rho = 7 sigma schedule. Explainer value: shows that "noise schedule" and "sampler" are separate design choices. Diffusers has `EulerDiscreteScheduler` / `HeunDiscreteScheduler` with `use_karras_sigmas`. Treated as an educational comparison, not a separate trained model (a DDPM checkpoint can be sampled with Karras sigmas through the epsilon-to-sigma conversion; to verify numerically).
- **SNR** `= abar_t / (1 - abar_t)`; show log-SNR in the schedule plot since it is what makes linear vs cosine comparable.
- Exactness requirement for the UI: every plotted schedule quantity must come from the same formulas as the Python (shared test vectors, section 8 of SPEC).

## 3. Libraries, current APIs (to verify against 0.43.0 in the build)

- Diffusers schedulers share an API: `set_timesteps(n)`, `scheduler.timesteps`, `scheduler.step(model_output, t, sample)` returning an output with `.prev_sample` and (for DDPM, DDIM, DPMSolver variants) `.pred_original_sample`; `scheduler.alphas_cumprod`, `scheduler.betas`, `scheduler.config`. `DDPMScheduler(beta_schedule="linear" | "squaredcos_cap_v2")` is the Diffusers name for the cosine schedule.
- `DDPMPipeline` (unconditional, UNet2DModel) has no per-step callback in the versions I know. Newer pipelines (Stable Diffusion family) use `callback_on_step_end(pipe, step, timestep, callback_kwargs)` with `callback_on_step_end_tensor_inputs`. Both facts are from memory and need verifying.
- **Design consequence:** do not depend on pipeline callbacks. Write our own sampling loop around `UNet2DModel` + any scheduler, which is about 30 lines and is exactly the "instrumented sampler". It also lets one trained network be sampled with several schedulers. For the optional latent model, use the callback API instead (that is where it matters).
- `scheduler.step` for multistep solvers keeps internal state (previous model outputs); the instrumented loop must call `set_timesteps` fresh per run and must not share scheduler instances across runs.
- Mixed solvers need care: DDIM/DPM-Solver `pred_original_sample` is available from step output; for Euler/Heun-style schedulers the x0 estimate must be computed from sigma. The recorder computes x0 itself from eps and abar where the scheduler does not return it, and tests assert agreement.

## 4. Existing explainers and what they lack

- **Diffusion Explainer** (Polo Club, Georgia Tech; Kim et al., arXiv 2305.03509, IJCAI 2024 demo 2404.16069; open source `poloclub/diffusion-explainer`). Excellent for the Stable Diffusion text-to-image pipeline: prompt, text encoder, UNet refinement over steps, guidance scale, side-by-side prompt comparison. Reported 10,000+ users. Gaps relative to this project: it works on precomputed Stable Diffusion runs and a diagram of the architecture; it does not expose the noise schedule maths, the predicted noise or x0 at every step as scrub-able images, a forward process, scheduler comparison, or anything you can train or run yourself. (I did not find an official limitations section; this comparison is my reading of the paper and demo, and the README should say so.)
- **Hugging Face Diffusion course** and the Annotated Diffusion post: strong written walkthroughs and notebooks, but static plots, not an interactive scrubber.
- **Lilian Weng, "What are diffusion models?"** and **Sander Dieleman, "Diffusion is spectral autoregression" / "Noise schedules considered harmful"**: the best conceptual references for the schedule and SNR discussion. Cite as further reading, not as code.
- **Karpathy-style minimal repos / `lucidrains/denoising-diffusion-pytorch`** (MIT): compact reference for from-scratch training. Useful to cross-check, not to copy.
- **k-diffusion** (Crowson): reference implementation of the EDM samplers.
- Gap this project fills: one instrumented trajectory with every internal quantity recorded, scrubbed with a playhead against the schedule curves, same noise seed across schedulers, plus an exact live 2-D toy that makes the reverse process visible as a point cloud flowing back onto a distribution.

## 5. Models and datasets (licences to re-confirm on the model cards when downloading)

| Item | Use | Licence | Size |
|---|---|---|---|
| `google/ddpm-cifar10-32` | Track A pretrained pixel DDPM, 32x32, ~36M params, linear schedule, T=1000. Verified Apache-2.0 on its Hub page. | Apache-2.0 (**verified**) | ~143 MB |
| CIFAR-10 | Only if comparing to the checkpoint's training data; likely not needed | No formal licence, free for research (Krizhevsky) | not downloaded |
| Fashion-MNIST (Zalando Research) | Track B from-scratch training data. 28x28 grayscale, padded to 32x32 | MIT | ~30 MB |
| MNIST digits as alternative | Simpler, but Fashion-MNIST samples are more visually convincing | CC BY-SA 3.0 (check) | ~11 MB |
| `google/ddpm-ema-church-256`, `celebahq-256` | Rejected: 256x256 is slow and large for a scrubber | Apache-2.0 | ~450 MB each |
| Latent T2I (SD 1.5 class, or SDXS/small distilled) | Optional track C | Model-dependent (CreativeML OpenRAIL-M for SD1.5, not MIT-compatible to redistribute; weights are not redistributed anyway) | 2 to 4 GB |

Dataset download: avoid torchvision. Fetch the four Fashion-MNIST IDX files over HTTPS (Zalando GitHub / mirror) with a small script and parse with numpy, or use the `datasets` library from the Hub. Both are free of extra heavy deps. Exact URL and checksums go into the script during the build.

## 6. Training from scratch: realistic plan

- Fashion-MNIST padded to 32x32, a small UNet (`diffusers.UNet2DModel`, about 8 to 15M params, channels 64/128/256, one attention block at 8x8) trained with epsilon-prediction MSE, AdamW 2e-4, EMA 0.999, bf16 autocast, T=1000.
- Throughput expectation on a 5060 Ti: order of 20 to 40 k images/s for a model this size at 32x32 (to measure). 60k images x about 40 epochs should be 15 to 30 minutes. Target: recognisable shirts, trousers, shoes. Report the loss curve and a FID-style number **only if** I compute it with a method I can describe honestly (a small-sample pixel-space or Inception FID is expensive and fragile; likely report the loss curve and a grid of samples, plus nearest-neighbour check against the training set to show it is not memorising).
- Also train the cosine-schedule variant (same architecture, same budget) so that linear vs cosine is a real trained comparison, not just a plot. If budget is tight, the schedule comparison can use the one network with reparameterised timesteps only where mathematically valid (it is not in general), so I prefer two trainings.

## 7. In-browser 2-D toy diffusion (the exact explainer)

- Target distribution: two moons / spiral / ring of Gaussians, 2-D points.
- Forward process exact in closed form. For the reverse process there are two honest options:
  1. **Analytic score of the empirical distribution (the "ideal denoiser").** For a dataset of N points, the optimal eps-predictor under the forward process is closed-form: posterior over data points is a softmax of Gaussian likelihoods. This needs no training, is exact, and shows the true marginal flow. It tends to memorise the points (an informative lesson).
  2. **Small MLP trained live in the browser** (hand-written JS, or a tiny tensor lib) on the same points, so users see a learned score approximate the ideal one.
- Plan: implement both; the analytic one is default (instant, exact), the MLP is a "train it yourself" panel. Optional: compare against the closed-form Gaussian case where everything can be verified by hand. Python reference implementation produces test vectors; JS must match to 1e-6.
- Rendering: canvas 2D, up to ~3000 points at 60 fps, vector-field arrows optional.

## 8. Static-demo asset budget

- Track A: for 4 seeds x 3 schedulers (DDPM 1000 steps subsampled to 50 frames, DDIM 50, DPM-Solver++ 20), record x_t, eps, x0 as 32x32 RGB. A 32x32 frame is 3 KB raw; as a sprite sheet in lossless WebP, a 50-frame row compresses to maybe 60 to 120 KB. Total for the whole recorded set should stay under about 5 MB. Upscale in the browser with crisp (nearest or smooth, to be decided) rendering and the halftone shader from the site.
- Schedule/SNR arrays go in one small JSON.
- Track B: same format; plus the loss curve JSON and a sample grid.
- Live mode in the local server streams frames over SSE or WebSocket as a sampler runs.

## 9. Risks found so far

1. Blackwell wheel behaviour on Windows cp313 (mitigation: runtime check early; cu130 fallback; CPU fallback with a tiny model).
2. Global `pip` install is forbidden; the CUDA torch lives only in the venv. The venv's `torch` must shadow the global one; the test suite must run in that venv.
3. CI (GitHub Actions) has no GPU: fast tests use tiny random-init models and numerical identities, not the real checkpoint.
4. HF Hub downloads can be rate limited or offline; the repo must degrade to the recorded demo and give a clear error.
5. Honest quality: a 15-minute from-scratch model will be recognisable but not state of the art. The README says so with the measured numbers.
6. Multistep solver state and scheduler differences when recording `x0`.
7. Latent track scope creep (kept optional, behind the 8 GB budget).

## 10. Sources

- DDPM: https://arxiv.org/abs/2006.11239
- Improved DDPM: https://arxiv.org/abs/2102.09672
- DDIM: https://arxiv.org/abs/2010.02502
- DPM-Solver: https://arxiv.org/abs/2206.00927 ; DPM-Solver++: https://arxiv.org/abs/2211.01095
- EDM: https://arxiv.org/abs/2206.00364
- Diffusion Explainer: https://arxiv.org/abs/2305.03509 , https://arxiv.org/abs/2404.16069 , https://github.com/poloclub/diffusion-explainer
- google/ddpm-cifar10-32: https://huggingface.co/google/ddpm-cifar10-32
- Diffusers docs: https://huggingface.co/docs/diffusers
- PyTorch Blackwell discussion: https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099
- PyTorch wheel indexes: https://download.pytorch.org/whl/cu128 , https://download.pytorch.org/whl/cu130
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
