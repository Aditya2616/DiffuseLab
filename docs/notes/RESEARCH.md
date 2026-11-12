# Research notes

What I read and decided before writing DiffuseLab. Numbers measured on my own machine are in the README, not here.

## The maths the app has to get exactly right

- **DDPM** (Ho, Jain, Abbeel 2020, [arXiv 2006.11239](https://arxiv.org/abs/2006.11239)). Forward process `q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I)` with `alpha_t = 1 - beta_t` and `abar_t = prod alpha_s`. The network predicts the noise, and the clean image it implies is `x0_hat = (x_t - sqrt(1 - abar_t) eps) / sqrt(abar_t)`. Linear schedule: beta from 1e-4 to 0.02 over T = 1000.
- **Improved DDPM** (Nichol and Dhariwal 2021, [arXiv 2102.09672](https://arxiv.org/abs/2102.09672)). Cosine schedule `abar_t = f(t) / f(0)`, `f(t) = cos^2(((t / T + s) / (1 + s)) pi / 2)`, s = 0.008, beta clipped at 0.999. It removes signal more gently than linear, so fewer steps are spent in almost-pure noise.
- **DDIM** (Song, Meng, Ermon 2020, [arXiv 2010.02502](https://arxiv.org/abs/2010.02502)). A deterministic (eta = 0) sampler that works on a subsequence of timesteps, so 20 to 50 steps replace 1000 with the same trained network.
- **DPM-Solver and DPM-Solver++** (Lu et al. 2022, [2206.00927](https://arxiv.org/abs/2206.00927), [2211.01095](https://arxiv.org/abs/2211.01095)). Higher-order ODE solvers in the log-SNR variable; the 2nd-order multistep variant gives usable samples in 10 to 20 steps.
- **EDM** (Karras et al. 2022, [arXiv 2206.00364](https://arxiv.org/abs/2206.00364)). Separates "noise schedule" from "sampler". I use its sigma spacing (rho = 7) with a plain first-order Euler step on a DDPM-trained network, as a way to show that the sampler is a separate choice from the schedule.
- **SNR** `abar_t / (1 - abar_t)`; I plot log-SNR because it is what makes linear and cosine comparable.

Every plotted schedule quantity in the UI comes from the same formulas as the Python code, and the two are checked against shared test vectors.

## Library behaviour I relied on (Diffusers 0.43)

- All schedulers share `set_timesteps(n)`, `scheduler.timesteps` and `scheduler.step(model_output, t, sample).prev_sample`. The cosine schedule is `beta_schedule="squaredcos_cap_v2"`.
- The unconditional `DDPMPipeline` has no per-step hook, so I wrote my own sampling loop around `UNet2DModel` and any scheduler. It is short, it records every internal quantity, and it lets one trained network be sampled with several schedulers.
- Multistep solvers keep state between steps, so each run builds a fresh scheduler and calls `set_timesteps` first (there is a test for state leaking between runs).
- Euler on Karras sigmas works on the variance-exploding variable `x_t / sqrt(abar_t)`. The sampler feeds the network `x / sqrt(sigma^2 + 1)`, derives `abar = 1 / (1 + sigma^2)` from sigma (Karras sigmas fall between integer timesteps), and computes x0 itself from the predicted noise. A test checks that an oracle noise predictor is recovered.

## Models and data

| Item | Use | Licence |
|---|---|---|
| `google/ddpm-cifar10-32` | Pretrained 32x32 pixel DDPM (about 36M parameters, linear schedule, T = 1000) | Apache-2.0 |
| Fashion-MNIST (Zalando Research) | Training data for the from-scratch model, padded 28 to 32 | MIT |

Fashion-MNIST is fetched as the four IDX files over HTTPS and parsed with numpy, so torchvision is not needed. I rejected the 256x256 DDPM checkpoints because they are too slow and too large to scrub through, and I did not build a text-to-image track: it would not add to what the explainer teaches and the weights are large.

## Training from scratch

`UNet2DModel` with channels 64/128/256, epsilon-prediction MSE, AdamW 2e-4, EMA 0.999, bf16 autocast on GPU, T = 1000, trained twice on the same budget (linear and cosine). Evaluation is the loss curve, sample grids, and a nearest-neighbour memorisation check against the training set. I do not report FID: a trustworthy FID needs an Inception network and many more samples than I would generate here.

## The 2-D toy

For an N-point dataset the optimal noise predictor under the forward process is closed-form: the posterior over data points is a softmax of Gaussian likelihoods. That gives an exact "ideal denoiser" with no training, and it shows the true flow of the marginal (including that it memorises the points). The Python and JavaScript implementations are checked against the same JSON vectors.

## Further reading

- Diffusion Explainer (Kim et al., [arXiv 2305.03509](https://arxiv.org/abs/2305.03509)), an interactive walkthrough of the Stable Diffusion pipeline.
- Hugging Face Diffusion course and The Annotated Diffusion Model.
- Lilian Weng, "What are diffusion models?"; Sander Dieleman's blog posts on noise schedules and diffusion as spectral autoregression.
- `lucidrains/denoising-diffusion-pytorch` and `k-diffusion` as compact reference implementations to cross-check against.
