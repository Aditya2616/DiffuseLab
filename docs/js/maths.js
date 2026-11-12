// The six equations the project implements, each tied to something visible on the page.
export const EQUATIONS = [
  { title: '1 · forward process', tex: String.raw`x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal N(0,I)`,
    on: 'The "Forward" section. One closed-form jump to any t. ᾱ<sub>t</sub> is the product of (1−β<sub>s</sub>) up to t.' },
  { title: '2 · what the network learns', tex: String.raw`\mathcal L=\mathbb E_{x_0,\,t,\,\epsilon}\,\big\|\epsilon-\epsilon_\theta(x_t,t)\big\|^2`,
    on: 'The loss curve under "Training". The middle panel of the scrubber is ε̂ = ε<sub>θ</sub>(x<sub>t</sub>, t) at every step.' },
  { title: '3 · predicted clean image', tex: String.raw`\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}`,
    on: 'The right-hand panel. Solving equation 1 for x<sub>0</sub> with the predicted noise. Early on it is a blurry average; it sharpens as t falls.' },
  { title: '4 · DDPM step', tex: String.raw`x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat\epsilon\Big)+\sigma_t z,\quad \sigma_t^2=\beta_t\tfrac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}`,
    on: 'The 1000-step column in "Compare", and η = 1 in the 2-D lab with all 1000 steps. Fresh noise z at every step, so it is stochastic.' },
  { title: '5 · DDIM step', tex: String.raw`x_{t'}=\sqrt{\bar\alpha_{t'}}\,\hat x_0+\sqrt{1-\bar\alpha_{t'}-\sigma^2}\,\hat\epsilon+\sigma z`,
    on: 'Any t′ &lt; t, so 50 or 20 steps. σ = 0 is deterministic DDIM (the η slider in the 2-D lab); σ from η = 1 on consecutive steps recovers equation 4 exactly, which the tests check.' },
  { title: '6 · signal-to-noise and the cosine schedule', tex: String.raw`\mathrm{SNR}(t)=\frac{\bar\alpha_t}{1-\bar\alpha_t},\qquad \bar\alpha_t=\frac{f(t)}{f(0)},\ f(t)=\cos^2\!\Big(\tfrac{t/T+s}{1+s}\cdot\tfrac{\pi}{2}\Big)`,
    on: 'The "Schedules" section and the curve under the scrubber. s = 0.008 and β<sub>t</sub> is clipped at 0.999.' },
];
