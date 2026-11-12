// The maths of diffusion, in plain float64 JavaScript. No dependencies, so Node can test it against the
// Python reference (tests/vectors/toy2d.json, see web-tests/math.test.mjs).

export const KINDS = ['linear', 'cosine'];

export function makeBetas(kind, T = 1000) {
  const b = new Float64Array(T);
  if (kind === 'linear') {
    for (let i = 0; i < T; i++) b[i] = 1e-4 + (0.02 - 1e-4) * (T === 1 ? 0 : i / (T - 1));
  } else if (kind === 'cosine') {
    const s = 0.008, f = u => Math.cos(((u + s) / (1 + s)) * Math.PI / 2) ** 2;
    for (let i = 0; i < T; i++) b[i] = Math.min(1 - f((i + 1) / T) / f(i / T), 0.999);
  } else throw new Error('unknown schedule ' + kind);
  return b;
}

export function makeSchedule(kind, T = 1000) {
  const betas = makeBetas(kind, T), alphas = new Float64Array(T), abar = new Float64Array(T);
  const snr = new Float64Array(T), logSnr = new Float64Array(T);
  let p = 1;
  for (let i = 0; i < T; i++) {
    alphas[i] = 1 - betas[i]; p *= alphas[i]; abar[i] = p;
    snr[i] = p / (1 - p); logSnr[i] = Math.log(snr[i]);
  }
  return { kind, T, betas, alphas, abar, snr, logSnr, abarAt: t => (t < 0 ? 1 : abar[t]) };
}

// Timesteps the Diffusers schedulers use here ("trailing" spacing): T-1 down in equal strides.
export function trailingTimesteps(T, steps) {
  const stride = T / steps, out = [];
  for (let i = 0; i < steps; i++) out.push(Math.round(T - i * stride) - 1);
  return out;
}

// ---- seeded randomness (for the forward-process demo and the toy) ----
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function gaussian(rand) {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

// ---- forward process, closed form ----
export const qSample = (x0, abar, eps) => Math.sqrt(abar) * x0 + Math.sqrt(1 - abar) * eps;
export const predictX0 = (xt, eps, abar) => (xt - Math.sqrt(1 - abar) * eps) / Math.sqrt(abar);

// ---- the 2-D toy: exact ideal denoiser for a finite point set (flat arrays [x0,y0,x1,y1,...]) ----
export function idealDenoiser(xt, t, data, sch) {
  const ab = sch.abar[t], sa = Math.sqrt(ab), v = 2 * (1 - ab), N = data.length / 2, M = xt.length / 2;
  const x0 = new Float64Array(xt.length), eps = new Float64Array(xt.length), logit = new Float64Array(N);
  const sb = Math.sqrt(1 - ab);
  for (let m = 0; m < M; m++) {
    const px = xt[2 * m], py = xt[2 * m + 1];
    let mx = -Infinity;
    for (let i = 0; i < N; i++) {
      const dx = px - sa * data[2 * i], dy = py - sa * data[2 * i + 1];
      logit[i] = -(dx * dx + dy * dy) / v;
      if (logit[i] > mx) mx = logit[i];
    }
    let z = 0, ax = 0, ay = 0;
    for (let i = 0; i < N; i++) {
      const w = Math.exp(logit[i] - mx);
      z += w; ax += w * data[2 * i]; ay += w * data[2 * i + 1];
    }
    x0[2 * m] = ax / z; x0[2 * m + 1] = ay / z;
    eps[2 * m] = (px - sa * x0[2 * m]) / sb; eps[2 * m + 1] = (py - sa * x0[2 * m + 1]) / sb;
  }
  return { x0, eps };
}

export function ddpmStep(xt, t, eps, sch, z) {
  const beta = sch.betas[t], alpha = sch.alphas[t], ab = sch.abar[t];
  const out = new Float64Array(xt.length), c = beta / Math.sqrt(1 - ab), ia = 1 / Math.sqrt(alpha);
  const sd = t === 0 ? 0 : Math.sqrt((beta * (1 - sch.abar[t - 1])) / (1 - ab));
  for (let i = 0; i < xt.length; i++) out[i] = (xt[i] - c * eps[i]) * ia + sd * (z ? z[i] : 0);
  return out;
}

// eta = 0 is deterministic DDIM; eta = 1 is the DDPM-like stochastic step on the same timesteps.
export function ddimStep(xt, t, tPrev, eps, sch, eta = 0, z = null) {
  const ab = sch.abar[t], abp = sch.abarAt(tPrev), out = new Float64Array(xt.length);
  const s1 = Math.sqrt(1 - ab), s0 = Math.sqrt(ab), a = Math.sqrt(abp);
  const sigma = eta * Math.sqrt((1 - abp) / (1 - ab)) * Math.sqrt(1 - ab / abp);
  const b = Math.sqrt(Math.max(1 - abp - sigma * sigma, 0));
  for (let i = 0; i < xt.length; i++) out[i] = a * ((xt[i] - s1 * eps[i]) / s0) + b * eps[i] + (eta > 0 && z ? sigma * z[i] : 0);
  return out;
}

// ---- toy datasets (shapes only; the maths above is what is verified against Python) ----
export function makeDataset(name, n, seed = 0) {
  const r = mulberry32(seed), pts = new Float64Array(2 * n);
  const g = () => gaussian(r);
  for (let i = 0; i < n; i++) {
    let x, y;
    if (name === 'moons') {
      const a = r() * Math.PI;
      if (i % 2) { x = 1 - Math.cos(a); y = 0.5 - Math.sin(a); } else { x = Math.cos(a); y = Math.sin(a); }
      x += 0.04 * g(); y += 0.04 * g();
    } else if (name === 'ring8') {
      const c = Math.floor(r() * 8) * Math.PI / 4;
      x = 1.6 * Math.cos(c) + 0.12 * g(); y = 1.6 * Math.sin(c) + 0.12 * g();
    } else if (name === 'spiral') {
      const rad = 0.15 + 0.85 * r(), a = rad * 3.5 * Math.PI;
      x = rad * Math.cos(a) + 0.02 * g(); y = rad * Math.sin(a) + 0.02 * g();
    } else throw new Error('unknown dataset ' + name);
    pts[2 * i] = x; pts[2 * i + 1] = y;
  }
  // centre and scale to unit standard deviation
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += pts[2 * i]; my += pts[2 * i + 1]; }
  mx /= n; my /= n;
  let v = 0;
  for (let i = 0; i < n; i++) { pts[2 * i] -= mx; pts[2 * i + 1] -= my; v += pts[2 * i] ** 2 + pts[2 * i + 1] ** 2; }
  const sd = Math.sqrt(v / (2 * n));
  for (let i = 0; i < 2 * n; i++) pts[i] /= sd;
  return pts;
}
