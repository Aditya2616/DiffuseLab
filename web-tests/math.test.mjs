// Run with: node --test web-tests
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { makeSchedule, idealDenoiser, ddpmStep, ddimStep, trailingTimesteps, qSample, predictX0 } from '../docs/js/math.js';

const V = JSON.parse(readFileSync(new URL('../tests/vectors/toy2d.json', import.meta.url)));
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${msg}: ${a} vs ${b}`);

test('schedules match the Python reference', () => {
  for (const kind of ['linear', 'cosine']) {
    const s = makeSchedule(kind), r = V.schedule[kind];
    r.t.forEach((t, i) => {
      close(s.betas[t], r.beta[i], 1e-12, `${kind} beta[${t}]`);
      close(s.abar[t], r.abar[i], 1e-12, `${kind} abar[${t}]`);
      close(s.logSnr[t], r.log_snr[i], 1e-9, `${kind} logsnr[${t}]`);
    });
  }
});

test('ideal denoiser matches the Python reference', () => {
  const d = V.denoiser, sch = makeSchedule(d.kind);
  const { x0, eps } = idealDenoiser(Float64Array.from(d.x_t.flat()), d.t, Float64Array.from(d.data.flat()), sch);
  d.x0.flat().forEach((v, i) => close(x0[i], v, 1e-10, `x0[${i}]`));
  d.eps.flat().forEach((v, i) => close(eps[i], v, 1e-10, `eps[${i}]`));
});

test('DDPM and DDIM steps match the Python reference', () => {
  const s = V.steps, sch = makeSchedule(s.kind), xt = Float64Array.from(V.denoiser.x_t.flat());
  const eps = Float64Array.from(s.eps.flat()), z = Float64Array.from(s.z.flat());
  const cmp = (got, ref, name) => ref.flat().forEach((v, i) => close(got[i], v, 1e-10, `${name}[${i}]`));
  cmp(ddpmStep(xt, s.t, eps, sch, z), s.ddpm, 'ddpm');
  cmp(ddimStep(xt, s.t, 550, eps, sch), s.ddim_to_550, 'ddim550');
  cmp(ddimStep(xt, s.t, -1, eps, sch), s.ddim_to_clean, 'ddimClean');
  cmp(ddimStep(xt, s.t, 550, eps, sch, 1, z), s.ddim_eta1_to_550, 'ddimEta1');
  cmp(ddimStep(xt, s.t, 550, eps, sch, 0.5, z), s.ddim_eta05_to_550, 'ddimEta05');
});

test('trailing timesteps match Diffusers (via Python vectors)', () => {
  for (const [n, ts] of Object.entries(V.trailing)) assert.deepEqual(trailingTimesteps(1000, +n), ts);
});

test('forward then predict_x0 with the true noise is the identity', () => {
  const sch = makeSchedule('cosine');
  for (const t of [0, 10, 500, 999]) close(predictX0(qSample(0.37, sch.abar[t], -1.2), -1.2, sch.abar[t]), 0.37, 1e-9, 'identity');
});
