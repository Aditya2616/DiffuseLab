// The forward process, computed live: x_t = sqrt(abar) x0 + sqrt(1 - abar) eps.
import { gaussian, makeSchedule, mulberry32, qSample } from './math.js';
import { BONE, LIME, drawPane, toBytes } from './render.js';
import { S, bindChips, chips, getRun, on, runId, seedsFor } from './state.js';
import { fmtAbar, fmtSnr } from './run.js';

const $ = id => document.getElementById(id);
const sch = { linear: makeSchedule('linear'), cosine: makeSchedule('cosine') };
let sources = [], cur = 0, kind = 'linear', eps = null, noiseSeed = 1;

function makeEps(n) {
  const r = mulberry32(noiseSeed * 7919), e = new Float64Array(n);
  for (let i = 0; i < n; i++) e[i] = gaussian(r);
  return e;
}

function draw() {
  const src = sources[cur];
  if (!src) return;
  const n = src.x0.length, t = +$('fwd-range').value, ab = sch[kind].abar[t];
  if (!eps || eps.length !== n) eps = makeEps(n);
  const sig = new Float32Array(n), noi = new Float32Array(n), sum = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    sig[i] = Math.sqrt(ab) * src.x0[i];
    noi[i] = Math.sqrt(1 - ab) * eps[i];
    sum[i] = qSample(src.x0[i], ab, eps[i]);
  }
  drawPane($('fw-0'), toBytes(sig), src.C, src.H, src.W, S.render, LIME);
  drawPane($('fw-1'), toBytes(noi), src.C, src.H, src.W, S.render, BONE);
  drawPane($('fw-2'), toBytes(sum), src.C, src.H, src.W, S.render, LIME);
  $('fw-t').textContent = String(t + 1);
  $('fw-abar').textContent = fmtAbar(ab);
  $('fw-snr').textContent = fmtSnr(sch[kind].snr[t]);
  $('fw-sn').textContent = `${Math.sqrt(ab).toFixed(2)} : ${Math.sqrt(1 - ab).toFixed(2)}`;
}

async function fashionReal() {
  try {
    const img = new Image();
    img.src = `${S.base}fashion_real.webp`;
    await img.decode();
    const cv = document.createElement('canvas');
    cv.width = img.width; cv.height = img.height;
    const cx = cv.getContext('2d', { willReadFrequently: true });
    cx.drawImage(img, 0, 0);
    const d = cx.getImageData(0, 0, img.width, img.height).data, out = [], n = img.width / 32;
    for (let k = 0; k < n; k++) {
      const x0 = new Float32Array(32 * 32);
      for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) x0[y * 32 + x] = d[(y * img.width + k * 32 + x) * 4] / 127.5 - 1;
      out.push({ label: `fashion ${k + 1}`, C: 1, H: 32, W: 32, x0 });
    }
    return out;
  } catch { return []; }
}

async function build() {
  const out = [];
  if (S.index.some(r => r.model === 'cifar10')) {
    for (const seed of seedsFor('cifar10')) {
      const id = runId('cifar10', 'ddim', seed);
      if (!id) continue;
      const r = await getRun(id), px = r.pixels(0, r.n - 1), x0 = new Float32Array(px.length);
      for (let i = 0; i < px.length; i++) x0[i] = px[i] / 127.5 - 1;
      out.push({ label: `generated ${out.length + 1}`, C: r.C, H: r.H, W: r.W, x0 });
    }
  }
  out.push(...(await fashionReal()));
  sources = out;
  chips($('fwd-src'), sources.map((s, i) => [i, s.label]), cur, v => { cur = +v; eps = null; draw(); });
  draw();
}

export function initForward() {
  bindChips($('fwd-kind'), v => { kind = v; draw(); });
  $('fwd-range').oninput = draw;
  $('fwd-noise').onclick = () => { noiseSeed++; eps = null; draw(); };
  window.addEventListener('resize', () => requestAnimationFrame(draw));
  on('render', draw);
  build();
}
