// Loading runs: recorded (WebP sprite + JSON, works on any static host) or live (SSE from the local server).
// Both produce the same Run object, so every view draws them identically.

const PANES = 3; // x_t, predicted noise, predicted x0

export class Run {
  constructor(meta) {
    this.meta = meta;
    this.C = meta.channels; this.H = meta.height; this.W = meta.width;
    this.t = meta.t ? meta.t.slice() : [];
    this.abar = meta.abar ? meta.abar.slice() : [];
    this.beta = meta.beta ? meta.beta.slice() : [];
    this.snr = meta.snr ? meta.snr.slice() : [];
    this.frames = [[], [], []]; // frames[pane][k] = Uint8Array(H*W*C), HWC order
  }
  get n() { return this.t.length; }
  get kind() { return this.meta.kind; }
  pixels(pane, k) { return this.frames[pane][Math.max(0, Math.min(this.n - 1, k))]; }
  // frame index whose timestep is the first one at or below t (the state "at" that noise level)
  frameAtT(t) {
    for (let k = 0; k < this.n; k++) if (this.t[k] <= t) return k;
    return this.n - 1;
  }
}

async function decodeSprite(url) {
  const blob = await (await fetch(url)).blob();
  const bmp = await createImageBitmap(blob);
  const cv = document.createElement('canvas');
  cv.width = bmp.width; cv.height = bmp.height;
  const cx = cv.getContext('2d', { willReadFrequently: true });
  cx.drawImage(bmp, 0, 0);
  return cx.getImageData(0, 0, bmp.width, bmp.height);
}

export async function loadRecorded(base, id) {
  const [meta, img] = await Promise.all([
    fetch(`${base}runs/${id}.json`).then(r => r.json()),
    decodeSprite(`${base}runs/${id}.webp`),
  ]);
  // JSON has null for the final (infinite) SNR
  meta.snr = meta.snr.map(v => (v === null ? Infinity : v));
  const run = new Run(meta);
  const { C, H, W } = run, n = meta.frames, d = img.data, sw = img.width;
  for (let p = 0; p < PANES; p++) {
    for (let k = 0; k < n; k++) {
      const out = new Uint8Array(H * W * C);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const src = ((p * H + y) * sw + k * W + x) * 4, dst = (y * W + x) * C;
        for (let c = 0; c < C; c++) out[dst + c] = d[src + c];
      }
      run.frames[p].push(out);
    }
  }
  return run;
}

// Live frames arrive as base64 of 3 panes, each C*H*W in CHW order.
export function appendLiveFrame(run, f) {
  const { C, H, W } = run, size = C * H * W;
  const raw = Uint8Array.from(atob(f.data), ch => ch.charCodeAt(0));
  for (let p = 0; p < PANES; p++) {
    const out = new Uint8Array(H * W * C);
    for (let c = 0; c < C; c++) for (let i = 0; i < H * W; i++) out[i * C + c] = raw[p * size + c * H * W + i];
    run.frames[p].push(out);
  }
  run.t.push(f.t); run.abar.push(f.abar); run.beta.push(f.beta); run.snr.push(f.snr === null ? Infinity : f.snr);
}

export async function detectLive(timeoutMs = 900) {
  try {
    const ctl = new AbortController();
    const to = setTimeout(() => ctl.abort(), timeoutMs);
    const r = await fetch('/api/health', { signal: ctl.signal });
    clearTimeout(to);
    if (!r.ok) return null;
    const j = await r.json();
    return j.status === 'ok' ? j : null;
  } catch { return null; }
}

export async function startLive(req, onFrame) {
  const r = await fetch('/api/runs', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(req) });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `server said ${r.status}`);
  const { run_id } = await r.json();
  return new Promise((resolve, reject) => {
    const es = new EventSource(`/api/runs/${run_id}/stream`);
    let run = null;
    es.addEventListener('meta', e => { run = new Run({ ...JSON.parse(e.data), frames: 0 }); });
    es.addEventListener('frame', e => { appendLiveFrame(run, JSON.parse(e.data)); onFrame && onFrame(run); });
    es.addEventListener('done', () => { es.close(); resolve(run); });
    es.addEventListener('error', e => { es.close(); run && run.n > 0 && e.data === undefined ? resolve(run) : reject(new Error(e.data ? JSON.parse(e.data).message : 'stream error')); });
  });
}

export const fmtSnr = v => (!isFinite(v) ? '∞' : v >= 100 ? v.toFixed(0) : v >= 1 ? v.toFixed(2) : v.toExponential(1).replace('e-', 'e−'));
export const fmtAbar = v => (v >= 0.9995 ? '1.000' : v >= 0.001 ? v.toFixed(3) : v.toExponential(1).replace('e-', 'e−'));
