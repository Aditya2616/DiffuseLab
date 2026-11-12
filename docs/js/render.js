// Canvas rendering for a pane of pixels. Two modes: "halftone" (square dots sized by brightness, the look of the
// portfolio scene) and "pixels" (the actual colours, hard-edged). Both read the same recorded pixel values.

export const DPR_CAP = 2;
export const LIME = [215, 255, 63], BONE = [237, 236, 230];

export function fitCanvas(cv) {
  const r = cv.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP);
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  return { w, h, dpr };
}

const off = document.createElement('canvas');

// px: Uint8Array(H*W*C) in HWC order. tint: [r,g,b] used for the halftone dots.
export function drawPane(cv, px, C, H, W, mode, tint = LIME) {
  const { w, h } = fitCanvas(cv), cx = cv.getContext('2d');
  cx.clearRect(0, 0, w, h);
  if (!px) return;
  if (mode === 'pixels') {
    off.width = W; off.height = H;
    const ox = off.getContext('2d'), id = ox.createImageData(W, H);
    for (let i = 0; i < H * W; i++) {
      id.data[i * 4] = px[i * C]; id.data[i * 4 + 1] = px[i * C + (C > 1 ? 1 : 0)]; id.data[i * 4 + 2] = px[i * C + (C > 1 ? 2 : 0)]; id.data[i * 4 + 3] = 255;
    }
    ox.putImageData(id, 0, 0);
    cx.imageSmoothingEnabled = false;
    cx.drawImage(off, 0, 0, w, h);
    return;
  }
  const cell = Math.min(w / W, h / H), ox0 = (w - cell * W) / 2, oy0 = (h - cell * H) / 2;
  const buckets = Array.from({ length: 12 }, () => new Path2D());
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = (y * W + x) * C;
    const lum = C > 1 ? (0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]) / 255 : px[i] / 255;
    if (lum < 0.03) continue;
    const s = cell * 0.94 * Math.sqrt(lum);
    buckets[Math.min(11, Math.round(lum * 11))].rect(ox0 + x * cell + (cell - s) / 2, oy0 + y * cell + (cell - s) / 2, s, s);
  }
  buckets.forEach((p, b) => { cx.fillStyle = `rgba(${tint[0]},${tint[1]},${tint[2]},${0.3 + 0.7 * (b / 11)})`; cx.fill(p); });
}

// Draw float pixels (Float32Array HWC, values in [-1,1] or arbitrary with scale) by converting to bytes first.
export function toBytes(f, scale = 1) {
  const out = new Uint8Array(f.length);
  for (let i = 0; i < f.length; i++) out[i] = Math.max(0, Math.min(255, Math.round(((f[i] / scale) + 1) * 127.5)));
  return out;
}
