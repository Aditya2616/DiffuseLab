// Small canvas line plots with a playhead. Everything drawn here is computed from the schedule maths in math.js.
import { fitCanvas } from './render.js';

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const mono = px => `${px}px "JetBrains Mono", ui-monospace, Consolas, monospace`;

// series: [{ y: Float64Array(T), color, label, dashed }], x axis is t from T-1 (left) to 0 (right) so that it
// reads the same direction as the denoising run. opts.log draws log10 y. marks: timesteps to tick on the axis.
export function drawPlot(cv, series, opts = {}) {
  const { w, h, dpr } = fitCanvas(cv), cx = cv.getContext('2d');
  const T = series[0].y.length, padL = 44 * dpr, padR = 12 * dpr, padT = 14 * dpr, padB = 26 * dpr;
  const iw = w - padL - padR, ih = h - padT - padB;
  cx.clearRect(0, 0, w, h);
  let lo = Infinity, hi = -Infinity;
  const tf = v => (opts.log ? Math.log10(Math.max(v, 1e-12)) : v);
  series.forEach(s => { for (let i = 0; i < T; i += 2) { const v = tf(s.y[i]); if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } } });
  if (opts.yMin !== undefined) lo = opts.yMin;
  if (opts.yMax !== undefined) hi = opts.yMax;
  if (hi === lo) hi = lo + 1;
  const X = t => padL + ((T - 1 - t) / (T - 1)) * iw, Y = v => padT + (1 - (tf(v) - lo) / (hi - lo)) * ih;

  cx.lineWidth = 1; cx.font = mono(9 * dpr); cx.fillStyle = 'rgba(237,236,230,.5)';
  cx.strokeStyle = 'rgba(237,236,230,.09)';
  cx.textAlign = 'right'; cx.textBaseline = 'middle';
  const ticks = opts.yTicks || [0, 0.25, 0.5, 0.75, 1].map(f => lo + f * (hi - lo));
  ticks.forEach(v => {
    const y = padT + (1 - (v - lo) / (hi - lo)) * ih;
    cx.beginPath(); cx.moveTo(padL, y); cx.lineTo(w - padR, y); cx.stroke();
    cx.fillText(opts.fmtY ? opts.fmtY(v) : (+v.toPrecision(2)).toString(), padL - 6 * dpr, y);
  });
  cx.textAlign = 'center'; cx.textBaseline = 'top';
  [999, 750, 500, 250, 0].forEach(t => {
    const x = X(t);
    cx.beginPath(); cx.moveTo(x, padT + ih); cx.lineTo(x, padT + ih + 4 * dpr); cx.stroke();
    cx.fillText(t === 999 ? '1000' : String(t), Math.min(Math.max(x, padL + 8 * dpr), w - padR - 8 * dpr), padT + ih + 8 * dpr);
  });

  series.forEach(s => {
    cx.beginPath(); cx.strokeStyle = s.color; cx.lineWidth = (s.width || 1.6) * dpr; cx.setLineDash(s.dashed ? [4 * dpr, 4 * dpr] : []);
    for (let i = 0; i < T; i++) { const t = T - 1 - i, x = X(t), y = Y(s.y[t]); i ? cx.lineTo(x, y) : cx.moveTo(x, y); }
    cx.stroke(); cx.setLineDash([]);
  });

  // timesteps the sampler actually visits
  if (opts.marks && opts.markSeries) {
    cx.fillStyle = 'rgba(215,255,63,.55)';
    opts.marks.forEach(t => { if (t >= 0) cx.fillRect(X(t) - 0.5 * dpr, padT + ih - 5 * dpr, 1 * dpr, 5 * dpr); });
  }

  if (opts.playT !== undefined && opts.playT !== null) {
    const t = Math.max(0, Math.min(T - 1, opts.playT)), x = X(t);
    cx.strokeStyle = 'rgba(215,255,63,.55)'; cx.lineWidth = 1 * dpr; cx.beginPath(); cx.moveTo(x, padT); cx.lineTo(x, padT + ih); cx.stroke();
    series.forEach(s => {
      cx.fillStyle = css('--lime') || '#d7ff3f'; cx.beginPath(); cx.arc(x, Y(s.y[t]), 3.5 * dpr, 0, 6.2832); cx.fill();
    });
  }
  return { X, Y, padL, padR, iw, T, w };
}

export function seriesLegend(el, series) {
  el.innerHTML = series.map(s => `<span class="key"><i style="background:${s.color};${s.dashed ? 'opacity:.7' : ''}"></i>${s.label}</span>`).join('');
}
