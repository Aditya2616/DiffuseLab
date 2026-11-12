// The 2-D lab. Exact ideal denoiser (math.js, verified against Python), DDIM/DDPM-like updates, no training.
import { ddimStep, gaussian, idealDenoiser, makeDataset, makeSchedule, mulberry32, trailingTimesteps } from './math.js';
import { fitCanvas } from './render.js';
import { bindChips, reduceMotion, whenVisible } from './state.js';
import { fmtAbar } from './run.js';

const $ = id => document.getElementById(id);
const NDATA = 240, NPART = 450, R = 3.3, TRAILS = 36;
let mode = 'reverse', dsName = 'moons', kind = 'linear', steps = 30, eta = 0;
let data = null, sch = null, ts = [], traj = [], fwd = [], pos = 0, playing = false, visible = true, last = 0, timer = 0;

function compute() {
  sch = makeSchedule(kind);
  data = makeDataset(dsName, NDATA, 1);
  ts = trailingTimesteps(1000, steps);
  // reverse: start from N(0, I), follow the sampler with the ideal denoiser at every step
  const rnd = mulberry32(5), x = new Float64Array(2 * NPART);
  for (let i = 0; i < x.length; i++) x[i] = gaussian(rnd);
  let cur = x;
  traj = [cur];
  for (let k = 0; k < ts.length; k++) {
    const { eps } = idealDenoiser(cur, ts[k], data, sch);
    const z = eta > 0 ? Float64Array.from(cur, () => gaussian(rnd)) : null;
    cur = ddimStep(cur, ts[k], k + 1 < ts.length ? ts[k + 1] : -1, eps, sch, eta, z);
    traj.push(cur);
  }
  // forward: the data itself, noised in closed form with one fixed eps per point
  const r2 = mulberry32(11), e = Float64Array.from(data, () => gaussian(r2));
  fwd = [];
  const tsF = [-1, ...ts.slice().reverse()]; // from clean to noisy
  for (const t of tsF) {
    const ab = sch.abarAt(t), out = new Float64Array(data.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.sqrt(ab) * data[i] + Math.sqrt(1 - ab) * e[i];
    fwd.push(out);
  }
  $('toy-t').max = steps;
  pos = Math.min(pos, steps);
}

function nearestMean(pts) {
  let s = 0;
  const M = pts.length / 2, N = data.length / 2;
  for (let m = 0; m < M; m++) {
    let best = Infinity;
    for (let i = 0; i < N; i++) { const dx = pts[2 * m] - data[2 * i], dy = pts[2 * m + 1] - data[2 * i + 1], d = dx * dx + dy * dy; if (d < best) best = d; }
    s += Math.sqrt(best);
  }
  return s / M;
}

function draw() {
  const cv = $('toy-cv'), { w, h, dpr } = fitCanvas(cv), cx = cv.getContext('2d');
  const k = Math.round(pos), sc = w / (2 * R), X = v => w / 2 + v * sc, Y = v => h / 2 - v * sc;
  cx.clearRect(0, 0, w, h);
  // axes, faint
  cx.strokeStyle = 'rgba(237,236,230,.07)'; cx.lineWidth = dpr;
  cx.beginPath(); cx.moveTo(X(0), 0); cx.lineTo(X(0), h); cx.moveTo(0, Y(0)); cx.lineTo(w, Y(0)); cx.stroke();
  // the data (what the denoiser knows)
  cx.strokeStyle = 'rgba(237,236,230,.38)'; cx.lineWidth = dpr;
  const dp = new Path2D();
  for (let i = 0; i < data.length; i += 2) { dp.moveTo(X(data[i]) + 3.2 * dpr, Y(data[i + 1])); dp.arc(X(data[i]), Y(data[i + 1]), 3.2 * dpr, 0, 6.2832); }
  cx.stroke(dp);

  const pts = mode === 'reverse' ? traj[k] : fwd[k];
  const t = mode === 'reverse' ? (k < ts.length ? ts[k] : -1) : (k === 0 ? -1 : ts[ts.length - k]);
  // denoiser pull: where the optimal denoiser would send a point sitting at each grid location, at this t
  if ($('toy-field').checked && mode === 'reverse' && k < ts.length) {
    const G = 15, grid = new Float64Array(2 * G * G);
    for (let a = 0; a < G; a++) for (let b = 0; b < G; b++) { grid[2 * (a * G + b)] = -R + (2 * R * (b + 0.5)) / G; grid[2 * (a * G + b) + 1] = -R + (2 * R * (a + 0.5)) / G; }
    const { x0 } = idealDenoiser(grid, ts[k], data, sch);
    cx.strokeStyle = 'rgba(215,255,63,.28)'; cx.lineWidth = 1.2 * dpr; cx.beginPath();
    for (let i = 0; i < G * G; i++) {
      const gx = grid[2 * i], gy = grid[2 * i + 1], dx = x0[2 * i] - gx, dy = x0[2 * i + 1] - gy, d = Math.hypot(dx, dy) || 1, len = Math.min(0.32, d * 0.5);
      cx.moveTo(X(gx), Y(gy)); cx.lineTo(X(gx + (dx / d) * len), Y(gy + (dy / d) * len));
    }
    cx.stroke();
  }
  // trails of a few particles up to now
  if (mode === 'reverse') {
    cx.strokeStyle = 'rgba(215,255,63,.16)'; cx.lineWidth = dpr; cx.beginPath();
    for (let m = 0; m < TRAILS; m++) { cx.moveTo(X(traj[0][2 * m]), Y(traj[0][2 * m + 1])); for (let j = 1; j <= k; j++) cx.lineTo(X(traj[j][2 * m]), Y(traj[j][2 * m + 1])); }
    cx.stroke();
  }
  cx.fillStyle = '#d7ff3f'; cx.beginPath();
  for (let m = 0; m < pts.length; m += 2) { cx.moveTo(X(pts[m]) + 2.3 * dpr, Y(pts[m + 1])); cx.arc(X(pts[m]), Y(pts[m + 1]), 2.3 * dpr, 0, 6.2832); }
  cx.fill();

  $('toy-tv').textContent = t < 0 ? '0' : String(t + 1);
  $('toy-abar').textContent = fmtAbar(sch.abarAt(t));
  $('toy-dist').textContent = nearestMean(pts).toFixed(3);
  const rng = $('toy-t');
  if (+rng.value !== k) rng.value = k;
}

function setPlaying(p) {
  playing = p;
  $('toy-play-ic').setAttribute('d', p ? 'M6 5h4v14H6zM14 5h4v14h-4z' : 'M7 5v14l12-7z');
}

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (!playing || !visible) return;
  pos += dt * Math.max(6, steps / 4.5);
  if (pos >= steps) { pos = steps; setPlaying(false); }
  draw();
}

function recompute(autoplay = true) {
  compute();
  pos = 0;
  setPlaying(autoplay && !reduceMotion());
  if (!playing) pos = steps;
  draw();
}

export function initToy() {
  bindChips($('toy-mode'), v => { mode = v; recompute(); });
  bindChips($('toy-data'), v => { dsName = v; recompute(); });
  bindChips($('toy-kind'), v => { kind = v; recompute(); });
  $('toy-steps').oninput = e => { steps = +e.target.value; $('toy-steps-v').textContent = steps; clearTimeout(timer); timer = setTimeout(recompute, 160); };
  $('toy-eta').oninput = e => { eta = +e.target.value; $('toy-eta-v').textContent = eta.toFixed(1); clearTimeout(timer); timer = setTimeout(recompute, 160); };
  $('toy-field').onchange = draw;
  $('toy-play').onclick = () => { if (!playing && pos >= steps) pos = 0; setPlaying(!playing); };
  $('toy-t').oninput = e => { setPlaying(false); pos = +e.target.value; draw(); };
  whenVisible($('toy-cv'), v => { visible = v; });
  window.addEventListener('resize', () => requestAnimationFrame(draw));
  compute();
  setPlaying(false);
  pos = steps;
  draw();
  // start the first run when the lab scrolls into view
  const io = new IntersectionObserver(es => { if (es[0].isIntersecting) { io.disconnect(); recompute(); } }, { threshold: 0.35 });
  io.observe($('toy-cv'));
  requestAnimationFrame(frame);
}
