// The main scrubber: three panes, a time slider, and the schedule curve with a playhead.
import { makeSchedule } from './math.js';
import { drawPlot } from './plots.js';
import { BONE, LIME, drawPane } from './render.js';
import { fmtAbar, fmtSnr, startLive } from './run.js';
import { MODELS, S, SCHEDS, bindChips, chips, emit, getRun, modelsAvailable, on, reduceMotion, runId, seedsFor, setOn, store, whenVisible } from './state.js';

const $ = id => document.getElementById(id);
const schedCache = {};
const sched = kind => (schedCache[kind] ||= makeSchedule(kind));
const TINT = [LIME, BONE, LIME];

let run = null, pos = 0, playing = false, speed = 1, visible = true, curve = 'abar', last = 0, liveBusy = false, loadToken = 0;

function framesPerSecond() { return Math.max(5, run.n / 5.5) * speed; }

function drawAll() {
  if (!run) return;
  const k = Math.round(Math.min(pos, run.n - 1));
  for (let i = 0; i < 3; i++) drawPane($(`cv-${i}`), run.pixels(i, k), run.C, run.H, run.W, S.render, TINT[i]);
  const t = run.t[k], ab = run.abar[k], done = t < 0;
  $('r-t').textContent = done ? '0' : String(t + 1);
  $('r-step').textContent = `${k} / ${run.n - 1}`;
  $('r-abar').textContent = fmtAbar(ab);
  $('r-snr').textContent = fmtSnr(run.snr[k]);
  $('r-sn').textContent = `${Math.sqrt(ab).toFixed(2)} : ${Math.sqrt(1 - ab).toFixed(2)}`;
  const rng = $('scrub-range');
  if (+rng.value !== k) rng.value = k;
  drawCurve(done ? 0 : t);
}

function drawCurve(playT) {
  const s = sched(run.kind);
  const defs = {
    abar: { y: s.abar, label: 'ᾱ_t', yMin: 0, yMax: 1, note: 'fraction of the original image variance that survives' },
    beta: { y: s.betas, label: 'β_t', yMin: 0, note: 'noise variance added at each forward step' },
    snr: { y: s.logSnr, label: 'log SNR', note: 'log of signal-to-noise ratio, ᾱ/(1−ᾱ)' },
  };
  const d = defs[curve];
  $('plot-note').textContent = d.note;
  drawPlot($('plot-hero'), [{ y: d.y, color: 'rgba(215,255,63,.9)', label: d.label }], {
    playT, marks: run.t, markSeries: true, yMin: d.yMin, yMax: d.yMax,
    yTicks: curve === 'snr' ? [-8, -4, 0, 4, 8] : undefined, fmtY: curve === 'beta' ? v => v.toFixed(3) : undefined,
  });
}

function setPlaying(p) {
  playing = p;
  $('play-ic').setAttribute('d', p ? 'M6 5h4v14H6zM14 5h4v14h-4z' : 'M7 5v14l12-7z');
}

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (!run || !visible || !playing || liveBusy) return;
  pos += dt * framesPerSecond();
  if (pos >= run.n - 1) { pos = run.n - 1; setPlaying(false); }
  drawAll();
}

async function load(autoplay = true) {
  const id = runId(S.model, S.sched, S.seed);
  if (!id) return;
  const token = ++loadToken;
  const r = await getRun(id);
  if (token !== loadToken) return;
  run = r; pos = 0;
  $('scrub-range').max = run.n - 1;
  const m = run.meta;
  $('run-note').textContent = `recorded run · ${MODELS[S.model].label} · ${SCHEDS[S.sched]} · seed ${m.seed} · ${m.steps} network calls · ${m.seconds}s on ${m.device}`;
  setPlaying(autoplay && !reduceMotion());
  if (!playing) pos = run.n - 1;
  drawAll();
  emit('run', run);
}

function renderChips() {
  const models = modelsAvailable();
  if (!models.includes(S.model)) S.model = models[0];
  const seeds = seedsFor(S.model);
  if (!seeds.includes(S.seed)) S.seed = seeds[0];
  chips($('model-chips'), models.map(m => [m, MODELS[m].label]), S.model, v => { S.model = v; S.seed = null; renderChips(); load(); emit('model'); });
  chips($('sched-chips'), Object.entries(SCHEDS).filter(([k]) => runId(S.model, k, S.seed)), S.sched, v => { S.sched = v; load(); });
  chips($('seed-chips'), seeds.map(s => [s, `seed ${s}`]), S.seed, v => { S.seed = +v; renderChips(); load(); emit('seed'); });
}

async function sampleLive() {
  if (!S.live || liveBusy) return;
  liveBusy = true; setPlaying(false);
  const btn = $('live-btn'), steps = { ddpm: 1000, ddim: 50, dpmpp: 20, euler: 20 }[S.sched];
  btn.textContent = 'sampling…'; btn.disabled = true;
  const seed = Math.floor(Math.random() * 1e6);
  try {
    await startLive({ model: S.model, scheduler: S.sched, steps, seed, record_every: S.sched === 'ddpm' ? 20 : 1 }, r => {
      run = r; pos = r.n - 1; $('scrub-range').max = Math.max(1, r.n - 1); drawAll();
    });
    run.meta = { ...run.meta, kind: run.meta.kind, seconds: '–', device: 'live', seed };
    $('run-note').textContent = `live run · fresh noise, seed ${seed} · ${MODELS[S.model].label} · ${SCHEDS[S.sched]}`;
    pos = 0; setPlaying(!reduceMotion()); drawAll();
  } catch (e) {
    $('run-note').innerHTML = `<span class="err">live sampling failed: ${String(e.message).replace(/</g, '&lt;')}</span>`;
  } finally { liveBusy = false; btn.textContent = 'sample fresh'; btn.disabled = false; }
}

export function initHero() {
  S.render = store('render') || 'halftone';
  setOn($('render-chips'), S.render);
  bindChips($('render-chips'), v => { S.render = v; store('render', v); drawAll(); emit('render'); });
  bindChips($('speed-chips'), v => { speed = +v; });
  bindChips($('curve-chips'), v => { curve = v; if (run) drawAll(); });
  $('play').onclick = () => { if (!run) return; if (!playing && pos >= run.n - 1) pos = 0; setPlaying(!playing); };
  $('scrub-range').oninput = e => { setPlaying(false); pos = +e.target.value; drawAll(); };
  $('step-back').onclick = () => { setPlaying(false); pos = Math.max(0, Math.round(pos) - 1); drawAll(); };
  $('step-fwd').onclick = () => { setPlaying(false); pos = Math.min(run.n - 1, Math.round(pos) + 1); drawAll(); };
  document.addEventListener('keydown', e => {
    if (!run || e.target.matches('input[type=text],select,textarea')) return;
    if (e.target.matches('input[type=range]') && e.target.id !== 'scrub-range') return;
    if (e.key === 'ArrowLeft') { $('step-back').click(); e.preventDefault(); }
    if (e.key === 'ArrowRight') { $('step-fwd').click(); e.preventDefault(); }
    if (e.key === ' ' && !e.target.matches('button')) { $('play').click(); e.preventDefault(); }
  });
  const panes = $('panes');
  panes.querySelectorAll('.pane').forEach(p => {
    p.onclick = () => {
      const i = p.dataset.i, focus = panes.dataset.focus === i ? '' : i;
      if (focus) panes.dataset.focus = focus; else panes.removeAttribute('data-focus');
      panes.querySelectorAll('.pane').forEach(q => q.classList.toggle('focus', q.dataset.i === focus));
      requestAnimationFrame(drawAll);
    };
  });
  if (window.matchMedia('(max-width: 640px)').matches) panes.querySelector('.pane[data-i="2"]').click(); // phones: the clean-image pane large, the other two stacked beneath
  $('live-btn').onclick = sampleLive;
  whenVisible($('stage'), v => { visible = v; });
  window.addEventListener('resize', () => requestAnimationFrame(drawAll));
  on('render', drawAll);
  renderChips();
  load();
  requestAnimationFrame(frame);
}

export const heroRun = () => run;
