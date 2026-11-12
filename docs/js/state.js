// Shared state and a tiny event bus. Recorded runs are cached after the first decode.
import { loadRecorded } from './run.js';

export const S = { base: 'data/', index: [], live: null, model: 'cifar10', sched: 'ddim', seed: null, render: 'halftone', playing: true };
const bus = new EventTarget();
export const on = (n, f) => bus.addEventListener(n, e => f(e.detail));
export const emit = (n, d) => bus.dispatchEvent(new CustomEvent(n, { detail: d }));

const cache = new Map();
export function getRun(id) {
  if (!cache.has(id)) cache.set(id, loadRecorded(S.base, id));
  return cache.get(id);
}

export const MODELS = {
  cifar10: { label: 'CIFAR-10 · pretrained', channels: 3 },
  'fashion-linear': { label: 'Fashion · trained here · linear', channels: 1 },
  'fashion-cosine': { label: 'Fashion · trained here · cosine', channels: 1 },
};
export const SCHED_NAMES = { ddpm: 'DDPM', ddim: 'DDIM', dpmpp: 'DPM-Solver++', euler: 'Euler Karras' };
export const SCHEDS = { ...SCHED_NAMES };
// Step counts come from the recorded runs (index.json), never from constants here.
export const stepsFor = s => S.index.find(r => r.scheduler === s)?.steps;
export function labelSamplers() {
  for (const k of Object.keys(SCHED_NAMES)) { const n = stepsFor(k); SCHEDS[k] = n ? `${SCHED_NAMES[k]} · ${n}` : SCHED_NAMES[k]; }
}
export const modelsAvailable = () => Object.keys(MODELS).filter(m => S.index.some(r => r.model === m));
export const seedsFor = model => [...new Set(S.index.filter(r => r.model === model).map(r => r.seed))].sort((a, b) => a - b);
export const runId = (model, sched, seed) => S.index.find(r => r.model === model && r.scheduler === sched && r.seed === seed)?.id;

export function store(key, val) {
  try { if (val === undefined) return localStorage.getItem('diffuselab.' + key); localStorage.setItem('diffuselab.' + key, val); } catch { /* storage blocked */ }
  return null;
}
export const reduceMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export function chips(el, items, current, onPick) {
  el.innerHTML = '';
  items.forEach(([v, label]) => {
    const b = document.createElement('button');
    b.className = 'chip' + (v === current ? ' on' : ''); b.textContent = label; b.dataset.v = v;
    b.setAttribute('role', 'radio'); b.setAttribute('aria-checked', v === current);
    b.onclick = () => { setOn(el, v); onPick(v); };
    el.appendChild(b);
  });
}
export function setOn(el, v) {
  el.querySelectorAll('.chip').forEach(b => { const on = b.dataset.v === String(v); b.classList.toggle('on', on); b.setAttribute('aria-checked', on); });
}
export function bindChips(el, onPick) {
  el.querySelectorAll('.chip').forEach(b => { b.setAttribute('role', 'radio'); b.onclick = () => { setOn(el, b.dataset.v); onPick(b.dataset.v); }; });
}

// Call fn(visible) when el enters or leaves the viewport, so animation loops can pause off screen.
export function whenVisible(el, fn) {
  const io = new IntersectionObserver(es => fn(es[0].isIntersecting), { rootMargin: '80px' });
  io.observe(el);
}

export async function getJSON(url) {
  try { const r = await fetch(url); return r.ok ? await r.json() : null; } catch { return null; }
}
