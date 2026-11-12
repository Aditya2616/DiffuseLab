// Same seed, same network, three samplers, linked by noise level t.
import { BONE, LIME, drawPane } from './render.js';
import { S, SCHEDS, chips, getJSON, getRun, on, runId, seedsFor } from './state.js';

const $ = id => document.getElementById(id);
const ORDER = ['ddpm', 'ddim', 'dpmpp', 'euler'];
let runs = {}, token = 0, stats = null;

function draw() {
  const t = +$('cmp-range').value;
  $('cmp-t').textContent = `t = ${t}`;
  ORDER.forEach((s, i) => {
    const r = runs[s];
    if (!r) return;
    const k = r.frameAtT(t === 1000 ? 999 : t === 0 ? -1 : t - 1);
    drawPane($(`cmp-x0-${i}`), r.pixels(2, k), r.C, r.H, r.W, S.render, LIME);
    drawPane($(`cmp-xt-${i}`), r.pixels(0, k), r.C, r.H, r.W, S.render, BONE);
    $(`cmp-cap-${i}`).textContent = `step ${k} / ${r.n - 1} · t = ${r.t[k] < 0 ? 0 : r.t[k] + 1}`;
  });
}

async function load() {
  const my = ++token;
  const seed = S.cmpSeed ?? seedsFor(S.model)[0];
  const rs = await Promise.all(ORDER.map(s => (runId(S.model, s, seed) ? getRun(runId(S.model, s, seed)) : null)));
  if (my !== token) return;
  runs = {};
  ORDER.forEach((s, i) => { if (rs[i]) runs[s] = rs[i]; });
  const calls = { ddpm: 1000, ddim: 50, dpmpp: 20, euler: 20 };
  $('cmp-table').querySelector('tbody').innerHTML = ORDER.filter(s => runs[s]).map(s => {
    const m = runs[s].meta;
    return `<tr><td>${SCHEDS[s]}</td><td class="num">${calls[s]}</td><td class="num">${m.seconds} s</td><td>${m.device}</td></tr>`;
  }).join('');
  let note = 'Wall times are from the recording run on the device shown; the first run on a GPU includes warm-up, so treat them as rough.';
  if (stats && S.model === 'cifar10') {
    const rows = stats.rows.map(r => `${r.steps} steps: DDIM ${r.ddim.toFixed(4)}, DPM-Solver++ ${r.dpmpp.toFixed(4)}${r.euler == null ? '' : `, Euler-Karras ${r.euler.toFixed(4)}`}`).join(' · ');
    note += ` Distance to a 1000-step DDIM reference (RMSE in [−1,1] pixel units, mean over ${stats.n_seeds} seeds): ${rows}.`;
  }
  $('cmp-note').textContent = note;
  draw();
}

function renderSeeds() {
  const seeds = seedsFor(S.model);
  if (!seeds.includes(S.cmpSeed)) S.cmpSeed = seeds[0];
  chips($('cmp-seed-chips'), seeds.map(s => [s, `seed ${s}`]), S.cmpSeed, v => { S.cmpSeed = +v; load(); });
}

export async function initCompare() {
  const host = $('cmp');
  host.innerHTML = ORDER.map((s, i) => `<div class="col"><h3>${SCHEDS[s]}</h3><canvas id="cmp-x0-${i}" aria-label="Predicted clean image, ${SCHEDS[s]}"></canvas><canvas id="cmp-xt-${i}" aria-label="Noisy sample, ${SCHEDS[s]}"></canvas><div class="cap" id="cmp-cap-${i}"></div></div>`).join('');
  stats = await getJSON(`${S.base}convergence.json`);
  $('cmp-range').oninput = draw;
  window.addEventListener('resize', () => requestAnimationFrame(draw));
  on('render', draw);
  on('model', () => { renderSeeds(); load(); });
  renderSeeds();
  await load();
}
