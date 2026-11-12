import { initCompare } from './compare.js';
import { initForward } from './forward.js';
import { initHero } from './hero.js';
import { EQUATIONS } from './maths.js';
import { detectLive } from './run.js';
import { initSchedules } from './schedules.js';
import { S, emit, getJSON } from './state.js';
import { initToy } from './toy.js';
import { initTraining } from './train.js';

const $ = id => document.getElementById(id);

function renderMaths() {
  $('eqs').innerHTML = EQUATIONS.map(e => `<article class="eq"><h3>${e.title}</h3><div class="tex" data-tex="${encodeURIComponent(e.tex)}"></div><p>${e.on}</p></article>`).join('');
  const go = () => document.querySelectorAll('.tex').forEach(el => {
    const tex = decodeURIComponent(el.dataset.tex);
    try { window.katex.render(tex, el, { displayMode: true, throwOnError: false }); } catch { el.textContent = tex; }
  });
  if (window.katex) go(); else { document.querySelectorAll('.tex').forEach(el => { el.textContent = decodeURIComponent(el.dataset.tex); }); window.addEventListener('load', () => window.katex && go()); }
}

function reveals() {
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.08 });
  document.querySelectorAll('.block > header, .block > .panel, .block > .wide, .block > .cols, .eq').forEach(el => { el.classList.add('reveal'); io.observe(el); });
}

async function boot() {
  const badge = $('mode-badge');
  S.index = (await getJSON(`${S.base}index.json`)) || [];
  S.live = await detectLive();
  badge.textContent = S.live ? `live · ${S.live.device}` : 'recorded run';
  badge.className = 'badge ' + (S.live ? 'live' : 'rec');
  badge.title = S.live ? 'Connected to the local server: "sample fresh" runs the model now.' : 'Replaying runs recorded by the Python code in the repo. No server involved.';
  if (S.live) $('live-btn').hidden = false;
  if (!S.index.length) { $('run-note').innerHTML = '<span class="err">No recorded runs found in data/. Run `diffuselab record`.</span>'; }
  renderMaths();
  initSchedules();
  initToy();
  if (S.index.length) { initHero(); await initCompare(); initForward(); }
  initTraining();
  reveals();
  emit('ready');
  window.__ready = true;
}
boot();
