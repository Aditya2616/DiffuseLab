// Linear vs cosine noise schedules, computed here from math.js.
import { makeSchedule } from './math.js';
import { drawPlot, seriesLegend } from './plots.js';
import { fmtAbar, fmtSnr } from './run.js';
import { bindChips } from './state.js';

const $ = id => document.getElementById(id);
let curve = 'abar';

function render() {
  const lin = makeSchedule('linear'), cos = makeSchedule('cosine'), t = +$('sched-t').value;
  const pick = s => ({ abar: s.abar, beta: s.betas, snr: s.logSnr })[curve];
  const series = [
    { y: pick(lin), color: 'rgba(237,236,230,.85)', label: 'linear (Ho et al. 2020)' },
    { y: pick(cos), color: 'rgba(215,255,63,.95)', label: 'cosine (Nichol & Dhariwal 2021)' },
  ];
  seriesLegend($('sched-legend'), series);
  drawPlot($('plot-sched'), series, {
    playT: t, yMin: curve === 'snr' ? undefined : 0, yMax: curve === 'abar' ? 1 : undefined,
    yTicks: curve === 'snr' ? [-12, -8, -4, 0, 4, 8] : undefined, fmtY: curve === 'beta' ? v => v.toFixed(2) : undefined,
  });
  $('sched-probe').textContent = `t = ${t + 1}`;
  const half = s => s.abar.findIndex(v => v < 0.5) + 1;
  const row = (label, f) => `<dt>${label}</dt><dd>${f(lin)}</dd><dd>${f(cos)}</dd>`;
  $('sched-facts').innerHTML =
    `<dt></dt><dd class="dim">linear</dd><dd class="dim">cosine</dd>` +
    row(`ᾱ<sub>${t + 1}</sub>`, s => fmtAbar(s.abar[t])) +
    row('SNR', s => fmtSnr(s.snr[t])) +
    row('signal : noise', s => `${Math.sqrt(s.abar[t]).toFixed(2)} : ${Math.sqrt(1 - s.abar[t]).toFixed(2)}`) +
    row('t where ᾱ &lt; ½', s => half(s)) +
    row('final ᾱ<sub>1000</sub>', s => s.abar[999].toExponential(1).replace('e-', 'e−'));
}

export function initSchedules() {
  bindChips($('sched-curve-chips'), v => { curve = v; render(); });
  $('sched-t').oninput = render;
  window.addEventListener('resize', () => requestAnimationFrame(render));
  render();
}
