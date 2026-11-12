// Training results: loss curves, run facts, sample grids, and the memorisation check. All numbers come from
// docs/data/training.json, written by scripts/build_training_report.py from the real runs.
import { fitCanvas } from './render.js';
import { S, chips, getJSON } from './state.js';

const $ = id => document.getElementById(id);
const COLORS = { linear: 'rgba(237,236,230,.9)', cosine: 'rgba(215,255,63,.95)' };

function drawLoss(cv, rep) {
  const { w, h, dpr } = fitCanvas(cv), cx = cv.getContext('2d');
  const padL = 46 * dpr, padR = 10 * dpr, padT = 12 * dpr, padB = 24 * dpr, iw = w - padL - padR, ih = h - padT - padB;
  cx.clearRect(0, 0, w, h);
  const kinds = Object.keys(rep.runs);
  let hi = 0, lo = Infinity, n = 0;
  kinds.forEach(k => { const L = rep.runs[k].loss_every_50_steps; n = Math.max(n, L.length); L.forEach(v => { hi = Math.max(hi, v); lo = Math.min(lo, v); }); });
  lo = Math.log10(lo * 0.9); hi = Math.log10(hi * 1.1);
  cx.font = `${9 * dpr}px "JetBrains Mono", monospace`; cx.fillStyle = 'rgba(237,236,230,.5)'; cx.strokeStyle = 'rgba(237,236,230,.09)'; cx.textAlign = 'right'; cx.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4, y = padT + (1 - i / 4) * ih;
    cx.beginPath(); cx.moveTo(padL, y); cx.lineTo(w - padR, y); cx.stroke();
    cx.fillText((10 ** v).toFixed(3), padL - 6 * dpr, y);
  }
  cx.textAlign = 'center'; cx.textBaseline = 'top';
  const steps = rep.runs[kinds[0]].steps;
  [0, 0.5, 1].forEach(f => { cx.textAlign = f === 1 ? 'right' : f === 0 ? 'left' : 'center'; cx.fillText((f ? `${(f * steps / 1000).toFixed(1)}k steps` : '0'), padL + f * iw, padT + ih + 7 * dpr); });
  kinds.forEach(k => {
    const L = rep.runs[k].loss_every_50_steps;
    cx.beginPath(); cx.strokeStyle = COLORS[k]; cx.lineWidth = 1.6 * dpr;
    L.forEach((v, i) => { const x = padL + (i / (n - 1)) * iw, y = padT + (1 - (Math.log10(v) - lo) / (hi - lo)) * ih; i ? cx.lineTo(x, y) : cx.moveTo(x, y); });
    cx.stroke();
  });
}

export async function initTraining() {
  const host = $('train-body');
  const rep = await getJSON(`${S.base}training.json`);
  if (!rep) { host.innerHTML = '<p class="wide dim">No training report is bundled with this build. Run <span class="mono">diffuselab train</span> and <span class="mono">python scripts/build_training_report.py</span>.</p>'; return; }
  const kinds = Object.keys(rep.runs);
  const f = (v, d = 0) => Number(v).toLocaleString('en', { maximumFractionDigits: d });
  const r0 = rep.runs[kinds[0]];
  const rows = kinds.map(k => {
    const r = rep.runs[k], nn = rep.nn_check?.[k];
    return `<tr><td>${k}</td><td class="num">${f(r.steps)}</td><td class="num">${f(r.seconds / 60, 1)} min</td><td class="num">${r.final_epoch_loss.toFixed(4)}</td>` +
      (nn ? `<td class="num">${nn.generated_to_train_median.toFixed(2)}</td><td class="num">${nn.heldout_to_train_median.toFixed(2)}</td>` : '<td></td><td></td>') + '</tr>';
  }).join('');
  host.innerHTML = `
    <p class="wide">A ${f(r0.params / 1e6, 1)}M-parameter UNet (<span class="mono">diffusers.UNet2DModel</span>) trained from scratch with the plain DDPM objective on ${f(r0.train_images)} Fashion-MNIST images padded to 32×32, once per schedule with identical settings: batch ${r0.batch}, AdamW ${r0.lr}, EMA ${r0.ema_decay}, ${r0.epochs_run} epochs, bf16 autocast, on ${r0.gpu || r0.device}. 5,000 images were held out.</p>
    <div class="panel">
      <canvas id="loss-cv" class="plot tall" aria-label="Training loss, linear and cosine"></canvas>
      <div class="legend">${kinds.map(k => `<span class="key"><i style="background:${COLORS[k]}"></i>${k} schedule</span>`).join('')}<span class="key dim">y: mean ε-prediction MSE per 50 steps (log)</span></div>
      <div class="table-wrap"><table><thead><tr><th>schedule</th><th>steps</th><th>wall time</th><th>final epoch loss</th><th>NN dist. generated</th><th>NN dist. held-out real</th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="note">${rep.notes.join(' ')} The two losses are not a quality comparison: the noise-prediction loss is weighted differently under different schedules, so a higher cosine loss does not mean worse samples. Wall times were measured on a shared GPU (the linear run had it almost to itself, the cosine run did not), so compare the settings, not the minutes.</p>
      <div class="group" id="grid-chips" role="radiogroup" aria-label="Sampler for the sample grids"></div>
      <div class="grids" id="grids"></div>
      <p class="note" id="grid-note"></p>
    </div>`;
  const names = { 'ddim-50': 'DDIM · 50', 'dpmpp-20': 'DPM-Solver++ · 20', 'euler-20': 'Euler Karras · 20', 'ddpm-1000': 'DDPM · 1000' };
  const showGrids = key => {
    const [sch, st] = key.split('-');
    $('grids').innerHTML = kinds.map(k => `<figure><img src="${S.base}grid_${k}_${sch}${st}.png" alt="64 samples, ${k} schedule, ${names[key]}" loading="lazy"><figcaption>${k} schedule · ${names[key]} · seed ${rep.grid_seed}. Not cherry-picked.</figcaption></figure>`).join('');
    $('grid-note').textContent = kinds.map(k => { const s = rep.samplers?.[k]?.[key]; return s ? `${k}: median nearest-training-image distance ${s.generated_to_train_median.toFixed(2)}` : ''; }).filter(Boolean).join(' · ') + `  (held-out real images: ${rep.nn_check[kinds[0]].heldout_to_train_median.toFixed(2)})`;
  };
  if (rep.samplers) {
    chips($('grid-chips'), Object.keys(names).map(k => [k, names[k]]), 'ddim-50', showGrids);
    showGrids('ddim-50');
  } else $('grids').innerHTML = kinds.map(k => `<figure><img src="${S.base}grid_${k}.png" alt="64 samples, ${k} schedule" loading="lazy"></figure>`).join('');
  const draw = () => drawLoss($('loss-cv'), rep);
  draw();
  window.addEventListener('resize', () => requestAnimationFrame(draw));
}
