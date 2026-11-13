// Training results: loss curves, run facts, sample grids, and the memorisation check. All numbers come from
// docs/data/training.json, written by scripts/build_training_report.py from the real runs.
import { fitCanvas } from './render.js';
import { S, SCHED_NAMES, chips, getJSON } from './state.js';

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

// FID and KID bars from fid.json: bar = mean over the sampling seeds, whisker = one standard deviation, plus the
// real-vs-real floor. Lower is better; comparable inside this repo only.
function fidBars(host, fid, kinds) {
  const fl = fid.floor, cells = Object.keys(fid.results[kinds[0]] || {});
  const metric = (m) => {
    const sc = m === 'fid' ? 1 : 1000, unit = m === 'fid' ? 'FID' : 'KID ×1000';
    const get = c => ({ v: c[`${m}_mean`] * sc, sd: (c[`${m}_std`] ?? 0) * sc });
    const all = [get(fl), ...kinds.flatMap(k => cells.map(c => get(fid.results[k][c])))];
    const max = Math.max(...all.map(a => a.v + a.sd)) * 1.05;
    const row = (label, a, cls) => `<div class="fb-row"><span class="fb-l">${label}</span><span class="fb-track"><i class="fb-bar ${cls}" style="width:${(100 * Math.max(a.v, 0)) / max}%"></i><b class="fb-sd" style="left:${(100 * Math.max(a.v - a.sd, 0)) / max}%;width:${(100 * 2 * a.sd) / max}%"></b></span><span class="fb-v">${a.v.toFixed(m === 'fid' ? 2 : 2)} ± ${a.sd.toFixed(2)}</span></div>`;
    const groups = cells.map(c => { const [s, n] = c.split('-'); return `<div class="fb-group"><div class="fb-h">${SCHED_NAMES[s]} · ${n} steps</div>${kinds.map(k => row(`${k}`, get(fid.results[k][c]), k)).join('')}</div>`; }).join('');
    return `<div class="fb-metric"><h4>${unit}</h4>${groups}<div class="fb-group"><div class="fb-h">real training images (floor)</div>${row('real', get(fl), 'floor')}</div></div>`;
  };
  host.innerHTML = `<h3 class="fb-title">Linear versus cosine: FID and KID</h3><div class="fb">${metric('fid')}${metric('kid')}</div>
    <p class="note">${f0(fid.meta.n_generated)} generated against ${f0(fid.meta.n_real)} Fashion-MNIST test images, mean and standard deviation over ${fid.meta.sampling_seeds.length} sampling seeds (a fresh set of samples per seed; the models are fixed, so this spread is sampling noise only, not retraining noise). Lower is better. The floor is the same metric for real images against the test set. Inception-v3 features (torch-fidelity ${fid.meta.torch_fidelity}) on small greyscale images replicated to RGB and resized, so compare numbers inside this repo only.</p>`;
}
const f0 = v => Number(v).toLocaleString('en');

export async function initTraining() {
  const host = $('train-body');
  const rep = await getJSON(`${S.base}training.json`);
  if (!rep) { host.innerHTML = '<p class="wide dim">No training report is bundled with this build. Run <span class="mono">diffuselab train</span> and <span class="mono">python scripts/build_training_report.py</span>.</p>'; return; }
  const fid = await getJSON(`${S.base}fid.json`);
  const kinds = Object.keys(rep.runs);
  const f = (v, d = 0) => Number(v).toLocaleString('en', { maximumFractionDigits: d });
  const r0 = rep.runs[kinds[0]];
  const rows = kinds.map(k => {
    const r = rep.runs[k], nn = rep.nn_check?.[k];
    return `<tr><td>${k}</td><td class="num">${f(r.steps)}</td><td class="num">${f(r.seconds / 60, 1)} min</td><td class="num">${r.final_epoch_loss.toFixed(4)}</td>` +
      (nn ? `<td class="num">${nn.generated_to_train_median.toFixed(2)}</td><td class="num">${nn.heldout_to_train_median.toFixed(2)}</td>` : '<td></td><td></td>') + '</tr>';
  }).join('');
  host.innerHTML = `
    <p class="wide">A ${f(r0.params / 1e6, 1)}M-parameter UNet (<span class="mono">diffusers.UNet2DModel</span>) trained from scratch with the plain DDPM objective on ${f(r0.train_images)} Fashion-MNIST images padded to 32×32, once per schedule with identical settings: batch ${r0.batch}, AdamW ${r0.lr}, EMA ${r0.ema_decay}, ${r0.epochs_run} epochs, bf16 autocast, on ${r0.gpu || r0.device} with torch ${r0.torch}. ${f(rep.holdout)} images were held out.</p>
    <div class="panel">
      <canvas id="loss-cv" class="plot tall" aria-label="Training loss, linear and cosine"></canvas>
      <div class="legend">${kinds.map(k => `<span class="key"><i style="background:${COLORS[k]}"></i>${k} schedule</span>`).join('')}<span class="key dim">y: mean ε-prediction MSE per 50 steps (log)</span></div>
      <div class="table-wrap"><table><thead><tr><th>schedule</th><th>steps</th><th>wall time</th><th>final epoch loss</th><th>NN dist. generated</th><th>NN dist. held-out real</th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="note">${rep.notes.join(' ')} The two losses are not a quality comparison: the noise-prediction loss is weighted differently under different schedules, so a higher cosine loss does not mean worse samples. Each wall time is one run on ${r0.gpu || r0.device} (the linear run had the GPU mostly to itself, the cosine run had it alone; the CPU was shared with other jobs), so treat the minutes as rough.</p>
      <div id="fid-box"></div>
      <div class="group" id="grid-chips" role="radiogroup" aria-label="Sampler for the sample grids"></div>
      <div class="grids" id="grids"></div>
      <p class="note" id="grid-note"></p>
    </div>`;
  // sampler keys look like "ddim-50" in training.json; the label is built from them
  const first = Object.keys(rep.samplers?.[kinds[0]] || {});
  const names = Object.fromEntries(first.map(key => { const [s, n] = key.split('-'); return [key, `${SCHED_NAMES[s]} · ${n}`]; }));
  const showGrids = key => {
    const [sch, st] = key.split('-');
    $('grids').innerHTML = kinds.map(k => `<figure><img src="${S.base}grid_${k}_${sch}${st}.png" alt="64 samples, ${k} schedule, ${names[key]}" loading="lazy"><figcaption>${k} schedule · ${names[key]} · seed ${rep.grid_seed}. Not cherry-picked.</figcaption></figure>`).join('');
    $('grid-note').textContent = kinds.map(k => { const s = rep.samplers?.[k]?.[key]; return s ? `${k}: median nearest-training-image distance ${s.generated_to_train_median.toFixed(2)}` : ''; }).filter(Boolean).join(' · ') + `  (held-out real images: ${rep.nn_check[kinds[0]].heldout_to_train_median.toFixed(2)})`;
  };
  if (rep.samplers) {
    chips($('grid-chips'), Object.keys(names).map(k => [k, names[k]]), first[0], showGrids);
    showGrids(first[0]);
  } else $('grids').innerHTML = kinds.map(k => `<figure><img src="${S.base}grid_${k}.png" alt="64 samples, ${k} schedule" loading="lazy"></figure>`).join('');
  if (fid) fidBars($('fid-box'), fid, kinds);
  const draw = () => drawLoss($('loss-cv'), rep);
  draw();
  window.addEventListener('resize', () => requestAnimationFrame(draw));
}
