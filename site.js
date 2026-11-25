// Companion page of the MTGP-VAE paper: skeleton players, latent knobs, outlier, quiz, per-sign recall.
// Data: window.DATA (data.js, built by make_data.py). Plotly.js only, no model runs in the browser.
"use strict";
const D = window.DATA;
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const GROUPS = ["body", "left_hand", "right_hand", "face"];
const GVAR = { body: "--body", left_hand: "--left-hand", right_hand: "--right-hand", face: "--face" };
const WIDTH = { body: 2, left_hand: 2, right_hand: 2, face: 1.2 };
const FPS = 10, HOLD = 6;
const EDGES = { body: [], left_hand: [], right_hand: [], face: [] };
for (const [p, es] of Object.entries(D.edges)) EDGES[D.skel_part[p]].push(...es);

function segs(f, edges) {
  const x = [], y = [];
  for (const [a, b] of edges) { x.push(f[2 * a], f[2 * b], null); y.push(f[2 * a + 1], f[2 * b + 1], null); }
  return [x, y];
}
function bounds(clips, pad = 0.05) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const c of clips) for (const f of c) for (let i = 0; i < f.length; i += 2) {
    x0 = Math.min(x0, f[i]); x1 = Math.max(x1, f[i]); y0 = Math.min(y0, f[i + 1]); y1 = Math.max(y1, f[i + 1]);
  }
  const px = pad * (x1 - x0), py = pad * (y1 - y0);
  return [x0 - px, x1 + px, y0 - py, y1 + py];
}
const baseLayout = () => ({
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", showlegend: false,
  font: { family: "system-ui, sans-serif", size: 13, color: css("--muted") }, margin: { l: 4, r: 4, t: 28, b: 4 },
  hovermode: false,
});
const CONFIG = { displayModeBar: false, responsive: true, staticPlot: false };

// ------------------------------------------------------------------ skeleton player
// panels: [{title, frames}], shown at the same relative time.
// opt: ncols, group {title, idx}, trail, highlight(i, k) -> colour,
// maxCell: max height of a panel (px), view: [x0, x1, y0, y1] (shared view), inset {l, r}: the time slider on its own
// row, its ends aligned with a chart below of those margins
function player(root, panels, opt = {}) {
  const n = panels.length, ncols = opt.ncols || n, nrows = Math.ceil(n / ncols);
  const nOut = Math.max(...panels.map((p) => p.frames.length));
  const [x0, x1, y0, y1] = opt.view || bounds(panels.map((p) => p.frames));
  const plot = document.createElement("div"); plot.className = "plot";
  const controls = document.createElement("div"); controls.className = "controls";
  const btn = document.createElement("button"); btn.textContent = "Pause";
  const slider = Object.assign(document.createElement("input"), { type: "range", min: 0, max: nOut - 1, value: 0 });
  slider.setAttribute("aria-label", "time");
  const label = document.createElement("span"); label.style.minWidth = "84px";
  if (opt.inset) {  // slider on its own row, thumb centres at the chart's plot-area edges (thumb radius 7px)
    controls.append(btn, label);
    const row = document.createElement("div"); row.className = "slider-row";
    row.style.margin = `4px ${opt.inset.r - 7}px 0 ${opt.inset.l - 7}px`;
    row.append(slider);
    root.append(plot, controls, row);
  } else {
    controls.append(btn, slider, label);
    root.append(plot, controls);
  }

  const cellW = (root.clientWidth || 900) / ncols, titleH = 26;
  const cellH = Math.min(cellW * (y1 - y0) / (x1 - x0), opt.maxCell || 340);
  const height = nrows * (cellH + titleH);
  const layout = baseLayout(); layout.height = height; layout.margin.t = 22;
  layout.annotations = []; layout.shapes = [];
  const traces = [];
  panels.forEach((p, i) => {
    const r = Math.floor(i / ncols), c = i % ncols, ax = i ? i + 1 : "";
    const gx = 0.01, gy = 0.03;
    layout[`xaxis${ax}`] = { domain: [c / ncols + gx, (c + 1) / ncols - gx], range: [x0, x1], visible: false, fixedrange: true };
    layout[`yaxis${ax}`] = { domain: [1 - (r + 1) / nrows + gy, 1 - r / nrows - gy], range: [y1, y0], visible: false,
                             fixedrange: true, scaleanchor: `x${ax}`, constrain: "domain" };
    if (p.title) layout.annotations.push({ text: p.title, xref: `x${ax} domain`, yref: `y${ax} domain`, x: 0.5, y: 1,
                                           yanchor: "bottom", showarrow: false, font: { size: 14, color: css("--ink") } });
    const xa = `x${ax}`, ya = `y${ax}`;
    traces.push({ x: [], y: [], xaxis: xa, yaxis: ya, mode: "lines", line: { color: css("--right-hand"), width: 3 },
                  opacity: 0.3 });  // trail of the right wrist
    for (const g of GROUPS) traces.push({ x: [], y: [], xaxis: xa, yaxis: ya, mode: "lines",
                                          line: { color: css(GVAR[g]), width: WIDTH[g] } });
  });
  if (opt.group) {
    const ax = opt.group.idx.map((i) => layout[`xaxis${i ? i + 1 : ""}`].domain), i0 = opt.group.idx[0];
    layout.annotations.push({ text: opt.group.title, xref: "paper", yref: `y${i0 ? i0 + 1 : ""} domain`,
                              x: (ax[0][0] + ax[ax.length - 1][1]) / 2, y: 1, yanchor: "bottom",
                              showarrow: false, font: { size: 14, color: css("--ink") } });
  }
  Plotly.newPlot(plot, traces, layout, CONFIG);

  let t = 0, tick = 0, playing = true, timer = null;
  const listeners = [];
  const per = 1 + GROUPS.length;
  function draw(tt) {
    t = tt; slider.value = tt; label.textContent = `frame ${tt + 1} / ${nOut}`;
    const xs = [], ys = [], cols = [];
    panels.forEach((p, i) => {
      const k = Math.round(tt / Math.max(nOut - 1, 1) * (p.frames.length - 1)), f = p.frames[k];
      const w = opt.trail === false ? [] : p.frames.slice(0, k + 1).map((g) => [g[2 * D.rw], g[2 * D.rw + 1]]);
      xs.push(w.map((q) => q[0])); ys.push(w.map((q) => q[1])); cols.push(css("--right-hand"));
      const hl = opt.highlight ? opt.highlight(i, k) : null;
      for (const g of GROUPS) {
        const [sx, sy] = segs(f, EDGES[g]); xs.push(sx); ys.push(sy);
        cols.push(hl && g === hl.group ? hl.color : css(GVAR[g]));
      }
    });
    Plotly.restyle(plot, { x: xs, y: ys, "line.color": cols });
    listeners.forEach((cb) => cb(tt));
  }
  function step() {
    tick += 1;
    if (tick > nOut - 1 + HOLD) tick = 0;
    draw(Math.min(tick, nOut - 1));
  }
  function play() { playing = true; btn.textContent = "Pause"; if (!timer) timer = setInterval(step, 1000 / FPS); }
  function pause() { playing = false; btn.textContent = "Play"; clearInterval(timer); timer = null; }
  btn.onclick = () => (playing ? pause() : play());
  slider.oninput = () => { pause(); tick = +slider.value; draw(tick); };
  new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting && playing) { if (!timer) timer = setInterval(step, 1000 / FPS); }
    else { clearInterval(timer); timer = null; }
  })).observe(root);
  draw(0);
  return {
    plot, nOut,
    setFrame(k) { pause(); tick = k; draw(k); },
    onFrame(cb) { listeners.push(cb); },
    setBoxes(boxes) {  // [{i, color, width, dash}]
      const shapes = boxes.map(({ i, color, width, dash }) => ({
        type: "rect", xref: `x${i ? i + 1 : ""} domain`, yref: `y${i ? i + 1 : ""} domain`, x0: 0, x1: 1, y0: 0, y1: 1,
        line: { color, width: width || 3, dash: dash || "solid" } }));
      Plotly.relayout(plot, { shapes });
    },
    redraw() { draw(t); },
  };
}

// ------------------------------------------------------------------ section 1: the three models
function models() {
  const root = document.getElementById("models");
  const P = D.players, names = { vae: "VAE", gp_vae: "GP-VAE", mtgp_vae: "MTGP-VAE" };
  const second = { vae: "reconstruction", gp_vae: "reconstruction", mtgp_vae: "mean process" };
  const caption = {
    vae: "The VAE reconstructs each frame well, but its prior has no notion of time: sampled frames are independent poses.",
    gp_vae: "The GP-VAE reconstructs the clip on a finer time grid (48 points for 32 frames), smoothly. Its samples move smoothly, but not like any sign.",
    mtgp_vae: "The MTGP-VAE's mean process is the sign as the model learned it; samples add a clip-specific deviation to it.",
  };
  const tabs = document.createElement("div"); tabs.className = "tabs"; tabs.setAttribute("role", "tablist");
  const holder = document.createElement("div"); holder.className = "player";
  const cap = document.createElement("figcaption");
  root.append(tabs, holder, cap);
  const view = bounds([P.real], 0.3);  // the same scale for the three models
  function show(m) {
    for (const b of tabs.children) b.setAttribute("aria-selected", b.dataset.m === m);
    holder.replaceChildren();
    player(holder, [{ title: "real", frames: P.real }, { title: second[m], frames: P[m].second },
                    { title: "", frames: P[m].samples[0] }, { title: "", frames: P[m].samples[1] }],
           { group: { title: "samples", idx: [2, 3] }, view });
    cap.textContent = caption[m];
  }
  for (const m of Object.keys(names)) {
    const b = document.createElement("button"); b.textContent = names[m]; b.dataset.m = m; b.setAttribute("role", "tab");
    b.onclick = () => show(m); tabs.append(b);
  }
  show(root.dataset.start || "vae");
}

// ------------------------------------------------------------------ section 2: latent knobs
function knobs() {
  const K = D.knobs, root = document.getElementById("knobs");
  const d = K.deltas.length, step = K.offsets[1] - K.offsets[0], lo = K.offsets[0], hi = K.offsets[K.offsets.length - 1];
  const vals = new Array(d).fill(0);
  const panel = document.createElement("div"); panel.className = "knob-panel";
  const stage = document.createElement("div"); stage.className = "player";
  root.append(stage, panel);
  const partName = { body: "body", left_hand: "left hand", right_hand: "right hand", face: "face" };
  const inputs = [];
  for (const [part, [a, b]] of Object.entries(K.parts)) {
    const h = document.createElement("h4"); h.textContent = partName[part]; h.style.setProperty("--c", css(GVAR[part]));
    panel.append(h);
    for (let k = a; k < b; k++) {
      const row = document.createElement("div"); row.className = "knob"; row.style.setProperty("--c", css(GVAR[part]));
      const lab = document.createElement("label"); lab.textContent = `z${k + 1}`;
      const inp = Object.assign(document.createElement("input"), { type: "range", min: lo, max: hi, step: 0.1, value: 0 });
      inp.setAttribute("aria-label", `latent dimension ${k + 1} (${partName[part]}), in standard deviations`);
      lab.htmlFor = inp.id = `z${k}`;
      const out = document.createElement("output"); out.textContent = "0σ";
      inp.oninput = () => { vals[k] = +inp.value; out.textContent = `${(+inp.value).toFixed(1)}σ`;
                            row.classList.toggle("moved", +inp.value !== 0); draw(); };
      inp.ondblclick = () => { inp.value = 0; inp.oninput(); };
      row.append(lab, inp, out); panel.append(row); inputs.push(inp);
    }
  }
  const reset = document.createElement("button"); reset.textContent = "Reset all knobs";
  reset.onclick = () => inputs.forEach((i) => { i.value = 0; i.oninput(); });
  panel.prepend(reset);

  function pose() {
    const f = K.base.slice();
    vals.forEach((v, k) => {
      if (!v) return;
      const pos = (v - lo) / step, i = Math.min(Math.floor(pos), K.offsets.length - 2), w = pos - i;
      const a = K.deltas[k][i], b = K.deltas[k][i + 1];
      for (let j = 0; j < f.length; j++) f[j] += a[j] * (1 - w) + b[j] * w;
    });
    return f;
  }
  // view: the base pose and every knob at +-2 sigma
  const i2 = K.offsets.indexOf(-2), j2 = K.offsets.indexOf(2);
  const extremes = [K.base];
  K.deltas.forEach((dk) => [i2, j2].forEach((o) => extremes.push(K.base.map((v, j) => v + dk[o][j]))));
  const [x0, x1, y0, y1] = bounds([extremes], 0.03);
  const plot = document.createElement("div"); stage.append(plot);
  const layout = baseLayout(); layout.margin.t = 8;
  layout.height = Math.min(560, (stage.clientWidth || 600) * (y1 - y0) / (x1 - x0));
  layout.xaxis = { range: [x0, x1], visible: false, fixedrange: true };
  layout.yaxis = { range: [y1, y0], visible: false, fixedrange: true, scaleanchor: "x", constrain: "domain" };
  const traces = GROUPS.map((g) => { const [x, y] = segs(K.base, EDGES[g]);
    return { x, y, mode: "lines", line: { color: css("--line"), width: WIDTH[g] + 1 } }; });  // mean pose, faint
  GROUPS.forEach((g) => traces.push({ x: [], y: [], mode: "lines", line: { color: css(GVAR[g]), width: WIDTH[g] + 0.5 } }));
  Plotly.newPlot(plot, traces, layout, CONFIG);
  function draw() {
    const f = pose(), xs = [], ys = [];
    for (const g of GROUPS) { const [x, y] = segs(f, EDGES[g]); xs.push(x); ys.push(y); }
    Plotly.restyle(plot, { x: xs, y: ys }, GROUPS.map((_, i) => GROUPS.length + i));
  }
  draw();
}

// ------------------------------------------------------------------ section 3: classification
function classification() {
  const R = D.results;
  const rows = [["Sign-VAE", "sign_vae"], ["Sign-GP-VAE", "sign_gp_vae"],
                ["Sign-GP-VAE + MTGP post-hoc", "sign_gp_vae_posthoc"], ["MTGP-VAE", "mtgp_vae"]];
  const tb = document.querySelector("#table tbody");
  for (const [label, k] of rows) {
    const tr = document.createElement("tr"); if (k === "mtgp_vae") tr.className = "ours";
    tr.innerHTML = `<td>${label}</td><td>${(100 * R[k].balanced_top1).toFixed(1)}</td><td>${(100 * R[k].balanced_top5).toFixed(1)}</td>`;
    tb.append(tr);
  }
  const a = R.sign_vae, b = R.mtgp_vae;
  const better = b.recall_top1.filter((v, i) => v > a.recall_top1[i]).length;
  const worse = b.recall_top1.filter((v, i) => v < a.recall_top1[i]).length;
  document.getElementById("recall-count").textContent =
    `The MTGP-VAE recognises ${better} of the ${b.signs.length} signs better than the Sign-VAE, ${worse} worse.`;
  const layout = baseLayout(); layout.height = 520; layout.hovermode = "closest"; layout.margin = { l: 64, r: 16, t: 12, b: 54 };
  const axis = (t) => ({ title: { text: t }, range: [0, 1.02], tickformat: ".0%", gridcolor: css("--line"),
                         zeroline: false, fixedrange: true, linecolor: css("--line") });
  layout.xaxis = axis("Sign-VAE: recall per sign (higher is better) →");
  layout.yaxis = axis("MTGP-VAE: recall per sign (higher is better) →");
  layout.shapes = [{ type: "line", x0: 0, y0: 0, x1: 1, y1: 1, line: { color: css("--muted"), width: 1, dash: "dot" } }];
  layout.annotations = [{ x: 0.2, y: 0.92, text: "MTGP-VAE better", showarrow: false, font: { color: css("--muted") } },
                        { x: 0.82, y: 0.08, text: "Sign-VAE better", showarrow: false, font: { color: css("--muted") } }];
  layout.hoverlabel = { bgcolor: css("--surface"), bordercolor: css("--line"), font: { color: css("--ink") } };
  Plotly.newPlot("recall", [{
    x: a.recall_top1, y: b.recall_top1, text: b.signs, mode: "markers", type: "scatter",
    marker: { size: 9, color: css("--body"), opacity: 0.8, line: { color: css("--surface"), width: 2 } },
    hovertemplate: "<b>%{text}</b><br>Sign-VAE %{x:.0%}<br>MTGP-VAE %{y:.0%}<extra></extra>",
  }], layout, CONFIG);
}

// ------------------------------------------------------------------ section 4: outlier
function outlier() {
  const O = D.outlier, root = document.getElementById("outlier");
  const corrected = O.frames.map((f, k) => (k === O.frame ? O.corrected : f));
  const holder = document.createElement("div"); holder.className = "player"; root.append(holder);
  const M = { l: 96, r: 16 };
  const p = player(holder, [{ title: "observed", frames: O.frames }, { title: "corrected", frames: corrected }],
                   { trail: false, maxCell: 300, inset: M,
                     highlight: (i, k) => (i === 1 && k === O.frame ? { group: O.hand, color: css("--accent") } : null) });
  const chart = document.createElement("div"); holder.append(chart);
  const layout = baseLayout(); layout.height = 220; layout.hovermode = "closest";
  layout.margin = { l: M.l, r: M.r, t: 10, b: 46 };
  layout.xaxis = { title: { text: "frame" }, gridcolor: css("--line"), zeroline: false, fixedrange: true,
                   range: [1, O.ll.length], dtick: 5, tick0: 0 };
  layout.yaxis = { title: { text: "log-likelihood<br>(lower = less likely)", standoff: 12 }, gridcolor: css("--line"),
                   zeroline: false, fixedrange: true, nticks: 4 };
  layout.hoverlabel = { bgcolor: css("--surface"), bordercolor: css("--line"), font: { color: css("--ink") } };
  const frames = O.ll.map((_, k) => k + 1);
  // outlier zone: below a threshold halfway between the outlier and the least likely other frame (indicative)
  const lo = Math.min(...O.ll), others = O.ll.filter((_, k) => k !== O.frame);
  const thr = (Math.min(...others) + O.ll[O.frame]) / 2, ymin = lo - 0.08 * (Math.max(...O.ll) - lo);
  layout.yaxis.range = [ymin, Math.max(...O.ll) + 0.05 * (Math.max(...O.ll) - lo)];
  layout.annotations = [{ xref: "paper", x: 1, xanchor: "right", y: (ymin + thr) / 2, text: "outliers", showarrow: false,
                          font: { color: css("--critical") } }];
  const zone = { type: "rect", xref: "paper", x0: 0, x1: 1, y0: ymin, y1: thr, layer: "below", line: { width: 0 },
                 fillcolor: css("--critical"), opacity: 0.12 };
  const cursor = (k) => [zone, { type: "line", x0: k + 1, x1: k + 1, yref: "paper", y0: 0, y1: 1,
                                 line: { color: css("--muted"), width: 1 } }];
  layout.shapes = cursor(0);
  Plotly.newPlot(chart, [{
    x: frames, y: O.ll, mode: "lines+markers", line: { color: css("--body"), width: 2 }, cliponaxis: false,
    marker: { size: 8, color: css("--body"), line: { color: css("--surface"), width: 2 } },
    hovertemplate: "frame %{x}: %{y:.0f}<extra></extra>",
  }], layout, CONFIG);
  p.onFrame((k) => Plotly.relayout(chart, { shapes: cursor(k) }));
  chart.on("plotly_click", (e) => p.setFrame(e.points[0].x - 1));
}

// ------------------------------------------------------------------ section 5: quiz
function quiz() {
  const Q = D.quiz, root = document.getElementById("quiz");
  const holder = document.createElement("div"); holder.className = "player"; root.append(holder);
  const p = player(holder, Q.clips.map((c, i) => ({ title: String(i + 1), frames: c })), { ncols: 3, maxCell: 240 });
  const bar = document.createElement("div"); bar.className = "quiz-buttons";
  const reveal = document.createElement("button"); reveal.textContent = "Reveal";
  const verdict = Object.assign(document.createElement("span"), { className: "verdict" });
  reveal.onclick = () => {
    reveal.disabled = true;
    p.setBoxes(Q.generated.map((g, i) => g && { i, color: css("--accent") }).filter(Boolean));
    verdict.textContent = "Framed in green: generated by the MTGP-VAE.";
  };
  bar.append(reveal, verdict);
  holder.append(bar);
}

models(); knobs(); classification(); outlier(); quiz();
