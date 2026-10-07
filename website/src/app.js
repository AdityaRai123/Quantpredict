(() => {
"use strict";

// =====================================================================================
// CONSTANTS
// =====================================================================================
const MODELS = ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost",
  "SVM", "KNN", "LSTM", "Vision Transformer"];
const MODEL_NOTES = {
  "Logistic Regression": "Linear baseline on the 37 features",
  "Random Forest": "200 trees, depth 10",
  "Gradient Boosting": "Boosted shallow trees",
  "XGBoost": "Regularised gradient boosting",
  "SVM": "RBF kernel with probability output",
  "KNN": "Votes of the nearest past days",
  "LSTM": "Recurrent net over 20-day windows",
  "Vision Transformer": "Reads each 20 × 37 window as a patched image",
};
const MAJ7 = "Majority Voting (7 = 6 classical + LSTM)";
const WP7 = "Weighted Voting - probabilities (7 models)";
const AP7 = "Adaptive Weighted Voting - probabilities (7 models)";
const ST7 = "Stacking (7 = 6 classical + LSTM)";
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = {
  ov: null, stocks: [], bySym: {}, cache: new Map(), current: null, data: null,
  tab: "overview", kind: "all", metric: "accuracy", daily: AP7, cmName: null,
  heatSort: "symbol", industry: "All", query: "", rendered: new Set(),
};

// =====================================================================================
// SMALL HELPERS
// =====================================================================================
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : (v * 100).toFixed(d) + "%");
const num = (v, d = 3) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const signed = (v, d = 1) => (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(d);
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
const quantile = (a, q) => { const s = [...a].sort((x, y) => x - y); const p = (s.length - 1) * q; const b = Math.floor(p); return s[b] + (s[Math.min(b + 1, s.length - 1)] - s[b]) * (p - b); };
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const fmtDate = (s) => { const [y, m, d] = s.split("-").map(Number); return `${d} ${MONTHS[m - 1]} ${y}`; };
const fmtMonth = (s) => { const [y, m] = s.split("-").map(Number); return `${MONTHS[m - 1]} ${y}`; };
const fmtInt = (n) => Number(n).toLocaleString("en-IN");
const fmtBytes = (b) => (b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : Math.round(b / 1e3) + " KB");
const axisPct = (v) => (Math.round(v * 1000) / 10) + "%";
const industryName = (s) => (s || "Other").replace(/\bIt\b/, "IT");

function shortName(n) {
  if (MODELS.includes(n)) return n;
  let m = n.match(/^(.+) \[(MI|Tree) Top-(\d+)\]$/);
  if (m) return `${m[1]} · ${m[2]} Top-${m[3]}`;
  const size = (n.match(/\((\d)/) || [])[1];
  const tail = size ? ` · ${size}` : "";
  if (/^Majority/.test(n)) return "Majority vote" + tail;
  if (/^Soft/.test(n)) return "Soft vote" + tail;
  if (/^Stacking/.test(n)) return "Stacking" + tail;
  m = n.match(/^(Adaptive Weighted|Weighted) Voting - (labels|probabilities)/);
  if (m) return `${m[1] === "Weighted" ? "Weighted" : "Adaptive"} (${m[2] === "labels" ? "labels" : "probs"})` + tail;
  return n;
}

function tokens() {
  const cs = getComputedStyle(document.documentElement);
  const g = (n) => cs.getPropertyValue(n).trim();
  const t = {};
  ["bg", "surface", "surface-2", "ink", "ink-2", "muted", "line", "line-2", "accent", "accent-ink",
   "up", "down", "band", "div-neg", "div-mid", "div-pos", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
    .forEach((k) => { t[k.replace(/-(\w)/g, (_, c) => c.toUpperCase())] = g("--" + k); });
  t.font = getComputedStyle(document.body).fontFamily;
  t.mono = g("--f-mono");
  return t;
}
const modelColor = (T, name) => T["s" + (MODELS.indexOf(name) + 1)] || T.ink2;

function hexA(hex, a) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

// =====================================================================================
// CHART PLUMBING (ECharts, theme-aware, lazily mounted)
// =====================================================================================
const charts = new Map();
const ro = typeof ResizeObserver !== "undefined"
  ? new ResizeObserver((entries) => entries.forEach((e) => { const r = charts.get(e.target.id); if (r) r.inst.resize(); }))
  : null;

function base(T) {
  return {
    animation: !REDUCED, animationDuration: 500,
    textStyle: { fontFamily: T.font, color: T.ink2 },
    tooltip: {
      backgroundColor: T.surface, borderColor: T.line2, borderWidth: 1, padding: [8, 10],
      textStyle: { color: T.ink, fontFamily: T.font, fontSize: 12.5 },
      extraCssText: "box-shadow:0 8px 24px -12px rgba(0,0,0,.35);border-radius:8px;",
    },
  };
}
function axis(T, extra = {}) {
  return Object.assign({
    axisLine: { lineStyle: { color: T.line2 } }, axisTick: { show: false },
    axisLabel: { color: T.muted, fontSize: 11.5, fontFamily: T.font },
    splitLine: { lineStyle: { color: T.line } }, nameTextStyle: { color: T.muted, fontSize: 11.5 },
  }, extra);
}

function mount(id, build) {
  const el = $(id);
  if (!el || typeof echarts === "undefined") return null;
  let rec = charts.get(id);
  if (!rec) {
    rec = { inst: echarts.init(el, null, { renderer: "canvas" }), build };
    charts.set(id, rec);
    if (ro) ro.observe(el);
  }
  rec.build = build;
  rec.inst.off("click");
  rec.inst.setOption(build(tokens()), { notMerge: true });
  return rec.inst;
}
function rethemeAll() {
  const T = tokens();
  charts.forEach((rec) => { rec.inst.setOption(rec.build(T), { notMerge: true }); });
}

// =====================================================================================
// TABLES
// =====================================================================================
function table(el, cols, rows, opts = {}) {
  let sortKey = opts.sortKey || null, dir = opts.dir || -1;
  const draw = () => {
    const data = [...rows];
    if (sortKey) {
      const c = cols.find((x) => x.key === sortKey);
      const val = c.sortVal || ((r) => r[sortKey]);
      data.sort((a, b) => { const x = val(a), y = val(b); return (x > y ? 1 : x < y ? -1 : 0) * dir; });
    }
    const head = cols.map((c) => `<th class="${c.num ? "num " : ""}${opts.sortable !== false ? "sortable" : ""}" data-k="${c.key}" scope="col">${esc(c.label)}${sortKey === c.key ? `<span class="arrow">${dir < 0 ? "↓" : "↑"}</span>` : ""}</th>`).join("");
    const body = data.map((r) => `<tr class="${opts.rowClass ? opts.rowClass(r) : ""}"${opts.rowAttr ? " " + opts.rowAttr(r) : ""}>${cols.map((c) => `<td class="${c.num ? "num" : ""}">${c.render ? c.render(r) : esc(r[c.key] ?? "–")}</td>`).join("")}</tr>`).join("");
    el.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    if (opts.sortable !== false) {
      el.querySelectorAll("th.sortable").forEach((th) => th.addEventListener("click", () => {
        const k = th.dataset.k; if (sortKey === k) dir = -dir; else { sortKey = k; dir = -1; } draw();
      }));
    }
    if (opts.onRow) el.querySelectorAll("tbody tr").forEach((tr, i) => tr.addEventListener("click", () => opts.onRow(data[i])));
  };
  draw();
}
const barCell = (v, lo = 0.42, hi = 0.58) => {
  const T = { accent: "var(--accent)", down: "var(--down)" };
  const clamp = (x) => Math.max(0, Math.min(1, (x - lo) / (hi - lo)));
  const a = clamp(0.5), b = clamp(v);
  const left = Math.min(a, b) * 100, width = Math.abs(b - a) * 100;
  return `<span class="bar-cell">${pct(v)}<span class="track"><span class="mid"></span><span class="fill" style="left:${left}%;width:${Math.max(width, 1)}%;background:${v >= 0.5 ? T.accent : T.down}"></span></span></span>`;
};
const winCell = (k, n) => `<span class="bar-cell"><span class="${k / n > 0.5 ? "win" : ""}">${k} / ${n}</span><span class="track"><span class="mid"></span><span class="fill" style="left:0;width:${(k / n) * 100}%;background:${k / n > 0.5 ? "var(--accent)" : "var(--muted)"}"></span></span></span>`;

// =====================================================================================
// DOWNLOADS (Artifact viewer capability first, plain link otherwise)
// =====================================================================================
const inViewer = !!(window.claude && typeof window.claude.use === "function");
const downloadsCap = inViewer ? window.claude.use("downloads").catch(() => null) : Promise.resolve(null);
const DL = {
  stock: (sym) => ({ zip: `downloads/stocks/${sym}.zip`, bundle: `bundles/${sym}.json`, name: `${sym}_quantpredict.zip`, sym }),
  all: () => ({ zip: "downloads/quantpredict_all_results.zip", bundle: "bundles/quantpredict_all_results.json", name: "quantpredict_all_results.zip" }),
};

// Inside the claude.ai viewer, .zip files cannot be hosted: fetch the JSON bundle, build the
// ZIP here with JSZip and hand it to the viewer's save prompt. Anywhere else: a plain link.
async function download(item) {
  const say = (m) => { $("dlStatus").textContent = m; $("dlStatusTop").textContent = m; };
  const cap = await downloadsCap;
  if (!cap) {
    if (inViewer) { say("Saving files is not available in this view. Open the page on claude.ai to download."); return; }
    const a = document.createElement("a");
    a.href = item.zip; a.download = item.name; a.rel = "noopener";
    document.body.appendChild(a); a.click(); a.remove();
    say(`Downloading ${item.name}…`);
    return;
  }
  try {
    if (typeof JSZip === "undefined") throw { code: "nozip" };
    say(`Preparing ${item.name}…`);
    const res = await fetch(item.bundle);
    if (!res.ok) throw { code: "fetch", message: res.status };
    const bundle = await res.json();
    const zip = new JSZip();
    bundle.files.forEach((f) => (f.b64 != null ? zip.file(f.path, f.b64, { base64: true }) : zip.file(f.path, f.text)));
    if (item.sym) {
      const d = await fetchStock(item.sym);
      zip.file(`${item.sym}/data/summary.json`, JSON.stringify(d.summary, null, 1));
      zip.file(`${item.sym}/data/series.json`, JSON.stringify(d.series));
    }
    const blob = await zip.generateAsync({ type: "blob", compression: "DEFLATE", compressionOptions: { level: 6 } });
    say(`${item.name} is ready (${fmtBytes(blob.size)}). Confirm the save prompt to download it.`);
    await cap.save({ filename: item.name, data: blob });
    say(`Saved ${item.name}.`);
  } catch (e) {
    if (e && e.code === "declined") say("Download cancelled.");
    else if (e && e.code === "rate_limited") say("A save prompt is already open. Finish that one first.");
    else if (e && e.code === "fetch") say(`Could not load the files for ${item.name} (HTTP ${e.message}). Try again.`);
    else if (e && e.code === "nozip") say("The ZIP builder did not load. Check your connection and reload the page.");
    else say(`The download did not complete${e && e.code ? ` (${e.code})` : ""}. Try again.`);
  }
}

// =====================================================================================
// OVERVIEW: hero, KPIs, method section, RQ1-4, heatmap
// =====================================================================================
function accOf(sym, name) { return state.ov.acc[sym][name]; }

function renderIntro() {
  const ov = state.ov, S = ov.stocks;
  const meanSingle = mean(S.map((s) => s.mean_individual_acc));
  const ens = ov.methods.filter((m) => m.kind === "Ensemble");
  const bestEns = ens[0];
  const ap = ov.methods.find((m) => m.name === AP7);
  const allPass = ov.checks_all_passed === ov.n_stocks;

  $("heroAnswer").innerHTML =
    `Across <b>${ov.n_stocks} stocks</b> and <b>${fmtInt(ov.test_days_total)} unseen test days</b>, the average single model scored <b>${pct(meanSingle)}</b> and the best voting rule on average, ${esc(shortName(bestEns.name))}, scored <b>${pct(bestEns.mean_acc)}</b>. ` +
    `Every method stays within a few points of a coin flip, and ${allPass ? "all" : ov.checks_all_passed + " of"} ${ov.n_stocks} runs passed all ${ov.n_checks} leakage checks.`;

  const y0 = ov.first_date.slice(0, 4), y1 = ov.last_date.slice(0, 4);
  const kpis = [
    [ov.n_stocks, "NIFTY-50 stocks, each run end to end"],
    [`${Number(y1) - Number(y0)} yrs`, `Daily NSE prices, ${fmtMonth(ov.first_date)} to ${fmtMonth(ov.last_date)}`],
    [fmtInt(ov.test_days_total), "Test days scored, never seen in training"],
    ["8 + 15", "Models and ensemble rules compared"],
    [ov.n_features, "Stationary technical features"],
    [`${ov.n_checks}/${ov.n_checks}`, allPass ? `Leakage checks passed on every stock` : `Checks passed on ${ov.checks_all_passed} stocks`],
  ];
  $("kpiRow").innerHTML = kpis.map(([v, l]) => `<div class="kpi"><dt>${esc(l)}</dt><dd>${esc(v)}</dd></div>`).join("");

  const caTotal = S.reduce((s, x) => s + x.corporate_actions, 0);
  const fill = { n_stocks: ov.n_stocks, ca_total: caTotal, ca_stocks: S.filter((s) => s.corporate_actions > 0).length,
    dl_backend: ov.dl_backend === "pytorch" ? "PyTorch" : "TensorFlow" };
  document.querySelectorAll("[data-fill]").forEach((el) => { const v = fill[el.dataset.fill]; if (v != null) el.textContent = v; });

  $("resultsLede").textContent =
    `Each stock has its own test period: the last 20% of its history, about ${fmtInt(Math.round(ov.test_days_total / ov.n_stocks))} trading days. ` +
    `On that many days, a coin flip lands anywhere within about ±${(ov.margin_range[1] * 100).toFixed(1)} points of 50% by luck alone. Keep that band in mind for every number below.`;

  $("genNote").textContent = `Results generated ${fmtDate(ov.generated)} by running Quantpredict.py on every stock.`;

  // model legend in the method section
  const T = tokens();
  $("modelList").innerHTML = MODELS.map((m) =>
    `<li><span class="dot" style="background:${modelColor(T, m)}"></span><span><b>${esc(m)}</b><span>${esc(MODEL_NOTES[m])}</span></span></li>`).join("");
}

function renderWalkForward() {
  // Proportions straight from make_walk_forward_folds(): initial 50% of training, 4 blocks of 12.5%.
  const W = 1000, rowH = 26, gap = 10, left = 118, right = 16, top = 30;
  const x = (f) => left + f * (W - left - right);
  const rows = [
    ...[0, 1, 2, 3].map((k) => ({ label: `Fold ${k + 1}`, train: [0, 0.4 + 0.1 * k], val: [0.4 + 0.1 * k, 0.5 + 0.1 * k] })),
    { label: "Final models", train: [0, 0.8], test: [0.8, 1] },
  ];
  const H = top + rows.length * (rowH + gap) + 34;
  let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
  s += `<rect x="${x(0.8)}" y="${top - 18}" width="${x(1) - x(0.8)}" height="${rows.length * (rowH + gap) + 18}" style="fill:var(--down-wash)" rx="6"></rect>`;
  s += `<text x="${x(0.4)}" y="${top - 6}" text-anchor="middle" style="fill:var(--muted);font:600 12px var(--f-mono);letter-spacing:.06em">TRAINING PERIOD (first 80%)</text>`;
  s += `<text x="${x(0.9)}" y="${top - 6}" text-anchor="middle" style="fill:var(--down-ink);font:600 12px var(--f-mono);letter-spacing:.06em">TEST (last 20%)</text>`;
  rows.forEach((r, i) => {
    const y = top + i * (rowH + gap);
    s += `<text x="${left - 12}" y="${y + rowH / 2 + 4}" text-anchor="end" style="fill:var(--ink-2);font:500 13px var(--f-body)">${r.label}</text>`;
    s += `<rect x="${x(0)}" y="${y}" width="${x(1) - x(0)}" height="${rowH}" rx="4" style="fill:var(--surface-2)"></rect>`;
    s += `<rect x="${x(r.train[0])}" y="${y}" width="${x(r.train[1]) - x(r.train[0]) - 2}" height="${rowH}" rx="4" style="fill:var(--accent-wash);stroke:var(--accent);stroke-width:1"></rect>`;
    s += `<text x="${x(r.train[0]) + 10}" y="${y + rowH / 2 + 4}" style="fill:var(--accent-ink);font:500 12px var(--f-body)">fit</text>`;
    if (r.val) {
      s += `<rect x="${x(r.val[0])}" y="${y}" width="${x(r.val[1]) - x(r.val[0])}" height="${rowH}" rx="4" style="fill:var(--accent)"></rect>`;
      s += `<text x="${(x(r.val[0]) + x(r.val[1])) / 2}" y="${y + rowH / 2 + 4}" text-anchor="middle" style="fill:#fff;font:600 12px var(--f-body)">validate</text>`;
    }
    if (r.test) {
      s += `<rect x="${x(r.test[0])}" y="${y}" width="${x(r.test[1]) - x(r.test[0])}" height="${rowH}" rx="4" style="fill:var(--down)"></rect>`;
      s += `<text x="${(x(r.test[0]) + x(r.test[1])) / 2}" y="${y + rowH / 2 + 4}" text-anchor="middle" style="fill:#fff;font:600 12px var(--f-body)">score once</text>`;
    }
  });
  const yb = top + rows.length * (rowH + gap) + 8;
  [0, 0.2, 0.4, 0.6, 0.8, 1].forEach((f) => {
    s += `<line x1="${x(f)}" x2="${x(f)}" y1="${yb - 4}" y2="${yb + 2}" style="stroke:var(--line-2)"></line>`;
    s += `<text x="${x(f)}" y="${yb + 16}" text-anchor="middle" style="fill:var(--muted);font:500 11px var(--f-mono)">${f * 100}%</text>`;
  });
  s += `</svg>`;
  $("wfDiagram").innerHTML = s;
}

function renderHero() {
  const ov = state.ov, S = ov.stocks;
  const narrow = $("heroChart").clientWidth < 460;
  const rows = [
    { label: narrow ? "Best single*" : "Best single model (hindsight)", get: (s) => s.best_individual_acc, muted: true },
    { label: narrow ? "Avg single" : "Average single model", get: (s) => s.mean_individual_acc },
    { label: narrow ? "Majority · 7" : "Majority vote · 7", get: (s) => accOf(s.symbol, MAJ7) },
    { label: narrow ? "Weighted · 7" : "Weighted (probs) · 7", get: (s) => accOf(s.symbol, WP7) },
    { label: narrow ? "Adaptive · 7" : "Adaptive (probs) · 7", get: (s) => accOf(s.symbol, AP7) },
    { label: "Stacking · 7", get: (s) => accOf(s.symbol, ST7) },
  ];
  const jitter = (sym, i) => { let h = 0; for (const c of sym + i) h = (h * 31 + c.charCodeAt(0)) % 9973; return (h / 9973 - 0.5) * 0.5; };
  const all = rows.flatMap((r) => S.map(r.get));
  const lo = Math.floor(Math.min(...all) * 100 - 1) / 100, hi = Math.ceil(Math.max(...all) * 100 + 1) / 100;
  const m = mean(S.map((s) => s.margin));
  mount("heroChart", (T) => ({
    ...base(T),
    grid: { left: 8, right: 16, top: 22, bottom: 8, containLabel: true },
    xAxis: axis(T, { type: "value", min: lo, max: hi, splitNumber: 6, axisLine: { onZero: false, lineStyle: { color: T.line2 } }, axisLabel: { color: T.muted, formatter: (v) => Math.round(v * 100) + "%" } }),
    yAxis: [
      axis(T, { type: "category", inverse: true, data: rows.map((r) => r.label), splitLine: { show: false }, axisLine: { show: false },
        axisLabel: { color: T.ink2, fontSize: 12 } }),
      { type: "value", inverse: true, min: -0.5, max: rows.length - 0.5, show: false },
    ],
    tooltip: { ...base(T).tooltip, trigger: "item",
      formatter: (p) => p.seriesName === "mean" ? `${esc(rows[p.value[1]].label)}<br><b>Mean ${pct(p.value[0])}</b>`
        : `<b>${esc(p.data.sym)}</b> · ${esc(rows[p.data.row].label)}<br>Accuracy <b>${pct(p.value[0])}</b>` },
    series: [
      { type: "scatter", name: "stocks", symbolSize: 7, yAxisIndex: 1,
        data: rows.flatMap((r, i) => S.map((s) => ({ value: [r.get(s), i + jitter(s.symbol, i)], sym: s.symbol, row: i,
          itemStyle: { color: r.muted ? hexA(T.muted, 0.55) : hexA(T.accent, 0.7) } }))),
        emphasis: { scale: 1.6, itemStyle: { borderColor: T.surface, borderWidth: 2 } },
        markArea: { silent: true, itemStyle: { color: T.band }, data: [[{ xAxis: 0.5 - m }, { xAxis: 0.5 + m }]] },
        markLine: { silent: true, symbol: "none", lineStyle: { color: T.ink2, width: 1, type: "solid" },
          label: { formatter: "coin flip", color: T.muted, fontSize: 11, position: "start", distance: 4 }, data: [{ xAxis: 0.5 }] } },
      { type: "scatter", name: "mean", symbol: "rect", symbolSize: [3, 22], z: 5, yAxisIndex: 1,
        itemStyle: { color: T.ink }, data: rows.map((r, i) => [mean(S.map(r.get)), i]) },
    ],
  }));
}

function renderFindings() {
  const ov = state.ov, S = ov.stocks, N = ov.n_stocks;
  const meanSingle = mean(S.map((s) => s.mean_individual_acc));
  const ensVs = [...ov.ensemble_vs].sort((a, b) => b.mean_acc - a.mean_acc);
  const best = ensVs[0];
  const ap = ov.ensemble_vs.find((e) => e.name === AP7);
  const above50 = ov.methods.filter((m) => m.kind === "Individual" && m.mean_acc > 0.5).length;
  const cards = [
    [pct(meanSingle), `Average accuracy of a single model across all ${N} stocks. A coin flip scores 50%; always guessing the majority direction averages ${pct(mean(S.map((s) => s.baseline)))}.`],
    [`${signed(best.delta_vs_mean_single)} pts`, `What the best voting rule on average (${shortName(best.name)}) added over the average single model.`],
    [`${ap.wins_vs_maj7} / ${N}`, `Stocks where adaptive weighted voting beat plain majority voting on the test period.`],
    [`${best.wins_vs_best_single} / ${N}`, `Stocks where ${shortName(best.name)} beat that stock’s best single model, chosen with hindsight.`],
  ];
  $("findings").innerHTML = cards.map(([big, txt]) => `<div class="finding card"><span class="big">${esc(big)}</span><p>${esc(txt)}</p></div>`).join("");
}

function renderRQ1() {
  const ov = state.ov, syms = Object.keys(ov.acc), N = syms.length;
  const stats = MODELS.map((m) => {
    const v = syms.map((s) => ov.acc[s][m]);
    const meta = ov.methods.find((x) => x.name === m);
    return { name: m, mean: mean(v), q1: quantile(v, 0.25), q3: quantile(v, 0.75), med: quantile(v, 0.5),
      beats: meta.beats_baseline, beats50: meta.beats_50, f1: meta.mean_f1, auc: meta.mean_auc };
  }).sort((a, b) => a.mean - b.mean);
  const baseline = mean(ov.stocks.map((s) => s.baseline));
  const top = stats[stats.length - 1];
  const totalBeats = stats.reduce((s, x) => s + x.beats, 0);
  const aboveBase = stats.filter((s) => s.mean > baseline).length, above50 = stats.filter((s) => s.mean > 0.5).length;
  $("rq1Title").textContent = aboveBase === 0 ? "Single models: above a coin flip, below the simplest baseline" : "Single models: a hair above chance, at best";
  $("rq1Answer").innerHTML =
    `<b>${aboveBase === 0 ? "Not on average." : aboveBase < 4 ? "Only a few, and barely." : "Barely."}</b> The strongest model on average is <b>${esc(top.name)}</b> at <b>${pct(top.mean)}</b>. ` +
    `${above50} of 8 models average above 50%, but always guessing a stock’s more common direction averages <b>${pct(baseline)}</b>, ` +
    `${aboveBase === 0 ? "higher than every model’s average" : `and only ${aboveBase} of 8 models clear that on average`}. ` +
    `Across all ${N * 8} stock-and-model pairs, a model beat its stock’s majority baseline ${totalBeats} times (${pct(totalBeats / (N * 8), 0)}).`;

  mount("rq1Chart", (T) => ({
    ...base(T),
    grid: { left: 8, right: 24, top: 26, bottom: 8, containLabel: true },
    xAxis: axis(T, { type: "value", scale: true, axisLabel: { color: T.muted, formatter: axisPct } }),
    yAxis: axis(T, { type: "category", data: stats.map((s) => s.name), axisLabel: { color: T.ink2, fontSize: 12 }, splitLine: { show: false }, axisLine: { show: false } }),
    tooltip: { ...base(T).tooltip, trigger: "item", formatter: (p) => { const s = stats[p.dataIndex]; return `<b>${esc(s.name)}</b><br>Mean ${pct(s.mean)} · median ${pct(s.med)}<br>Middle 50%: ${pct(s.q1)} – ${pct(s.q3)}<br>Beat majority baseline on ${s.beats}/${N} stocks`; } },
    series: [{
      type: "custom", name: "models",
      renderItem: (params, api) => {
        const i = params.dataIndex, s = stats[i];
        const y = api.coord([0, i])[1];
        const x1 = api.coord([s.q1, i])[0], x3 = api.coord([s.q3, i])[0], xm = api.coord([s.mean, i])[0];
        const c = modelColor(T, s.name);
        return { type: "group", children: [
          { type: "rect", shape: { x: x1, y: y - 4, width: Math.max(x3 - x1, 2), height: 8, r: 4 }, style: { fill: hexA(c, 0.28) } },
          { type: "circle", shape: { cx: xm, cy: y, r: 6.5 }, style: { fill: c, stroke: T.surface, lineWidth: 2 } },
          { type: "text", style: { x: xm + 10, y: y - 14, text: pct(s.mean), fill: T.ink2, font: `500 11px ${T.font}` } },
        ] };
      },
      encode: { x: [1, 2, 3], y: 0 },
      data: stats.map((s, i) => [i, s.q1, s.q3, s.mean]),
      markLine: { silent: true, symbol: "none", data: [
        { xAxis: 0.5, lineStyle: { color: T.ink2, type: "solid", width: 1 }, label: { formatter: "50%", color: T.muted, fontSize: 11 } },
        { xAxis: baseline, lineStyle: { color: T.muted, type: [4, 4], width: 1 }, label: { formatter: "baseline", color: T.muted, fontSize: 11 } },
      ] },
    }],
  }));
  table($("rq1Table"), [
    { key: "name", label: "Model", render: (r) => `<span class="swatch" style="background:var(--s${MODELS.indexOf(r.name) + 1})"></span>${esc(r.name)}` },
    { key: "mean", label: "Mean acc.", num: true, render: (r) => pct(r.mean) },
    { key: "med", label: "Median", num: true, render: (r) => pct(r.med) },
    { key: "q1", label: "Middle 50%", num: true, render: (r) => `${pct(r.q1)} – ${pct(r.q3)}` },
    { key: "beats", label: "Beat baseline", num: true, render: (r) => `${r.beats}/${N}` },
    { key: "beats50", label: "Above 50%", num: true, render: (r) => `${r.beats50}/${N}` },
    { key: "f1", label: "Mean F1", num: true, render: (r) => num(r.f1) },
    { key: "auc", label: "Mean AUC", num: true, render: (r) => num(r.auc) },
  ], [...stats].reverse(), { sortKey: "mean" });

  const wins = MODELS.map((m) => ({ name: m, n: ov.best_counts[m] })).sort((a, b) => a.n - b.n);
  mount("rq1Wins", (T) => ({
    ...base(T),
    grid: { left: 8, right: 36, top: 8, bottom: 28, containLabel: true },
    xAxis: axis(T, { type: "value", minInterval: 1, axisLabel: { color: T.muted } }),
    yAxis: axis(T, { type: "category", data: wins.map((w) => w.name), axisLabel: { color: T.ink2, fontSize: 12 }, splitLine: { show: false }, axisLine: { show: false } }),
    tooltip: { ...base(T).tooltip, trigger: "item", formatter: (p) => `<b>${esc(p.name)}</b><br>Best single model on ${p.value} of ${N} stocks` },
    series: [{ type: "bar", barWidth: 12, data: wins.map((w) => ({ value: w.n, itemStyle: { color: modelColor(T, w.name), borderRadius: [0, 4, 4, 0] } })),
      label: { show: true, position: "right", color: T.ink2, fontSize: 11.5, fontFamily: T.font } }],
  }));
}

function renderRQ2() {
  const ov = state.ov, N = ov.n_stocks, P = ov.vit_pairs;
  const vit = ov.methods.find((m) => m.name === "Vision Transformer");
  const ranked = ov.methods.filter((m) => m.kind === "Individual").sort((a, b) => b.mean_acc - a.mean_acc);
  const rank = ranked.findIndex((m) => m.name === "Vision Transformer") + 1;
  const w = P.reduce((s, p) => s + p.wins8, 0), n = P.reduce((s, p) => s + p.n, 0);
  const deltas = P.map((p) => p.mean_delta);
  $("rq2Answer").innerHTML =
    `<b>${Math.abs(mean(deltas)) < 0.003 ? "Not measurably." : mean(deltas) > 0 ? "Slightly." : "No."}</b> On its own the ViT averages <b>${pct(vit.mean_acc)}</b>, ranking ${rank} of 8 single models. ` +
    `Adding it to an ensemble moved mean accuracy by ${signed(Math.min(...deltas), 2)} to ${signed(Math.max(...deltas), 2)} points depending on the rule, and the 8-model version won ${w} of ${n} stock-by-rule comparisons (${pct(w / n, 0)}).`;
  const labels = P.map((p) => p.family === "Weighted" || p.family === "Adaptive" ? `${p.family} (${p.rule === "labels" ? "labels" : "probs"})` : p.family);
  const vals = P.flatMap((p) => [p.mean7, p.mean8]);
  mount("rq2Chart", (T) => ({
    ...base(T),
    legend: { top: 0, right: 0, textStyle: { color: T.ink2, fontSize: 12 }, itemWidth: 12, itemHeight: 12, data: ["7 models (no ViT)", "8 models (with ViT)"] },
    grid: { left: 8, right: 24, top: 32, bottom: 28, containLabel: true },
    xAxis: axis(T, { type: "value", min: Math.floor(Math.min(...vals) * 200 - 1) / 200, max: Math.ceil(Math.max(...vals) * 200 + 1) / 200, axisLabel: { color: T.muted, formatter: (v) => (v * 100).toFixed(1) + "%" } }),
    yAxis: axis(T, { type: "category", data: labels, inverse: true, axisLabel: { color: T.ink2, fontSize: 12 }, splitLine: { show: false }, axisLine: { show: false } }),
    tooltip: { ...base(T).tooltip, trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: T.band } },
      formatter: (ps) => { const p = P[ps[0].dataIndex]; return `<b>${esc(labels[ps[0].dataIndex])}</b><br>7 models: ${pct(p.mean7, 2)}<br>8 models: ${pct(p.mean8, 2)}<br>ViT version higher on ${p.wins8}/${p.n} stocks`; } },
    series: [
      { type: "custom", silent: true, renderItem: (params, api) => {
          const i = params.dataIndex, p = P[i];
          const a = api.coord([p.mean7, i]), b = api.coord([p.mean8, i]);
          return { type: "line", shape: { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, style: { stroke: T.line2, lineWidth: 2 } };
        }, data: P.map((p, i) => [i]), encode: { y: 0 } },
      { type: "scatter", name: "7 models (no ViT)", symbolSize: 12, itemStyle: { color: T.ink2, borderColor: T.surface, borderWidth: 2 }, data: P.map((p) => p.mean7), z: 3 },
      { type: "scatter", name: "8 models (with ViT)", symbolSize: 12, itemStyle: { color: modelColor(T, "Vision Transformer"), borderColor: T.surface, borderWidth: 2 }, data: P.map((p) => p.mean8), z: 4 },
    ],
  }));
  table($("rq2Table"), [
    { key: "label", label: "Rule" },
    { key: "mean7", label: "7 models", num: true, render: (r) => pct(r.mean7, 2) },
    { key: "mean8", label: "8 with ViT", num: true, render: (r) => pct(r.mean8, 2) },
    { key: "mean_delta", label: "Change", num: true, render: (r) => `${signed(r.mean_delta, 2)} pts` },
    { key: "wins8", label: "ViT version higher", num: true, render: (r) => winCell(r.wins8, r.n) },
  ], P.map((p, i) => ({ ...p, label: labels[i] })), { sortable: false });
}

function renderRQ3() {
  const ov = state.ov, N = ov.n_stocks;
  const E = [...ov.ensemble_vs].sort((a, b) => a.mean_acc - b.mean_acc);
  const meanSingle = mean(ov.stocks.map((s) => s.mean_individual_acc));
  const best = E[E.length - 1];
  const fixed7 = ov.ensemble_vs.filter((e) => e.size === 7);
  const minMean = Math.min(...fixed7.map((e) => e.wins_vs_mean_single)), maxMean = Math.max(...fixed7.map((e) => e.wins_vs_mean_single));
  $("rq3Answer").innerHTML =
    `<b>They beat the average model, not the best one.</b> The 7-model rules beat the average single model on ${minMean}–${maxMean} of ${N} stocks. ` +
    `But even the strongest rule on average, <b>${esc(shortName(best.name))}</b> (${pct(best.mean_acc)}), beat the hindsight-best single model on only ${best.wins_vs_best_single} of ${N}. ` +
    `Combining models smooths out a bad pick; it does not create signal that none of the models has.`;
  const lo = Math.min(...E.map((e) => e.mean_acc), meanSingle), hi = Math.max(...E.map((e) => e.mean_acc));
  mount("rq3Chart", (T) => ({
    ...base(T),
    legend: { bottom: 0, left: "center", textStyle: { color: T.ink2, fontSize: 12 }, itemWidth: 12, itemHeight: 12, data: ["6 or 7 models", "8 models (with ViT)"] },
    grid: { left: 8, right: 30, top: 26, bottom: 36, containLabel: true },
    xAxis: axis(T, { type: "value", min: Math.floor(lo * 200 - 1) / 200, max: Math.ceil(hi * 200 + 1) / 200, axisLabel: { color: T.muted, formatter: (v) => (v * 100).toFixed(1) + "%" } }),
    yAxis: axis(T, { type: "category", data: E.map((e) => shortName(e.name)), axisLabel: { color: T.ink2, fontSize: 12 }, splitLine: { show: false }, axisLine: { show: false } }),
    tooltip: { ...base(T).tooltip, trigger: "item", formatter: (p) => { const e = E[p.dataIndex]; return `<b>${esc(shortName(e.name))}</b><br>Mean accuracy ${pct(e.mean_acc, 2)}<br>vs majority · 7: ${signed(e.delta_vs_maj7, 2)} pts<br>vs average single: ${signed(e.delta_vs_mean_single, 2)} pts`; } },
    series: [
      { type: "scatter", name: "6 or 7 models", symbolSize: 11, itemStyle: { color: T.accent, borderColor: T.surface, borderWidth: 2 },
        data: E.map((e, i) => (e.size === 8 ? ["-", i] : [e.mean_acc, i])),
        markLine: { silent: true, symbol: "none", data: [{ xAxis: meanSingle, lineStyle: { color: T.muted, type: [4, 4] }, label: { formatter: "avg single model", color: T.muted, fontSize: 11 } }] } },
      { type: "scatter", name: "8 models (with ViT)", symbolSize: 11, itemStyle: { color: T.surface, borderColor: T.accent, borderWidth: 2 },
        data: E.map((e, i) => (e.size === 8 ? [e.mean_acc, i] : ["-", i])) },
    ],
  }));
  table($("rq3Table"), [
    { key: "name", label: "Rule", render: (r) => esc(shortName(r.name)) },
    { key: "wins_vs_maj7", label: "Majority · 7", num: true, render: (r) => (r.name === MAJ7 ? "–" : winCell(r.wins_vs_maj7, N)) },
    { key: "wins_vs_mean_single", label: "Avg single", num: true, render: (r) => winCell(r.wins_vs_mean_single, N) },
    { key: "wins_vs_best_single", label: "Best single", num: true, render: (r) => winCell(r.wins_vs_best_single, N) },
  ], [...E].reverse(), { sortKey: "wins_vs_mean_single" });
}

function renderRQ4() {
  const ov = state.ov, F = ov.feature_selection;
  const sets = Object.entries(ov.fs_best_set_counts).sort((a, b) => b[1] - a[1]);
  const gains = [];
  F.forEach((f) => { ["mi", "tree"].forEach((k) => { if (f[k] != null) gains.push({ model: f.model, set: k === "mi" ? "MI Top-20" : "Tree Top-20", d: f[k] - f[k + "_all"], wins: f[k + "_wins"], n: f[k + "_n"] }); }); });
  gains.sort((a, b) => b.d - a.d);
  const avgMI = mean(gains.filter((g) => g.set === "MI Top-20").map((g) => g.d)), avgTree = mean(gains.filter((g) => g.set === "Tree Top-20").map((g) => g.d));
  const g0 = gains[0], g1 = gains[gains.length - 1];
  $("rq4Answer").innerHTML =
    `<b>${Math.abs(avgMI) < 0.003 && Math.abs(avgTree) < 0.003 ? "Not on average." : avgMI > 0 || avgTree > 0 ? "A little, for some models." : "No."}</b> Averaged over models, MI Top-20 changed accuracy by ${signed(avgMI, 2)} points and Tree Top-20 by ${signed(avgTree, 2)}. ` +
    `The biggest gain was ${esc(g0.model)} with ${g0.set} (${signed(g0.d, 2)}), the biggest loss ${esc(g1.model)} with ${g1.set} (${signed(g1.d, 2)}). ` +
    `Walk-forward validation preferred ${sets.map(([k, v]) => `${k} on ${v} stocks`).join(", ")}.`;
  const rows = F.filter((f) => f.mi != null || f.tree != null);
  const vals = rows.flatMap((f) => [f.all, f.mi, f.tree]).filter((v) => v != null);
  mount("rq4Chart", (T) => ({
    ...base(T),
    legend: { top: 0, right: 0, textStyle: { color: T.ink2, fontSize: 12 }, itemWidth: 12, itemHeight: 12 },
    grid: { left: 8, right: 24, top: 32, bottom: 28, containLabel: true },
    xAxis: axis(T, { type: "value", min: Math.floor(Math.min(...vals) * 200 - 1) / 200, max: Math.ceil(Math.max(...vals) * 200 + 1) / 200, axisLabel: { color: T.muted, formatter: (v) => (v * 100).toFixed(1) + "%" } }),
    yAxis: axis(T, { type: "category", data: rows.map((f) => f.model), inverse: true, axisLabel: { color: T.ink2, fontSize: 12 }, splitLine: { show: false }, axisLine: { show: false } }),
    tooltip: { ...base(T).tooltip, trigger: "axis", axisPointer: { type: "shadow", shadowStyle: { color: T.band } },
      formatter: (ps) => { const f = rows[ps[0].dataIndex]; return `<b>${esc(f.model)}</b><br>All 37 features: ${pct(f.all, 2)}` + (f.mi != null ? `<br>MI Top-20: ${pct(f.mi, 2)} (higher on ${f.mi_wins}/${f.mi_n})` : "") + (f.tree != null ? `<br>Tree Top-20: ${pct(f.tree, 2)} (higher on ${f.tree_wins}/${f.tree_n})` : ""); } },
    series: [
      { type: "custom", silent: true, renderItem: (params, api) => {
          const i = params.dataIndex, f = rows[i]; const vs = [f.all, f.mi, f.tree].filter((v) => v != null);
          const a = api.coord([Math.min(...vs), i]), b = api.coord([Math.max(...vs), i]);
          return { type: "line", shape: { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, style: { stroke: T.line2, lineWidth: 2 } };
        }, data: rows.map((f, i) => [i]), encode: { y: 0 } },
      { type: "scatter", name: "All 37 features", symbol: "circle", symbolSize: 12, itemStyle: { color: T.surface, borderColor: T.ink, borderWidth: 2 }, data: rows.map((f) => f.all), z: 3 },
      { type: "scatter", name: "MI Top-20", symbol: "circle", symbolSize: 11, itemStyle: { color: T.accent, borderColor: T.surface, borderWidth: 1.5 }, data: rows.map((f) => f.mi), z: 4 },
      { type: "scatter", name: "Tree Top-20", symbol: "diamond", symbolSize: 13, itemStyle: { color: T.ink2, borderColor: T.surface, borderWidth: 1.5 }, data: rows.map((f) => f.tree), z: 4 },
    ],
  }));
  table($("rq4Table"), [
    { key: "model", label: "Model" },
    { key: "all", label: "All 37", num: true, render: (r) => pct(r.all, 2) },
    { key: "mi", label: "MI Top-20", num: true, render: (r) => (r.mi == null ? "–" : `${pct(r.mi, 2)} <span class="muted">(${r.mi_wins}/${r.mi_n})</span>`) },
    { key: "tree", label: "Tree Top-20", num: true, render: (r) => (r.tree == null ? "–" : `${pct(r.tree, 2)} <span class="muted">(${r.tree_wins}/${r.tree_n})</span>`) },
  ], rows, { sortable: false });
}

function renderHeat() {
  const ov = state.ov;
  const cols = [...MODELS, "", ...ov.ensembles.map((e) => e.name)];
  const order = [...ov.stocks];
  const key = state.heatSort;
  if (key === "symbol") order.sort((a, b) => a.symbol.localeCompare(b.symbol));
  if (key === "adaptive") order.sort((a, b) => accOf(b.symbol, AP7) - accOf(a.symbol, AP7));
  if (key === "best") order.sort((a, b) => b.best_individual_acc - a.best_individual_acc);
  if (key === "industry") order.sort((a, b) => industryName(a.industry).localeCompare(industryName(b.industry)) || a.symbol.localeCompare(b.symbol));
  const data = [];
  order.forEach((s, y) => cols.forEach((c, x) => { if (c) data.push([x, y, accOf(s.symbol, c)]); }));
  const h = order.length * 18 + 170;
  $("heatChart").style.height = h + "px";
  const inst = mount("heatChart", (T) => ({
    ...base(T),
    grid: { left: 8, right: 12, top: 56, bottom: 8, containLabel: true },
    xAxis: { type: "category", position: "top", data: cols.map((c) => (c ? shortName(c) : "")), axisTick: { show: false }, axisLine: { show: false }, splitArea: { show: false },
      axisLabel: { color: T.ink2, fontSize: 11, rotate: 40, interval: 0, fontFamily: T.font } },
    yAxis: { type: "category", data: order.map((s) => s.symbol), inverse: true, axisTick: { show: false }, axisLine: { show: false },
      axisLabel: { color: T.ink2, fontSize: 11, fontFamily: T.mono, interval: 0 } },
    visualMap: { type: "continuous", min: 0.44, max: 0.56, calculable: false, orient: "horizontal", left: "center", top: 0, itemWidth: 10, itemHeight: 180,
      text: ["56%", "44%"], textStyle: { color: T.muted, fontSize: 11 }, inRange: { color: [T.divNeg, T.divMid, T.divPos] }, show: true },
    tooltip: { ...base(T).tooltip, trigger: "item", formatter: (p) => { const s = order[p.value[1]]; return `<b>${esc(s.symbol)}</b> · ${esc(s.company)}<br>${esc(shortName(cols[p.value[0]]))}: <b>${pct(p.value[2])}</b><br><span style="color:${T.muted}">Majority baseline ${pct(s.baseline)} · click to open</span>`; } },
    series: [{ type: "heatmap", data, itemStyle: { borderColor: T.surface, borderWidth: 2, borderRadius: 2 }, emphasis: { itemStyle: { borderColor: T.ink, borderWidth: 1 } }, progressive: 0 }],
  }));
  if (inst) {
    inst.on("click", (p) => { const s = order[p.value[1]]; if (s) { selectStock(s.symbol); $("explorer").scrollIntoView({ behavior: REDUCED ? "auto" : "smooth" }); } });
    inst.resize();
  }
}

// =====================================================================================
// EXPLORER: board + per-stock panel
// =====================================================================================
function renderChips() {
  const inds = ["All", ...[...new Set(state.ov.stocks.map((s) => industryName(s.industry)))].sort()];
  $("industryChips").innerHTML = inds.map((i) => `<button type="button" class="chip${i === state.industry ? " is-on" : ""}" data-ind="${esc(i)}">${esc(i)}</button>`).join("");
  $("industryChips").querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => { state.industry = b.dataset.ind; renderChips(); renderBoard(); }));
}

function visibleStocks() {
  const q = state.query.trim().toLowerCase();
  return state.stocks.filter((s) => (state.industry === "All" || industryName(s.industry) === state.industry) &&
    (!q || s.symbol.toLowerCase().includes(q) || s.company.toLowerCase().includes(q)));
}

function renderBoard() {
  const list = visibleStocks();
  if (!list.length) { $("board").innerHTML = `<p class="board-empty">No stock matches “${esc(state.query)}”. Try a symbol like TCS or a name like Bank.</p>`; return; }
  $("board").innerHTML = list.map((s) => {
    const v = accOf(s.symbol, AP7);
    const span = 0.06, c = Math.max(-1, Math.min(1, (v - 0.5) / span));
    const left = c >= 0 ? 50 : 50 + c * 50, width = Math.abs(c) * 50;
    return `<button type="button" role="option" class="tile${s.symbol === state.current ? " is-on" : ""}" aria-selected="${s.symbol === state.current}" data-sym="${esc(s.symbol)}">
      <span class="sym">${esc(s.symbol)}</span><span class="co">${esc(s.company.replace(/ Ltd\.?$/, ""))}</span>
      <span class="meter"><span class="bar"><i style="left:${left}%;width:${Math.max(width, 2)}%;background:${v >= 0.5 ? "var(--div-pos)" : "var(--div-neg)"}"></i></span><b>${pct(v)}</b></span></button>`;
  }).join("");
  $("board").querySelectorAll(".tile").forEach((b) => b.addEventListener("click", () => selectStock(b.dataset.sym, true)));
  const on = $("board").querySelector(".tile.is-on");
  if (on && $("board").scrollWidth > $("board").clientWidth) $("board").scrollLeft = on.offsetLeft - $("board").clientWidth / 2 + on.clientWidth / 2;
}

async function fetchStock(sym) {
  if (state.cache.has(sym)) return state.cache.get(sym);
  const p = fetch(`data/stocks/${encodeURIComponent(sym)}.json`).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
  state.cache.set(sym, p);
  p.catch(() => state.cache.delete(sym));
  return p;
}

async function selectStock(sym, focusPanel) {
  if (!state.bySym[sym]) return;
  state.current = sym;
  renderBoard();
  $("dlStock").value = sym;
  const panel = $("stockPanel");
  panel.classList.add("is-loading");
  try { history.replaceState(null, "", "#stock-" + sym); } catch (e) { /* sandboxed history */ }
  let d;
  try { d = await fetchStock(sym); }
  catch (e) { $("sName").textContent = `Could not load ${sym}`; $("sMeta").textContent = "Check your connection and pick the stock again."; panel.classList.remove("is-loading"); return; }
  if (state.current !== sym) return;
  state.data = d; state.rendered = new Set(); state.cmName = d.summary.best.individual.name;
  renderStockHead();
  renderTab(state.tab);
  panel.classList.remove("is-loading");
  if (focusPanel && window.innerWidth < 900) panel.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "start" });
}

function renderStockHead() {
  const s = state.data.summary, o = state.bySym[s.symbol];
  $("sSym").textContent = `${s.symbol} · ${industryName(s.industry)}`;
  $("sName").textContent = s.company || s.symbol;
  $("sMeta").textContent = `${fmtDate(s.date_range.start)} – ${fmtDate(s.date_range.end)} · ${fmtInt(s.rows.clean)} trading days · test period ${fmtDate(s.test.start)} – ${fmtDate(s.test.end)} (${fmtInt(s.test.n)} days)`;
  const acc = Object.fromEntries(s.results.map((r) => [r.name, r]));
  const tiles = [
    ["Best single model", pct(s.best.individual.accuracy), s.best.individual.name + " · hindsight"],
    ["Best ensemble", pct(s.best.ensemble.accuracy), shortName(s.best.ensemble.name) + " · hindsight"],
    ["Adaptive vote · 7", pct(acc[AP7].accuracy), "Fixed rule"],
    ["Majority baseline", pct(s.majority_baseline), `Always guess ${s.test_up_rate >= 0.5 ? "UP" : "DOWN"}`],
    ["Chance band", "±" + (s.noise_margin * 100).toFixed(1) + " pts", "95% range around 50%"],
  ];
  $("sTiles").innerHTML = tiles.map(([k, v, sub]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd><span class="sub" title="${esc(sub)}">${esc(sub)}</span></div>`).join("");
}

function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll("#tabs [role=tab]").forEach((b) => {
    const on = b.id === "tab-" + tab;
    b.setAttribute("aria-selected", on); b.tabIndex = on ? 0 : -1;
    $(b.getAttribute("aria-controls")).hidden = !on;
  });
  if (state.data) renderTab(tab);
}

function renderTab(tab) {
  if (state.rendered.has(tab)) { charts.forEach((r) => r.inst.resize()); return; }
  state.rendered.add(tab);
  ({ overview: renderPrice, models: renderModels, voting: renderVoting, daily: renderDaily, features: renderFeatures,
     training: renderTraining, next: renderNext, checks: renderChecks }[tab])();
}

// ---- Price -------------------------------------------------------------------------
function renderPrice() {
  const { summary: s, series } = state.data, P = series.price;
  const testStart = s.test.start;
  const ca = s.corporate_actions;
  mount("priceChart", (T) => ({
    ...base(T),
    grid: { left: 8, right: 16, top: 16, bottom: 8, containLabel: true },
    xAxis: axis(T, { type: "category", data: P.date, boundaryGap: false, splitLine: { show: false }, axisLabel: { color: T.muted, formatter: (v) => v.slice(0, 4), interval: "auto", hideOverlap: true } }),
    yAxis: axis(T, { type: "value", scale: true, axisLabel: { color: T.muted, formatter: (v) => "₹" + fmtInt(v) } }),
    tooltip: { ...base(T).tooltip, trigger: "axis", axisPointer: { type: "line", lineStyle: { color: T.line2 } },
      formatter: (ps) => `${fmtDate(ps[0].axisValue)}<br>Close <b>₹${Number(ps[0].value).toLocaleString("en-IN", { maximumFractionDigits: 2 })}</b>${ps[0].axisValue >= testStart ? `<br><span style="color:${T.muted}">test period</span>` : ""}` },
    dataZoom: [{ type: "inside", throttle: 30, zoomOnMouseWheel: false, moveOnMouseWheel: false }],
    series: [{
      type: "line", data: P.close, showSymbol: false, sampling: "lttb", lineStyle: { color: T.accent, width: 1.6 },
      areaStyle: { color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: hexA(T.accent, 0.18) }, { offset: 1, color: hexA(T.accent, 0) }] } },
      markArea: { silent: true, itemStyle: { color: T.band }, label: { show: true, position: "insideTop", color: T.muted, fontSize: 11, formatter: "test period" },
        data: [[{ xAxis: P.date.find((d) => d >= testStart) }, { xAxis: P.date[P.date.length - 1] }]] },
      markLine: { symbol: "none", silent: false, lineStyle: { color: T.down, width: 1, type: "solid" },
        label: { formatter: (p) => `×${ca[p.dataIndex].factor}`, color: T.down, fontSize: 10.5, position: "insideEndTop" },
        tooltip: { formatter: (p) => `Corporate action ${fmtDate(ca[p.dataIndex].date)}<br>Overnight gap ${ca[p.dataIndex].gap} → factor ${ca[p.dataIndex].factor}` },
        data: ca.map((c) => ({ xAxis: P.date.find((d) => d >= c.date) })) },
    }],
  }));
  mount("volumeChart", (T) => ({
    ...base(T),
    grid: { left: 8, right: 16, top: 6, bottom: 40, containLabel: true },
    xAxis: axis(T, { type: "category", data: P.date, splitLine: { show: false }, axisLabel: { show: false } }),
    yAxis: axis(T, { type: "value", splitNumber: 2, axisLabel: { color: T.muted, formatter: (v) => (v >= 1e7 ? (v / 1e7).toFixed(0) + " cr" : v >= 1e5 ? (v / 1e5).toFixed(0) + " L" : fmtInt(v)) } }),
    tooltip: { ...base(T).tooltip, trigger: "axis", formatter: (ps) => `${fmtDate(ps[0].axisValue)}<br>Volume <b>${fmtInt(ps[0].value)}</b> shares (adjusted)` },
    dataZoom: [{ type: "inside", throttle: 30, zoomOnMouseWheel: false, moveOnMouseWheel: false }, { type: "slider", height: 20, bottom: 8, borderColor: T.line, backgroundColor: T.surface2, fillerColor: hexA(T.accent, 0.12),
      dataBackground: { lineStyle: { color: T.line2 }, areaStyle: { color: T.line } }, handleStyle: { color: T.surface, borderColor: T.line2 },
      moveHandleStyle: { color: T.line2 }, textStyle: { color: T.muted, fontSize: 10 }, labelFormatter: (i) => P.date[i] ? P.date[i].slice(0, 7) : "" }],
    series: [{ type: "bar", data: P.volume, large: true, itemStyle: { color: hexA(T.muted, 0.55) }, barCategoryGap: "10%" }],
  }));
  const pc = charts.get("priceChart"), vc = charts.get("volumeChart");
  if (pc && vc) { pc.inst.group = vc.inst.group = "pv"; echarts.connect("pv"); }

  const f = [
    ["Training period", `${fmtMonth(s.train.start)} – ${fmtMonth(s.train.end)}`, `${fmtInt(s.train.n)} days`],
    ["Test period", `${fmtMonth(s.test.start)} – ${fmtMonth(s.test.end)}`, `${fmtInt(s.test.n)} days`],
    ["UP days overall", pct(s.class_balance.up), `${pct(s.test_up_rate)} in the test period`],
    ["Corporate actions", ca.length ? ca.map((c) => `${fmtMonth(c.date)} ×${c.factor}`).join(", ") : "None detected", ca.length ? "Earlier prices rescaled" : "Prices used as traded"],
  ];
  $("splitFacts").innerHTML = f.map(([k, v, sub]) => `<div class="fact"><span>${esc(k)}</span><b>${esc(v)}</b><small class="muted">${esc(sub)}</small></div>`).join("");
}

// ---- Models ------------------------------------------------------------------------
function modelOfRow(name) { return MODELS.find((m) => name === m || name.startsWith(m + " [")); }

function renderModels() {
  const s = state.data.summary, metric = state.metric;
  const label = $("metricSel").selectedOptions[0].textContent;
  const rows = s.results.filter((r) => state.kind === "all" || r.kind === state.kind)
    .filter((r) => r[metric] != null).sort((a, b) => a[metric] - b[metric]);
  $("modelsTitle").textContent = `Test ${label.toLowerCase()} · ${state.kind === "all" ? "all rows" : state.kind === "Individual" ? "single models" : state.kind === "Ensemble" ? "ensembles" : "feature-selected variants"}`;
  const h = Math.max(260, rows.length * 22 + 70);
  $("modelsChart").style.height = h + "px";
  const vals = rows.map((r) => r[metric]);
  const lo = Math.min(0.5, ...vals), hi = Math.max(0.5, ...vals);
  const pad = (hi - lo) * 0.08 + 0.005;
  const inst = mount("modelsChart", (T) => ({
    ...base(T),
    grid: { left: 8, right: 56, top: 8, bottom: 28, containLabel: true },
    xAxis: axis(T, { type: "value", min: Math.max(0, lo - pad), max: Math.min(1, hi + pad), axisLabel: { color: T.muted, formatter: (v) => (v * 100).toFixed(0) + "%" } }),
    yAxis: axis(T, { type: "category", data: rows.map((r) => shortName(r.name)), axisLabel: { color: T.ink2, fontSize: 11.5, interval: 0 }, splitLine: { show: false }, axisLine: { show: false } }),
    tooltip: { ...base(T).tooltip, trigger: "item", formatter: (p) => { const r = rows[p.dataIndex]; return `<b>${esc(shortName(r.name))}</b> <span style="color:${T.muted}">${esc(r.kind)}</span><br>Accuracy ${pct(r.accuracy)} · F1 ${num(r.f1)}<br>Precision ${num(r.precision)} · Recall ${num(r.recall)}<br>ROC-AUC ${num(r.roc_auc)}`; } },
    series: [{
      type: "custom", name: "rows",
      renderItem: (params, api) => {
        const i = params.dataIndex, r = rows[i];
        const y = api.coord([0.5, i])[1], x0 = api.coord([0.5, i])[0], x1 = api.coord([r[metric], i])[0];
        const m = modelOfRow(r.name);
        const c = r.kind === "Ensemble" ? T.ink2 : modelColor(T, m);
        const sel = r.name === state.cmName;
        return { type: "group", children: [
          { type: "rect", shape: { x: Math.min(x0, x1), y: y - 6, width: Math.max(Math.abs(x1 - x0), 2), height: 12, r: 3 },
            style: { fill: r.kind === "Feature-selected" ? hexA(c, 0.45) : c, stroke: sel ? T.ink : null, lineWidth: sel ? 1.5 : 0 } },
          { type: "text", style: { x: x1 + (x1 >= x0 ? 6 : -6), y: y, text: pct(r[metric]), fill: sel ? T.ink : T.muted, font: `${sel ? 600 : 500} 11px ${T.font}`, align: x1 >= x0 ? "left" : "right", verticalAlign: "middle" } },
        ] };
      },
      encode: { x: 1, y: 0 }, data: rows.map((r, i) => [i, r[metric]]),
      markLine: { silent: true, symbol: "none", data: [{ xAxis: 0.5, lineStyle: { color: T.ink, width: 1, type: "solid" }, label: { formatter: "0.50", color: T.muted, fontSize: 10.5 } },
        ...(metric === "accuracy" ? [{ xAxis: s.majority_baseline, lineStyle: { color: T.muted, type: [4, 4] }, label: { formatter: "baseline", color: T.muted, fontSize: 10.5 } }] : [])] },
    }],
  }));
  if (inst) { inst.on("click", (p) => { if (rows[p.dataIndex]) { state.cmName = rows[p.dataIndex].name; renderConfusion(); renderModelsChartOnly(); } }); inst.resize(); }
  renderConfusion();
  const best = Math.max(...s.results.filter((r) => state.kind === "all" || r.kind === state.kind).map((r) => r.accuracy));
  table($("resultsTable"), [
    { key: "name", label: "Model / rule", render: (r) => esc(shortName(r.name)) },
    { key: "kind", label: "Type", render: (r) => `<span class="kind-pill">${esc(r.kind === "Individual" ? "Single" : r.kind === "Feature-selected" ? "Top-20" : "Ensemble")}</span>` },
    { key: "accuracy", label: "Accuracy", num: true, render: (r) => barCell(r.accuracy) },
    { key: "precision", label: "Precision", num: true, render: (r) => num(r.precision) },
    { key: "recall", label: "Recall", num: true, render: (r) => num(r.recall) },
    { key: "f1", label: "F1", num: true, render: (r) => num(r.f1) },
    { key: "roc_auc", label: "ROC-AUC", num: true, render: (r) => num(r.roc_auc) },
  ], s.results.filter((r) => state.kind === "all" || r.kind === state.kind), {
    sortKey: "accuracy", rowClass: (r) => `clickable${r.accuracy === best ? " is-best" : ""}`,
    onRow: (r) => { state.cmName = r.name; renderConfusion(); renderModelsChartOnly(); },
  });
}
function renderModelsChartOnly() { const r = charts.get("modelsChart"); if (r) r.inst.setOption(r.build(tokens()), { notMerge: true }); }

function renderConfusion() {
  const s = state.data.summary, name = state.cmName;
  const cm = s.confusion[name];
  const r = s.results.find((x) => x.name === name);
  if (!cm || !r) {
    $("cmCard").innerHTML = `<p class="cm-title">${esc(shortName(name))}</p><p class="muted small">Confusion matrices are stored for the eight models and fifteen ensembles. Feature-selected variants show their scores in the table.</p>`;
    return;
  }
  const [[tn, fp], [fn, tp]] = cm, n = tn + fp + fn + tp;
  const cell = (v, ok) => `<div class="cell" style="background:${ok ? `rgba(12,163,12,${0.08 + 0.32 * v / n * 2})` : `rgba(208,59,59,${0.06 + 0.3 * v / n * 2})`}"><b>${fmtInt(v)}</b><span>${pct(v / n)}</span></div>`;
  $("cmCard").innerHTML = `
    <p class="cm-title">${esc(shortName(name))}</p>
    <div class="cm" role="table" aria-label="Confusion matrix">
      <span></span><span class="lab">Called DOWN</span><span class="lab">Called UP</span>
      <span class="lab">Was DOWN</span>${cell(tn, true)}${cell(fp, false)}
      <span class="lab">Was UP</span>${cell(fn, false)}${cell(tp, true)}
    </div>
    <p class="cm-axis">Green: right calls · red: wrong calls · ${fmtInt(n)} test days</p>
    <div class="cm-stats">
      <div><span>Accuracy</span><b>${pct(r.accuracy)}</b></div><div><span>F1</span><b>${num(r.f1)}</b></div>
      <div><span>Precision</span><b>${num(r.precision)}</b></div><div><span>Recall</span><b>${num(r.recall)}</b></div>
      <div><span>Called UP</span><b>${pct((fp + tp) / n)}</b></div><div><span>Was UP</span><b>${pct((fn + tp) / n)}</b></div>
    </div>`;
}

// ---- Voting ------------------------------------------------------------------------
function rolling(hits, w) {
  const out = new Array(hits.length).fill(null); let sum = 0;
  for (let i = 0; i < hits.length; i++) { sum += hits[i]; if (i >= w) sum -= hits[i - w]; if (i >= w - 1) out[i] = sum / w; }
  return out;
}

function renderVoting() {
  const { summary: s, series } = state.data, t = series.test;
  const V = s.voting;
  const best7 = Math.max(...V.filter((v) => v.models === 7).map((v) => v.accuracy));
  table($("votingTable"), [
    { key: "method", label: "Method", render: (r) => `${esc(r.method)}${r.models === 8 ? ' <span class="kind-pill">8 · ViT</span>' : ""}` },
    { key: "rule", label: "Combines" },
    { key: "weights", label: "Weights" },
    { key: "accuracy", label: "Accuracy", num: true, render: (r) => barCell(r.accuracy) },
    { key: "f1", label: "F1", num: true, render: (r) => num(r.f1) },
    { key: "roc_auc", label: "ROC-AUC", num: true, render: (r) => num(r.roc_auc) },
  ], V, { sortable: false, rowClass: (r) => (r.models === 7 && r.accuracy === best7 ? "is-best" : "") });

  const a7 = s.adaptive["7"];
  $("adaptiveNote").innerHTML = `Adaptive settings were tuned on the training period only: each day, weights come from the last <b>${a7.window} known days</b> with sharpening <b>&beta; = ${a7.beta}</b>.` +
    (a7.beta === 0 ? " &beta; = 0 means adaptation did not help in validation, so the adaptive rule fell back to equal weights." : " Higher &beta; shifts more of the vote to whichever models have been right lately.");

  const names7 = MODELS.slice(0, 7);
  mount("weightsChart", (T) => ({
    ...base(T),
    legend: { top: 0, left: 0, textStyle: { color: T.ink2, fontSize: 11.5 }, itemWidth: 10, itemHeight: 10, icon: "roundRect" },
    grid: { left: 8, right: 16, top: 44, bottom: 8, containLabel: true },
    xAxis: axis(T, { type: "category", data: t.date, boundaryGap: false, splitLine: { show: false }, axisLabel: { color: T.muted, formatter: (v) => fmtMonth(v), hideOverlap: true } }),
    yAxis: axis(T, { type: "value", min: 0, max: 1, axisLabel: { color: T.muted, formatter: (v) => Math.round(v * 100) + "%" } }),
    tooltip: { ...base(T).tooltip, trigger: "axis", axisPointer: { type: "line", lineStyle: { color: T.ink2 } }, order: "valueDesc",
      formatter: (ps) => `${fmtDate(ps[0].axisValue)}<br>` + ps.map((p) => `${p.marker}${esc(p.seriesName)} <b>${pct(p.value)}</b>`).join("<br>") },
    series: names7.map((m) => ({ name: m, type: "line", stack: "w", data: t.adaptive_weights_7[m], showSymbol: false, sampling: "lttb",
      lineStyle: { width: 0.8, color: T.surface }, areaStyle: { color: modelColor(T, m), opacity: 0.92 }, itemStyle: { color: modelColor(T, m) }, emphasis: { focus: "series" } })),
  }));

  const methods = [[MAJ7, "Majority", "muted"], [WP7, "Weighted (static)", "ink2"], [AP7, "Adaptive weighted", "accent"]];
  mount("rollingChart", (T) => ({
    ...base(T),
    legend: { top: 0, left: 0, textStyle: { color: T.ink2, fontSize: 11.5 }, itemWidth: 14, itemHeight: 3 },
    grid: { left: 8, right: 16, top: 34, bottom: 8, containLabel: true },
    xAxis: axis(T, { type: "category", data: t.date, boundaryGap: false, splitLine: { show: false }, axisLabel: { color: T.muted, formatter: (v) => fmtMonth(v), hideOverlap: true } }),
    yAxis: axis(T, { type: "value", scale: true, axisLabel: { color: T.muted, formatter: (v) => Math.round(v * 100) + "%" } }),
    tooltip: { ...base(T).tooltip, trigger: "axis", axisPointer: { type: "line", lineStyle: { color: T.ink2 } },
      formatter: (ps) => `${fmtDate(ps[0].axisValue)}<br>` + ps.map((p) => `${p.marker}${esc(p.seriesName)} <b>${p.value == null ? "–" : pct(p.value)}</b>`).join("<br>") },
    series: methods.map(([n, label, tok], i) => ({
      name: label, type: "line", showSymbol: false, data: rolling(t.ensemble_pred[n].map((p, j) => (p === t.actual[j] ? 1 : 0)), 60),
      lineStyle: { width: 2, color: T[tok] }, itemStyle: { color: T[tok] },
      ...(i === 0 ? { markLine: { silent: true, symbol: "none", data: [{ yAxis: 0.5 }], lineStyle: { color: T.ink2, type: "dotted", width: 1 }, label: { show: false } } } : {}),
    })),
  }));

  const val = s.validation;
  table($("weightsTable"), [
    { key: "name", label: "Model", render: (r) => `<span class="swatch" style="background:var(--s${MODELS.indexOf(r.name) + 1})"></span>${esc(r.name)}` },
    { key: "f1", label: "Validation F1", num: true, render: (r) => num(r.f1) },
    { key: "w", label: "Static weight", num: true, render: (r) => pct(r.w) },
    { key: "am", label: "Adaptive mean", num: true, render: (r) => pct(r.am) },
    { key: "an", label: "Adaptive, next day", num: true, render: (r) => pct(r.an) },
  ], names7.map((m, j) => ({ name: m, f1: val[m].val_f1, w: s.weights.static_7[m], am: a7.mean_weights[j], an: a7.next_weights[j] })), { sortKey: "w" });
}

// ---- Day by day --------------------------------------------------------------------
function fillDailySelect() {
  const s = state.data.summary;
  const ens = s.results.filter((r) => r.kind === "Ensemble").map((r) => r.name);
  const sel = $("dailySel");
  sel.innerHTML = `<optgroup label="Ensembles">${ens.map((n) => `<option value="${esc(n)}">${esc(shortName(n))}</option>`).join("")}</optgroup>` +
    `<optgroup label="Single models">${MODELS.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("")}</optgroup>`;
  if (![...ens, ...MODELS].includes(state.daily)) state.daily = AP7;
  sel.value = state.daily;
}

function renderDaily() {
  fillDailySelect();
  const t = state.data.series.test, s = state.data.summary, key = state.daily;
  const pred = t.ensemble_pred[key] || t.model_pred[key];
  const hits = pred.map((p, i) => (p === t.actual[i] ? 1 : 0));
  const roll = rolling(hits, 60).filter((v) => v != null);
  $("dailyStats").innerHTML = [
    ["Right calls", pct(mean(hits))],
    ["Called UP", pct(mean(pred))],
    ["Actually UP", pct(mean(t.actual))],
    ["Best 60 days", pct(Math.max(...roll))],
    ["Worst 60 days", pct(Math.min(...roll))],
  ].map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  const n = t.date.length, startPct = Math.max(0, 100 - (120 / n) * 100);
  mount("dailyChart", (T) => ({
    ...base(T),
    grid: { left: 8, right: 16, top: 12, bottom: 48, containLabel: true },
    xAxis: axis(T, { type: "category", data: t.date, splitLine: { show: false }, axisLabel: { color: T.muted, formatter: (v) => { const [y, m, d] = v.split("-"); return `${+d} ${MONTHS[+m - 1]} ’${y.slice(2)}`; }, hideOverlap: true } }),
    yAxis: axis(T, { type: "value", scale: true, axisLabel: { color: T.muted, formatter: (v) => "₹" + fmtInt(v) } }),
    tooltip: { ...base(T).tooltip, trigger: "axis", axisPointer: { type: "line", lineStyle: { color: T.line2 } },
      formatter: (ps) => { const i = ps[0].dataIndex; const next = i + 1 < n ? t.close[i + 1] : null;
        return `${fmtDate(t.date[i])} · close ₹${Number(t.close[i]).toLocaleString("en-IN")}<br>Called <b>${pred[i] ? "UP" : "DOWN"}</b>, next day was <b>${t.actual[i] ? "UP" : "DOWN"}</b>` +
          `<br><span style="color:${hits[i] ? T.up : T.down};font-weight:600">${hits[i] ? "Right" : "Wrong"}</span>`; } },
    dataZoom: [{ type: "inside", start: startPct, end: 100, zoomOnMouseWheel: false, moveOnMouseWheel: false }, { type: "slider", start: startPct, end: 100, height: 20, bottom: 8, borderColor: T.line, backgroundColor: T.surface2,
      fillerColor: hexA(T.accent, 0.12), dataBackground: { lineStyle: { color: T.line2 }, areaStyle: { color: T.line } }, handleStyle: { color: T.surface, borderColor: T.line2 },
      textStyle: { color: T.muted, fontSize: 10 }, labelFormatter: (i) => (t.date[i] ? fmtMonth(t.date[i]) : "") }],
    series: [
      { type: "line", data: t.close, showSymbol: false, lineStyle: { color: T.line2, width: 1.5 }, z: 1, silent: true },
      { type: "scatter", z: 3, symbolSize: 10,
        data: t.close.map((c, i) => ({ value: c, symbol: "triangle", symbolRotate: pred[i] ? 0 : 180,
          itemStyle: { color: hits[i] ? T.up : T.down, borderColor: T.surface, borderWidth: 1 } })) },
    ],
  }));
  mount("dailyRolling", (T) => ({
    ...base(T),
    grid: { left: 8, right: 16, top: 12, bottom: 8, containLabel: true },
    xAxis: axis(T, { type: "category", data: t.date, boundaryGap: false, splitLine: { show: false }, axisLabel: { color: T.muted, formatter: (v) => fmtMonth(v), hideOverlap: true } }),
    yAxis: axis(T, { type: "value", scale: true, axisLabel: { color: T.muted, formatter: (v) => Math.round(v * 100) + "%" } }),
    tooltip: { ...base(T).tooltip, trigger: "axis", formatter: (ps) => `${fmtDate(ps[0].axisValue)}<br>Last 60 days: <b>${ps[0].value == null ? "–" : pct(ps[0].value)}</b>` },
    series: [{ type: "line", data: rolling(hits, 60), showSymbol: false, lineStyle: { color: T.accent, width: 2 }, itemStyle: { color: T.accent },
      areaStyle: { color: hexA(T.accent, 0.08), origin: 0.5 },
      markLine: { silent: true, symbol: "none", data: [{ yAxis: 0.5, label: { formatter: "50%" } }, { yAxis: s.majority_baseline, label: { formatter: "baseline" }, lineStyle: { type: [4, 4], color: T.muted } }],
        lineStyle: { color: T.ink2, width: 1, type: "dotted" }, label: { color: T.muted, fontSize: 10.5, position: "insideEndTop" } } }],
  }));
}

// ---- Features ----------------------------------------------------------------------
function renderFeatures() {
  const s = state.data.summary, fs = s.feature_selection;
  $("fsNote").innerHTML = `Walk-forward validation picked <b>${esc(fs.best_set)}</b> as the best feature set for this stock. The LSTM and ViT were re-trained on <b>${esc(fs.eval_set)}</b> for the feature-selected comparison.`;
  const featChart = (id, scores, selected) => {
    const rows = s.features.map((f, i) => ({ f, v: scores[i], on: selected.includes(f) })).sort((a, b) => a.v - b.v);
    mount(id, (T) => ({
      ...base(T),
      grid: { left: 8, right: 24, top: 4, bottom: 24, containLabel: true },
      xAxis: axis(T, { type: "value", axisLabel: { color: T.muted, fontSize: 10.5 } }),
      yAxis: axis(T, { type: "category", data: rows.map((r) => r.f), axisLabel: { color: T.ink2, fontSize: 10.5, fontFamily: T.mono, interval: 0 }, splitLine: { show: false }, axisLine: { show: false } }),
      tooltip: { ...base(T).tooltip, trigger: "item", formatter: (p) => `<b>${esc(rows[p.dataIndex].f)}</b><br>Score ${num(rows[p.dataIndex].v, 4)}<br>${rows[p.dataIndex].on ? "Kept in Top-20" : "Dropped"}` },
      series: [{ type: "bar", barWidth: 9, data: rows.map((r) => ({ value: r.v, itemStyle: { color: r.on ? T.accent : T.line2, borderRadius: [0, 3, 3, 0] } })) }],
    }));
  };
  featChart("miChart", fs.mi_scores, fs.mi_selected);
  featChart("treeChart", fs.tree_importances, fs.tree_selected);
  const sets = Object.keys(fs.walk_forward_f1);
  const models = Object.keys(fs.walk_forward_f1[sets[0]]);
  const bestSet = sets.reduce((a, b) => (fs.walk_forward_f1[b].MEAN > fs.walk_forward_f1[a].MEAN ? b : a));
  table($("fsTable"), [
    { key: "m", label: "Model", render: (r) => (r.m === "MEAN" ? "<b>Mean</b>" : esc(r.m)) },
    ...sets.map((k) => ({ key: k, label: k, num: true, render: (r) => (k === bestSet && r.m === "MEAN" ? `<b>${num(r[k])}</b>` : num(r[k])) })),
  ], models.map((m) => ({ m, ...Object.fromEntries(sets.map((k) => [k, fs.walk_forward_f1[k][m]])) })), { sortable: false, rowClass: (r) => (r.m === "MEAN" ? "is-best" : "") });
}

// ---- Training ----------------------------------------------------------------------
function renderTraining() {
  const s = state.data.summary, tc = s.training_curves;
  const lossChart = (id, h) => {
    const best = h.val_loss.indexOf(Math.min(...h.val_loss));
    mount(id, (T) => ({
      ...base(T),
      legend: { top: 0, right: 0, textStyle: { color: T.ink2, fontSize: 11.5 }, itemWidth: 14, itemHeight: 3 },
      grid: { left: 8, right: 16, top: 30, bottom: 24, containLabel: true },
      xAxis: axis(T, { type: "category", data: h.loss.map((_, i) => i + 1), name: "epoch", nameLocation: "middle", nameGap: 26, splitLine: { show: false }, boundaryGap: false }),
      yAxis: axis(T, { type: "value", scale: true, axisLabel: { color: T.muted, formatter: (v) => v.toFixed(3) } }),
      tooltip: { ...base(T).tooltip, trigger: "axis", formatter: (ps) => `Epoch ${ps[0].axisValue}<br>` + ps.map((p) => `${p.marker}${p.seriesName} <b>${num(p.value, 4)}</b>`).join("<br>") },
      series: [
        { name: "train loss", type: "line", data: h.loss, showSymbol: false, lineStyle: { color: T.ink2, width: 2 }, itemStyle: { color: T.ink2 } },
        { name: "validation loss", type: "line", data: h.val_loss, showSymbol: false, lineStyle: { color: T.accent, width: 2 }, itemStyle: { color: T.accent },
          markPoint: { symbol: "circle", symbolSize: 9, itemStyle: { color: T.accent, borderColor: T.surface, borderWidth: 2 },
            label: { show: true, position: best === 0 ? "right" : "top", formatter: `best: epoch ${best + 1}`, color: T.ink2, fontSize: 11 }, data: [{ coord: [best, h.val_loss[best]] }] } },
      ],
    }));
  };
  lossChart("lstmChart", tc.lstm);
  lossChart("vitChart", tc.vit);
  table($("valTable"), [
    { key: "name", label: "Model", render: (r) => `<span class="swatch" style="background:var(--s${MODELS.indexOf(r.name) + 1})"></span>${esc(r.name)}` },
    { key: "val_accuracy", label: "Val. accuracy", num: true, render: (r) => pct(r.val_accuracy) },
    { key: "val_balanced_accuracy", label: "Balanced acc.", num: true, render: (r) => pct(r.val_balanced_accuracy) },
    { key: "val_f1", label: "Val. F1", num: true, render: (r) => num(r.val_f1) },
    { key: "val_roc_auc", label: "Val. ROC-AUC", num: true, render: (r) => num(r.val_roc_auc) },
  ], MODELS.map((m) => ({ name: m, ...s.validation[m] })), { sortKey: "val_f1" });
}

// ---- Next-day call -----------------------------------------------------------------
function renderNext() {
  const L = state.data.summary.latest;
  $("nextDate").textContent = fmtDate(L.date);
  $("nextClose").textContent = `Last close ₹${Number(L.close).toLocaleString("en-IN", { minimumFractionDigits: 2 })} · ${MODELS.slice(0, 7).filter((m) => L.models[m].pred).length} of 7 main models say UP`;
  $("nextModels").innerHTML = MODELS.map((m) => {
    const c = L.models[m], up = c.pred === 1;
    return `<div class="call"><div class="call-top"><span class="call-name"><span class="swatch" style="background:var(--s${MODELS.indexOf(m) + 1})"></span>${esc(m)}</span>
      <span class="verdict ${up ? "up" : "down"}">${up ? "▲ UP" : "▼ DOWN"}</span></div>
      <div class="prob" aria-hidden="true"><i style="width:${(c.prob * 100).toFixed(1)}%"></i></div><small>P(UP) = ${num(c.prob, 3)}</small></div>`;
  }).join("");
  table($("nextEnsembles"), [
    { key: "name", label: "Rule", render: (r) => esc(r.name.replace(" [experimental]", " · experimental")) },
    { key: "pred", label: "Decision", render: (r) => `<span class="verdict ${r.pred ? "up" : "down"}">${r.pred ? "▲ UP" : "▼ DOWN"}</span>` },
    { key: "why", label: "Why", render: (r) => esc(r.detail || (r.name.includes("labels") ? `weighted vote S = ${num(r.score)} (UP if > 0.50)` : `weighted P(UP) = ${num(r.score)} (UP if ≥ 0.50)`)) },
  ], Object.entries(L.ensembles).map(([name, v]) => ({ name, ...v })), { sortable: false });
}

// ---- Integrity ---------------------------------------------------------------------
function renderChecks() {
  const C = state.data.summary.checks, ok = C.filter((c) => c.pass).length;
  $("checksNote").innerHTML = `<b>${ok} of ${C.length} checks passed.</b> STEP 17 re-verifies every no-look-ahead promise in code. A single failure would mean the results above cannot be trusted.`;
  $("checksList").innerHTML = C.map((c) => `<li><span class="badge ${c.pass ? "pass" : "fail"}">${c.pass ? "PASS" : "FAIL"}</span><span>${esc(c.label)}</span></li>`).join("");
}

// =====================================================================================
// WIRING
// =====================================================================================
function wire() {
  $("stockSearch").addEventListener("input", (e) => { state.query = e.target.value; renderBoard(); });
  $("stockSearch").addEventListener("keydown", (e) => { if (e.key === "Enter") { const v = visibleStocks(); if (v.length) selectStock(v[0].symbol, true); } });
  const step = (d) => { const list = visibleStocks().length ? visibleStocks() : state.stocks; const i = list.findIndex((s) => s.symbol === state.current); selectStock(list[(i + d + list.length) % list.length].symbol); };
  $("prevStock").addEventListener("click", () => step(-1));
  $("nextStock").addEventListener("click", () => step(1));
  $("stockZip").addEventListener("click", () => { if (state.current) { $("dlStock").value = state.current; download(DL.stock(state.current)); } });
  $("dlStockBtn").addEventListener("click", () => download(DL.stock($("dlStock").value)));
  $("dlAllBtn").addEventListener("click", () => download(DL.all()));

  const tabs = [...document.querySelectorAll("#tabs [role=tab]")];
  tabs.forEach((b, i) => {
    b.addEventListener("click", () => setTab(b.id.replace("tab-", "")));
    b.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (d) { const n = tabs[(i + d + tabs.length) % tabs.length]; n.focus(); setTab(n.id.replace("tab-", "")); }
    });
  });
  $("kindSeg").querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => {
    $("kindSeg").querySelectorAll(".seg-btn").forEach((x) => x.classList.toggle("is-on", x === b));
    state.kind = b.dataset.kind; renderModels();
  }));
  $("metricSel").addEventListener("change", (e) => { state.metric = e.target.value; renderModels(); });
  $("dailySel").addEventListener("change", (e) => { state.daily = e.target.value; renderDaily(); });
  document.querySelectorAll(".heat-head .seg-btn").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".heat-head .seg-btn").forEach((x) => x.classList.toggle("is-on", x === b));
    state.heatSort = b.dataset.sort; renderHeat();
  }));

  // theme: explicit toggle (remembered per viewer), plus OS / host changes
  const root = document.documentElement;
  try { const saved = localStorage.getItem("qp-theme"); if (saved && !root.hasAttribute("data-theme")) root.setAttribute("data-theme", saved); } catch (e) { /* storage blocked */ }
  $("themeToggle").addEventListener("click", () => {
    const dark = root.getAttribute("data-theme") ? root.getAttribute("data-theme") === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.setAttribute("data-theme", dark ? "light" : "dark");
    try { localStorage.setItem("qp-theme", dark ? "light" : "dark"); } catch (e) { /* storage blocked */ }
  });
  new MutationObserver(() => requestAnimationFrame(rethemeAll)).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => requestAnimationFrame(rethemeAll));

  // highlight the section in view
  const links = [...document.querySelectorAll(".nav a")];
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) links.forEach((a) => a.classList.toggle("is-active", a.getAttribute("href") === "#" + e.target.id)); }), { rootMargin: "-45% 0px -50% 0px" });
    ["question", "method", "results", "explorer", "downloads", "limits"].forEach((id) => io.observe($(id)));
  }
}

async function start() {
  wire();
  renderWalkForward();
  let ov;
  try { ov = await fetch("data/overview.json").then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }); }
  catch (e) {
    $("heroAnswer").textContent = "The results data could not be loaded. If you opened this file directly from disk, serve the folder over HTTP instead (for example: python -m http.server).";
    return;
  }
  state.ov = ov;
  state.stocks = [...ov.stocks].sort((a, b) => a.symbol.localeCompare(b.symbol));
  state.bySym = Object.fromEntries(state.stocks.map((s) => [s.symbol, s]));
  $("dlStock").innerHTML = state.stocks.map((s) => `<option value="${esc(s.symbol)}">${esc(s.symbol)} · ${esc(s.company)}${ov.download_bytes ? ` (${fmtBytes(ov.download_bytes[s.symbol])})` : ""}</option>`).join("");
  if (ov.download_bytes) $("dlAllSize").textContent = fmtBytes(ov.download_bytes._all) + " ZIP";

  renderIntro(); renderHero(); renderFindings(); renderRQ1(); renderRQ2(); renderRQ3(); renderRQ4(); renderHeat();
  renderChips(); renderBoard();
  const fromHash = (location.hash.match(/^#stock-([A-Za-z0-9-]+)$/) || [])[1];
  const first = fromHash && state.bySym[fromHash.toUpperCase()] ? fromHash.toUpperCase() : (state.bySym.RELIANCE ? "RELIANCE" : state.stocks[0].symbol);
  await selectStock(first);
  if (fromHash) $("explorer").scrollIntoView();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
