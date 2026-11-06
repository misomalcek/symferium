// Symferium UI: plain JS, no build step. Everything comes from the control API; live numbers
// arrive over Server-Sent Events (/api/events).
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const GB = 1024 ** 3;
const gb = (b) => (b == null ? "–" : (b / GB).toFixed(2) + " GB");
const num = (x, d = 1) => (x ? Number(x).toFixed(d) : "–");

let token = localStorage.getItem("symferium.token") || "";
const state = { machine: null, budget: null, instances: [], local: [], settings: null, catalog: [] };

function toast(msg, bad = false) {
  const d = document.createElement("div");
  d.textContent = msg;
  if (bad) d.className = "bad";
  $("#toast").appendChild(d);
  setTimeout(() => d.remove(), bad ? 12000 : 5000);
}

async function api(path, opts = {}) {
  const headers = { "content-type": "application/json", ...(opts.headers || {}) };
  if (token) headers.authorization = "Bearer " + token;
  const r = await fetch(path, { ...opts, headers, body: opts.body && typeof opts.body !== "string" ? JSON.stringify(opts.body) : opts.body });
  if (r.status === 401) {
    const t = prompt("Symferium API token (server.admin_token):");
    if (t != null) { token = t; localStorage.setItem("symferium.token", t); return api(path, opts); }
  }
  const text = await r.text();
  let data; try { data = JSON.parse(text); } catch { data = text; }
  if (!r.ok) { const e = new Error(errText(data)); e.data = data; throw e; }
  return data;
}
function errText(d) {
  if (d && d.reasons) return d.error + ":\n• " + d.reasons.join("\n• ");
  if (d && d.error) return typeof d.error === "string" ? d.error : JSON.stringify(d.error);
  return typeof d === "string" ? d : JSON.stringify(d);
}

// ---------------------------------------------------------------- tabs
$$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(name) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === name));
  localStorage.setItem("symferium.tab", name);
  if (name === "models") loadModels();
  if (name === "settings") loadSettings();
  if (name === "bench") loadBench();
  if (name === "downloads") renderDownloads();
}

// ---------------------------------------------------------------- overview
function renderMachine() {
  const m = state.machine, b = state.budget;
  if (!m) return;
  const pressureCls = { normal: "ok", warn: "warn", critical: "bad" }[m.pressure] || "";
  const cards = [
    ["Machine", m.model_identifier || "–", m.cpu_brand],
    ["Memory", gb(m.total_bytes), "available " + gb(m.available_bytes)],
    ["Pressure", `<span class="pill ${pressureCls}">${esc(m.pressure)}</span>`, "swap used " + gb(m.swap_used_bytes)],
    ["Metal working set", gb(m.metal_working_set_bytes), m.gpu_cores ? m.gpu_cores + " GPU cores" : ""],
    ["CPU", `${m.perf_cores}P + ${m.efficiency_cores}E`, m.logical_cpus + " threads"],
    ["Headroom", gb(b.headroom_bytes), "largest load allowed now"],
  ];
  $("#machine").innerHTML = cards.map(([k, v, s]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="k">${esc(s)}</div></div>`).join("");
  const total = m.total_bytes;
  const committed = b.committed_bytes, other = Math.max(0, total - m.available_bytes - committed);
  const seg = (v, c) => `<div style="width:${(100 * Math.max(0, v)) / total}%;background:${c}"></div>`;
  $("#budget").innerHTML = `<div class="bar">${seg(other, "#98a2b3")}${seg(committed, "var(--accent)")}${seg(Math.max(0, m.available_bytes - b.min_free_bytes), "#7fd4a0")}${seg(Math.min(b.min_free_bytes, m.available_bytes), "#f4c27a")}</div>
    <div class="legend"><span><i style="background:#98a2b3"></i>macOS + other apps ${gb(other)}</span><span><i style="background:var(--accent)"></i>Symferium models ${gb(committed)}</span>
    <span><i style="background:#7fd4a0"></i>usable ${gb(Math.max(0, m.available_bytes - b.min_free_bytes))}</span><span><i style="background:#f4c27a"></i>kept free ${gb(b.min_free_bytes)}</span>
    <span>ceiling ${gb(b.cap_bytes)}</span><span>starting ${gb(b.pending_bytes)}</span></div>`;
}

function stateCls(s) { return { ready: "ok", starting: "warn", restarting: "warn", stopping: "warn", failed: "bad" }[s] || ""; }

function renderInstances() {
  const list = state.instances;
  if (!list.length) { $("#instances").innerHTML = `<p class="muted">Nothing loaded. Load a model from the Models tab.</p>`; return; }
  $("#instances").innerHTML = `<div class="table-wrap"><table><thead><tr><th>Model</th><th>State</th><th>Endpoint</th><th>Engine</th><th class="num">Ctx</th>
    <th class="num">Memory est. / measured</th><th class="num">Decode t/s</th><th class="num">Prefill t/s</th><th class="num">Requests</th><th></th></tr></thead><tbody>
    ${list.map((i) => `<tr><td><b>${esc(i.id)}</b><br><span class="muted">${esc(i.class)} · ${esc(i.kind)}${i.vision ? " · vision" : ""}</span></td>
      <td><span class="pill ${stateCls(i.status.state)}">${esc(i.status.state)}</span>${i.status.restarts ? `<br><span class="muted">${i.status.restarts} restarts</span>` : ""}
        ${i.status.last_error && i.status.state !== "ready" ? `<br><span class="muted">${esc(i.status.last_error)}</span>` : ""}</td>
      <td><code>${esc(i.url)}</code></td><td>${esc(i.engine)}</td><td class="num">${i.ctx}</td>
      <td class="num">${gb(i.estimate.total)}<br><span class="muted">${i.status.measured_bytes ? gb(i.status.measured_bytes) : "–"}</span></td>
      <td class="num">${num(i.metrics.last_decode_tps)}<br><span class="muted">avg ${num(i.metrics.avg_decode_tps)}</span></td>
      <td class="num">${num(i.metrics.last_prefill_tps)}</td><td class="num">${i.metrics.requests}${i.metrics.in_flight > 0 ? `<br><span class="muted">${i.metrics.in_flight} active</span>` : ""}</td>
      <td><button data-logs="${esc(i.id)}">Logs</button> <button class="danger" data-unload="${esc(i.id)}">Unload</button></td></tr>`).join("")}
    </tbody></table></div>`;
}
document.addEventListener("click", async (e) => {
  const t = e.target;
  if (t.dataset.unload) {
    if (!confirm(`Unload ${t.dataset.unload}? It will also be removed from the set restored at login.`)) return;
    try { await api("/api/unload", { method: "POST", body: { id: t.dataset.unload } }); toast("Unloaded " + t.dataset.unload); } catch (err) { toast(err.message, true); }
  }
  if (t.dataset.logs) {
    try {
      const l = await api(`/api/instances/${encodeURIComponent(t.dataset.logs)}/logs?n=400`);
      const w = window.open("", "_blank");
      w.document.write(`<title>${esc(l.id)} logs</title><pre style="white-space:pre-wrap;font:12px ui-monospace,monospace">${esc(l.lines.join("\n"))}</pre>`);
    } catch (err) { toast(err.message, true); }
  }
  if (t.dataset.load) openLoad(t.dataset.load);
  if (t.dataset.repo) openRepo(t.dataset.repo);
  if (t.dataset.dl) download(t.dataset.dl, t.dataset.quant);
  if (t.dataset.cancel) api("/api/downloads/" + t.dataset.cancel, { method: "DELETE" }).catch((err) => toast(err.message, true));
});

// ---------------------------------------------------------------- models
async function loadModels() {
  const presets = await api("/api/presets");
  $("#preset-select").innerHTML = Object.entries(presets).map(([k, p]) => `<option value="${esc(k)}" title="${esc(p.description)}">${esc(k)} (${p.models.length})</option>`).join("");
  const d = await api("/api/models");
  state.local = d.local;
  state.prefs = d.engine_preferences || [];
  renderModels();
}
$("#model-filter").addEventListener("input", renderModels);
$("#preset-load").addEventListener("click", async () => {
  const name = $("#preset-select").value;
  if (!name || !confirm(`Load every model of ${name}, one at a time?`)) return;
  toast("Loading set " + name + "…");
  showTab("overview");
  try {
    const r = await api(`/api/presets/${encodeURIComponent(name)}/load`, { method: "POST" });
    r.forEach((x) => toast(x.ok ? `${x.id}: ${x.url}` : `${x.id}: ${x.error}`, !x.ok));
  } catch (err) { toast(err.message, true); }
});
$("#rescan").addEventListener("click", async () => { const r = await api("/api/models/rescan", { method: "POST" }); toast(`${r.count} models found`); loadModels(); });

function flagTags(f) {
  return ["vision", "thinking", "tools", "agentic", "coding", "audio"].filter((k) => f && f[k]).map((k) => `<span class="tag">${k}</span>`).join("");
}
function renderModels() {
  const q = $("#model-filter").value.toLowerCase();
  const list = state.local.filter((m) => !q || (m.id + m.name + m.class + m.arch).toLowerCase().includes(q));
  const loaded = new Set(state.instances.map((i) => i.model_id));
  const pref = Object.fromEntries((state.prefs || []).map((p) => [p.family, p]));
  $("#local-models").innerHTML = list.length ? `<div class="table-wrap"><table><thead><tr><th>Model</th><th>Class</th><th>Format</th><th>Quant</th><th class="num">Size</th><th class="num">Params</th><th class="num">Trained ctx</th><th>Capabilities</th><th></th></tr></thead><tbody>
    ${list.map((m) => `<tr><td><b>${esc(m.id)}</b><br><span class="muted" title="${esc(m.path)}">${esc(m.name)}${m.read_only ? " · read-only folder" : ""}</span>
        ${pref[m.family] ? `<br><span class="tag">bench: ${esc(pref[m.family].engine)} fastest</span>` : ""}</td>
      <td>${esc(m.class)}</td><td>${esc(m.format)}</td><td>${esc(m.quant || "")}</td><td class="num">${gb(m.size_bytes)}</td>
      <td class="num">${m.params ? (m.params / 1e9).toFixed(1) + " B" : "–"}</td><td class="num">${m.ctx_train || "–"}</td>
      <td title="${esc(Object.entries(m.flag_sources || {}).map(([k, v]) => k + ": " + v).join("\n"))}">${flagTags(m.flags)}</td>
      <td>${loaded.has(m.id) ? `<span class="pill ok">loaded</span>` : `<button class="primary" data-load="${esc(m.id)}">Load…</button>`}</td></tr>`).join("")}
    </tbody></table></div>` : `<p class="muted">No models found in the models folders. Download one from the Catalog tab, or add a folder in Settings → paths.</p>`;
}

const CACHE_TYPES = ["", "f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "turbo4", "turbo3", "turbo2"];
let loadingModel = null;
async function openLoad(id) {
  loadingModel = state.local.find((m) => m.id === id);
  if (!state.settings) state.settings = await api("/api/settings");
  const f = $("#load-form");
  f.reset();
  $("#load-title").textContent = "Load " + id;
  const engines = Object.entries(state.settings.values.engines).filter(([, p]) => p.enabled && (!p.formats.length || p.formats.includes(loadingModel.format)));
  f.engine.innerHTML = `<option value="">default for ${esc(loadingModel.class)}</option>` + engines.map(([k]) => `<option>${esc(k)}</option>`).join("");
  for (const n of ["cache_type_k", "cache_type_v"]) f[n].innerHTML = CACHE_TYPES.map((c) => `<option value="${c}">${c || "class default"}</option>`).join("");
  f.mmproj.value = loadingModel.mmproj ? "auto" : "none";
  $("#load-estimate").innerHTML = "";
  $("#load-dialog").showModal();
  estimate();
}
function loadReq() {
  const f = $("#load-form");
  const params = {};
  f.params.value.split("\n").map((l) => l.trim()).filter(Boolean).forEach((l) => { const i = l.indexOf("="); if (i > 0) params[l.slice(0, i).trim()] = l.slice(i + 1).trim(); });
  if (f.cache_type_k.value) params.cache_type_k = f.cache_type_k.value;
  if (f.cache_type_v.value) params.cache_type_v = f.cache_type_v.value;
  return {
    model: loadingModel.id, alias: f.alias.value || null, engine: f.engine.value || null,
    port: f.port.value ? Number(f.port.value) : null, ctx: f.ctx.value ? Number(f.ctx.value) : null,
    mmproj: f.mmproj.value || null, params, persist: f.persist.checked,
  };
}
async function estimate() {
  try {
    const e = await api("/api/estimate", { method: "POST", body: loadReq() });
    const x = e.estimate;
    const parts = [["weights", x.weights], ["projector", x.projector], ["KV cache", x.kv], ["SWA checkpoints", x.swa_checkpoints], ["prompt cache", x.prompt_cache], ["compute", x.compute], ["overhead", x.overhead]]
      .filter(([, v]) => v).map(([k, v]) => `${k} ${gb(v)}`).join(" · ");
    $("#load-estimate").innerHTML = `<div class="${e.allowed ? "yes" : "no"}"><b>${e.allowed ? "Fits" : "Refused"}: ${gb(x.total)}</b> on <b>${esc(e.engine)}</b>, ctx ${e.ctx}, headroom ${gb(e.budget.headroom_bytes)}</div>
      <div class="muted">${parts}${x.correction !== 1 ? ` · ×${x.correction.toFixed(2)} learned correction` : ""}</div>
      ${e.reasons.map((r) => `<div class="no">• ${esc(r)}</div>`).join("")}${x.notes.map((n) => `<div class="muted">• ${esc(n)}</div>`).join("")}`;
  } catch (err) { $("#load-estimate").innerHTML = `<div class="no">${esc(err.message)}</div>`; }
}
$("#estimate-btn").addEventListener("click", estimate);
$("#load-form").addEventListener("change", estimate);
$("#load-btn").addEventListener("click", async () => {
  try {
    const v = await api("/api/load", { method: "POST", body: loadReq() });
    $("#load-dialog").close();
    toast(`Loading ${v.id} on ${v.url}`);
    showTab("overview");
  } catch (err) { toast(err.message, true); }
});

// ---------------------------------------------------------------- catalog
$("#catalog-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target, p = new URLSearchParams();
  for (const n of ["q", "class", "format", "author"]) if (f[n].value) p.set(n, f[n].value);
  for (const n of ["vision", "tools", "thinking"]) if (f[n].checked) p.set(n, "true");
  $("#catalog-results").innerHTML = `<p class="muted">Searching…</p>`;
  try { state.catalog = await api("/api/catalog/search?" + p); renderCatalog(); } catch (err) { $("#catalog-results").innerHTML = `<p class="no">${esc(err.message)}</p>`; }
});
function fitPill(q) {
  const c = { fits: "ok", needs_unload: "warn", too_big: "bad" }[q.fit] || "";
  return `<span class="pill ${c}">${esc(q.quant)}${q.size ? " " + gb(q.size) : ""}${q.fit && q.fit !== "unknown" ? " · " + q.fit.replace("_", " ") : ""}</span>`;
}
function renderCatalog() {
  const l = state.catalog;
  $("#catalog-results").innerHTML = l.length ? `<div class="table-wrap"><table><thead><tr><th>Repository</th><th>Class</th><th>Format</th><th class="num">Params</th><th>Capabilities</th><th>Quants</th><th class="num">Downloads</th><th></th></tr></thead><tbody>
    ${l.map((m) => `<tr><td><b>${esc(m.repo)}</b>${m.gated ? ` <span class="tag">gated</span>` : ""}${m.moe ? ` <span class="tag">MoE</span>` : ""}<br><span class="muted">${esc(m.pipeline_tag || "")}</span></td>
      <td>${esc(m.class)}</td><td>${esc(m.format)}</td><td class="num">${m.params ? (m.params / 1e9).toFixed(1) + " B" : "–"}</td>
      <td title="${esc(Object.entries(m.flag_sources || {}).map(([k, v]) => k + ": " + v).join("\n"))}">${flagTags(m.flags)}${m.mmproj.length ? `<span class="tag">mmproj</span>` : ""}</td>
      <td>${m.quants.slice(0, 8).map((q) => `<span class="tag">${esc(q.quant)}</span>`).join("")}${m.quants.length > 8 ? " …" : ""}</td>
      <td class="num">${m.downloads.toLocaleString()}</td><td><button data-repo="${esc(m.repo)}">Details…</button></td></tr>`).join("")}
    </tbody></table></div>` : `<p class="muted">No results.</p>`;
}
async function openRepo(repo) {
  $("#repo-detail").innerHTML = `<p class="muted">Loading ${esc(repo)}…</p>`;
  $("#repo-dialog").showModal();
  try {
    const m = await api("/api/catalog/model/" + repo);
    $("#repo-detail").innerHTML = `<h3>${esc(m.repo)}</h3>
      <p>${esc(m.class)} · ${esc(m.kind)} · ${esc(m.format)}${m.params ? " · " + (m.params / 1e9).toFixed(1) + " B params" : ""}${m.ctx ? " · " + m.ctx + " ctx" : ""}${m.base_model ? " · base " + esc(m.base_model) : ""}</p>
      <p>${flagTags(m.flags)}</p><p class="muted">${Object.entries(m.flag_sources).map(([k, v]) => `${esc(k)} from ${esc(v)}`).join(" · ")}</p>
      ${m.gated ? `<p class="no">Gated repository: accept the licence on huggingface.co and set catalog.hf_token in Settings.</p>` : ""}
      <table><thead><tr><th>Quant</th><th class="num">Size</th><th class="num">Est. memory</th><th>Fit</th><th></th></tr></thead><tbody>
      ${m.quants.map((q) => `<tr><td>${esc(q.quant)}${q.recommended ? ` <span class="tag">recommended</span>` : ""}<br><span class="muted">${q.files.length} file(s)</span></td>
        <td class="num">${gb(q.size)}</td><td class="num">${gb(q.estimate_bytes)}</td><td>${fitPill(q)}</td>
        <td>${q.fit === "too_big" ? "" : `<button class="primary" data-dl="${esc(m.repo)}" data-quant="${esc(q.quant)}">Download${m.mmproj.length && m.flags.vision ? " + mmproj" : ""}</button>`}</td></tr>`).join("")}
      </tbody></table>
      ${m.mmproj.length ? `<p class="muted">Projector: ${esc(m.mmproj[0].path)} (${gb(m.mmproj[0].size)}) is downloaded with the model.</p>` : ""}`;
  } catch (err) { $("#repo-detail").innerHTML = `<p class="no">${esc(err.message)}</p>`; }
}
async function download(repo, quant) {
  try { const r = await api("/api/downloads", { method: "POST", body: { repo, quant, include_mmproj: true } }); toast(`Downloading ${repo} ${r.quant || quant}`); $("#repo-dialog").close(); showTab("downloads"); }
  catch (err) { toast(err.message, true); }
}

// ---------------------------------------------------------------- downloads
function renderDownloads() {
  const l = state.downloads || [];
  $("#download-list").innerHTML = l.length ? `<table><thead><tr><th>Repository</th><th>State</th><th>Progress</th><th class="num">Speed</th><th></th></tr></thead><tbody>
    ${l.map((d) => { const s = d.status, pct = s.bytes_total ? (100 * s.bytes_done) / s.bytes_total : 0;
      return `<tr><td><b>${esc(d.repo)}</b><br><span class="muted">${esc(s.current_file || d.files.map((f) => f.path).join(", "))}</span></td>
      <td><span class="pill ${{ done: "ok", failed: "bad", running: "warn" }[s.state] || ""}">${esc(s.state)}</span>${s.error ? `<br><span class="muted">${esc(s.error)}</span>` : ""}</td>
      <td style="min-width:200px"><div class="bar"><div style="width:${pct}%;background:var(--accent)"></div></div><span class="muted">${gb(s.bytes_done)} / ${gb(s.bytes_total)}</span></td>
      <td class="num">${s.speed_bps ? (s.speed_bps / 1024 / 1024).toFixed(1) + " MB/s" : "–"}</td>
      <td>${["queued", "running"].includes(s.state) ? `<button data-cancel="${esc(d.id)}">Cancel</button>` : ""}</td></tr>`; }).join("")}</tbody></table>`
    : `<p class="muted">No downloads yet. Interrupted downloads resume where they stopped.</p>`;
}

// ---------------------------------------------------------------- bench
async function loadBench() {
  if (!state.local.length) await loadModels();
  $("#bench-models").innerHTML = state.local.filter((m) => m.kind === "chat").map((m) => `<option value="${esc(m.id)}">${esc(m.id)} (${esc(m.format)})</option>`).join("");
  const r = await api("/api/bench");
  $("#bench-results").innerHTML = r.length ? `<table><thead><tr><th>Model</th><th>Family</th><th>Engine</th><th>Quant</th><th class="num">Prefill t/s</th><th class="num">Decode t/s</th><th class="num">TTFT ms</th><th>When</th></tr></thead><tbody>
    ${r.slice().reverse().map((b) => `<tr><td>${esc(b.model)}</td><td>${esc(b.family)}</td><td>${esc(b.engine)}</td><td>${esc(b.quant || "")}</td><td class="num">${num(b.prefill_tps)}</td>
      <td class="num"><b>${num(b.decode_tps)}</b></td><td class="num">${num(b.ttft_ms, 0)}</td><td>${new Date(b.ts_ms).toLocaleString()}</td></tr>`).join("")}</tbody></table>`
    : `<p class="muted">No results yet.</p>`;
}
$("#bench-run").addEventListener("click", async () => {
  const models = $$("#bench-models option").filter((o) => o.selected).map((o) => o.value);
  if (!models.length) return toast("Pick at least one model", true);
  await api("/api/bench/compare", { method: "POST", body: { models } });
  toast("Benchmark started: each model loads alone, runs, and unloads. Results appear here.");
});

// ---------------------------------------------------------------- settings
async function loadSettings() {
  state.settings = await api("/api/settings");
  $("#config-path").textContent = state.settings.path ? "Saved to " + state.settings.path : "";
  renderSettings();
}
$("#settings-filter").addEventListener("input", renderSettings);
$("#agent-only").addEventListener("change", renderSettings);
function getPath(obj, key) { return key.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj); }
function inputFor(s, v) {
  const id = "set-" + s.key;
  if (s.kind === "bool") return `<input type="checkbox" id="${id}" ${v ? "checked" : ""}>`;
  if (s.choices) return `<select id="${id}">${s.choices.map((c) => `<option ${c === v ? "selected" : ""}>${esc(c)}</option>`).join("")}</select>`;
  if (s.kind === "int" || s.kind === "float") return `<input type="number" id="${id}" value="${esc(v)}" ${s.min != null ? `min="${s.min}"` : ""} ${s.max != null ? `max="${s.max}"` : ""} step="${s.kind === "int" ? 1 : "any"}">`;
  if (s.kind === "string_list") return `<textarea id="${id}" rows="${Math.min(8, Math.max(2, (v || []).length))}">${esc((v || []).join("\n"))}</textarea>`;
  if (s.kind === "map") return `<textarea id="${id}" rows="2">${esc(JSON.stringify(v || {}))}</textarea>`;
  return `<input type="${s.secret ? "password" : "text"}" id="${id}" value="${esc(v)}">`;
}
function renderSettings() {
  const q = $("#settings-filter").value.toLowerCase(), agentOnly = $("#agent-only").checked;
  const { values, schema } = state.settings;
  let html = "", section = "";
  for (const s of schema) {
    if (q && !(s.key + " " + s.description).toLowerCase().includes(q)) continue;
    if (agentOnly && !s.agent_writable) continue;
    const top = s.key.split(".").slice(0, s.section === "engines" || s.section === "classes" ? 2 : 1).join(".");
    if (top !== section) { section = top; html += `<div class="section-title">${esc(top)}</div>`; }
    const v = getPath(values, s.key);
    const range = s.min != null ? ` · ${s.min}…${s.max}` : "";
    html += `<div class="setting"><div><div class="key">${esc(s.key)}</div><div class="desc">${esc(s.description)}${s.unit ? " (" + esc(s.unit) + ")" : ""}${range}
      ${s.agent_writable ? ` <span class="tag">agent</span>` : ""}${s.restart_required ? ` <span class="tag">restart</span>` : ""}</div></div>
      <div>${inputFor(s, v)}</div><div><button data-save="${esc(s.key)}">Save</button> <button data-reset="${esc(s.key)}" title="Back to the machine default">↺</button></div></div>`;
  }
  $("#settings-form").innerHTML = html || `<p class="muted">No matching settings.</p>`;
}
$("#settings-form").addEventListener("click", async (e) => {
  const key = e.target.dataset.save || e.target.dataset.reset;
  if (!key) return;
  const s = state.settings.schema.find((x) => x.key === key);
  try {
    if (e.target.dataset.reset) {
      await api("/api/settings/reset", { method: "POST", body: { key } });
      toast("Reset " + key);
    } else {
      const el = document.getElementById("set-" + key);
      let value;
      if (s.kind === "bool") value = el.checked;
      else if (s.kind === "int" || s.kind === "float") value = Number(el.value);
      else if (s.kind === "string_list") value = el.value.split("\n").map((x) => x.trim()).filter(Boolean);
      else if (s.kind === "map") value = JSON.parse(el.value || "{}");
      else value = el.value;
      const r = await api("/api/settings", { method: "PUT", body: { key, value } });
      toast("Saved " + key + (r.restart_required.length ? " (takes effect after a restart)" : ""));
    }
    loadSettings();
  } catch (err) { toast(err.message, true); }
});

// ---------------------------------------------------------------- live updates
function logEvent(ev, quiet = false) {
  const line = `${new Date(ev.ts_ms).toLocaleTimeString()}  ${ev.kind}  ${JSON.stringify(ev.data)}`;
  const el = $("#event-log");
  el.textContent = (line + "\n" + el.textContent).slice(0, 200000);
  if (quiet) return;
  if (ev.kind === "governor.refused") toast("Refused: " + (ev.data.reasons || []).join("; "), true);
  if (ev.kind === "model.failed") toast(`${ev.data.id} failed: ${ev.data.error}`, true);
  if (ev.kind === "model.ready") toast(`${ev.data.id} ready on :${ev.data.port} (${ev.data.load_seconds.toFixed(1)} s)`);
  if (ev.kind === "download.done") { toast("Downloaded " + ev.data.repo); loadModels(); }
  if (ev.kind === "download.failed") toast(`Download failed: ${ev.data.error}`, true);
  if (ev.kind.startsWith("bench.") && $("#bench").classList.contains("active")) loadBench();
}
function connect() {
  const es = new EventSource("/api/events" + (token ? "?token=" + encodeURIComponent(token) : ""));
  es.addEventListener("open", () => { $("#conn").textContent = "live"; $("#conn").className = "pill ok"; });
  es.addEventListener("error", () => { $("#conn").textContent = "reconnecting…"; $("#conn").className = "pill warn"; });
  es.addEventListener("metrics", (e) => {
    const d = JSON.parse(e.data);
    Object.assign(state, { machine: d.machine, budget: d.budget, instances: d.instances, downloads: d.downloads });
    renderMachine(); renderInstances();
    if ($("#downloads").classList.contains("active")) renderDownloads();
  });
  es.addEventListener("event", (e) => logEvent(JSON.parse(e.data)));
}

(async function init() {
  try {
    const m = await api("/api/metrics");
    Object.assign(state, { machine: m.machine, budget: m.budget, instances: m.instances });
    renderMachine(); renderInstances();
    (await api("/api/events/history")).forEach((e) => logEvent(e, true));
    const s = await api("/api/settings");
    state.settings = s;
    $("#author-select").innerHTML += s.values.catalog.authors.map((a) => `<option>${esc(a)}</option>`).join("");
    loadModels();
  } catch (err) { toast(err.message, true); }
  connect();
  showTab(localStorage.getItem("symferium.tab") || "overview");
})();
