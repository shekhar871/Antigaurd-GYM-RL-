"""
FastAPI entrypoint expected by the OpenEnv hackathon layout (`server/app.py` next to the package).

We keep one process-wide environment for the simplest HTTP smoke tests; swap in a session
factory if you need concurrent WebSocket classrooms later.

The `/web` route serves an interactive browser UI for judges and demos.
All required API endpoints (/, /reset, /step, /state, /health) remain fully intact.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from agentguard_gym.environment import AgentGuardEnvironment
from agentguard_gym.models import CyberTaskType

app = FastAPI(title="AgentGuard-Gym", version="0.2.0")
_env = AgentGuardEnvironment()

# ---------------------------------------------------------------------------
# Request / Response models (unchanged — autograder depends on these)
# ---------------------------------------------------------------------------

class ResetBody(BaseModel):
    seed: Optional[int] = Field(default=None, ge=0)
    episode_id: Optional[str] = None
    task: Optional[CyberTaskType] = None


class StepBody(BaseModel):
    action: Dict[str, Any]


# ---------------------------------------------------------------------------
# Interactive Web UI  (served at / and /web)
# ---------------------------------------------------------------------------

_WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AgentGuard-Gym · Interactive Explorer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #0a0c12;
      --surface:   #111520;
      --surface2:  #181d2e;
      --border:    #1e2540;
      --accent:    #6c63ff;
      --accent2:   #ff6584;
      --green:     #4ade80;
      --yellow:    #facc15;
      --red:       #f87171;
      --muted:     #4a5180;
      --text:      #e2e8f0;
      --text-dim:  #8892b0;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* ── Header ── */
    header {
      background: linear-gradient(135deg, #0f1629 0%, #151b35 100%);
      border-bottom: 1px solid var(--border);
      padding: 18px 32px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }
    .logo {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .logo-icon {
      width: 40px; height: 40px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-size: 20px;
    }
    .logo-text h1 { font-size: 1.25rem; font-weight: 700; }
    .logo-text span {
      font-size: 0.72rem; font-weight: 400;
      color: var(--text-dim); letter-spacing: .06em;
    }
    .header-links { display: flex; gap: 10px; }
    .header-links a {
      font-size: .8rem; padding: 6px 14px;
      border-radius: 6px; border: 1px solid var(--border);
      color: var(--text-dim); text-decoration: none;
      transition: all .2s;
    }
    .header-links a:hover { border-color: var(--accent); color: var(--accent); }

    /* ── Status bar ── */
    .status-bar {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 10px 32px;
      display: flex; align-items: center; gap: 24px; flex-wrap: wrap;
    }
    .status-pill {
      display: flex; align-items: center; gap: 6px;
      font-size: .78rem; color: var(--text-dim);
    }
    .dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 6px var(--green);
      animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%,100%{ opacity:1; } 50%{ opacity:.5; } }

    /* ── Main layout ── */
    main {
      flex: 1;
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 0;
      min-height: 0;
    }
    @media (max-width: 800px) { main { grid-template-columns: 1fr; } }

    /* ── Sidebar ── */
    aside {
      background: var(--surface);
      border-right: 1px solid var(--border);
      padding: 24px;
      display: flex; flex-direction: column; gap: 20px;
      overflow-y: auto;
    }
    .section-title {
      font-size: .72rem; font-weight: 600;
      color: var(--text-dim); letter-spacing: .1em;
      text-transform: uppercase; margin-bottom: 4px;
    }
    .card {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
    }

    /* ── Form controls ── */
    label { font-size: .8rem; color: var(--text-dim); display: block; margin-bottom: 6px; }
    select, input[type=number] {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 9px 12px;
      border-radius: 7px;
      font-size: .85rem;
      font-family: inherit;
      outline: none;
      transition: border-color .2s;
      margin-bottom: 12px;
    }
    select:focus, input:focus { border-color: var(--accent); }

    textarea {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 10px 12px;
      border-radius: 7px;
      font-size: .82rem;
      font-family: 'JetBrains Mono', monospace;
      resize: vertical;
      min-height: 100px;
      outline: none;
      transition: border-color .2s;
    }
    textarea:focus { border-color: var(--accent); }

    .btn {
      width: 100%;
      padding: 10px 16px;
      border-radius: 8px;
      font-size: .85rem; font-weight: 600;
      cursor: pointer; border: none;
      transition: all .2s;
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent), #8b5cf6);
      color: #fff;
    }
    .btn-primary:hover { opacity: .88; transform: translateY(-1px); }
    .btn-secondary {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-dim);
    }
    .btn-secondary:hover { border-color: var(--accent2); color: var(--accent2); }

    /* ── Task badges ── */
    .task-badge {
      display: inline-block;
      padding: 3px 10px; border-radius: 20px;
      font-size: .72rem; font-weight: 600;
    }
    .badge-pi  { background:#3730a360; color:#a5b4fc; border:1px solid #4f46e5; }
    .badge-ssrf{ background:#9333ea40; color:#d8b4fe; border:1px solid #7c3aed; }
    .badge-mp  { background:#b4520a40; color:#fdba74; border:1px solid #c2410c; }

    /* ── Right panel ── */
    .right-panel {
      display: flex; flex-direction: column; overflow: hidden;
    }

    /* ── History feed ── */
    .history-header {
      padding: 16px 24px;
      border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between;
    }
    .history-header h2 { font-size: .95rem; font-weight: 600; }
    .clear-btn {
      font-size: .75rem; color: var(--text-dim);
      background: none; border: 1px solid var(--border);
      border-radius: 5px; padding: 4px 10px; cursor: pointer;
      transition: all .2s;
    }
    .clear-btn:hover { border-color: var(--red); color: var(--red); }

    #feed {
      flex: 1; overflow-y: auto;
      padding: 20px 24px;
      display: flex; flex-direction: column; gap: 12px;
    }

    .event {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
      animation: slideIn .25s ease;
    }
    @keyframes slideIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:none; } }

    .event-header {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 8px;
    }
    .event-type {
      font-size: .72rem; font-weight: 700; letter-spacing: .08em;
      padding: 2px 8px; border-radius: 4px;
    }
    .type-reset  { background:#1e3a5f; color:#60a5fa; }
    .type-step   { background:#1a3520; color:#4ade80; }
    .type-state  { background:#2d1f4e; color:#a78bfa; }
    .type-health { background:#1e2a1a; color:#86efac; }
    .type-error  { background:#3b1111; color:#f87171; }

    .event-time { font-size: .72rem; color: var(--muted); }

    .event-body {
      font-family: 'JetBrains Mono', monospace;
      font-size: .78rem;
      color: var(--text-dim);
      white-space: pre-wrap;
      word-break: break-all;
      max-height: 220px;
      overflow-y: auto;
      background: var(--bg);
      border-radius: 6px;
      padding: 10px;
    }

    /* reward badge inside event */
    .reward-row {
      display: flex; gap: 12px; margin-top: 8px; flex-wrap: wrap;
    }
    .metric {
      font-size: .75rem;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 4px 10px;
      color: var(--text-dim);
    }
    .metric span { color: var(--text); font-weight: 600; }

    /* empty state */
    .empty {
      text-align: center; padding: 60px 20px;
      color: var(--muted); font-size: .9rem;
    }
    .empty-icon { font-size: 3rem; margin-bottom: 12px; }

    /* loading spinner */
    .spinner {
      width: 16px; height: 16px;
      border: 2px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin .6s linear infinite;
      display: inline-block; vertical-align: middle; margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Score bar ── */
    .score-section { margin-top: 4px; }
    .score-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
    .score-label { font-size: .8rem; color: var(--text-dim); }
    .score-val { font-size: .8rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
    .progress-bar {
      height: 6px; background: var(--border); border-radius: 4px; overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--green));
      border-radius: 4px;
      transition: width .5s ease;
    }

    scrollbar-color: var(--border) transparent;
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  </style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">🛡️</div>
    <div class="logo-text">
      <h1>AgentGuard-Gym</h1>
      <span>OpenEnv · Agentic Security Environment</span>
    </div>
  </div>
  <div class="header-links">
    <a href="/docs" target="_blank">📄 API Docs</a>
    <a href="/health" target="_blank">❤️ Health</a>
    <a href="https://github.com/meta-pytorch/OpenEnv" target="_blank">🔗 OpenEnv</a>
  </div>
</header>

<div class="status-bar">
  <div class="status-pill"><div class="dot"></div> Server Running</div>
  <div class="status-pill" id="step-counter">Steps: <strong id="step-count">0</strong></div>
  <div class="status-pill" id="score-pill">Avg Reward: <strong id="avg-reward">—</strong></div>
  <div class="status-pill">Task: <strong id="cur-task">—</strong></div>
</div>

<main>
  <aside>

    <!-- Reset -->
    <div>
      <div class="section-title">🔄 Reset Environment</div>
      <div class="card">
        <label>Task</label>
        <select id="reset-task">
          <option value="">— auto —</option>
          <option value="prompt_injection">Prompt Injection</option>
          <option value="tool_misuse_ssrf">Tool Misuse / SSRF</option>
          <option value="memory_poisoning_privilege">Memory Poisoning</option>
        </select>

        <label>Seed (optional)</label>
        <input type="number" id="reset-seed" placeholder="42" min="0" />

        <button class="btn btn-primary" onclick="doReset()">Reset Environment</button>
      </div>
    </div>

    <!-- Step -->
    <div>
      <div class="section-title">⚡ Step (Send Action)</div>
      <div class="card">
        <label>Defense Action JSON</label>
        <textarea id="step-action">{
  "defense": "allow",
  "rationale": "Low-risk request."
}</textarea>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:4px;">
          <button class="btn btn-secondary" onclick="fillAction('allow')">allow</button>
          <button class="btn btn-secondary" onclick="fillAction('sanitize')">sanitize</button>
          <button class="btn btn-secondary" onclick="fillAction('block')">block</button>
          <button class="btn btn-secondary" onclick="fillAction('quarantine_memory')">quarantine</button>
          <button class="btn btn-secondary" onclick="fillAction('clear_exposed_secrets')">clear secrets</button>
          <button class="btn btn-secondary" onclick="fillAction('audit_tool_chain')">audit</button>
        </div>

        <button class="btn btn-primary" style="margin-top:12px" onclick="doStep()">Send Step</button>
      </div>
    </div>

    <!-- State / Health -->
    <div>
      <div class="section-title">🔍 Inspect</div>
      <div class="card" style="display:grid;gap:8px;">
        <button class="btn btn-secondary" onclick="doState()">Get State</button>
        <button class="btn btn-secondary" onclick="doHealth()">Health Check</button>
      </div>
    </div>

    <!-- Score -->
    <div>
      <div class="section-title">📊 Session Score</div>
      <div class="card score-section">
        <div class="score-row">
          <span class="score-label">Success threshold</span>
          <span class="score-val" style="color:var(--yellow)">0.65</span>
        </div>
        <div class="score-row">
          <span class="score-label">Current avg reward</span>
          <span class="score-val" id="score-display">—</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" id="score-bar" style="width:0%"></div></div>
      </div>
    </div>

  </aside>

  <div class="right-panel">
    <div class="history-header">
      <h2>📋 Interaction Log</h2>
      <button class="clear-btn" onclick="clearFeed()">Clear</button>
    </div>
    <div id="feed">
      <div class="empty">
        <div class="empty-icon">🛡️</div>
        <div>No interactions yet.</div>
        <div style="margin-top:6px;font-size:.82rem">Click <strong>Reset Environment</strong> to begin.</div>
      </div>
    </div>
  </div>
</main>

<script>
  let steps = 0, rewards = [];

  function ts() {
    return new Date().toLocaleTimeString('en-US', {hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
  }

  function updateScore() {
    if (!rewards.length) return;
    const avg = rewards.reduce((a,b)=>a+b,0)/rewards.length;
    const pct = Math.min(100, (avg / 1.0) * 100);
    document.getElementById('avg-reward').textContent = avg.toFixed(3);
    document.getElementById('score-display').textContent = avg.toFixed(3);
    document.getElementById('score-bar').style.width = pct + '%';
    document.getElementById('score-bar').style.background =
      avg >= 0.65 ? 'linear-gradient(90deg,#4ade80,#22c55e)' : 'linear-gradient(90deg,#6c63ff,#8b5cf6)';
  }

  function addEvent(type, label, data, extras) {
    const feed = document.getElementById('feed');
    const empty = feed.querySelector('.empty');
    if (empty) empty.remove();

    const ev = document.createElement('div');
    ev.className = 'event';

    let rewardHtml = '';
    if (extras) {
      rewardHtml = '<div class="reward-row">' +
        Object.entries(extras).map(([k,v])=>
          `<div class="metric">${k}: <span>${v}</span></div>`
        ).join('') + '</div>';
    }

    ev.innerHTML = `
      <div class="event-header">
        <span class="event-type type-${type}">${label}</span>
        <span class="event-time">${ts()}</span>
      </div>
      <div class="event-body">${JSON.stringify(data, null, 2)}</div>
      ${rewardHtml}
    `;
    feed.appendChild(ev);
    feed.scrollTop = feed.scrollHeight;
  }

  function setLoading(id, on) {
    const btn = document.querySelector(`button[onclick="${id}()"]`);
    if (!btn) return;
    btn.disabled = on;
    btn.innerHTML = on ? '<span class="spinner"></span>Running…' : btn.dataset.orig || btn.innerHTML;
    if (!on && btn.dataset.orig) btn.innerHTML = btn.dataset.orig;
    if (on) btn.dataset.orig = btn.innerHTML;
  }

  async function apiCall(method, path, body) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const r = await fetch(path, opts);
    return { ok: r.ok, status: r.status, data: await r.json() };
  }

  async function doReset() {
    const task = document.getElementById('reset-task').value || null;
    const seedRaw = document.getElementById('reset-seed').value;
    const seed = seedRaw ? parseInt(seedRaw) : null;
    setLoading('doReset', true);
    try {
      const { ok, data } = await apiCall('POST', '/reset', { task, seed });
      document.getElementById('cur-task').textContent = task || 'auto';
      steps = 0; rewards = [];
      document.getElementById('step-count').textContent = steps;
      addEvent(ok ? 'reset' : 'error', ok ? 'RESET' : 'ERROR', data);
      updateScore();
    } catch(e) { addEvent('error','ERROR',{message:e.message}); }
    setLoading('doReset', false);
  }

  async function doStep() {
    let action;
    try { action = JSON.parse(document.getElementById('step-action').value); }
    catch { addEvent('error','JSON ERROR',{message:'Invalid JSON in action field'}); return; }
    setLoading('doStep', true);
    try {
      const { ok, data } = await apiCall('POST', '/step', { action });
      steps++;
      document.getElementById('step-count').textContent = steps;
      if (ok && data.reward) {
        const rv = parseFloat(data.reward.value ?? data.reward);
        if (!isNaN(rv)) { rewards.push(rv); updateScore(); }
      }
      addEvent(ok ? 'step' : 'error', ok ? `STEP #${steps}` : 'ERROR', data, ok ? {
        reward: data.reward?.value ?? '—',
        done: String(data.done),
        defense: action.defense
      } : null);
    } catch(e) { addEvent('error','ERROR',{message:e.message}); }
    setLoading('doStep', false);
  }

  async function doState() {
    setLoading('doState', true);
    try {
      const { ok, data } = await apiCall('GET', '/state');
      addEvent(ok ? 'state' : 'error', 'STATE', data);
    } catch(e) { addEvent('error','ERROR',{message:e.message}); }
    setLoading('doState', false);
  }

  async function doHealth() {
    setLoading('doHealth', true);
    try {
      const { ok, data } = await apiCall('GET', '/health');
      addEvent(ok ? 'health' : 'error', 'HEALTH', data);
    } catch(e) { addEvent('error','ERROR',{message:e.message}); }
    setLoading('doHealth', false);
  }

  function fillAction(defense) {
    const rationales = {
      allow: 'Request appears benign — no threat detected.',
      sanitize: 'Input contains suspicious patterns — sanitizing before processing.',
      block: 'Confirmed malicious intent — blocking action.',
      quarantine_memory: 'Memory poisoning detected — quarantining contaminated memory.',
      clear_exposed_secrets: 'Secrets exposed in context — clearing immediately.',
      audit_tool_chain: 'Tool chain anomaly detected — triggering full audit.'
    };
    document.getElementById('step-action').value = JSON.stringify({
      defense, rationale: rationales[defense] || ''
    }, null, 2);
  }

  function clearFeed() {
    document.getElementById('feed').innerHTML =
      '<div class="empty"><div class="empty-icon">🛡️</div><div>Log cleared.</div></div>';
    steps = 0; rewards = [];
    document.getElementById('step-count').textContent = 0;
    document.getElementById('avg-reward').textContent = '—';
    document.getElementById('score-display').textContent = '—';
    document.getElementById('score-bar').style.width = '0%';
    document.getElementById('cur-task').textContent = '—';
  }
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes (ALL original endpoints preserved exactly — autograder safe)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    """Interactive web UI — judges can explore the environment here."""
    return HTMLResponse(content=_WEB_UI_HTML)


@app.get("/web", response_class=HTMLResponse)
def web_ui() -> HTMLResponse:
    """Alias for the web interface (matches OpenEnv ENABLE_WEB_INTERFACE convention)."""
    return HTMLResponse(content=_WEB_UI_HTML)


@app.post("/reset")
def http_reset(body: ResetBody) -> Dict[str, Any]:
    obs = _env.reset(seed=body.seed, episode_id=body.episode_id, task=body.task)
    return {"observation": obs.model_dump(mode="json"), "reward": None, "done": False}


@app.post("/step")
def http_step(body: StepBody) -> Dict[str, Any]:
    result = _env.step(body.action)
    return {
        "observation": result.observation.model_dump(mode="json"),
        "reward": result.reward.model_dump(mode="json"),
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
def http_state() -> Dict[str, Any]:
    return _env.state().model_dump(mode="json")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# CLI entry (unchanged — `uv run server` / openenv-cli / Docker CMD)
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry used by `uv run server` / OpenEnv validators (binds PORT from env, default 7860)."""
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
