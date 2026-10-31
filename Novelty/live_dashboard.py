"""
live_dashboard.py  –  Ablation-Replay "Live Kubernetes" Dashboard
==================================================================
Replaces the broken real-kubectl live page with an emulator-style dashboard
that replays pre-computed ablation CSVs from Novelty/results/paper_outputs_complete/
using the exact same 4-chart design as the main simulator.

Endpoints (mounted at /live by web_dashboard.py):
  GET  /live              – full HTML page
  GET  /api/live/variants – list available (dataset, variant) pairs
  POST /api/live/load     – load a CSV into replay state  {dataset, variant}
  POST /api/live/step     – advance one tick, return chart data point
  POST /api/live/reset    – reset to tick 0
  GET  /api/live/status   – current replay state / KPIs
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
RESULTS_ROOT = _HERE / "results" / "paper_outputs_complete"

DATASET_DIRS = {
    "worldcup98":       RESULTS_ROOT / "worldcup98_fifa_ablation",
    "azurefunctions2019": RESULTS_ROOT / "azurefunctions2019_wikipedia_like_ablation",
}

SKIP_FILES = {"summary.csv", "novelty_effects_vs_paper.csv", "pairwise_vs_paper.csv"}
# Inplace-resize variants are excluded from the dashboard (N4/inplace is not used)
SKIP_PATTERNS = ["inplace", "novelty_4_inplace"]

# ── In-memory replay state ─────────────────────────────────────────────────────
class ReplayState:
    def __init__(self):
        self.rows: list[dict] = []
        self.tick_idx: int = 0
        self.dataset: str = ""
        self.variant: str = ""
        self.is_playing: bool = False
        self.cumulative_cost: float = 0.0
        self.slo_violations: int = 0
        self.total_ticks: int = 0

    def load(self, dataset: str, variant: str):
        path = DATASET_DIRS.get(dataset, DATASET_DIRS["worldcup98"]) / f"{variant}.csv"
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)
        self.tick_idx = 0
        self.dataset = dataset
        self.variant = variant
        self.is_playing = False
        self.cumulative_cost = 0.0
        self.slo_violations = 0
        self.total_ticks = len(self.rows)

    def step(self) -> Optional[dict]:
        if self.tick_idx >= len(self.rows):
            self.is_playing = False
            return None
        row = self.rows[self.tick_idx]
        self.tick_idx += 1
        cost = float(row.get("resource_cost") or 0)
        self.cumulative_cost += cost
        if str(row.get("slo_violation", "False")).lower() == "true":
            self.slo_violations += 1
        return row

    def reset(self):
        self.tick_idx = 0
        self.cumulative_cost = 0.0
        self.slo_violations = 0
        self.is_playing = False

_state = ReplayState()

# ── Helper: list variants ──────────────────────────────────────────────────────
def _list_variants() -> dict[str, list[str]]:
    result = {}
    for ds, path in DATASET_DIRS.items():
        if not path.exists():
            result[ds] = []
            continue
        files = sorted(
            f.stem for f in path.glob("*.csv")
            if f.name not in SKIP_FILES
            and not any(pat in f.stem for pat in SKIP_PATTERNS)
        )
        result[ds] = files
    return result

# ── Friendly display names ─────────────────────────────────────────────────────
def _display_name(variant: str) -> str:
    mapping = {
        "paper_hybrid_bilstm_dtr_rolling_fixed_monitor": "Paper Baseline (Hybrid BiLSTM+DTR)",
        "reactive_hpa_fixed_cpu":                        "Reactive HPA (Fixed CPU)",
        "proactive_hpa_bilstm_fixed_cpu":                "Proactive HPA (BiLSTM)",
        "burst_aware_hpa_bilstm_fixed_cpu":              "Burst-Aware HPA (BiLSTM)",
        "novelty_1_tuned_adaptive_burst":                "N1: Adaptive Burst Detection",
        "novelty_2_quantile_provisioning":               "N2: Quantile Provisioning",
        "novelty_3_adaptive_monitoring":                 "N4: Adaptive Monitoring",
        "novelty_4_inplace_resize":                      "N3: In-Place Resize (archived)",
        "optimized_novelty_n1_burst_n2_quantile":        "N1+N2: Burst+Quantile",
        "optimized_novelty_n1_burst_n2_quantile_n3_monitoring": "N1+N2+N4: +Monitoring",
        "optimized_novelty_n1_burst_n2_quantile_n3_monitoring_n4_inplace": "N1+N2+N4+N3: Full Stack",
        "optimized_novelty_n1_burst_n2_quantile_n4_inplace": "N1+N2+N3: +InPlace",
        "optimized_novelty_n1_burst_n3_monitoring":      "N1+N4: Burst+Monitoring",
        "optimized_novelty_n1_burst_n3_monitoring_n4_inplace": "N1+N4+N3: +InPlace",
        "optimized_novelty_n1_burst_n4_inplace":         "N1+N3: Burst+InPlace",
        "optimized_novelty_n2_quantile_n3_monitoring":   "N2+N4: Quantile+Monitoring",
        "optimized_novelty_n2_quantile_n3_monitoring_n4_inplace": "N2+N4+N3: +InPlace",
        "optimized_novelty_n2_quantile_n4_inplace":      "N2+N3: Quantile+InPlace",
        "optimized_novelty_n3_monitoring_n4_inplace":    "N4+N3: Monitoring+InPlace",
    }
    return mapping.get(variant, variant.replace("_", " ").title())

# ── API Endpoints ──────────────────────────────────────────────────────────────

@router.get("/api/live/variants")
def api_variants():
    variants = _list_variants()
    # Add display names
    result = {}
    for ds, vlist in variants.items():
        result[ds] = [{"id": v, "label": _display_name(v)} for v in vlist]
    return result


@router.post("/api/live/load")
def api_load(body: dict):
    dataset = body.get("dataset", "worldcup98")
    variant = body.get("variant", "paper_hybrid_bilstm_dtr_rolling_fixed_monitor")
    _state.load(dataset, variant)
    return {"status": "ok", "total_ticks": _state.total_ticks, "variant": variant}


@router.post("/api/live/step")
def api_step():
    row = _state.step()
    if row is None:
        return {"done": True}

    util = float(row.get("utilization") or 0)
    return {
        "done": False,
        "tick": int(row.get("tick") or _state.tick_idx - 1),
        "actual_qps": float(row.get("actual_request_rate") or 0),
        "predicted_qps": float(row.get("predicted_demand") or 0),
        "p10": float(row.get("p10") or 0),
        "p50": float(row.get("p50") or 0),
        "p90": float(row.get("p90") or 0),
        "replicas": int(row.get("replicas") or 1),
        "utilization_pct": round(util * 100, 1),
        "response_ms": float(row.get("response_ms") or 0),
        "slo_violation": str(row.get("slo_violation", "False")).lower() == "true",
        "resource_cost": float(row.get("resource_cost") or 0),
        "cumulative_cost": round(_state.cumulative_cost, 3),
        "burst": str(row.get("burst", "False")).lower() == "true",
        "execution_backend": row.get("execution_backend", "rolling"),
        # KPI running totals
        "slo_violation_count": _state.slo_violations,
        "slo_pct": round(100.0 * _state.slo_violations / _state.tick_idx, 3),
        "progress_pct": round(100.0 * _state.tick_idx / max(_state.total_ticks, 1), 1),
    }


@router.post("/api/live/reset")
def api_reset():
    _state.reset()
    return {"status": "ok"}


@router.post("/api/live/play")
def api_play(body: dict):
    _state.is_playing = body.get("playing", True)
    return {"status": "ok", "is_playing": _state.is_playing}


@router.get("/api/live/status")
def api_status():
    return {
        "dataset": _state.dataset,
        "variant": _state.variant,
        "display_name": _display_name(_state.variant) if _state.variant else "",
        "tick_idx": _state.tick_idx,
        "total_ticks": _state.total_ticks,
        "is_playing": _state.is_playing,
        "cumulative_cost": round(_state.cumulative_cost, 3),
        "slo_violations": _state.slo_violations,
        "slo_pct": round(100.0 * _state.slo_violations / max(_state.tick_idx, 1), 3),
        "progress_pct": round(100.0 * _state.tick_idx / max(_state.total_ticks, 1), 1),
    }


# ── HTML Page ──────────────────────────────────────────────────────────────────

@router.get("/live", response_class=HTMLResponse)
def live_page():
    return HTMLResponse(content=_HTML)


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Ablation Replay — Live Kubernetes</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
  tailwind.config = {
    theme: {
      extend: {
        colors: {
          accentBlue:   '#3b82f6',
          accentGreen:  '#22c55e',
          accentYellow: '#eab308',
          accentRed:    '#ef4444',
          accentPurple: '#a855f7',
          accentOrange: '#f97316',
        }
      }
    }
  }
</script>
<style>
  body { background: #0f172a; color: #e2e8f0; font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; }
  canvas { max-height: 200px; }
  .kpi-val { font-size: 1.5rem; font-weight: 700; line-height: 1.2; }
  .kpi-lbl { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; }
  .badge { display:inline-block; padding:1px 7px; border-radius:4px; font-size:0.65rem; font-weight:600; letter-spacing:0.06em; }
  select, input[type=range] { accent-color: #3b82f6; }
  ::-webkit-scrollbar { width:5px; } ::-webkit-scrollbar-track { background:#1e293b; } ::-webkit-scrollbar-thumb { background:#334155; border-radius:3px; }
</style>
</head>
<body class="min-h-screen flex flex-col">

<!-- Header -->
<header class="flex items-center justify-between px-5 py-2.5 bg-slate-900 border-b border-slate-800 shrink-0">
  <div class="flex items-center gap-3">
    <span class="text-accentBlue font-bold text-sm tracking-tight">&#9881; AutoScale Research</span>
    <span class="badge bg-accentPurple/20 text-accentPurple border border-accentPurple/30">ABLATION REPLAY</span>
    <span id="badge-playing" class="badge bg-slate-700 text-slate-300 border border-slate-600">PAUSED</span>
  </div>
  <nav class="flex gap-3 text-xs text-slate-400">
    <a href="/" class="hover:text-slate-200 transition-colors">&#8592; Simulator</a>
    <span class="text-slate-600">|</span>
    <span class="text-accentPurple font-medium">Ablation Replay</span>
  </nav>
</header>

<!-- Body -->
<div class="flex flex-1 overflow-hidden">

  <!-- Sidebar -->
  <aside class="w-56 shrink-0 bg-slate-900 border-r border-slate-800 overflow-y-auto flex flex-col gap-0 p-3 text-xs">

    <!-- Dataset picker -->
    <div class="mb-3">
      <h2 class="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1.5 border-b border-slate-800 pb-1">Dataset</h2>
      <select id="dataset-selector" onchange="onDatasetChange()" class="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-accentBlue">
        <option value="worldcup98">FIFA World Cup 98</option>
        <option value="azurefunctions2019">Azure Functions 2019</option>
      </select>
    </div>

    <!-- Variant picker -->
    <div class="mb-3">
      <h2 class="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1.5 border-b border-slate-800 pb-1">Variant</h2>
      <select id="variant-selector" class="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-accentBlue" style="height:auto;">
      </select>
    </div>

    <button onclick="loadVariant()" class="w-full bg-accentBlue hover:bg-blue-600 text-white font-semibold py-1.5 rounded text-xs mb-3 transition-colors">
      Load Variant
    </button>

    <!-- Controls -->
    <div class="mb-3">
      <h2 class="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-2 border-b border-slate-800 pb-1">Playback</h2>
      <div class="flex gap-1.5 mb-2">
        <button id="btn-play" onclick="togglePlay()" class="flex-1 bg-accentGreen/20 hover:bg-accentGreen/30 text-accentGreen border border-accentGreen/30 font-semibold py-1.5 rounded text-xs transition-colors">&#9654; Play</button>
        <button onclick="doStep()" class="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 font-semibold py-1.5 rounded text-xs transition-colors">&#9654;&#124; Step</button>
      </div>
      <button onclick="doReset()" class="w-full bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 font-semibold py-1.5 rounded text-xs transition-colors mb-2">&#8635; Reset</button>

      <div class="flex items-center justify-between mb-1">
        <label class="text-slate-400">Speed</label>
        <span id="speed-label" class="text-slate-300">300ms</span>
      </div>
      <input type="range" id="speed-slider" min="50" max="2000" value="300" step="50"
             oninput="document.getElementById('speed-label').innerText=this.value+'ms'"
             class="w-full h-1.5 rounded-full bg-slate-700">
    </div>

    <!-- Progress -->
    <div class="mb-3">
      <h2 class="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1.5 border-b border-slate-800 pb-1">Progress</h2>
      <div class="w-full bg-slate-800 rounded-full h-1.5 mb-1">
        <div id="progress-bar" class="bg-accentPurple h-1.5 rounded-full transition-all" style="width:0%"></div>
      </div>
      <div class="flex justify-between text-slate-500 text-[10px]">
        <span id="tick-label">Tick 0</span>
        <span id="total-label">/ 0</span>
      </div>
    </div>

    <!-- KPI panels -->
    <div class="space-y-2">
      <h2 class="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1.5 border-b border-slate-800 pb-1">Live KPIs</h2>

      <div class="bg-slate-800 rounded p-2">
        <div class="kpi-lbl">Replicas</div>
        <div class="kpi-val text-accentGreen" id="kpi-replicas">—</div>
      </div>
      <div class="bg-slate-800 rounded p-2">
        <div class="kpi-lbl">CPU Utilisation</div>
        <div class="kpi-val text-accentYellow" id="kpi-util">—</div>
      </div>
      <div class="bg-slate-800 rounded p-2">
        <div class="kpi-lbl">Response Time</div>
        <div class="kpi-val text-accentGreen" id="kpi-rt">—</div>
      </div>
      <div class="bg-slate-800 rounded p-2">
        <div class="kpi-lbl">Cumulative Cost</div>
        <div class="kpi-val text-accentPurple" id="kpi-cost">—</div>
      </div>
      <div class="bg-slate-800 rounded p-2">
        <div class="kpi-lbl">SLO Violations</div>
        <div class="kpi-val" id="kpi-slo">—</div>
      </div>
      <div class="bg-slate-800 rounded p-2">
        <div class="kpi-lbl">Backend</div>
        <div class="text-xs font-semibold mt-0.5" id="kpi-backend">—</div>
      </div>
    </div>

    <div class="mt-3 p-2 bg-slate-800/50 rounded text-[10px] text-slate-500">
      Replaying pre-computed ablation results. No live cluster connection required.
    </div>
  </aside>

  <!-- Charts -->
  <main class="flex-1 overflow-y-auto p-4 grid grid-cols-2 grid-rows-2 gap-4">

    <!-- Pods / QPS Chart -->
    <div class="bg-slate-900 border border-slate-800 rounded-lg p-3 flex flex-col">
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-xs font-semibold text-slate-300">Pod Replicas &amp; QPS</h3>
        <span class="badge bg-accentBlue/10 text-accentBlue border border-accentBlue/20">SCALING</span>
      </div>
      <div class="flex-1 relative"><canvas id="chart-pods"></canvas></div>
    </div>

    <!-- Utilisation Chart -->
    <div class="bg-slate-900 border border-slate-800 rounded-lg p-3 flex flex-col">
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-xs font-semibold text-slate-300">CPU Utilisation %</h3>
        <span class="badge bg-accentYellow/10 text-accentYellow border border-accentYellow/20">RESOURCE</span>
      </div>
      <div class="flex-1 relative"><canvas id="chart-util"></canvas></div>
    </div>

    <!-- Latency Chart -->
    <div class="bg-slate-900 border border-slate-800 rounded-lg p-3 flex flex-col">
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-xs font-semibold text-slate-300">Response Time (ms)</h3>
        <span class="badge bg-accentGreen/10 text-accentGreen border border-accentGreen/20">LATENCY</span>
      </div>
      <div class="flex-1 relative"><canvas id="chart-latency"></canvas></div>
    </div>

    <!-- Cost Chart -->
    <div class="bg-slate-900 border border-slate-800 rounded-lg p-3 flex flex-col">
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-xs font-semibold text-slate-300">Cumulative Resource Cost</h3>
        <span class="badge bg-accentPurple/10 text-accentPurple border border-accentPurple/20">COST</span>
      </div>
      <div class="flex-1 relative"><canvas id="chart-cost"></canvas></div>
    </div>

  </main>
</div>

<script>
// ── Chart setup ──────────────────────────────────────────────────────────────
const CHART_OPTS = {
  animation: false,
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { labels: { color: '#94a3b8', font: { size: 10 }, boxWidth: 12 } } },
  scales: {
    x: { ticks: { color: '#475569', font: { size: 9 }, maxTicksLimit: 8 }, grid: { color: '#1e293b' } },
    y: { ticks: { color: '#475569', font: { size: 9 } }, grid: { color: '#1e293b' } }
  }
};

function makeChart(id, datasets, yOpts) {
  return new Chart(document.getElementById(id), {
    type: 'line',
    data: { labels: [], datasets },
    options: { ...CHART_OPTS, scales: { ...CHART_OPTS.scales, y: { ...CHART_OPTS.scales.y, ...yOpts } } }
  });
}

const MAX_POINTS = 200;

const chartPods = makeChart('chart-pods', [
  { label: 'Replicas',   data: [], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', fill: false, tension: 0.3, pointRadius: 0, yAxisID: 'y' },
  { label: 'Actual QPS', data: [], borderColor: '#f97316', backgroundColor: 'transparent', fill: false, tension: 0.3, pointRadius: 0, borderDash: [3,3], yAxisID: 'y1' },
  { label: 'Pred QPS',   data: [], borderColor: '#a855f7', backgroundColor: 'transparent', fill: false, tension: 0.3, pointRadius: 0, yAxisID: 'y1' },
], {});

// Two y-axes for pods chart
chartPods.options.scales.y1 = { position: 'right', ticks: { color: '#475569', font: { size: 9 } }, grid: { drawOnChartArea: false } };

const chartUtil = makeChart('chart-util', [
  { label: 'Utilisation %', data: [], borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.1)', fill: true, tension: 0.3, pointRadius: 0 },
], { min: 0, max: 100 });

const chartLatency = makeChart('chart-latency', [
  { label: 'Response ms', data: [], borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.1)', fill: true, tension: 0.3, pointRadius: 0 },
  { label: 'SLO 200ms',   data: [], borderColor: '#ef4444', backgroundColor: 'transparent', fill: false, tension: 0, pointRadius: 0, borderDash: [4,2] },
], { min: 0 });

const chartCost = makeChart('chart-cost', [
  { label: 'Cumulative Cost', data: [], borderColor: '#a855f7', backgroundColor: 'rgba(168,85,247,0.15)', fill: true, tension: 0.3, pointRadius: 0 },
], {});

function pushPoint(chart, label, ...values) {
  chart.data.labels.push(label);
  values.forEach((v, i) => chart.data.datasets[i].data.push(v));
  if (chart.data.labels.length > MAX_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets.forEach(ds => ds.data.shift());
  }
  chart.update('none');
}

function clearCharts() {
  [chartPods, chartUtil, chartLatency, chartCost].forEach(c => {
    c.data.labels = [];
    c.data.datasets.forEach(ds => ds.data = []);
    c.update('none');
  });
}

// ── State ────────────────────────────────────────────────────────────────────
let variantMap = {};
let playTimer = null;
let isPlaying = false;

// ── On load ──────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  await fetchVariants();
  populateVariants('worldcup98');
  // Auto-load first variant
  await loadVariant();
});

async function fetchVariants() {
  const res = await fetch('/api/live/variants');
  variantMap = await res.json();
}

function onDatasetChange() {
  const ds = document.getElementById('dataset-selector').value;
  populateVariants(ds);
}

function populateVariants(ds) {
  const sel = document.getElementById('variant-selector');
  sel.innerHTML = '';
  const list = variantMap[ds] || [];
  list.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v.id;
    opt.textContent = v.label;
    sel.appendChild(opt);
  });
}

async function loadVariant() {
  stopPlay();
  clearCharts();
  const dataset = document.getElementById('dataset-selector').value;
  const variant = document.getElementById('variant-selector').value;
  const res = await fetch('/api/live/load', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ dataset, variant })
  });
  const data = await res.json();
  document.getElementById('total-label').textContent = '/ ' + data.total_ticks;
  document.getElementById('tick-label').textContent = 'Tick 0';
  document.getElementById('progress-bar').style.width = '0%';
  // Clear KPIs
  ['kpi-replicas','kpi-util','kpi-rt','kpi-cost','kpi-slo','kpi-backend'].forEach(id => {
    document.getElementById(id).textContent = '—';
  });
}

async function doStep() {
  const res = await fetch('/api/live/step', { method: 'POST' });
  const d = await res.json();
  if (d.done) { stopPlay(); return false; }
  applyPoint(d);
  return true;
}

function applyPoint(d) {
  const tick = d.tick;
  // Charts
  pushPoint(chartPods,    tick, d.replicas, d.actual_qps, d.predicted_qps + 100);
  pushPoint(chartUtil,    tick, d.utilization_pct);
  pushPoint(chartLatency, tick, d.response_ms, 200);
  pushPoint(chartCost,    tick, d.cumulative_cost);

  // KPIs
  document.getElementById('kpi-replicas').textContent = d.replicas;
  document.getElementById('kpi-util').textContent     = d.utilization_pct.toFixed(1) + '%';
  document.getElementById('kpi-rt').textContent       = d.response_ms.toFixed(0) + ' ms';
  document.getElementById('kpi-cost').textContent     = d.cumulative_cost.toFixed(2);

  const sloEl = document.getElementById('kpi-slo');
  sloEl.textContent = d.slo_pct.toFixed(2) + '%';
  sloEl.className = 'kpi-val ' + (d.slo_pct > 1 ? 'text-accentRed' : d.slo_pct > 0 ? 'text-accentYellow' : 'text-accentGreen');

  const be = d.execution_backend || 'rolling';
  const beEl = document.getElementById('kpi-backend');
  beEl.textContent = be.toUpperCase();
  beEl.className = 'text-xs font-semibold mt-0.5 ' + (be === 'inplace' ? 'text-accentPurple' : 'text-accentBlue');

  // Progress
  document.getElementById('tick-label').textContent = 'Tick ' + tick;
  document.getElementById('progress-bar').style.width = d.progress_pct + '%';
}

function togglePlay() {
  if (isPlaying) stopPlay(); else startPlay();
}

function startPlay() {
  if (isPlaying) return;
  isPlaying = true;
  document.getElementById('btn-play').textContent = '⏸ Pause';
  document.getElementById('btn-play').className = document.getElementById('btn-play').className
    .replace('accentGreen', 'accentRed');
  document.getElementById('badge-playing').textContent = 'PLAYING';
  document.getElementById('badge-playing').className = 'badge bg-accentGreen/20 text-accentGreen border border-accentGreen/30';
  scheduleStep();
}

function stopPlay() {
  isPlaying = false;
  clearTimeout(playTimer);
  document.getElementById('btn-play').textContent = '▶ Play';
  document.getElementById('btn-play').className = document.getElementById('btn-play').className
    .replace('accentRed', 'accentGreen');
  document.getElementById('badge-playing').textContent = 'PAUSED';
  document.getElementById('badge-playing').className = 'badge bg-slate-700 text-slate-300 border border-slate-600';
}

function scheduleStep() {
  if (!isPlaying) return;
  const speed = parseInt(document.getElementById('speed-slider').value);
  playTimer = setTimeout(async () => {
    const ok = await doStep();
    if (ok) scheduleStep();
    else stopPlay();
  }, speed);
}

async function doReset() {
  stopPlay();
  await fetch('/api/live/reset', { method: 'POST' });
  clearCharts();
  document.getElementById('tick-label').textContent = 'Tick 0';
  document.getElementById('progress-bar').style.width = '0%';
}
</script>
</body>
</html>
"""
