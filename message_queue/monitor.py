"""
Real-time Monitoring Dashboard for the Pipeline.

This module implements a lightweight HTTP server that serves a dynamic 
web dashboard. It visualizes the pipeline topology, node performance (TPS), 
queue backlogs, and shared memory pool usage.
"""
import http.server
import json
import threading
import time
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Pipeline Monitor Pro</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #020617;
            --card-bg: #0f172a;
            --card-border: #1e293b;
            --accent-primary: #38bdf8;
            --accent-secondary: #818cf8;
            --text-main: #f8fafc;
            --text-muted: #64748b;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --font-sans: 'Inter', -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * { box-sizing: border-box; }
        body { 
            font-family: var(--font-sans); 
            background: var(--bg-color); 
            color: var(--text-main);
            margin: 0; 
            padding: 32px;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 32px;
        }

        .logo-area {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 20px;
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
        }

        h1 { 
            font-size: 24px; 
            font-weight: 700; 
            margin: 0;
            letter-spacing: -0.02em;
        }

        #conn-status { 
            font-size: 11px; 
            padding: 4px 12px; 
            border-radius: 20px; 
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        #conn-status::before {
            content: '';
            width: 6px;
            height: 6px;
            background: currentColor;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px currentColor;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(1.2); }
            100% { opacity: 1; transform: scale(1); }
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 24px;
            margin-bottom: 24px;
        }

        .card { 
            background: var(--card-bg); 
            padding: 24px; 
            border-radius: 20px; 
            border: 1px solid var(--card-border);
            transition: border-color 0.3s ease;
        }

        .card:hover {
            border-color: rgba(56, 189, 248, 0.3);
        }

        .card-title { 
            margin: 0 0 20px 0; 
            font-size: 12px; 
            text-transform: uppercase; 
            letter-spacing: 0.1em; 
            color: var(--text-muted);
            font-weight: 700;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .metric-group {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .metric-value { 
            font-size: 36px; 
            font-weight: 700; 
            color: var(--text-main);
            font-family: var(--font-mono);
            letter-spacing: -0.05em;
        }

        .metric-sub { 
            color: var(--text-muted); 
            font-size: 13px; 
            font-weight: 500;
        }

        /* Topology Canvas */
        #topology-container { 
            width: 100%; 
            height: 500px; 
            background: #020617; 
            background-image: 
                linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size: 40px 40px;
            border-radius: 24px; 
            position: relative; 
            margin-bottom: 32px;
            overflow: hidden;
            border: 1px solid var(--card-border);
            box-shadow: inset 0 0 40px rgba(0,0,0,0.5);
        }

        .topo-node {
            position: absolute;
            width: 180px;
            height: 95px;
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(8px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            display: flex;
            flex-direction: column;
            padding: 12px 16px;
            z-index: 2;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: default;
        }

        .topo-node:hover {
            border-color: var(--accent-primary);
            background: rgba(30, 41, 59, 0.9);
            transform: translateY(-4px);
            box-shadow: 0 12px 24px -8px rgba(0,0,0,0.5);
        }

        .node-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .node-name { 
            font-size: 13px; 
            font-weight: 700; 
            color: var(--text-main);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .node-status-dot {
            width: 6px;
            height: 6px;
            background: var(--success);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--success);
        }

        .node-body {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }

        .node-rate { 
            font-size: 18px; 
            font-weight: 700; 
            color: var(--accent-primary); 
            font-family: var(--font-mono);
        }

        .node-latency { 
            font-size: 11px; 
            color: var(--text-muted); 
            font-weight: 600;
        }

        .topo-line {
            position: absolute;
            height: 2px;
            background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
            opacity: 0.3;
            z-index: 1;
            transform-origin: left center;
            transition: opacity 0.3s ease;
        }

        .marble {
            position: absolute;
            width: 8px;
            height: 8px;
            background: #fff;
            border-radius: 50%;
            z-index: 3;
            box-shadow: 0 0 12px #fff, 0 0 24px var(--accent-primary);
        }

        .topo-label {
            position: absolute;
            font-size: 10px;
            font-weight: 800;
            color: var(--text-muted);
            background: var(--card-border);
            padding: 4px 10px;
            border-radius: 6px;
            z-index: 2;
            letter-spacing: 0.1em;
            top: 20px;
        }

        .chart-container { 
            width: 100%; 
            height: 300px; 
            margin-top: 16px;
        }

        .node-stat-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid var(--card-border);
        }

        .node-stat-row:last-child { border: none; }

        .stat-label { font-size: 13px; font-weight: 600; color: var(--text-main); }
        .stat-value { font-family: var(--font-mono); font-size: 13px; color: var(--accent-primary); }
    </style>
</head>
<body>
    <header>
        <div class="logo-area">
            <div class="logo-icon">P</div>
            <h1>Pipeline Monitor Pro</h1>
        </div>
        <div id="conn-status">Live System</div>
    </header>
    
    <div id="topology-container">
        <div id="topo-start" class="topo-label">SOURCE</div>
        <div id="topo-end" class="topo-label">SINK</div>
    </div>

    <div class="dashboard-grid">
        <div class="card">
            <div class="card-title">Memory Pool</div>
            <div class="metric-group">
                <div class="metric-value" id="pool-usage">0 / 0</div>
                <div class="metric-sub" id="pool-sub">Shared Memory Usage (Used / Total)</div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-title">Global Throughput</div>
            <div style="display: flex; gap: 48px; margin-bottom: 12px;">
                <div class="metric-group">
                    <div class="metric-value" id="p-in-rate">0.00</div>
                    <div class="metric-sub">Input TPS</div>
                </div>
                <div class="metric-group">
                    <div class="metric-value" id="p-out-rate" style="color: var(--success);">0.00</div>
                    <div class="metric-sub">Output TPS</div>
                </div>
            </div>
            <div style="font-size: 11px; color: var(--text-muted); font-family: var(--font-mono); border-top: 1px solid var(--card-border); pt: 8px; margin-top: 8px; display: flex; gap: 20px;">
                <span>TOTAL IN: <span id="p-in-total" style="color: var(--text-main)">0</span></span>
                <span>TOTAL OUT: <span id="p-out-total" style="color: var(--success)">0</span></span>
            </div>
        </div>

        <div class="card">
            <div class="card-title">Node Health</div>
            <div id="node-stats"></div>
        </div>
    </div>

    <div class="dashboard-grid" style="grid-template-columns: 1.6fr 1fr;">
        <div class="card">
            <div class="card-title">Performance Trends</div>
            <div class="chart-container">
                <canvas id="rateChart"></canvas>
            </div>
        </div>
        <div class="card">
            <div class="card-title">Queue Backlogs</div>
            <div class="chart-container">
                <canvas id="backlogChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        /**
         * Pipeline Monitor Frontend Logic
         * 
         * This script handles:
         * 1. Chart initialization (Chart.js)
         * 2. Dynamic topology generation (DAG layout)
         * 3. Real-time data polling from /api/metrics
         * 4. Marble animations for data flow visualization
         */
        let rateChart, backlogChart;
        const PRESET_COLORS = [
            '#38bdf8', '#818cf8', '#34d399', '#fbbf24', '#f472b6', '#fb7185', '#2dd4bf', '#94a3b8'
        ];

        function initCharts() {
            // ... (Chart.js configuration)
            Chart.defaults.color = '#64748b';
            Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
            Chart.defaults.font.family = "'Inter', sans-serif";

            const rateCtx = document.getElementById('rateChart').getContext('2d');
            rateChart = new Chart(rateCtx, {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { intersect: false, mode: 'index' },
                    scales: { 
                        y: { 
                            beginAtZero: true, 
                            ticks: { font: { family: 'JetBrains Mono', size: 11 } }
                        },
                        x: { display: false }
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            align: 'end',
                            labels: { boxWidth: 6, usePointStyle: true, padding: 15, font: { size: 11, weight: '600' } }
                        }
                    },
                    animation: { duration: 0 }
                }
            });

            const backlogCtx = document.getElementById('backlogChart').getContext('2d');
            backlogChart = new Chart(backlogCtx, {
                type: 'bar',
                data: { labels: [], datasets: [{ label: 'Backlog', data: [], borderRadius: 8, backgroundColor: '#38bdf8' }] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: { 
                        y: { beginAtZero: true, ticks: { font: { family: 'JetBrains Mono', size: 11 } } },
                        x: { ticks: { font: { size: 10, weight: '600' } } }
                    },
                    plugins: { legend: { display: false } },
                    animation: { duration: 0 }
                }
            });
        }

        const nodeColors = {};
        const lastSuccessCount = {};
        const lastProducedCount = {};
        const nodePositions = {};

        function animateMarble(fromPos, toPos) {
            const topoContainer = document.getElementById('topology-container');
            const marble = document.createElement('div');
            marble.className = 'marble';
            
            const startX = fromPos.x;
            const startY = fromPos.y;
            const endX = toPos.x;
            const endY = toPos.y;
            
            marble.style.left = `${startX}px`;
            marble.style.top = `${startY}px`;
            topoContainer.appendChild(marble);

            const start = performance.now();
            const duration = 500;

            function step(timestamp) {
                const progress = Math.min((timestamp - start) / duration, 1);
                const ease = progress * (2 - progress); // easeOutQuad
                marble.style.left = `${startX + (endX - startX) * ease}px`;
                marble.style.top = `${startY + (endY - startY) * ease}px`;
                
                if (progress < 1) {
                    requestAnimationFrame(step);
                } else {
                    marble.remove();
                }
            }
            requestAnimationFrame(step);
        }

        function getColor(name) {
            if (!nodeColors[name]) {
                const index = Object.keys(nodeColors).length % PRESET_COLORS.length;
                nodeColors[name] = PRESET_COLORS[index];
            }
            return nodeColors[name];
        }

        /**
         * Automatically layout nodes based on their input/output partitions.
         * Uses a simple BFS-based layering algorithm to position nodes from left to right.
         */
        function updateTopology(nodesData) {
            const topoContainer = document.getElementById('topology-container');
            const nodes = Object.keys(nodesData);
            if (nodes.length === 0) return;

            const adj = {};
            const revAdj = {};
            nodes.forEach(n => { adj[n] = []; revAdj[n] = []; });

            nodes.forEach(u => {
                nodes.forEach(v => {
                    if (nodesData[u].output === nodesData[v].input && nodesData[u].output) {
                        adj[u].push(v);
                        revAdj[v].push(u);
                    }
                });
            });

            const layers = {};
            const queue = [];
            nodes.forEach(n => {
                if (revAdj[n].length === 0) {
                    layers[n] = 0;
                    queue.push(n);
                }
            });

            while (queue.length > 0) {
                const u = queue.shift();
                adj[u].forEach(v => {
                    layers[v] = Math.max(layers[v] || 0, layers[u] + 1);
                    queue.push(v);
                });
            }

            const nodesPerLayer = {};
            nodes.forEach(n => {
                const l = layers[n] || 0;
                nodesPerLayer[l] = (nodesPerLayer[l] || 0) + 1;
            });

            const currentLayerIdx = {};
            nodes.forEach(n => {
                const l = layers[n] || 0;
                const idx = currentLayerIdx[l] || 0;
                const totalInLayer = nodesPerLayer[l];
                
                const x = 100 + l * 320;
                const spacing = 140;
                const startY = (500 - (totalInLayer * spacing - (spacing - 95))) / 2;
                const y = startY + idx * spacing;
                
                nodePositions[n] = { x, y, centerX: x + 90, centerY: y + 47.5 };
                currentLayerIdx[l] = idx + 1;

                let nodeEl = document.getElementById(`topo-${n}`);
                if (!nodeEl) {
                    nodeEl = document.createElement('div');
                    nodeEl.id = `topo-${n}`;
                    nodeEl.className = 'topo-node';
                    topoContainer.appendChild(nodeEl);
                }
                nodeEl.style.left = `${x}px`;
                nodeEl.style.top = `${y}px`;
                const latencyMs = ((nodesData[n].latency || 0) * 1000).toFixed(1);
                const inRate = nodesData[n].rate || 0;
                const outRate = nodesData[n].produced_rate || 0;
                
                nodeEl.innerHTML = `
                    <div class="node-header">
                        <div class="node-name">${n}</div>
                        <div class="node-status-dot"></div>
                    </div>
                    <div class="node-body" style="display: flex; justify-content: space-between; align-items: flex-end;">
                        <div style="display: flex; flex-direction: column; gap: 2px;">
                            <div style="font-size: 14px; font-weight: 700; color: var(--accent-primary); font-family: var(--font-mono);">IN: ${inRate.toFixed(2)}</div>
                            <div style="font-size: 12px; font-weight: 600; color: var(--success); font-family: var(--font-mono);">OUT: ${outRate.toFixed(2)}</div>
                        </div>
                        <div class="node-latency">${latencyMs}ms</div>
                    </div>
                `;
            });

            nodes.forEach(u => {
                if (revAdj[u].length === 0) {
                    const startPos = { x: 40, y: nodePositions[u].centerY };
                    const endPos = { x: nodePositions[u].x, y: nodePositions[u].centerY };
                    drawLink(`line-in-${u}`, startPos, endPos);
                    const startLabel = document.getElementById('topo-start');
                    startLabel.style.left = '20px';
                }

                adj[u].forEach(v => {
                    const startPos = { x: nodePositions[u].x + 180, y: nodePositions[u].centerY };
                    const endPos = { x: nodePositions[v].x, y: nodePositions[v].centerY };
                    drawLink(`line-${u}-${v}`, startPos, endPos);
                    
                    const diff = nodesData[u].produced - (lastProducedCount[u] || 0);
                    if (lastProducedCount[u] !== undefined && diff > 0) {
                        const numMarbles = Math.min(diff, 5);
                        for (let i = 0; i < numMarbles; i++) {
                            setTimeout(() => animateMarble(startPos, endPos), i * (1000 / numMarbles));
                        }
                    }
                });

                if (adj[u].length === 0) {
                    const startPos = { x: nodePositions[u].x + 180, y: nodePositions[u].centerY };
                    const endPos = { x: nodePositions[u].x + 280, y: nodePositions[u].centerY };
                    drawLink(`line-${u}-out`, startPos, endPos);
                    const endLabel = document.getElementById('topo-end');
                    endLabel.style.left = `${endPos.x + 10}px`;
                    
                    const diff = nodesData[u].produced - (lastProducedCount[u] || 0);
                    if (lastProducedCount[u] !== undefined && diff > 0) {
                        const numMarbles = Math.min(diff, 5);
                        for (let i = 0; i < numMarbles; i++) {
                            setTimeout(() => animateMarble(startPos, endPos), i * (1000 / numMarbles));
                        }
                    }
                }
                lastProducedCount[u] = nodesData[u].produced;
                lastSuccessCount[u] = nodesData[u].success;
            });
        }

        function drawLink(id, start, end) {
            const topoContainer = document.getElementById('topology-container');
            let line = document.getElementById(id);
            if (!line) {
                line = document.createElement('div');
                line.id = id;
                line.className = 'topo-line';
                topoContainer.appendChild(line);
            }
            const dx = end.x - start.x;
            const dy = end.y - start.y;
            const length = Math.sqrt(dx * dx + dy * dy);
            const angle = Math.atan2(dy, dx) * 180 / Math.PI;
            line.style.width = `${length}px`;
            line.style.left = `${start.x}px`;
            line.style.top = `${start.y}px`;
            line.style.transform = `rotate(${angle}deg)`;
        }

        function formatBytes(bytes, decimals = 2) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const dm = decimals < 0 ? 0 : decimals;
            const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
        }

        async function update() {
            try {
                const resp = await fetch('/api/metrics');
                if (!resp.ok) throw new Error(`HTTP error! status: ${resp.status}`);
                const data = await resp.json();
                
                const statusEl = document.getElementById('conn-status');
                statusEl.innerText = 'Live System';

                if (data.pool) {
                    const used = data.pool.used_size || 0;
                    const total = data.pool.total_size || 0;
                    const allocs = data.pool.num_allocations || 0;
                    document.getElementById('pool-usage').innerText = `${formatBytes(used)} / ${formatBytes(total)}`;
                    document.getElementById('pool-sub').innerText = `Shared Memory: ${allocs} active allocations`;
                }

                if (data.pipeline) {
                    document.getElementById('p-in-rate').innerText = (data.pipeline.input_rate || 0).toFixed(2);
                    document.getElementById('p-out-rate').innerText = (data.pipeline.output_rate || 0).toFixed(2);
                    document.getElementById('p-in-total').innerText = data.pipeline.input_count || 0;
                    document.getElementById('p-out-total').innerText = data.pipeline.output_count || 0;
                    window.p_in_count = data.pipeline.input_count;
                    window.p_out_count = data.pipeline.output_count;
                }

                if (data.nodes) {
                    updateTopology(data.nodes);
                    
                    let html = '';
                    const now = Date.now() / 1000;
                    Object.entries(data.nodes).forEach(([name, stats]) => {
                        html += `
                            <div class="node-stat-row">
                                <span class="stat-label">${name}</span>
                                <div style="text-align: right">
                                    <div class="stat-value">${stats.success} OK / ${stats.fail} ERR</div>
                                    <div style="font-size: 10px; color: var(--text-muted); font-family: var(--font-mono)">
                                        IN: ${(stats.rate || 0).toFixed(2)} | OUT: ${(stats.produced_rate || 0).toFixed(2)}
                                    </div>
                                </div>
                            </div>
                        `;

                        if (rateChart) {
                            let dataset = rateChart.data.datasets.find(d => d.label === name);
                            if (!dataset) {
                                const color = getColor(name);
                                dataset = {
                                    label: name,
                                    data: [],
                                    borderColor: color,
                                    backgroundColor: color + '20',
                                    borderWidth: 2,
                                    fill: true,
                                    tension: 0.4,
                                    pointRadius: 0
                                };
                                rateChart.data.datasets.push(dataset);
                            }
                            dataset.data.push({x: now, y: stats.rate || 0});
                            if (dataset.data.length > 60) dataset.data.shift();
                        }
                    });
                    document.getElementById('node-stats').innerHTML = html;
                }

                if (backlogChart && data.partitions) {
                    const partitions = Object.keys(data.partitions);
                    backlogChart.data.labels = partitions.map(p => p.split('/').pop());
                    backlogChart.data.datasets[0].data = partitions.map(p => data.partitions[p].size);
                    backlogChart.update();
                }

                if (rateChart) {
                    if (rateChart.data.labels.length > 60) rateChart.data.labels.shift();
                    rateChart.data.labels.push(new Date().toLocaleTimeString());
                    rateChart.update();
                }

                // Animate input marble
                if (window.lastInputCount !== undefined && window.p_in_count > window.lastInputCount) {
                    const diff = window.p_in_count - window.lastInputCount;
                    const numMarbles = Math.min(diff, 5);
                    Object.keys(data.nodes).forEach(n => {
                        const isEntry = !Object.values(data.nodes).some(other => other.output === data.nodes[n].input && other.output);
                        if (isEntry && nodePositions[n]) {
                            for (let i = 0; i < numMarbles; i++) {
                                setTimeout(() => animateMarble({x: 40, y: nodePositions[n].centerY}, {x: nodePositions[n].x, y: nodePositions[n].centerY}), i * (1000 / numMarbles));
                            }
                        }
                    });
                }
                window.lastInputCount = window.p_in_count;

            } catch (e) { 
                console.error("Update failed:", e);
                document.getElementById('conn-status').innerText = 'Offline';
                document.getElementById('conn-status').style.color = 'var(--danger)';
            }
        }

        initCharts();
        setInterval(update, 1000);
    </script>
</body>
</html>
"""

class MetricsHandler(http.server.BaseHTTPRequestHandler):
    pipeline_metrics = {}

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode('utf-8'))
        elif self.path == '/api/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            try:
                # Convert proxy dict to real dict for JSON serialization
                # We use dict() to ensure we have a snapshot of the current state
                nodes = {}
                nodes_proxy = self.pipeline_metrics.get('nodes', {})
                for k, v in nodes_proxy.items():
                    nodes[k] = dict(v) if hasattr(v, 'items') else v

                data = {
                    'nodes': nodes,
                    'pool': dict(self.pipeline_metrics.get('pool', {})),
                    'partitions': dict(self.pipeline_metrics.get('partitions', {})),
                    'pipeline': dict(self.pipeline_metrics.get('pipeline', {}))
                }
                self.wfile.write(json.dumps(data).encode())
            except Exception as e:
                # Fallback for partial data or errors during serialization
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        return # Silent

def start_monitor_server(metrics: Dict, port: int = 5000):
    MetricsHandler.pipeline_metrics = metrics
    class ReuseAddrHTTPServer(http.server.HTTPServer):
        allow_reuse_address = True
        
    try:
        server = ReuseAddrHTTPServer(('0.0.0.0', port), MetricsHandler)
        logger.info(f"Monitor dashboard available at http://localhost:{port}")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Failed to start monitor server on port {port}: {e}")
