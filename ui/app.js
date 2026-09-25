/**
 * AgentTrace UI Application
 */

const API_BASE = '';

// State
let currentRun = null;
let graphData = null;
let selectedNode = null;
let simulation = null;
let svg = null;
let g = null;
let zoom = null;

// Node type colors
const NODE_COLORS = {
    'agent_input': '#3b82f6',
    'agent_output': '#8b5cf6',
    'tool_call': '#f59e0b',
    'tool_result': '#10b981',
    'error': '#ef4444',
    'state_read': '#06b6d4',
    'state_write': '#ec4899',
    'decision': '#6366f1',
    'checkpoint': '#14b8a6',
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initGraph();
    initEventListeners();
    loadRuns();
});

// Initialize graph visualization
function initGraph() {
    const container = document.getElementById('graph-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    svg = d3.select('#graph-svg')
        .attr('viewBox', [0, 0, width, height]);

    // Define arrow marker
    svg.append('defs').append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 20)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .append('path')
        .attr('d', 'M 0,-5 L 10,0 L 0,5')
        .attr('class', 'edge-arrow');

    g = svg.append('g');

    // Zoom behavior
    zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);
}

// Initialize event listeners
function initEventListeners() {
    // Run selection
    document.getElementById('run-select').addEventListener('change', (e) => {
        if (e.target.value) {
            loadGraph(e.target.value);
        }
    });

    // Refresh button
    document.getElementById('refresh-btn').addEventListener('click', () => {
        loadRuns();
        if (currentRun) {
            loadGraph(currentRun);
        }
    });

    // Infer relationships
    document.getElementById('infer-btn').addEventListener('click', () => {
        if (currentRun) {
            inferRelationships(currentRun);
        }
    });

    // Zoom controls
    document.getElementById('zoom-in').addEventListener('click', () => {
        svg.transition().call(zoom.scaleBy, 1.3);
    });

    document.getElementById('zoom-out').addEventListener('click', () => {
        svg.transition().call(zoom.scaleBy, 0.7);
    });

    document.getElementById('zoom-reset').addEventListener('click', () => {
        svg.transition().call(zoom.transform, d3.zoomIdentity);
    });

    // Tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            const tabId = e.target.dataset.tab;
            activateTab(tabId);
        });
    });

    // Trace buttons
    document.getElementById('trace-forward-btn').addEventListener('click', () => {
        const nodeId = document.getElementById('trace-node-id').value;
        if (nodeId) traceForward(nodeId);
    });

    document.getElementById('trace-backward-btn').addEventListener('click', () => {
        const nodeId = document.getElementById('trace-node-id').value;
        if (nodeId) traceBackward(nodeId);
    });

    // Timeline slider
    document.getElementById('timeline-slider').addEventListener('input', (e) => {
        updateTimeline(parseInt(e.target.value));
    });

    // Counterfactual modal
    document.getElementById('cf-cancel-btn').addEventListener('click', () => {
        document.getElementById('cf-modal').classList.add('hidden');
    });

    document.getElementById('cf-analyze-btn').addEventListener('click', () => {
        runCounterfactual();
    });
}

// Tab switching
function activateTab(tabId) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

    document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
    document.getElementById(tabId).classList.add('active');
}

// API calls
async function loadRuns() {
    try {
        const response = await fetch(`${API_BASE}/api/runs`);
        const data = await response.json();

        const select = document.getElementById('run-select');
        select.innerHTML = '<option value="">Select a run...</option>';

        data.runs.forEach(run => {
            const option = document.createElement('option');
            option.value = run.run_id;
            option.textContent = `${run.run_id} (${run.node_count} nodes)`;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load runs:', error);
    }
}

async function loadGraph(runId) {
    try {
        const response = await fetch(`${API_BASE}/api/runs/${runId}/graph`);
        graphData = await response.json();
        currentRun = runId;

        renderGraph();
        loadErrors();
        loadStats();
    } catch (error) {
        console.error('Failed to load graph:', error);
    }
}

async function traceForward(nodeId) {
    try {
        const response = await fetch(`${API_BASE}/api/runs/${currentRun}/trace/forward`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ node_id: nodeId }),
        });
        const data = await response.json();
        displayTraceResults(data.affected_nodes, 'forward');
        highlightNodes(data.affected_nodes.map(n => n.id));
    } catch (error) {
        console.error('Failed to trace forward:', error);
    }
}

async function traceBackward(nodeId) {
    try {
        const response = await fetch(`${API_BASE}/api/runs/${currentRun}/trace/backward`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ node_id: nodeId }),
        });
        const data = await response.json();
        displayTraceResults(data.cause_nodes, 'backward');
        highlightNodes(data.cause_nodes.map(n => n.id));
    } catch (error) {
        console.error('Failed to trace backward:', error);
    }
}

async function inferRelationships(runId) {
    try {
        const response = await fetch(`${API_BASE}/api/runs/${runId}/infer`, {
            method: 'POST',
        });
        const data = await response.json();
        alert(`Inferred ${data.count} new relationships`);
        loadGraph(runId);
    } catch (error) {
        console.error('Failed to infer relationships:', error);
    }
}

async function loadErrors() {
    try {
        const response = await fetch(`${API_BASE}/api/runs/${currentRun}/errors`);
        const data = await response.json();
        displayErrors(data.errors);
    } catch (error) {
        console.error('Failed to load errors:', error);
    }
}

async function loadStats() {
    const stats = graphData.statistics;
    displayStats(stats);
}

async function analyzeError(errorNodeId) {
    try {
        const response = await fetch(`${API_BASE}/api/runs/${currentRun}/analyze-error/${errorNodeId}`);
        const data = await response.json();
        displayErrorAnalysis(data);
    } catch (error) {
        console.error('Failed to analyze error:', error);
    }
}

async function runCounterfactual() {
    const nodeId = document.getElementById('cf-node-id').textContent;
    const alternativeData = document.getElementById('cf-alternative-data').value;

    try {
        const response = await fetch(`${API_BASE}/api/runs/${currentRun}/counterfactual`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: nodeId,
                alternative_data: JSON.parse(alternativeData),
            }),
        });
        const data = await response.json();
        displayCounterfactualResults(data);
    } catch (error) {
        console.error('Failed to run counterfactual:', error);
        alert('Failed to parse alternative data as JSON');
    }
}

// Rendering functions
function renderGraph() {
    if (!graphData) return;

    const container = document.getElementById('graph-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Clear previous content
    g.selectAll('*').remove();

    const nodes = graphData.nodes.map(n => ({ ...n }));
    const edges = graphData.edges.map(e => ({
        ...e,
        source: e.source_id,
        target: e.target_id,
    }));

    // Create force simulation
    simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(edges).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(30));

    // Draw edges
    const link = g.append('g')
        .selectAll('line')
        .data(edges)
        .join('line')
        .attr('class', 'edge-line')
        .attr('marker-end', 'url(#arrowhead)')
        .attr('stroke-opacity', d => d.confidence || 0.6);

    // Draw nodes
    const node = g.append('g')
        .selectAll('circle')
        .data(nodes)
        .join('circle')
        .attr('class', 'node-circle')
        .attr('r', 12)
        .attr('fill', d => NODE_COLORS[d.type] || '#666')
        .call(drag(simulation))
        .on('click', (event, d) => selectNode(d));

    // Node labels
    const label = g.append('g')
        .selectAll('text')
        .data(nodes)
        .join('text')
        .attr('class', 'node-label')
        .attr('dy', 25)
        .attr('text-anchor', 'middle')
        .text(d => d.agent_id.substring(0, 10));

    // Update positions on tick
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);

        label
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });

    // Update timeline range
    if (nodes.length > 0) {
        document.getElementById('timeline-slider').max = nodes.length - 1;
        document.getElementById('timeline-slider').value = nodes.length - 1;
    }
}

// Drag behavior
function drag(simulation) {
    function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }

    function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }

    function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }

    return d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended);
}

function selectNode(node) {
    selectedNode = node;

    // Update visual selection
    g.selectAll('.node-circle')
        .classed('selected', d => d.id === node.id);

    // Display node details
    displayNodeDetails(node);

    // Update trace input
    document.getElementById('trace-node-id').value = node.id;

    // Switch to details tab
    activateTab('node-details');
}

function displayNodeDetails(node) {
    const container = document.getElementById('node-info');
    container.innerHTML = `
        <div class="node-info-card">
            <h3>Type</h3>
            <span class="node-type ${node.type}">${node.type}</span>
        </div>
        <div class="node-info-card">
            <h3>Agent</h3>
            <div class="value">${node.agent_id}</div>
        </div>
        <div class="node-info-card">
            <h3>Timestamp</h3>
            <div class="value">${new Date(node.timestamp).toLocaleString()}</div>
        </div>
        <div class="node-info-card">
            <h3>Node ID</h3>
            <div class="value">${node.id}</div>
        </div>
        <div class="node-info-card">
            <h3>Data</h3>
            <pre class="value">${JSON.stringify(node.data, null, 2)}</pre>
        </div>
        <div class="node-info-card">
            <h3>Metadata</h3>
            <pre class="value">${JSON.stringify(node.metadata, null, 2)}</pre>
        </div>
        <div style="margin-top: 1rem;">
            <button class="btn" onclick="openCounterfactual('${node.id}')">What If...?</button>
        </div>
    `;
}

function displayTraceResults(nodes, direction) {
    const container = document.getElementById('trace-results');
    const arrow = direction === 'forward' ? '→' : '←';

    if (nodes.length === 0) {
        container.innerHTML = '<div class="placeholder">No nodes found</div>';
        return;
    }

    container.innerHTML = nodes.map(node => `
        <div class="trace-item" onclick="selectNodeById('${node.id}')">
            <span class="trace-arrow">${arrow}</span>
            <span class="node-type ${node.type}">${node.type}</span>
            <span>${node.agent_id}</span>
        </div>
    `).join('');
}

function displayErrors(errors) {
    const container = document.getElementById('error-list');

    if (errors.length === 0) {
        container.innerHTML = '<div class="placeholder">No errors found</div>';
        return;
    }

    container.innerHTML = errors.map(error => `
        <div class="error-item" onclick="analyzeError('${error.id}')">
            <h4>${error.data.error_type || 'Error'}</h4>
            <p>${error.data.error_message || 'No message'}</p>
            <small>${error.agent_id} - ${new Date(error.timestamp).toLocaleString()}</small>
        </div>
    `).join('');
}

function displayErrorAnalysis(analysis) {
    const node = analysis.error_node;
    const container = document.getElementById('node-info');

    container.innerHTML = `
        <div class="node-info-card">
            <h3>Error Analysis</h3>
            <div class="value">${node.data.error_type}: ${node.data.error_message}</div>
        </div>
        <div class="node-info-card">
            <h3>Root Causes (${analysis.root_causes.length})</h3>
            ${analysis.root_causes.map(rc => `
                <div class="trace-item" onclick="selectNodeById('${rc.id}')">
                    <span class="node-type ${rc.type}">${rc.type}</span>
                    <span>${rc.agent_id}</span>
                </div>
            `).join('')}
        </div>
        <div class="node-info-card">
            <h3>Likely Causes</h3>
            ${analysis.likely_causes.map(lc => `
                <div class="trace-item" onclick="selectNodeById('${lc.node.id}')">
                    <span>Score: ${(lc.score * 100).toFixed(0)}%</span>
                    <ul style="margin-left: 1rem; font-size: 0.875rem;">
                        ${lc.reasons.map(r => `<li>${r}</li>`).join('')}
                    </ul>
                </div>
            `).join('')}
        </div>
    `;

    activateTab('node-details');

    // Highlight the causal chain
    const nodeIds = analysis.root_causes.map(n => n.id);
    nodeIds.push(node.id);
    highlightNodes(nodeIds);
}

function displayStats(stats) {
    const container = document.getElementById('stats-content');

    container.innerHTML = `
        <div class="stat-card">
            <h3>Run ID</h3>
            <div class="value" style="font-size: 0.875rem; word-break: break-all;">${stats.run_id}</div>
        </div>
        <div class="stat-card">
            <h3>Nodes</h3>
            <div class="stat-value">${stats.node_count}</div>
        </div>
        <div class="stat-card">
            <h3>Edges</h3>
            <div class="stat-value">${stats.edge_count}</div>
        </div>
        <div class="stat-card">
            <h3>Agents</h3>
            <div class="stat-list">
                ${stats.agents.map(a => `<div class="stat-list-item"><span>${a}</span></div>`).join('')}
            </div>
        </div>
        <div class="stat-card">
            <h3>Node Types</h3>
            <div class="stat-list">
                ${Object.entries(stats.node_types).map(([type, count]) => `
                    <div class="stat-list-item">
                        <span class="node-type ${type}">${type}</span>
                        <span>${count}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

function displayCounterfactualResults(results) {
    const container = document.getElementById('cf-results');
    container.innerHTML = `
        <h4>Comparison Results</h4>
        <p>Common nodes: ${results.comparison.common_nodes}</p>
        <p>Different data: ${results.comparison.different_data.length} nodes</p>
        ${results.comparison.different_data.map(diff => `
            <div class="node-info-card">
                <h3>Node ${diff.node_id.substring(0, 8)}...</h3>
                <p>Original: ${JSON.stringify(diff.self_data)}</p>
                <p>Counterfactual: ${JSON.stringify(diff.other_data)}</p>
            </div>
        `).join('')}
        <p>Counterfactual run saved as: ${results.counterfactual_run_id}</p>
    `;
}

function highlightNodes(nodeIds) {
    const nodeSet = new Set(nodeIds);

    g.selectAll('.node-circle')
        .attr('stroke', d => nodeSet.has(d.id) ? '#e94560' : null)
        .attr('stroke-width', d => nodeSet.has(d.id) ? 3 : 2);

    g.selectAll('.edge-line')
        .classed('highlighted', d =>
            nodeSet.has(d.source.id || d.source) &&
            nodeSet.has(d.target.id || d.target)
        );
}

function updateTimeline(index) {
    if (!graphData) return;

    const nodes = graphData.nodes.slice(0, index + 1);
    const nodeIds = new Set(nodes.map(n => n.id));

    // Fade out nodes beyond timeline
    g.selectAll('.node-circle')
        .attr('opacity', d => nodeIds.has(d.id) ? 1 : 0.2);

    g.selectAll('.node-label')
        .attr('opacity', d => nodeIds.has(d.id) ? 1 : 0.2);

    g.selectAll('.edge-line')
        .attr('opacity', d =>
            nodeIds.has(d.source.id || d.source) &&
            nodeIds.has(d.target.id || d.target) ? 0.6 : 0.1
        );

    // Update label
    if (nodes.length > 0) {
        const lastNode = nodes[nodes.length - 1];
        document.getElementById('timeline-label').textContent =
            new Date(lastNode.timestamp).toLocaleTimeString();
    }
}

function openCounterfactual(nodeId) {
    const node = graphData.nodes.find(n => n.id === nodeId);
    if (!node) return;

    document.getElementById('cf-node-id').textContent = nodeId;
    document.getElementById('cf-original-data').textContent = JSON.stringify(node.data, null, 2);
    document.getElementById('cf-alternative-data').value = JSON.stringify(node.data, null, 2);
    document.getElementById('cf-results').innerHTML = '';
    document.getElementById('cf-modal').classList.remove('hidden');
}

// Global helpers
window.selectNodeById = function(nodeId) {
    const node = graphData.nodes.find(n => n.id === nodeId);
    if (node) selectNode(node);
};

window.openCounterfactual = openCounterfactual;
window.analyzeError = analyzeError;
