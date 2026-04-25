const socket = io();

// Risk thresholds for joint angle highlighting
const RISK_THRESHOLDS = {
    Neck:       30,
    Back:       20,
    R_Shoulder: 45,
    L_Shoulder: 45,
    R_Elbow:    60,
    L_Elbow:    60,
    R_Wrist:    30,
    L_Wrist:    30,
};

const jointList = [
    'Neck', 'Back',
    'R_Shoulder', 'L_Shoulder',
    'R_Elbow',    'L_Elbow',
    'R_Wrist',    'L_Wrist',
];

const jointComponents = {
    Neck:       ['Neck', 'Neck_Roll', 'Neck_Yaw'],
    R_Shoulder: ['R_Shoulder', 'R_Shoulder_Abduction'],
    L_Shoulder: ['L_Shoulder', 'L_Shoulder_Abduction'],
    R_Elbow:    ['R_Elbow', 'R_Elbow_Roll'],
    L_Elbow:    ['L_Elbow', 'L_Elbow_Roll'],
    R_Wrist:    ['R_Wrist', 'R_Wrist_Roll', 'R_Wrist_Yaw'],
    L_Wrist:    ['L_Wrist', 'L_Wrist_Roll', 'L_Wrist_Yaw'],
    Back:       ['Trunk_Pitch', 'Trunk_Roll', 'Trunk_Yaw'],
    Legs:       ['legs_score'],
};

const componentColors = {
    Neck:               '#00d4ff',
    Roll:               '#ffaa00',
    Yaw:                '#ff6b6b',
    Abduction:          '#27ae60',
    Trunk_Pitch:        '#00d4ff',
    Trunk_Roll:         '#ffaa00',
    Trunk_Yaw:          '#ff6b6b',
    legs_score:         '#e67e22',
    R_Shoulder:         '#00d4ff',
    L_Shoulder:         '#00d4ff',
    R_Elbow:            '#00d4ff',
    L_Elbow:            '#00d4ff',
    R_Wrist:            '#00d4ff',
    L_Wrist:            '#00d4ff',
    R_Elbow_Roll:       '#ffaa00',
    L_Elbow_Roll:       '#ffaa00',
    R_Wrist_Roll:       '#ffaa00',
    L_Wrist_Roll:       '#ffaa00',
    R_Wrist_Yaw:        '#ff6b6b',
    L_Wrist_Yaw:        '#ff6b6b',
    R_Shoulder_Abduction: '#27ae60',
    L_Shoulder_Abduction: '#27ae60',
    Neck_Roll:          '#ffaa00',
    Neck_Yaw:           '#ff6b6b',
};

function colorForComp(comp) {
    if (componentColors[comp]) return componentColors[comp];
    const suffix = comp.split('_').pop();
    return componentColors[suffix] || '#00d4ff';
}

const history = {};
for (const joint in jointComponents) {
    jointComponents[joint].forEach(comp => { history[comp] = []; });
}

const rulaRightHistory = [];
const rulaLeftHistory = [];
const rebaRightHistory = [];
const rebaLeftHistory = [];

const MAX_HISTORY = 6000;
const trendCharts = {};

let rulaChartRight, rulaChartLeft, rulaChartBoth;
let rebaChartRight, rebaChartLeft, rebaChartBoth;

function initTrendCharts() {
    for (const joint in jointComponents) {
        const canvas = document.getElementById(`trend-${joint}`);
        if (!canvas) continue;
        const ctx        = canvas.getContext('2d');
        const components = jointComponents[joint];
        const datasets   = components.map(comp => ({
            label:           comp.replace(/_/g, ' '),
            data:            [],
            borderColor:     colorForComp(comp),
            backgroundColor: 'transparent',
            fill:            false,
            tension:         0,
            pointRadius:     0,
            borderWidth:     1.5,
        }));
        trendCharts[joint] = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets },
            options: {
                responsive:          true,
                maintainAspectRatio: false,
                animation:           false,
                scales: {
                    y: { min: -180, max: 180, ticks: { stepSize: 45 }, title: { display: true, text: 'Degrees' } },
                    x: { type: 'linear', ticks: { display: false } },
                },
                plugins: { legend: { display: true, position: 'top', labels: { font: { size: 8 } } } },
            },
        });
    }
}

function initRulaCharts() {
    const ctxRight = document.getElementById('trend-rula-right')?.getContext('2d');
    if (ctxRight) {
        rulaChartRight = new Chart(ctxRight, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Right RULA', data: [], borderColor: '#E84545', backgroundColor: 'rgba(232,69,69,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 1.5 }] },
            options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { y: { min: 1, max: 7, ticks: { stepSize: 1 } }, x: { ticks: { display: false } } }, plugins: { legend: { display: false } } }
        });
    }
    const ctxLeft = document.getElementById('trend-rula-left')?.getContext('2d');
    if (ctxLeft) {
        rulaChartLeft = new Chart(ctxLeft, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Left RULA', data: [], borderColor: '#2980B9', backgroundColor: 'rgba(41,128,185,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 1.5 }] },
            options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { y: { min: 1, max: 7, ticks: { stepSize: 1 } }, x: { ticks: { display: false } } }, plugins: { legend: { display: false } } }
        });
    }
    const ctxBoth = document.getElementById('trend-rula-both')?.getContext('2d');
    if (ctxBoth) {
        rulaChartBoth = new Chart(ctxBoth, {
            type: 'line',
            data: { labels: [], datasets: [
                { label: 'Right RULA', data: [], borderColor: '#E84545', backgroundColor: 'transparent', tension: 0, pointRadius: 0, borderWidth: 2 },
                { label: 'Left RULA', data: [], borderColor: '#2980B9', backgroundColor: 'transparent', tension: 0, pointRadius: 0, borderWidth: 2 }
            ] },
            options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { y: { min: 1, max: 7, ticks: { stepSize: 1 } }, x: { ticks: { display: false } } }, plugins: { legend: { labels: { color: '#fff' } } } }
        });
    }
}

function initRebaCharts() {
    const ctxRight = document.getElementById('trend-reba-right')?.getContext('2d');
    if (ctxRight) {
        rebaChartRight = new Chart(ctxRight, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Right REBA', data: [], borderColor: '#E84545', backgroundColor: 'rgba(232,69,69,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 1.5 }] },
            options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { y: { min: 1, max: 15, ticks: { stepSize: 2 } }, x: { ticks: { display: false } } }, plugins: { legend: { display: false } } }
        });
    }
    const ctxLeft = document.getElementById('trend-reba-left')?.getContext('2d');
    if (ctxLeft) {
        rebaChartLeft = new Chart(ctxLeft, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Left REBA', data: [], borderColor: '#2980B9', backgroundColor: 'rgba(41,128,185,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 1.5 }] },
            options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { y: { min: 1, max: 15, ticks: { stepSize: 2 } }, x: { ticks: { display: false } } }, plugins: { legend: { display: false } } }
        });
    }
    const ctxBoth = document.getElementById('trend-reba-both')?.getContext('2d');
    if (ctxBoth) {
        rebaChartBoth = new Chart(ctxBoth, {
            type: 'line',
            data: { labels: [], datasets: [
                { label: 'Right REBA', data: [], borderColor: '#E84545', backgroundColor: 'transparent', tension: 0, pointRadius: 0, borderWidth: 2 },
                { label: 'Left REBA', data: [], borderColor: '#2980B9', backgroundColor: 'transparent', tension: 0, pointRadius: 0, borderWidth: 2 }
            ] },
            options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { y: { min: 1, max: 15, ticks: { stepSize: 2 } }, x: { ticks: { display: false } } }, plugins: { legend: { labels: { color: '#fff' } } } }
        });
    }
}

initTrendCharts();
initRulaCharts();
initRebaCharts();

// Connection status
const statusEl = document.getElementById('conn-status');
if (!statusEl) {
    const newStatus = document.createElement('div');
    newStatus.id = 'conn-status';
    newStatus.style.position = 'fixed';
    newStatus.style.top = '10px';
    newStatus.style.right = '10px';
    newStatus.style.background = '#12151d';
    newStatus.style.border = '1px solid #1e2330';
    newStatus.style.padding = '4px 12px';
    newStatus.style.borderRadius = '20px';
    newStatus.style.fontFamily = 'monospace';
    newStatus.style.fontSize = '12px';
    newStatus.style.zIndex = '9999';
    document.body.appendChild(newStatus);
}

socket.on('connect', () => {
    const el = document.getElementById('conn-status');
    if (el) el.innerText = '⬤ Connected';
    console.log('Socket connected');
});
socket.on('disconnect', () => {
    const el = document.getElementById('conn-status');
    if (el) el.innerText = '⬤ Disconnected';
    console.log('Socket disconnected');
});

// ── Debug panel for raw sensor data ──
let rawDebugDiv = null;
function createRawDebugPanel() {
    if (rawDebugDiv) return;
    const div = document.createElement('div');
    div.id = 'raw-debug-panel';
    div.style.position = 'fixed';
    div.style.bottom = '10px';
    div.style.right = '10px';
    div.style.background = '#0b0d12';
    div.style.border = '1px solid #00d4ff';
    div.style.borderRadius = '6px';
    div.style.padding = '6px 10px';
    div.style.fontFamily = 'JetBrains Mono, monospace';
    div.style.fontSize = '10px';
    div.style.color = '#00d4ff';
    div.style.maxWidth = '300px';
    div.style.backgroundColor = 'rgba(0,0,0,0.8)';
    div.style.backdropFilter = 'blur(4px)';
    div.style.zIndex = '9998';
    div.style.pointerEvents = 'none';
    div.innerHTML = '<strong>📡 Raw sensors</strong><br>Waiting for data...';
    document.body.appendChild(div);
    rawDebugDiv = div;
}
createRawDebugPanel();

// Listen for raw sensor data (emitted by fixed data_processor.py)
socket.on('raw_sensors', (data) => {
    console.log('[RAW]', data);
    if (rawDebugDiv) {
        let html = '<strong>📡 Raw sensors</strong><br>';
        for (const [sid, vals] of Object.entries(data)) {
            html += `${sid}: r=${vals.roll.toFixed(1)}° p=${vals.pitch.toFixed(1)}° y=${vals.yaw.toFixed(1)}°<br>`;
        }
        rawDebugDiv.innerHTML = html;
        rawDebugDiv.style.opacity = '0.9';
        setTimeout(() => { if (rawDebugDiv) rawDebugDiv.style.opacity = '0.6'; }, 2000);
    }
});

// Main angles event (joint angles, RULA, REBA)
socket.on('angles', data => {
    if (rawDebugDiv) {
        rawDebugDiv.style.borderColor = '#00e5a0';
        setTimeout(() => { if (rawDebugDiv) rawDebugDiv.style.borderColor = '#00d4ff'; }, 1000);
    }
    if (data.angles)  updateAngles(data.angles);
    if (data.angles)  updateTrends(data.angles, data.legs_score);
    if (data.rula)    updateRULA(data.rula);
    if (data.reba)    updateREBA(data.reba);
    // Update RULA/REBA trends
    if (data.rula) {
        if (data.rula.right && data.rula.right.final !== undefined) {
            rulaRightHistory.push(data.rula.right.final);
            if (rulaRightHistory.length > 600) rulaRightHistory.shift();
        }
        if (data.rula.left && data.rula.left.final !== undefined) {
            rulaLeftHistory.push(data.rula.left.final);
            if (rulaLeftHistory.length > 600) rulaLeftHistory.shift();
        }
        if (rulaChartRight) {
            rulaChartRight.data.labels = Array.from({ length: rulaRightHistory.length }, (_, i) => i);
            rulaChartRight.data.datasets[0].data = rulaRightHistory;
            rulaChartRight.update();
        }
        if (rulaChartLeft) {
            rulaChartLeft.data.labels = Array.from({ length: rulaLeftHistory.length }, (_, i) => i);
            rulaChartLeft.data.datasets[0].data = rulaLeftHistory;
            rulaChartLeft.update();
        }
        if (rulaChartBoth) {
            const maxLen = Math.max(rulaRightHistory.length, rulaLeftHistory.length);
            rulaChartBoth.data.labels = Array.from({ length: maxLen }, (_, i) => i);
            rulaChartBoth.data.datasets[0].data = rulaRightHistory;
            rulaChartBoth.data.datasets[1].data = rulaLeftHistory;
            rulaChartBoth.update();
        }
    }
    if (data.reba) {
        if (data.reba.right && data.reba.right.final !== undefined) {
            rebaRightHistory.push(data.reba.right.final);
            if (rebaRightHistory.length > 600) rebaRightHistory.shift();
        }
        if (data.reba.left && data.reba.left.final !== undefined) {
            rebaLeftHistory.push(data.reba.left.final);
            if (rebaLeftHistory.length > 600) rebaLeftHistory.shift();
        }
        if (rebaChartRight) {
            rebaChartRight.data.labels = Array.from({ length: rebaRightHistory.length }, (_, i) => i);
            rebaChartRight.data.datasets[0].data = rebaRightHistory;
            rebaChartRight.update();
        }
        if (rebaChartLeft) {
            rebaChartLeft.data.labels = Array.from({ length: rebaLeftHistory.length }, (_, i) => i);
            rebaChartLeft.data.datasets[0].data = rebaLeftHistory;
            rebaChartLeft.update();
        }
        if (rebaChartBoth) {
            const maxLen = Math.max(rebaRightHistory.length, rebaLeftHistory.length);
            rebaChartBoth.data.labels = Array.from({ length: maxLen }, (_, i) => i);
            rebaChartBoth.data.datasets[0].data = rebaRightHistory;
            rebaChartBoth.data.datasets[1].data = rebaLeftHistory;
            rebaChartBoth.update();
        }
    }
});

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.innerText = (val === undefined || val === null) ? '-' : val;
}

function updateAngles(angles) {
    let html = '';
    let any  = false;
    for (const joint of jointList) {
        const value = (joint === 'Back') ? angles['Trunk_Pitch'] : angles[joint];
        if (value === undefined || value === null) continue;
        any = true;
        const threshold = RISK_THRESHOLDS[joint] || 999;
        const risky     = Math.abs(value) > threshold;
        const clz       = risky ? 'joint-item risky' : 'joint-item';
        
        html += `<div class="${clz}">
            <div class="jname">${joint.replace('_', ' ')}</div>
            <div class="jval">${value.toFixed(1)}°</div>
            <div class="jwarn"><i class="fas fa-exclamation-triangle"></i> Threshold exceeded</div>
        </div>`;
    }
    if (!any) html = '<p style="color:var(--text-dim);">No joint data yet.</p>';
    const el = document.getElementById('angles');
    if (el) el.innerHTML = html;
}

function updateTrends(angles, legs_score) {
    const merged = Object.assign({}, angles, { legs_score: legs_score ?? null });
    for (const joint in jointComponents) {
        jointComponents[joint].forEach(comp => {
            const val = merged[comp];
            if (val !== undefined && val !== null) {
                history[comp].push(val);
                if (history[comp].length > MAX_HISTORY) history[comp].shift();
            }
        });
    }
    for (const joint in jointComponents) {
        const chart = trendCharts[joint];
        if (!chart) continue;
        const components = jointComponents[joint];
        let maxLen = 0;
        components.forEach(comp => { if (history[comp].length > maxLen) maxLen = history[comp].length; });
        chart.data.labels = Array.from({ length: maxLen }, (_, i) => i);
        components.forEach((comp, idx) => {
            const slice  = history[comp].slice(-maxLen);
            const padded = Array(maxLen - slice.length).fill(null).concat(slice);
            chart.data.datasets[idx].data = padded;
        });
        chart.update();
    }
}

function updateRULA(rula) {
    ['right', 'left'].forEach(side => {
        const s = rula[side];
        if (!s) return;
        const p = `rula-${side}`;
        setText(`${p}`,              s.final);
        setText(`${p}-action`,       s.action);
        setText(`${p}-ua`,           s.upper_arm_score);
        setText(`${p}-fa`,           s.forearm_score);
        setText(`${p}-w`,            s.wrist_score);
        setText(`${p}-n`,            s.neck_score);
        setText(`${p}-t`,            s.trunk_score);
        setText(`${p}-a`,            s.score_a);
        setText(`${p}-b`,            s.score_b);
        setText(`${p}-c`,            s.score_c);
        setText(`${p}-d`,            s.score_d);
        setText(`${p}-shoulder-flex`, typeof s.shoulder_flexion   === 'number' ? s.shoulder_flexion.toFixed(1)   + '°' : '-');
        setText(`${p}-shoulder-abd`,  typeof s.shoulder_abduction  === 'number' ? s.shoulder_abduction.toFixed(1)  + '°' : '-');
        setText(`${p}-elbow-flex`,    typeof s.elbow_flexion        === 'number' ? s.elbow_flexion.toFixed(1)        + '°' : '-');
        setText(`${p}-wrist-flex`,    typeof s.wrist_flexion        === 'number' ? s.wrist_flexion.toFixed(1)        + '°' : '-');
        setText(`${p}-wrist-dev`,     typeof s.wrist_deviation      === 'number' ? s.wrist_deviation.toFixed(1)      + '°' : '-');
        setText(`${p}-wrist-pron`,    typeof s.wrist_pronation      === 'number' ? s.wrist_pronation.toFixed(1)      + '°' : '-');
        setText(`${p}-neck-flex`,     typeof s.neck_flexion         === 'number' ? s.neck_flexion.toFixed(1)         + '°' : '-');
        setText(`${p}-trunk-flex`,    typeof s.trunk_flexion        === 'number' ? s.trunk_flexion.toFixed(1)        + '°' : '-');
    });
}

function updateREBA(reba) {
    ['right', 'left'].forEach(side => {
        const s = reba[side];
        if (!s) return;
        const p = `reba-${side}`;
        setText(`${p}`,              s.final);
        setText(`${p}-final`,        s.final);
        setText(`${p}-action`,       s.action);
        setText(`${p}-trunk`,        s.trunk_score);
        setText(`${p}-neck`,         s.neck_score);
        setText(`${p}-legs`,         s.legs_score);
        setText(`${p}-ua`,           s.upper_arm_score);
        setText(`${p}-fa`,           s.forearm_score);
        setText(`${p}-w`,            s.wrist_score);
        setText(`${p}-a`,            s.score_a);
        setText(`${p}-b`,            s.score_b);
        setText(`${p}-c`,            s.score_c);
        setText(`${p}-trunk-flex`,   typeof s.trunk_flexion        === 'number' ? s.trunk_flexion.toFixed(1)        + '°' : '-');
        setText(`${p}-neck-flex`,    typeof s.neck_flexion         === 'number' ? s.neck_flexion.toFixed(1)         + '°' : '-');
        setText(`${p}-ua-flex`,      typeof s.upper_arm_flexion    === 'number' ? s.upper_arm_flexion.toFixed(1)    + '°' : '-');
        setText(`${p}-ua-abd`,       typeof s.upper_arm_abduction  === 'number' ? s.upper_arm_abduction.toFixed(1)  + '°' : '-');
        setText(`${p}-fa-flex`,      typeof s.forearm_flexion      === 'number' ? s.forearm_flexion.toFixed(1)      + '°' : '-');
        setText(`${p}-wrist-flex`,   typeof s.wrist_flexion        === 'number' ? s.wrist_flexion.toFixed(1)        + '°' : '-');
        setText(`${p}-wrist-dev`,    typeof s.wrist_deviation      === 'number' ? s.wrist_deviation.toFixed(1)      + '°' : '-');
        setText(`${p}-wrist-pron`,   typeof s.wrist_pronation      === 'number' ? s.wrist_pronation.toFixed(1)      + '°' : '-');
    });
}

function downloadCSV() {
    window.location.href = '/api/csv/latest';
}

function generateReport() {
    fetch('/api/report/generate', { method: 'POST' })
        .then(r => r.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a   = document.createElement('a');
            a.href     = url;
            a.download = 'report.pdf';
            a.click();
        })
        .catch(err => console.error('Report generation failed:', err));
}