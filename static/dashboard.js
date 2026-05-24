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
    R_Thigh:    45,
    L_Thigh:    45,
    R_Knee:     60,
    L_Knee:     60,
};

const jointList = [
    'Neck', 'Back',
    'R_Shoulder', 'L_Shoulder',
    'R_Elbow',    'L_Elbow',
    'R_Wrist',    'L_Wrist',
    'R_Thigh',    'L_Thigh',
    'R_Knee',     'L_Knee',
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
    R_Thigh:    ['R_Thigh', 'R_Thigh_Roll', 'R_Thigh_Yaw'],
    L_Thigh:    ['L_Thigh', 'L_Thigh_Roll', 'L_Thigh_Yaw'],
    R_Knee:     ['R_Knee'],
    L_Knee:     ['L_Knee'],
};

const componentColors = {
    Neck:               '#00d4ff',
    Roll:               '#ffaa00',
    Yaw:                '#ff6b6b',
    Abduction:          '#27ae60',
    Trunk_Pitch:        '#00d4ff',
    Trunk_Roll:         '#ffaa00',
    Trunk_Yaw:          '#ff6b6b',
    R_Thigh:            '#00d4ff',
    L_Thigh:            '#00d4ff',
    R_Knee:             '#00d4ff',
    L_Knee:             '#00d4ff',
    R_Thigh_Roll:       '#ffaa00',
    L_Thigh_Roll:       '#ffaa00',
    R_Thigh_Yaw:        '#ff6b6b',
    L_Thigh_Yaw:        '#ff6b6b',
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

const rulaHistory = [];
const rebaHistory = [];

const MAX_HISTORY = 6000;
const trendCharts = {};

let rulaChartUnified;
let rebaChartUnified;

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
                plugins: { legend: { display: true, position: 'top', labels: { font: { size: window.innerWidth < 768 ? 6 : 8 } } } },
            },
        });
    }
}

function initRulaCharts() {
    const ctxUnified = document.getElementById('trend-rula-unified')?.getContext('2d');
    if (ctxUnified) {
        rulaChartUnified = new Chart(ctxUnified, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'RULA Score', data: [], borderColor: '#00e5ff', backgroundColor: 'rgba(0,229,255,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 2 }] },
            options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { y: { min: 1, max: 7, ticks: { stepSize: 1 } }, x: { ticks: { display: false } } }, plugins: { legend: { display: false } } }
        });
    }
}

function initRebaCharts() {
    const ctxUnified = document.getElementById('trend-reba-unified')?.getContext('2d');
    if (ctxUnified) {
        rebaChartUnified = new Chart(ctxUnified, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'REBA Score', data: [], borderColor: '#00e5ff', backgroundColor: 'rgba(0,229,255,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 2 }] },
            options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { y: { min: 1, max: 15, ticks: { stepSize: 2 } }, x: { ticks: { display: false } } }, plugins: { legend: { display: false } } }
        });
    }
}

// Defer all chart + socket wiring until the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    // Ensure conn-status badge exists
    if (!document.getElementById('conn-status')) {
        const newStatus = document.createElement('div');
        newStatus.id = 'conn-status';
        newStatus.style.cssText = 'position:fixed;top:10px;right:10px;background:#12151d;border:1px solid #1e2330;padding:4px 12px;border-radius:20px;font-family:monospace;font-size:12px;z-index:9999;';
        newStatus.innerText = '⬤ Connecting...';
        document.body.appendChild(newStatus);
    }
    initTrendCharts();
    initRulaCharts();
    initRebaCharts();
    createRawDebugPanel();
    wireSocketEvents();

    // Start active patient tracking session timer
    if (document.getElementById('patient-session-timer')) {
        window.sessionStartTime = Date.now();
        setInterval(() => {
            const elapsed = Math.floor((Date.now() - window.sessionStartTime) / 1000);
            const hrs = String(Math.floor(elapsed / 3600)).padStart(2, '0');
            const mins = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
            const secs = String(elapsed % 60).padStart(2, '0');
            const timerEl = document.getElementById('patient-session-timer');
            if (timerEl) timerEl.innerText = `${hrs}:${mins}:${secs}`;
            
            // Goal progress (30 minutes goal)
            const goalMins = 30;
            const goalPct = Math.min(100, Math.round((elapsed / (goalMins * 60)) * 100));
            setText('patient-goal-val', `${goalPct}%`);
            const goalFill = document.getElementById('patient-goal-progress');
            if (goalFill) goalFill.style.width = `${goalPct}%`;
        }, 1000);
    }
});

// ── Debug panel for raw sensor data ──
let rawDebugDiv = null;
let rawDebugBody = null;
let rawDebugMinimized = false;

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

    const header = document.createElement('div');
    header.style.display = 'flex';
    header.style.justifyContent = 'space-between';
    header.style.alignItems = 'center';
    header.style.cursor = 'pointer';
    header.innerHTML = '<strong>📡 Raw sensors</strong><span id="raw-minimize" style="margin-left:10px; font-weight:bold;">−</span>';
    
    header.addEventListener('click', () => {
        rawDebugMinimized = !rawDebugMinimized;
        rawDebugBody.style.display = rawDebugMinimized ? 'none' : 'block';
        document.getElementById('raw-minimize').innerText = rawDebugMinimized ? '+' : '−';
    });

    rawDebugBody = document.createElement('div');
    rawDebugBody.innerHTML = 'Waiting for data...';
    rawDebugBody.style.marginTop = '5px';

    div.appendChild(header);
    div.appendChild(rawDebugBody);
    document.body.appendChild(div);
    rawDebugDiv = div;
}
// ── Wire all socket events (called after DOM ready) ──────────────────────
function wireSocketEvents() {
    socket.on('connect', () => {
        const el = document.getElementById('conn-status');
        const elMob = document.getElementById('conn-status-mobile');
        if (el) { el.innerText = '⬤ Connected'; el.classList.add('live'); }
        if (elMob) { elMob.innerText = '⬤ Live'; elMob.classList.add('live'); }
        console.log('Socket connected');
    });
    socket.on('disconnect', () => {
        const el = document.getElementById('conn-status');
        const elMob = document.getElementById('conn-status-mobile');
        if (el) { el.innerText = '⬤ Disconnected'; el.classList.remove('live'); }
        if (elMob) { elMob.innerText = '⬤ Disconnected'; elMob.classList.remove('live'); }
        console.log('Socket disconnected');
    });

socket.on('raw_sensors', (data) => {
    console.log('[RAW]', data);
    
    // Update active sensor count for mobile
    const sensorCount = Object.keys(data).length;
    const countEl = document.getElementById('active-sensors-count');
    if (countEl) countEl.innerText = sensorCount;

    if (rawDebugDiv && rawDebugBody && !rawDebugMinimized) {
        let html = '';
        for (const [sid, vals] of Object.entries(data)) {
            html += `${sid}: r=${vals.roll.toFixed(1)}° p=${vals.pitch.toFixed(1)}° y=${vals.yaw.toFixed(1)}°<br>`;
        }
        rawDebugBody.innerHTML = html;
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

    // Calibrate posture score from angles
    if (data.angles) {
        let postureScore = 100;
        let worstJoint = 'Normal';
        let maxDev = 0;
        let activeJoints = 0;
        
        for (const joint of jointList) {
            const val = (joint === 'Back') ? data.angles['Trunk_Pitch'] : data.angles[joint];
            if (val === undefined || val === null) continue;
            activeJoints++;
            const threshold = RISK_THRESHOLDS[joint] || 999;
            const dev = Math.abs(val) - threshold;
            if (dev > 0) {
                const penalty = Math.min(25, (dev / threshold) * 20);
                postureScore -= penalty;
                if (dev > maxDev) {
                    maxDev = dev;
                    worstJoint = joint;
                }
            }
        }
        postureScore = Math.max(0, Math.round(postureScore));
        
        // --- Update Patient HUD (if elements exist) ---
        const ring = document.getElementById('score-ring-fill');
        if (ring) {
            const circumference = 80 * 2 * Math.PI; // r=80
            ring.style.strokeDasharray = circumference;
            const offset = circumference - (postureScore / 100) * circumference;
            ring.style.strokeDashoffset = offset;
            
            // color mapping
            if (postureScore >= 85) {
                ring.style.stroke = 'var(--ok)';
            } else if (postureScore >= 60) {
                ring.style.stroke = 'var(--warn)';
            } else {
                ring.style.stroke = 'var(--danger)';
            }
        }
        
        setText('patient-score-text', `${postureScore}%`);
        const label = document.getElementById('patient-score-label');
        if (label) {
            if (postureScore >= 85) {
                label.innerText = 'EXCELLENT';
                label.style.color = 'var(--ok)';
            } else if (postureScore >= 60) {
                label.innerText = 'ADJUST POSTURE';
                label.style.color = 'var(--warn)';
            } else {
                label.innerText = 'HIGH TENSION';
                label.style.color = 'var(--danger)';
            }
        }
        
        const adviceEl = document.getElementById('patient-coach-advice');
        if (adviceEl) {
            let advice = 'Looking great! Sitting upright with excellent spinal alignment. Keep it up!';
            if (postureScore < 85) {
                if (worstJoint === 'Neck') {
                    advice = 'Your neck is leaning forward. Try raising your chin and alignment back to standard.';
                } else if (worstJoint === 'Back') {
                    advice = 'Trunk slumping detected. Sit tall and align your spine back.';
                } else if (worstJoint.includes('Shoulder')) {
                    advice = 'Shoulder extension/abduction strain. Bring your arms closer and lower shoulders.';
                } else {
                    advice = `Tension detected at ${worstJoint.replace('_', ' ')}. Adjust your position.`;
                }
            }
            adviceEl.innerText = advice;
        }
        
        // Track safe alignment ratio
        if (typeof window.totalFrames === 'undefined') {
            window.totalFrames = 0;
            window.safeFrames = 0;
        }
        window.totalFrames++;
        if (postureScore >= 85) window.safeFrames++;
        const ratio = Math.round((window.safeFrames / window.totalFrames) * 100);
        setText('patient-ratio-val', `${ratio}%`);
        const ratioFill = document.getElementById('patient-ratio-progress');
        if (ratioFill) ratioFill.style.width = `${ratio}%`;
        
        // --- Update Doctor Diagnostics Panel (if elements exist) ---
        if (data.ai_predictions) {
            const ai = data.ai_predictions;
            const prob = (ai.risk_10d * 100).toFixed(1);
            setText('ai-risk-prob-text', `${prob}%`);
            const probFill = document.getElementById('ai-risk-gauge-fill');
            if (probFill) probFill.style.width = `${prob}%`;
            
            const badge = document.getElementById('ai-risk-level-badge');
            if (badge) {
                badge.innerText = ai.risk_level || 'SAFE';
                badge.className = 'badge clinical ' + (ai.risk_level || 'safe').toLowerCase();
            }
            
            setText('ai-anomaly-state', ai.top_anomaly && ai.top_anomaly !== 'normal' ? ai.top_anomaly.replace(/_/g, ' ') : 'Normal');
            
            let crit = ai.critical_joint;
            if (!crit || crit === 'None') {
                crit = (worstJoint !== 'Normal') ? worstJoint.replace('_', ' ') : 'None';
            }
            setText('ai-critical-joint', crit);
        } else {
            // Warm-up/fallback
            let crit = (worstJoint !== 'Normal') ? worstJoint.replace('_', ' ') : 'None';
            setText('ai-critical-joint', crit);
        }
    }
    // Update RULA/REBA trends
    if (data.rula) {
        if (data.rula.final !== undefined) {
            rulaHistory.push(data.rula.final);
            if (rulaHistory.length > 600) rulaHistory.shift();
        }
        if (rulaChartUnified) {
            rulaChartUnified.data.labels = Array.from({ length: rulaHistory.length }, (_, i) => i);
            rulaChartUnified.data.datasets[0].data = rulaHistory;
            rulaChartUnified.update('none');
        }
    }
    if (data.reba) {
        if (data.reba.final !== undefined) {
            rebaHistory.push(data.reba.final);
            if (rebaHistory.length > 600) rebaHistory.shift();
        }
        if (rebaChartUnified) {
            rebaChartUnified.data.labels = Array.from({ length: rebaHistory.length }, (_, i) => i);
            rebaChartUnified.data.datasets[0].data = rebaHistory;
            rebaChartUnified.update('none');
        }
    }
});
} // end wireSocketEvents

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
        chart.update('none');  // no animation for real-time speed
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

// ── Active Stretching Timer Logic ──
let stretchInterval = null;
window.startStretch = function(btn, type) {
    if (stretchInterval) clearInterval(stretchInterval);
    
    // Reset all stretching buttons
    document.querySelectorAll('.btn-stretch').forEach(b => {
        b.innerHTML = '<i class="fas fa-play"></i> Start Stretch (15s)';
        b.classList.remove('active');
        b.disabled = false;
    });
    
    btn.disabled = true;
    btn.classList.add('active');
    let left = 15;
    btn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Stretching... (${left}s)`;
    
    stretchInterval = setInterval(() => {
        left--;
        if (left > 0) {
            btn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Stretching... (${left}s)`;
        } else {
            clearInterval(stretchInterval);
            stretchInterval = null;
            btn.innerHTML = '<i class="fas fa-check"></i> Completed!';
            btn.disabled = false;
            btn.classList.remove('active');
            
            const card = btn.closest('.stretch-card');
            if (card) {
                card.classList.add('completed-flash');
                setTimeout(() => card.classList.remove('completed-flash'), 1000);
            }
        }
    }, 1000);
};