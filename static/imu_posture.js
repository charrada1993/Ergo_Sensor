/**
 * imu_posture.js — IMU Posture & Lombo-Pelvi-Fémoral Assessment
 * Handles WebSocket data reception, biomechanical score computation,
 * DOM updates and Chart.js rendering.
 */

'use strict';

// ══════════════════════════════════════════
//  Constants
// ══════════════════════════════════════════

// Sensor ID mapping: Firebase key → anatomical position
const SENSOR_MAP = {
    NECK:       'C1',
    UPPER_BACK: 'C7',
    R_BICEPS:   'T5',
    R_FOREARM:  'T12',
    R_HAND:     'L3',
    R_THIGH:    'S1',
    R_SHANK:    'R Thigh',
    L_BICEPS:   'R Shank',
    L_FOREARM:  'R Foot',
    L_HAND:     'L Thigh',
    L_THIGH:    'L Shank',
    L_SHANK:    'L Foot',
};

// Score thresholds
const INTERP = [
    { min: 90, id: 'interp-90', label: 'Excellent',            color: '#10b981' },
    { min: 80, id: 'interp-80', label: 'Very Good',            color: '#34d399' },
    { min: 70, id: 'interp-70', label: 'Normal',               color: '#a78bfa' },
    { min: 60, id: 'interp-60', label: 'Mild Alteration',      color: '#fbbf24' },
    { min: 50, id: 'interp-50', label: 'Moderate Alteration',  color: '#f97316' },
    { min:  0, id: 'interp-0',  label: 'Significant Alteration', color: '#ef4444' },
];

const MAX_HISTORY = 600; // ~10 min at 1 Hz

// ══════════════════════════════════════════
//  State
// ══════════════════════════════════════════

const state = {
    globalHistory:    [],
    cervicalHistory:  [],
    thoracicHistory:  [],
    lumbarHistory:    [],
    calibrated:       false,
    // Latest raw pitch values per sensor (in degrees)
    pitch: {},
    roll:  {},
    yaw:   {},
};

// ══════════════════════════════════════════
//  Socket.IO
// ══════════════════════════════════════════

const socket = io();

socket.on('connect', () => {
    console.log('[IMU Posture] WebSocket connected');
    fetchSensorStatus();
    setInterval(fetchSensorStatus, 5000);
});

socket.on('angles', (data) => {
    // Store raw angles per sensor
    if (data.sensors) {
        for (const [sid, vals] of Object.entries(data.sensors)) {
            if (vals.pitch !== undefined) state.pitch[sid] = vals.pitch;
            if (vals.roll  !== undefined) state.roll[sid]  = vals.roll;
            if (vals.yaw   !== undefined) state.yaw[sid]   = vals.yaw;
        }
    }

    // Fallback: reconstruct from legacy per-field events
    if (data.neck_flexion    !== undefined) state.pitch['NECK']       = data.neck_flexion;
    if (data.trunk_flexion   !== undefined) state.pitch['UPPER_BACK'] = data.trunk_flexion;

    // Compute biomechanical parameters
    const params = computeParams();

    // Compute sub-scores
    const scores = computeScores(params);

    // Update DOM
    updateDOM(params, scores);

    // Update charts
    updateCharts(scores, params);
});

// ══════════════════════════════════════════
//  Sensor status
// ══════════════════════════════════════════

async function fetchSensorStatus() {
    try {
        const res = await fetch('/api/sensors');
        const sensors = await res.json();
        let onlineCount = 0;
        sensors.forEach(s => {
            const el = document.getElementById(`status-${s.sensor_id}`);
            if (el) {
                el.innerHTML = s.online
                    ? '<span class="badge online"><i class="fas fa-circle"></i> Online</span>'
                    : '<span class="badge offline"><i class="fas fa-circle"></i> Offline</span>';
            }
            if (s.online) onlineCount++;
        });
        updateCalibBadge(onlineCount, sensors.length);
    } catch (err) { console.error('[IMU Posture] Sensor fetch error:', err); }
}

function updateCalibBadge(online, total) {
    const badge = document.getElementById('calib-badge');
    if (!badge) return;
    if (online >= 12) {
        badge.style.borderColor  = 'rgba(16,185,129,0.3)';
        badge.style.background   = 'rgba(16,185,129,0.08)';
        badge.style.color        = '#10b981';
        badge.innerHTML = '<i class="fas fa-check-circle"></i><span>12/12 Connected</span>';
    } else {
        badge.style.borderColor  = 'rgba(251,191,36,0.3)';
        badge.style.background   = 'rgba(251,191,36,0.06)';
        badge.style.color        = '#fbbf24';
        badge.innerHTML = `<i class="fas fa-circle-notch fa-spin"></i><span>${online}/${total} Connected</span>`;
    }
}

// ══════════════════════════════════════════
//  Biomechanical parameter computation
// ══════════════════════════════════════════

/**
 * Returns angle difference between two IMU pitches.
 * Uses pitch (sagittal flexion/extension) as primary spinal angle.
 */
function angleDiff(id1, id2) {
    const p1 = state.pitch[id1];
    const p2 = state.pitch[id2];
    if (p1 === undefined || p2 === undefined) return null;
    return Math.abs(p1 - p2);
}

function symmetryIndex(dVal, gVal) {
    if (dVal === undefined || gVal === undefined) return null;
    const avg = (Math.abs(dVal) + Math.abs(gVal)) / 2;
    if (avg === 0) return 0;
    return (Math.abs(Math.abs(dVal) - Math.abs(gVal)) / avg) * 100;
}

function computeParams() {
    // ── Spinal angles ──
    const cervical   = angleDiff('NECK',    'UPPER_BACK');   // C1–C7
    const thoracique = angleDiff('R_BICEPS','R_FOREARM');    // T5–T12
    const lombaire   = angleDiff('R_HAND',  'R_THIGH');      // L3–S1
    const tronc      = angleDiff('NECK',    'R_THIGH');      // C1–S1

    // ── Pelvic parameters (from S1 roll/pitch/yaw) ──
    const pelvicTilt      = state.pitch['R_THIGH']   !== undefined ? state.pitch['R_THIGH'].toFixed(1) + '°' : null;
    const pelvicObliquity = state.roll['R_THIGH']    !== undefined ? state.roll['R_THIGH'].toFixed(1)  + '°' : null;
    const pelvicRotation  = state.yaw['R_THIGH']     !== undefined ? state.yaw['R_THIGH'].toFixed(1)   + '°' : null;

    // ── Symmetry Indices ──
    const siHanche   = symmetryIndex(state.pitch['R_SHANK'],   state.pitch['L_HAND']);
    const siGenou    = symmetryIndex(state.pitch['L_BICEPS'],  state.pitch['L_THIGH']);
    const siCheville = symmetryIndex(state.pitch['L_FOREARM'], state.pitch['L_SHANK']);

    // ── Stability: trunk oscillation (std of tronc pitch over recent window) ──
    // Use the rolling variance of C1 pitch as proxy for postural sway
    const c1vals = state.cervicalHistory.slice(-30);
    const stability = c1vals.length > 5 ? stdDev(c1vals) : null;

    return {
        cervical, thoracique, lombaire, tronc,
        pelvicTilt, pelvicObliquity, pelvicRotation,
        siHanche, siGenou, siCheville,
        stability,
    };
}

// ══════════════════════════════════════════
//  Score computation (0–100)
// ══════════════════════════════════════════

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function computeScores(p) {
    // ── Spine Sub-score (30 pts) ──
    // Average error across the 3 spinal angles vs reference (0°)
    const angles = [p.cervical, p.thoracique, p.lombaire].filter(v => v !== null);
    let rachis = 15; // default mid-range when no data
    if (angles.length > 0) {
        const avgErr = angles.reduce((s, v) => s + v, 0) / angles.length;
        rachis = clamp(30 * (1 - avgErr / 25), 0, 30);
    }

    // ── Symmetry Sub-score (20 pts) ──
    const sis = [p.siHanche, p.siGenou, p.siCheville].filter(v => v !== null);
    let symetrie = 10; // default
    if (sis.length > 0) {
        const avgSI = sis.reduce((s, v) => s + v, 0) / sis.length;
        symetrie = clamp(20 * (1 - avgSI / 20), 0, 20);
    }

    // ── Stability Sub-score (20 pts) ──
    let stabilite = 10; // default
    if (p.stability !== null) {
        stabilite = clamp(20 * (1 - p.stability / 15), 0, 20);
    }

    // ── Gait Sub-score (15 pts) — static default, updated when gait data available ──
    const marche   = 15;

    // ── Mobility Sub-score (15 pts) — static default, updated with ROM data ──
    const mobilite = 15;

    const total = rachis + symetrie + stabilite + marche + mobilite;

    return {
        rachis:    Math.round(rachis),
        symetrie:  Math.round(symetrie),
        stabilite: Math.round(stabilite),
        marche:    Math.round(marche),
        mobilite:  Math.round(mobilite),
        total:     Math.round(clamp(total, 0, 100)),
    };
}

// ══════════════════════════════════════════
//  DOM update
// ══════════════════════════════════════════

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.innerText = (value === undefined || value === null) ? '-' : value;
}

function setBar(id, value, max) {
    const el = document.getElementById(id);
    if (el) el.style.width = (clamp(value / max, 0, 1) * 100).toFixed(1) + '%';
}

function interpFromScore(score) {
    for (const row of INTERP) {
        if (score >= row.min) return row;
    }
    return INTERP[INTERP.length - 1];
}

function updateDOM(p, s) {
    // Global score
    setText('imu-final',  s.total);
    setText('imu-final2', s.total);

    // Action label + color
    const interp = interpFromScore(s.total);
    const actionEl = document.getElementById('imu-action');
    if (actionEl) {
        actionEl.innerText = interp.label;
        actionEl.style.color       = interp.color;
        actionEl.style.background  = interp.color + '18';
        actionEl.style.borderColor = interp.color + '44';
    }
    const finalNum = document.getElementById('imu-final');
    if (finalNum) finalNum.style.textShadow = `0 0 30px ${interp.color}88`;

    // Interpretation text
    setText('imu-interpretation', interp.label);
    const interpEl = document.getElementById('imu-interpretation');
    if (interpEl) interpEl.style.color = interp.color;

    // Highlight active row in table
    INTERP.forEach(row => {
        const tr = document.getElementById(row.id);
        if (tr) tr.classList.remove('active-row');
    });
    const activeRow = document.getElementById(interp.id);
    if (activeRow) activeRow.classList.add('active-row');

    // Sub-scores
    setText('score-rachis',    s.rachis);
    setText('score-symetrie',  s.symetrie);
    setText('score-stabilite', s.stabilite);
    setText('score-marche',    s.marche);
    setText('score-mobilite',  s.mobilite);
    setBar('bar-rachis',    s.rachis,    30);
    setBar('bar-symetrie',  s.symetrie,  20);
    setBar('bar-stabilite', s.stabilite, 20);
    setBar('bar-marche',    s.marche,    15);
    setBar('bar-mobilite',  s.mobilite,  15);

    // Spinal angles
    setText('imu-cervical',   p.cervical   !== null ? p.cervical.toFixed(1)   + '°' : '-');
    setText('imu-thoracique', p.thoracique !== null ? p.thoracique.toFixed(1) + '°' : '-');
    setText('imu-lombaire',   p.lombaire   !== null ? p.lombaire.toFixed(1)   + '°' : '-');
    setText('imu-tronc',      p.tronc      !== null ? p.tronc.toFixed(1)      + '°' : '-');

    // Pelvic
    setText('imu-pelvic-tilt',      p.pelvicTilt      || '-');
    setText('imu-pelvic-obliquity', p.pelvicObliquity || '-');
    setText('imu-pelvic-rotation',  p.pelvicRotation  || '-');

    // Symmetry
    setText('imu-si-hanche',   p.siHanche   !== null ? p.siHanche.toFixed(1)   + ' %' : '-');
    setText('imu-si-genou',    p.siGenou    !== null ? p.siGenou.toFixed(1)    + ' %' : '-');
    setText('imu-si-cheville', p.siCheville !== null ? p.siCheville.toFixed(1) + ' %' : '-');

    // Gait (placeholder — will be driven by gait events when available)
    setText('imu-vitesse',  '-');
    setText('imu-cadence',  '-');
    setText('imu-pas',      '-');
}

// ══════════════════════════════════════════
//  Charts
// ══════════════════════════════════════════

let chartGlobal, chartRadar, chartRachis;

function initCharts() {
    const gridColor  = 'rgba(255,255,255,0.05)';
    const tickColor  = '#5a6280';
    const purple     = '#a78bfa';
    const purpleFill = 'rgba(124,58,237,0.12)';

    // ── Chart 1: Global score trend ──
    const ctxG = document.getElementById('trend-imu-global')?.getContext('2d');
    if (ctxG) {
        chartGlobal = new Chart(ctxG, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Global Score',
                    data: [],
                    borderColor: purple,
                    backgroundColor: purpleFill,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                scales: {
                    y: { min: 0, max: 100, ticks: { color: tickColor, stepSize: 20 }, grid: { color: gridColor } },
                    x: { ticks: { display: false }, grid: { display: false } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    // ── Chart 2: Radar sub-scores ──
    const ctxR = document.getElementById('trend-imu-subscores')?.getContext('2d');
    if (ctxR) {
        chartRadar = new Chart(ctxR, {
            type: 'radar',
            data: {
                labels: ['Spine (30)', 'Symmetry (20)', 'Stability (20)', 'Gait (15)', 'Mobility (15)'],
                datasets: [{
                    label: 'Sub-scores',
                    data: [0, 0, 0, 0, 0],
                    borderColor: purple,
                    backgroundColor: purpleFill,
                    pointBackgroundColor: purple,
                    pointBorderColor: '#fff',
                    borderWidth: 2,
                    pointRadius: 3,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false, animation: { duration: 400 },
                scales: {
                    r: {
                        min: 0, max: 30,
                        ticks: { color: tickColor, stepSize: 10, backdropColor: 'transparent' },
                        grid: { color: gridColor },
                        pointLabels: { color: '#8890aa', font: { size: 11 } }
                    }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    // ── Chart 3: Spinal angles (3 lines) ──
    const ctxA = document.getElementById('trend-imu-rachis')?.getContext('2d');
    if (ctxA) {
        chartRachis = new Chart(ctxA, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Cervical (C1–C7)',
                        data: [],
                        borderColor: '#a78bfa', backgroundColor: 'transparent',
                        fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2,
                    },
                    {
                        label: 'Thoracic (T5–T12)',
                        data: [],
                        borderColor: '#38bdf8', backgroundColor: 'transparent',
                        fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2,
                    },
                    {
                        label: 'Lumbar (L3–S1)',
                        data: [],
                        borderColor: '#fb923c', backgroundColor: 'transparent',
                        fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2,
                    },
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                scales: {
                    y: { min: 0, max: 60, ticks: { color: tickColor, callback: v => v + '°' }, grid: { color: gridColor } },
                    x: { ticks: { display: false }, grid: { display: false } }
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: '#8890aa', boxWidth: 12, font: { size: 11 } }
                    }
                }
            }
        });
    }
}

function pushHistory(arr, value) {
    arr.push(value);
    if (arr.length > MAX_HISTORY) arr.shift();
}

function updateCharts(scores, params) {
    const label = state.globalHistory.length;

    // Chart 1 — global trend
    if (chartGlobal) {
        pushHistory(state.globalHistory, scores.total);
        chartGlobal.data.labels = Array.from({ length: state.globalHistory.length }, (_, i) => i);
        chartGlobal.data.datasets[0].data = state.globalHistory;
        chartGlobal.update();
    }

    // Chart 2 — radar
    if (chartRadar) {
        chartRadar.data.datasets[0].data = [
            scores.rachis,
            scores.symetrie * (30 / 20),    // normalise to /30 scale for radar
            scores.stabilite * (30 / 20),
            scores.marche   * (30 / 15),
            scores.mobilite * (30 / 15),
        ];
        chartRadar.update();
    }

    // Chart 3 — spinal angles
    if (chartRachis) {
        if (params.cervical   !== null) pushHistory(state.cervicalHistory,  params.cervical);
        if (params.thoracique !== null) pushHistory(state.thoracicHistory,  params.thoracique);
        if (params.lombaire   !== null) pushHistory(state.lumbarHistory,    params.lombaire);

        const n = Math.max(state.cervicalHistory.length, state.thoracicHistory.length, state.lumbarHistory.length);
        chartRachis.data.labels             = Array.from({ length: n }, (_, i) => i);
        chartRachis.data.datasets[0].data   = state.cervicalHistory;
        chartRachis.data.datasets[1].data   = state.thoracicHistory;
        chartRachis.data.datasets[2].data   = state.lumbarHistory;
        chartRachis.update();
    }
}

// ══════════════════════════════════════════
//  Math helpers
// ══════════════════════════════════════════

function stdDev(arr) {
    if (arr.length < 2) return 0;
    const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
    const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
    return Math.sqrt(variance);
}

// ══════════════════════════════════════════
//  Init
// ══════════════════════════════════════════

initCharts();
