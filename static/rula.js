const socket = io();

const rulaRightHistory = [];
const rulaLeftHistory = [];

let rulaChartRight, rulaChartLeft, rulaChartBoth;

function initCharts() {
    const ctxRight = document.getElementById('trend-rula-right')?.getContext('2d');
    if (ctxRight) {
        rulaChartRight = new Chart(ctxRight, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Right RULA', data: [], borderColor: '#E84545', backgroundColor: 'rgba(232,69,69,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 1.5 }] },
            options: {
                responsive: false,   // FIXED SIZE – canvas will not resize
                maintainAspectRatio: false,
                animation: false,
                scales: { y: { min: 1, max: 7, ticks: { stepSize: 1 } }, x: { ticks: { display: false } } },
                plugins: { legend: { display: false } }
            }
        });
    }
    const ctxLeft = document.getElementById('trend-rula-left')?.getContext('2d');
    if (ctxLeft) {
        rulaChartLeft = new Chart(ctxLeft, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Left RULA', data: [], borderColor: '#2980B9', backgroundColor: 'rgba(41,128,185,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 1.5 }] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: { y: { min: 1, max: 7, ticks: { stepSize: 1 } }, x: { ticks: { display: false } } },
                plugins: { legend: { display: false } }
            }
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
            options: {
                responsive: false,
                maintainAspectRatio: false,
                animation: false,
                scales: { y: { min: 1, max: 7, ticks: { stepSize: 1 } }, x: { ticks: { display: false } } },
                plugins: { legend: { labels: { color: '#fff' } } }
            }
        });
    }
}

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.innerText = (value === undefined || value === null) ? '-' : value;
}

async function fetchSensorStatus() {
    try {
        const response = await fetch('/api/sensors');
        const sensors = await response.json();
        sensors.forEach(s => {
            const el = document.getElementById(`status-${s.sensor_id}`);
            if (el) {
                el.innerHTML = s.online ? '<span class="badge online"><i class="fas fa-circle"></i> Online</span>' : '<span class="badge offline"><i class="fas fa-circle"></i> Offline</span>';
            }
        });
    } catch (err) { console.error(err); }
}

socket.on('connect', () => {
    console.log('RULA page WebSocket connected');
    fetchSensorStatus();
    setInterval(fetchSensorStatus, 5000);
});

socket.on('angles', (data) => {
    if (data.rula) updateRULA(data.rula);
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
});

function updateRULA(rula) {
    const sideMap = {
        right: {
            final: 'rula-right-final', action: 'rula-right-action',
            ua: 'rula-right-ua', fa: 'rula-right-fa', w: 'rula-right-w',
            n: 'rula-right-n', t: 'rula-right-t', a: 'rula-right-a', b: 'rula-right-b', c: 'rula-right-c', d: 'rula-right-d',
            shoulder_flex: 'rula-right-shoulder-flex', shoulder_abd: 'rula-right-shoulder-abd',
            elbow_flex: 'rula-right-elbow-flex', wrist_flex: 'rula-right-wrist-flex',
            wrist_dev: 'rula-right-wrist-dev', wrist_pron: 'rula-right-wrist-pron',
            neck_flex: 'rula-right-neck-flex', trunk_flex: 'rula-right-trunk-flex', final2: 'rula-right-final2'
        },
        left: {
            final: 'rula-left-final', action: 'rula-left-action',
            ua: 'rula-left-ua', fa: 'rula-left-fa', w: 'rula-left-w',
            n: 'rula-left-n', t: 'rula-left-t', a: 'rula-left-a', b: 'rula-left-b', c: 'rula-left-c', d: 'rula-left-d',
            shoulder_flex: 'rula-left-shoulder-flex', shoulder_abd: 'rula-left-shoulder-abd',
            elbow_flex: 'rula-left-elbow-flex', wrist_flex: 'rula-left-wrist-flex',
            wrist_dev: 'rula-left-wrist-dev', wrist_pron: 'rula-left-wrist-pron',
            neck_flex: 'rula-left-neck-flex', trunk_flex: 'rula-left-trunk-flex', final2: 'rula-left-final2'
        }
    };
    for (const [side, s] of Object.entries(rula)) {
        if (!s) continue;
        const ids = sideMap[side];
        if (!ids) continue;
        setText(ids.final, s.final);
        setText(ids.action, s.action);
        setText(ids.ua, s.upper_arm);
        setText(ids.fa, s.forearm);
        setText(ids.w, s.wrist);
        setText(ids.n, s.neck);
        setText(ids.t, s.trunk);
        setText(ids.a, s.score_a);
        setText(ids.b, s.score_b);
        setText(ids.c, s.score_c);
        setText(ids.d, s.score_d);
        setText(ids.shoulder_flex, s.shoulder_flexion !== undefined ? s.shoulder_flexion.toFixed(1) + '°' : '-');
        setText(ids.shoulder_abd, s.shoulder_abduction !== undefined ? s.shoulder_abduction.toFixed(1) + '°' : '-');
        setText(ids.elbow_flex, s.elbow_flexion !== undefined ? s.elbow_flexion.toFixed(1) + '°' : '-');
        setText(ids.wrist_flex, s.wrist_flexion !== undefined ? s.wrist_flexion.toFixed(1) + '°' : '-');
        setText(ids.wrist_dev, s.wrist_deviation !== undefined ? s.wrist_deviation.toFixed(1) + '°' : '-');
        setText(ids.wrist_pron, s.wrist_pronation !== undefined ? s.wrist_pronation.toFixed(1) + '°' : '-');
        setText(ids.neck_flex, s.neck_flexion !== undefined ? s.neck_flexion.toFixed(1) + '°' : '-');
        setText(ids.trunk_flex, s.trunk_flexion !== undefined ? s.trunk_flexion.toFixed(1) + '°' : '-');
        setText(ids.final2, s.final);
    }
}

initCharts();