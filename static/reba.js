const socket = io();

const rebaRightHistory = [];
const rebaLeftHistory = [];

let rebaChartRight, rebaChartLeft, rebaChartBoth;

function initCharts() {
    const ctxRight = document.getElementById('trend-reba-right')?.getContext('2d');
    if (ctxRight) {
        rebaChartRight = new Chart(ctxRight, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Right REBA', data: [], borderColor: '#E84545', backgroundColor: 'rgba(232,69,69,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 1.5 }] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: { y: { min: 1, max: 15, ticks: { stepSize: 2 } }, x: { ticks: { display: false } } },
                plugins: { legend: { display: false } }
            }
        });
    }
    const ctxLeft = document.getElementById('trend-reba-left')?.getContext('2d');
    if (ctxLeft) {
        rebaChartLeft = new Chart(ctxLeft, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Left REBA', data: [], borderColor: '#2980B9', backgroundColor: 'rgba(41,128,185,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 1.5 }] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: { y: { min: 1, max: 15, ticks: { stepSize: 2 } }, x: { ticks: { display: false } } },
                plugins: { legend: { display: false } }
            }
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
            options: {
                responsive: false,
                maintainAspectRatio: false,
                animation: false,
                scales: { y: { min: 1, max: 15, ticks: { stepSize: 2 } }, x: { ticks: { display: false } } },
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
    console.log('REBA page WebSocket connected');
    fetchSensorStatus();
    setInterval(fetchSensorStatus, 5000);
});

socket.on('angles', (data) => {
    if (data.reba) updateREBA(data.reba);
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

function updateREBA(reba) {
    const sideMap = {
        right: {
            final: 'reba-right-final', action: 'reba-right-action',
            trunk: 'reba-right-trunk', neck: 'reba-right-neck', legs: 'reba-right-legs',
            a: 'reba-right-a', ua: 'reba-right-ua', fa: 'reba-right-fa', w: 'reba-right-w',
            b: 'reba-right-b', c: 'reba-right-c', act: 'reba-right-act',
            trunk_flex: 'reba-right-trunk-flex', neck_flex: 'reba-right-neck-flex',
            ua_flex: 'reba-right-ua-flex', ua_abd: 'reba-right-ua-abd',
            fa_flex: 'reba-right-fa-flex', wrist_flex: 'reba-right-wrist-flex',
            wrist_dev: 'reba-right-wrist-dev', wrist_pron: 'reba-right-wrist-pron'
        },
        left: {
            final: 'reba-left-final', action: 'reba-left-action',
            trunk: 'reba-left-trunk', neck: 'reba-left-neck', legs: 'reba-left-legs',
            a: 'reba-left-a', ua: 'reba-left-ua', fa: 'reba-left-fa', w: 'reba-left-w',
            b: 'reba-left-b', c: 'reba-left-c', act: 'reba-left-act',
            trunk_flex: 'reba-left-trunk-flex', neck_flex: 'reba-left-neck-flex',
            ua_flex: 'reba-left-ua-flex', ua_abd: 'reba-left-ua-abd',
            fa_flex: 'reba-left-fa-flex', wrist_flex: 'reba-left-wrist-flex',
            wrist_dev: 'reba-left-wrist-dev', wrist_pron: 'reba-left-wrist-pron'
        }
    };
    for (const [side, s] of Object.entries(reba)) {
        if (!s) continue;
        const ids = sideMap[side];
        if (!ids) continue;
        setText(ids.final, s.final);
        setText(ids.action, s.action);
        setText(ids.trunk, s.trunk);
        setText(ids.neck, s.neck);
        setText(ids.legs, s.legs);
        setText(ids.a, s.score_a);
        setText(ids.ua, s.upper_arm);
        setText(ids.fa, s.forearm);
        setText(ids.w, s.wrist);
        setText(ids.b, s.score_b);
        setText(ids.c, s.score_c);
        setText(ids.act, s.activity !== undefined ? s.activity : '-');
        setText(ids.trunk_flex, s.trunk_flexion !== undefined ? s.trunk_flexion.toFixed(1) + '°' : '-');
        setText(ids.neck_flex, s.neck_flexion !== undefined ? s.neck_flexion.toFixed(1) + '°' : '-');
        setText(ids.ua_flex, s.upper_arm_flexion !== undefined ? s.upper_arm_flexion.toFixed(1) + '°' : '-');
        setText(ids.ua_abd, s.upper_arm_abduction !== undefined ? s.upper_arm_abduction.toFixed(1) + '°' : '-');
        setText(ids.fa_flex, s.forearm_flexion !== undefined ? s.forearm_flexion.toFixed(1) + '°' : '-');
        setText(ids.wrist_flex, s.wrist_flexion !== undefined ? s.wrist_flexion.toFixed(1) + '°' : '-');
        setText(ids.wrist_dev, s.wrist_deviation !== undefined ? s.wrist_deviation.toFixed(1) + '°' : '-');
        setText(ids.wrist_pron, s.wrist_pronation !== undefined ? s.wrist_pronation.toFixed(1) + '°' : '-');
    }
}

initCharts();