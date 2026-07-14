const socket = io();

const rebaHistory = [];

let rebaChartUnified;

function initCharts() {
    const ctxUnified = document.getElementById('trend-reba-unified')?.getContext('2d');
    if (ctxUnified) {
        rebaChartUnified = new Chart(ctxUnified, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'REBA Score', data: [], borderColor: '#00e5ff', backgroundColor: 'rgba(0,229,255,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 2 }] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: { y: { min: 1, max: 15, ticks: { stepSize: 2 } }, x: { ticks: { display: false } } },
                plugins: { legend: { display: false } }
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
    if (data.reba) {
        updateREBA(data.reba);
        if (data.reba.final !== undefined) {
            rebaHistory.push(data.reba.final);
            if (rebaHistory.length > 600) rebaHistory.shift();
        }
        if (rebaChartUnified) {
            rebaChartUnified.data.labels = Array.from({ length: rebaHistory.length }, (_, i) => i);
            rebaChartUnified.data.datasets[0].data = rebaHistory;
            rebaChartUnified.update();
        }
    }
});

function updateREBA(reba) {
    if (!reba) return;
    setText('reba-final', reba.final);
    const actionEl = document.getElementById('reba-action');
    if (actionEl) {
        actionEl.innerText = reba.action || '-';
        actionEl.className = 'action-label'; // reset
        if (reba.final >= 8) actionEl.classList.add('badge', 'high-risk');
        else if (reba.final >= 4) actionEl.classList.add('badge', 'med-risk');
        else actionEl.classList.add('badge', 'low-risk');
    }
    setText('reba-side', reba.side || 'None');
    setText('reba-trunk', reba.trunk_score);
    setText('reba-neck', reba.neck_score);
    setText('reba-legs', reba.legs_score);
    setText('reba-a', reba.score_a);
    setText('reba-ua', reba.upper_arm_score);
    setText('reba-fa', reba.forearm_score);
    setText('reba-w', reba.wrist_score);
    setText('reba-b', reba.score_b);
    setText('reba-c', reba.score_c);
    setText('reba-trunk-flex', reba.trunk_flexion !== undefined ? reba.trunk_flexion.toFixed(1) + '°' : '-');
    setText('reba-neck-flex', reba.neck_flexion !== undefined ? reba.neck_flexion.toFixed(1) + '°' : '-');
    setText('reba-ua-flex', reba.upper_arm_flexion !== undefined ? reba.upper_arm_flexion.toFixed(1) + '°' : '-');
    setText('reba-ua-abd', reba.upper_arm_abduction !== undefined ? reba.upper_arm_abduction.toFixed(1) + '°' : '-');
    setText('reba-fa-flex', reba.forearm_flexion !== undefined ? reba.forearm_flexion.toFixed(1) + '°' : '-');
    setText('reba-wrist-flex', reba.wrist_flexion !== undefined ? reba.wrist_flexion.toFixed(1) + '°' : '-');
    setText('reba-wrist-dev', reba.wrist_deviation !== undefined ? reba.wrist_deviation.toFixed(1) + '°' : '-');
    setText('reba-wrist-pron', reba.wrist_pronation !== undefined ? reba.wrist_pronation.toFixed(1) + '°' : '-');
}

initCharts();