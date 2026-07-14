const socket = io();

const rulaHistory = [];

let rulaChartUnified;

function initCharts() {
    const ctxUnified = document.getElementById('trend-rula-unified')?.getContext('2d');
    if (ctxUnified) {
        rulaChartUnified = new Chart(ctxUnified, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'RULA Score', data: [], borderColor: '#00e5ff', backgroundColor: 'rgba(0,229,255,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 2 }] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: { y: { min: 1, max: 7, ticks: { stepSize: 1 } }, x: { ticks: { display: false } } },
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
    console.log('RULA page WebSocket connected');
    fetchSensorStatus();
    setInterval(fetchSensorStatus, 5000);
});

socket.on('angles', (data) => {
    if (data.rula) {
        updateRULA(data.rula);
        if (data.rula.final !== undefined) {
            rulaHistory.push(data.rula.final);
            if (rulaHistory.length > 600) rulaHistory.shift();
        }
        if (rulaChartUnified) {
            rulaChartUnified.data.labels = Array.from({ length: rulaHistory.length }, (_, i) => i);
            rulaChartUnified.data.datasets[0].data = rulaHistory;
            rulaChartUnified.update();
        }
    }
});

function updateRULA(rula) {
    if (!rula) return;
    setText('rula-final', rula.final);
    const actionEl = document.getElementById('rula-action');
    if (actionEl) {
        actionEl.innerText = rula.action || '-';
        actionEl.className = 'action-label'; // reset
        if (rula.final >= 7) actionEl.classList.add('badge', 'high-risk');
        else if (rula.final >= 5) actionEl.classList.add('badge', 'med-risk');
        else if (rula.final >= 3) actionEl.classList.add('badge', 'med-risk'); // RULA 3-4 is further investigation, let's treat as med or low-med.
        else actionEl.classList.add('badge', 'low-risk');
    }
    setText('rula-side', rula.side || 'None');
    setText('rula-ua', rula.upper_arm_score);
    setText('rula-fa', rula.forearm_score);
    setText('rula-w', rula.wrist_score);
    setText('rula-n', rula.neck_score);
    setText('rula-t', rula.trunk_score);
    setText('rula-a', rula.score_a);
    setText('rula-b', rula.score_b);
    setText('rula-c', rula.score_c);
    setText('rula-d', rula.score_d);
    setText('rula-shoulder-flex', rula.shoulder_flexion !== undefined ? rula.shoulder_flexion.toFixed(1) + '°' : '-');
    setText('rula-shoulder-abd', rula.shoulder_abduction !== undefined ? rula.shoulder_abduction.toFixed(1) + '°' : '-');
    setText('rula-elbow-flex', rula.elbow_flexion !== undefined ? rula.elbow_flexion.toFixed(1) + '°' : '-');
    setText('rula-wrist-flex', rula.wrist_flexion !== undefined ? rula.wrist_flexion.toFixed(1) + '°' : '-');
    setText('rula-wrist-dev', rula.wrist_deviation !== undefined ? rula.wrist_deviation.toFixed(1) + '°' : '-');
    setText('rula-wrist-pron', rula.wrist_pronation !== undefined ? rula.wrist_pronation.toFixed(1) + '°' : '-');
    setText('rula-neck-flex', rula.neck_flexion !== undefined ? rula.neck_flexion.toFixed(1) + '°' : '-');
    setText('rula-trunk-flex', rula.trunk_flexion !== undefined ? rula.trunk_flexion.toFixed(1) + '°' : '-');
    setText('rula-final2', rula.final);
}

initCharts();