const socket = io();

const MAX_HISTORY = 600;
const rulaRightHistory = [];
const rulaLeftHistory = [];

let chartRight, chartLeft, chartBoth;

function initCharts() {
    // Right RULA chart
    const ctxRight = document.getElementById('trend-rula-right').getContext('2d');
    chartRight = new Chart(ctxRight, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Right RULA', data: [], borderColor: '#00d4ff', backgroundColor: 'rgba(0,212,255,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 1.5 }] },
        options: {
            responsive: false,                 // <-- FIXED SIZE
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: { min: 1, max: 7, ticks: { stepSize: 1 } },
                x: { ticks: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // Left RULA chart
    const ctxLeft = document.getElementById('trend-rula-left').getContext('2d');
    chartLeft = new Chart(ctxLeft, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Left RULA', data: [], borderColor: '#ffaa00', backgroundColor: 'rgba(255,170,0,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 1.5 }] },
        options: {
            responsive: false,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: { min: 1, max: 7, ticks: { stepSize: 1 } },
                x: { ticks: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // Combined chart
    const ctxBoth = document.getElementById('trend-rula-both').getContext('2d');
    chartBoth = new Chart(ctxBoth, {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'Right RULA', data: [], borderColor: '#00d4ff', backgroundColor: 'transparent', tension: 0, pointRadius: 0, borderWidth: 2 },
            { label: 'Left RULA', data: [], borderColor: '#ffaa00', backgroundColor: 'transparent', tension: 0, pointRadius: 0, borderWidth: 2 }
        ]},
        options: {
            responsive: false,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: { min: 1, max: 7, ticks: { stepSize: 1 } },
                x: { ticks: { display: false } }
            },
            plugins: { legend: { labels: { color: '#fff' } } }
        }
    });
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
    } catch (error) {
        console.error('Failed to fetch sensor status:', error);
    }
}

socket.on('connect', () => {
    console.log('RULA WebSocket connected');
    fetchSensorStatus();
    setInterval(fetchSensorStatus, 5000);
});

socket.on('angles', (data) => {
    if (data.rula) updateRULA(data.rula);
});

function updateRULA(rula) {
    if (rula.right) {
        document.getElementById('rula-right-final').innerText = rula.right.final;
        document.getElementById('rula-right-action').innerText = rula.right.action;
        document.getElementById('rula-right-ua').innerText = rula.right.upper_arm;
        document.getElementById('rula-right-fa').innerText = rula.right.forearm;
        document.getElementById('rula-right-w').innerText = rula.right.wrist;
        document.getElementById('rula-right-n').innerText = rula.right.neck;
        document.getElementById('rula-right-t').innerText = rula.right.trunk;
        document.getElementById('rula-right-a').innerText = rula.right.score_a;
        document.getElementById('rula-right-b').innerText = rula.right.score_b;
        document.getElementById('rula-right-c').innerText = rula.right.score_c;
        rulaRightHistory.push(rula.right.final);
        if (rulaRightHistory.length > MAX_HISTORY) rulaRightHistory.shift();
    }
    if (rula.left) {
        document.getElementById('rula-left-final').innerText = rula.left.final;
        document.getElementById('rula-left-action').innerText = rula.left.action;
        document.getElementById('rula-left-ua').innerText = rula.left.upper_arm;
        document.getElementById('rula-left-fa').innerText = rula.left.forearm;
        document.getElementById('rula-left-w').innerText = rula.left.wrist;
        document.getElementById('rula-left-n').innerText = rula.left.neck;
        document.getElementById('rula-left-t').innerText = rula.left.trunk;
        document.getElementById('rula-left-a').innerText = rula.left.score_a;
        document.getElementById('rula-left-b').innerText = rula.left.score_b;
        document.getElementById('rula-left-c').innerText = rula.left.score_c;
        rulaLeftHistory.push(rula.left.final);
        if (rulaLeftHistory.length > MAX_HISTORY) rulaLeftHistory.shift();
    }
    if (chartRight) {
        chartRight.data.labels = Array.from({ length: rulaRightHistory.length }, (_, i) => i);
        chartRight.data.datasets[0].data = rulaRightHistory;
        chartRight.update();
    }
    if (chartLeft) {
        chartLeft.data.labels = Array.from({ length: rulaLeftHistory.length }, (_, i) => i);
        chartLeft.data.datasets[0].data = rulaLeftHistory;
        chartLeft.update();
    }
    if (chartBoth) {
        const maxLen = Math.max(rulaRightHistory.length, rulaLeftHistory.length);
        chartBoth.data.labels = Array.from({ length: maxLen }, (_, i) => i);
        chartBoth.data.datasets[0].data = rulaRightHistory;
        chartBoth.data.datasets[1].data = rulaLeftHistory;
        chartBoth.update();
    }
}

initCharts();