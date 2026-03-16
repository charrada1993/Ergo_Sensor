const socket = io();

const MAX_HISTORY = 600;
const rebaRightHistory = [];
const rebaLeftHistory = [];

let chartRight, chartLeft, chartBoth;

function initCharts() {
    // Right REBA chart
    const ctxRight = document.getElementById('trend-reba-right').getContext('2d');
    chartRight = new Chart(ctxRight, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Right REBA', data: [], borderColor: '#00d4ff', backgroundColor: 'rgba(0,212,255,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 1.5 }] },
        options: {
            responsive: false,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: { min: 1, max: 15, ticks: { stepSize: 2 } },
                x: { ticks: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // Left REBA chart
    const ctxLeft = document.getElementById('trend-reba-left').getContext('2d');
    chartLeft = new Chart(ctxLeft, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Left REBA', data: [], borderColor: '#ffaa00', backgroundColor: 'rgba(255,170,0,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 1.5 }] },
        options: {
            responsive: false,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: { min: 1, max: 15, ticks: { stepSize: 2 } },
                x: { ticks: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // Combined chart
    const ctxBoth = document.getElementById('trend-reba-both').getContext('2d');
    chartBoth = new Chart(ctxBoth, {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'Right REBA', data: [], borderColor: '#00d4ff', backgroundColor: 'transparent', tension: 0, pointRadius: 0, borderWidth: 2 },
            { label: 'Left REBA', data: [], borderColor: '#ffaa00', backgroundColor: 'transparent', tension: 0, pointRadius: 0, borderWidth: 2 }
        ]},
        options: {
            responsive: false,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: { min: 1, max: 15, ticks: { stepSize: 2 } },
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

        // Optionally handle lower body sensors that might not be in EXPECTED_SENSORS yet
        // If you add them to config later, they will appear automatically.
    } catch (error) {
        console.error('Failed to fetch sensor status:', error);
    }
}

socket.on('connect', () => {
    console.log('REBA WebSocket connected');
    fetchSensorStatus();
    setInterval(fetchSensorStatus, 5000);
});

socket.on('angles', (data) => {
    if (data.reba) updateREBA(data.reba);
});

function updateREBA(reba) {
    if (reba.right) {
        document.getElementById('reba-right-final').innerText = reba.right.final;
        document.getElementById('reba-right-action').innerText = reba.right.action;
        document.getElementById('reba-right-trunk').innerText = reba.right.trunk;
        document.getElementById('reba-right-neck').innerText = reba.right.neck;
        document.getElementById('reba-right-legs').innerText = reba.right.legs;
        document.getElementById('reba-right-ua').innerText = reba.right.upper_arm;
        document.getElementById('reba-right-fa').innerText = reba.right.forearm;
        document.getElementById('reba-right-w').innerText = reba.right.wrist;
        document.getElementById('reba-right-a').innerText = reba.right.score_a;
        document.getElementById('reba-right-b').innerText = reba.right.score_b;
        rebaRightHistory.push(reba.right.final);
        if (rebaRightHistory.length > MAX_HISTORY) rebaRightHistory.shift();
    }
    if (reba.left) {
        document.getElementById('reba-left-final').innerText = reba.left.final;
        document.getElementById('reba-left-action').innerText = reba.left.action;
        document.getElementById('reba-left-trunk').innerText = reba.left.trunk;
        document.getElementById('reba-left-neck').innerText = reba.left.neck;
        document.getElementById('reba-left-legs').innerText = reba.left.legs;
        document.getElementById('reba-left-ua').innerText = reba.left.upper_arm;
        document.getElementById('reba-left-fa').innerText = reba.left.forearm;
        document.getElementById('reba-left-w').innerText = reba.left.wrist;
        document.getElementById('reba-left-a').innerText = reba.left.score_a;
        document.getElementById('reba-left-b').innerText = reba.left.score_b;
        rebaLeftHistory.push(reba.left.final);
        if (rebaLeftHistory.length > MAX_HISTORY) rebaLeftHistory.shift();
    }
    if (chartRight) {
        chartRight.data.labels = Array.from({ length: rebaRightHistory.length }, (_, i) => i);
        chartRight.data.datasets[0].data = rebaRightHistory;
        chartRight.update();
    }
    if (chartLeft) {
        chartLeft.data.labels = Array.from({ length: rebaLeftHistory.length }, (_, i) => i);
        chartLeft.data.datasets[0].data = rebaLeftHistory;
        chartLeft.update();
    }
    if (chartBoth) {
        const maxLen = Math.max(rebaRightHistory.length, rebaLeftHistory.length);
        chartBoth.data.labels = Array.from({ length: maxLen }, (_, i) => i);
        chartBoth.data.datasets[0].data = rebaRightHistory;
        chartBoth.data.datasets[1].data = rebaLeftHistory;
        chartBoth.update();
    }
}

initCharts();