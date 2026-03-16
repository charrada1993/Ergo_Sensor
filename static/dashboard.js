const socket = io();

// Risk thresholds for highlighting
const RISK_THRESHOLDS = {
    Neck: 30,
    Back: 20,
    R_Shoulder: 45,
    L_Shoulder: 45,
    R_Elbow: 60,
    L_Elbow: 60,
    R_Wrist: 30,
    L_Wrist: 30
};

// Joints we track
const jointList = ['Neck', 'Back', 'R_Shoulder', 'L_Shoulder', 'R_Elbow', 'L_Elbow', 'R_Wrist', 'L_Wrist'];

// Store up to 10 minutes of data at 10 Hz = 6000 points
const MAX_HISTORY = 6000;
const history = {};
jointList.forEach(j => history[j] = []);

// Additional data histories
const trunkHistory = [];
const legsHistory = [];

// Global risk history (last 60 seconds = 600 points) – kept for risk trend if needed
const MAX_RISK_HISTORY = 600;
const riskHistory = [];

let radarChart;
let riskTrendChart;
const trendCharts = {};

// Initialize all joint trend charts on page load
function initTrendCharts() {
    // Joint charts
    jointList.forEach(joint => {
        const canvasId = `trend-${joint}`;
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.warn(`Canvas not found: ${canvasId}`);
            return;
        }
        const ctx = canvas.getContext('2d');
        trendCharts[joint] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: `${joint} (°)`,
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    fill: true,
                    tension: 0,
                    pointRadius: 0,
                    borderWidth: 1.5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    y: { min: 0, max: 180, ticks: { stepSize: 30 } },
                    x: { type: 'linear', ticks: { display: false } }
                },
                plugins: { legend: { display: false } }
            }
        });
    });

    // Trunk angle chart
    const trunkCanvas = document.getElementById('trend-trunk');
    if (trunkCanvas) {
        const ctx = trunkCanvas.getContext('2d');
        trendCharts['trunk'] = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Trunk (°)', data: [], borderColor: '#00d4ff', backgroundColor: 'rgba(0,212,255,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 1.5 }] },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                scales: { y: { min: 0, max: 90, ticks: { stepSize: 15 } }, x: { ticks: { display: false } } },
                plugins: { legend: { display: false } }
            }
        });
    }

    // Legs score chart
    const legsCanvas = document.getElementById('trend-legs');
    if (legsCanvas) {
        const ctx = legsCanvas.getContext('2d');
        trendCharts['legs'] = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Legs Score', data: [], borderColor: '#ffaa00', backgroundColor: 'rgba(255,170,0,0.1)', fill: true, tension: 0, pointRadius: 0, borderWidth: 1.5 }] },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                scales: { y: { min: 1, max: 4, ticks: { stepSize: 1 } }, x: { ticks: { display: false } } },
                plugins: { legend: { display: false } }
            }
        });
    }
}

// Initialize global risk trend chart (optional, kept if you still want it)
function initRiskTrendChart() {
    const canvas = document.getElementById('riskTrendChart');
    if (!canvas) {
        console.warn('Canvas riskTrendChart not found');
        return;
    }
    const ctx = canvas.getContext('2d');
    riskTrendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Global Risk (%)',
                data: [],
                borderColor: '#00d4ff',
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                fill: true,
                tension: 0,
                pointRadius: 0,
                borderWidth: 1.5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: { min: 0, max: 100, ticks: { stepSize: 20 } },
                x: { type: 'linear', ticks: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// Call initializations
initTrendCharts();
initRiskTrendChart();

socket.on('connect', () => {
    document.getElementById('conn-status').innerText = 'Connected';
    console.log('WebSocket connected');
});

socket.on('disconnect', () => {
    document.getElementById('conn-status').innerText = 'Disconnected';
    console.log('WebSocket disconnected');
});

socket.on('angles', (data) => {
    console.log('Received angles:', data.angles);
    console.log('Received risk:', data.risk);
    if (data.angles) updateAngles(data.angles);
    if (data.risk) updateRisk(data.risk);
    if (data.angles) updateTrends(data.angles);
    if (data.rula) updateRULA(data.rula);
    // New data
    if (data.trunk_angle !== undefined) {
        trunkHistory.push(data.trunk_angle);
        if (trunkHistory.length > MAX_HISTORY) trunkHistory.shift();
        const chart = trendCharts['trunk'];
        if (chart) {
            chart.data.labels = Array.from({ length: trunkHistory.length }, (_, i) => i);
            chart.data.datasets[0].data = trunkHistory;
            chart.update();
        }
    }
    if (data.legs_score !== undefined) {
        legsHistory.push(data.legs_score);
        if (legsHistory.length > MAX_HISTORY) legsHistory.shift();
        const chart = trendCharts['legs'];
        if (chart) {
            chart.data.labels = Array.from({ length: legsHistory.length }, (_, i) => i);
            chart.data.datasets[0].data = legsHistory;
            chart.update();
        }
    }
});

function updateRULA(rula) {
    if (rula.right) {
        document.getElementById('rula-right').innerText = rula.right.final;
        document.getElementById('rula-right-action').innerText = rula.right.action;
    }
    if (rula.left) {
        document.getElementById('rula-left').innerText = rula.left.final;
        document.getElementById('rula-left-action').innerText = rula.left.action;
    }
}

function updateAngles(angles) {
    let html = '<ul>';
    let anyData = false;
    for (let joint of jointList) {
        const value = angles[joint];
        if (value === undefined) continue;
        anyData = true;
        const threshold = RISK_THRESHOLDS[joint] || 999;
        const isRisky = value > threshold;
        const color = isRisky ? '#ff6b6b' : '#00d4ff';
        const icon = isRisky ? '<i class="fas fa-exclamation-triangle" style="color: #ffaa00; margin-right: 5px;"></i>' : '';
        html += `<li style="color: ${color};">${icon}${joint}: ${value.toFixed(1)}°</li>`;
    }
    html += '</ul>';
    if (!anyData) html = '<p>No joint data available yet.</p>';
    document.getElementById('angles').innerHTML = html;
}

function updateRisk(risk) {
    if (!risk) {
        console.warn('Risk is null or undefined');
        return;
    }
    const globalPercent = risk.global * 100;
    console.log('Global percent:', globalPercent);
    const pointer = document.getElementById('gauge-pointer');
    if (pointer) pointer.style.left = globalPercent + '%';
    document.getElementById('risk-level').innerHTML = `Global Risk: ${globalPercent.toFixed(1)}% (${risk.level})`;

    // Update radar chart
    if (radarChart) {
        radarChart.data.datasets[0].data = [risk.R1*100, risk.R2*100, risk.R3*100, risk.R4*100, risk.R5*100];
        radarChart.update();
    } else {
        const ctx = document.getElementById('radarChart').getContext('2d');
        if (ctx) {
            radarChart = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['R1 Static', 'R2 Repetition', 'R3 Angle', 'R4 Effort', 'R5 Recovery'],
                    datasets: [{
                        label: 'Risk Components (%)',
                        data: [risk.R1*100, risk.R2*100, risk.R3*100, risk.R4*100, risk.R5*100],
                        backgroundColor: 'rgba(0, 212, 255, 0.2)',
                        borderColor: '#00d4ff',
                        pointBackgroundColor: '#00d4ff'
                    }]
                },
                options: { scales: { r: { min: 0, max: 100 } } }
            });
        }
    }

    // Update global risk trend
    riskHistory.push(globalPercent);
    if (riskHistory.length > MAX_RISK_HISTORY) riskHistory.shift();
    if (riskTrendChart) {
        riskTrendChart.data.labels = Array.from({ length: riskHistory.length }, (_, i) => i);
        riskTrendChart.data.datasets[0].data = riskHistory;
        riskTrendChart.update();
    }
}

function updateTrends(angles) {
    // Update history for each joint
    jointList.forEach(joint => {
        const val = angles[joint];
        if (val !== undefined) {
            history[joint].push(val);
            if (history[joint].length > MAX_HISTORY) history[joint].shift();
        }
    });

    // Update each trend chart with the full history
    jointList.forEach(joint => {
        const chart = trendCharts[joint];
        if (chart) {
            const labels = Array.from({ length: history[joint].length }, (_, i) => i);
            chart.data.labels = labels;
            chart.data.datasets[0].data = history[joint];
            chart.update();
        }
    });
}

function downloadCSV() {
    window.location.href = '/api/csv/latest';
}

function generateReport() {
    fetch('/api/report/generate', { method: 'POST' })
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'report.pdf';
            a.click();
        })
        .catch(err => console.error('Report generation failed:', err));
}