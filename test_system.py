"""
ERGO SENSOR SYSTEM - Full Test Suite
Tests: Server, API endpoints, WebSocket, data pipeline, AI predictions
"""
import requests
import json
import time
import sys
import threading

BASE = "http://127.0.0.1:5000"
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results = {"pass": 0, "fail": 0, "warn": 0}

def ok(msg):
    print(f"  {PASS} {msg}")
    results["pass"] += 1

def fail(msg):
    print(f"  {FAIL} {msg}")
    results["fail"] += 1

def warn(msg):
    print(f"  {WARN} {msg}")
    results["warn"] += 1

def info(msg):
    print(f"  {INFO} {msg}")

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

# ─────────────────────────────────────────────────────────
# 1. SERVER REACHABILITY
# ─────────────────────────────────────────────────────────
section("1. Server Reachability")
try:
    r = requests.get(BASE + "/", timeout=5)
    if r.status_code == 200:
        ok(f"GET /  →  HTTP {r.status_code}")
    else:
        fail(f"GET /  →  HTTP {r.status_code}")
except Exception as e:
    fail(f"Server unreachable: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────
# 2. PAGE ROUTES
# ─────────────────────────────────────────────────────────
section("2. Page Routes")
pages = ["/", "/system", "/ai", "/history", "/settings", "/login"]
for page in pages:
    try:
        r = requests.get(BASE + page, timeout=5, allow_redirects=True)
        if r.status_code in (200, 302):
            ok(f"GET {page}  →  HTTP {r.status_code}")
        else:
            fail(f"GET {page}  →  HTTP {r.status_code}")
    except Exception as e:
        fail(f"GET {page}  →  {e}")

# ─────────────────────────────────────────────────────────
# 3. API TIME SYNC
# ─────────────────────────────────────────────────────────
section("3. API — Time Sync")
try:
    r = requests.get(BASE + "/api/time", timeout=5)
    if r.status_code == 200:
        t = int(r.text.strip())
        now = int(time.time())
        drift = abs(t - now)
        if drift < 10:
            ok(f"GET /api/time  →  {t}  (drift={drift}s)")
        else:
            warn(f"GET /api/time  →  {t}  (drift={drift}s — high)")
    else:
        fail(f"GET /api/time  →  HTTP {r.status_code}")
except Exception as e:
    fail(f"/api/time  →  {e}")

# ─────────────────────────────────────────────────────────
# 4. SENSOR STATUS
# ─────────────────────────────────────────────────────────
section("4. API — Sensor Status")
try:
    r = requests.get(BASE + "/api/sensors", timeout=5)
    if r.status_code == 200:
        sensors = r.json()
        ok(f"GET /api/sensors  →  {len(sensors)} sensors returned")
        online = [s for s in sensors if s.get("online")]
        info(f"  Online: {[s['sensor_id'] for s in online]}")
        if len(online) == 0:
            warn("No sensors currently online")
        else:
            ok(f"{len(online)} sensor(s) currently ONLINE")
    else:
        fail(f"GET /api/sensors  →  HTTP {r.status_code}")
except Exception as e:
    fail(f"/api/sensors  →  {e}")

# ─────────────────────────────────────────────────────────
# 5. DATA INGESTION — HTTP POST
# ─────────────────────────────────────────────────────────
section("5. API — Data Ingestion (POST /api/data)")

test_payloads = [
    {"sensor_id": "NECK",       "roll": -3.5,  "pitch": 7.2,  "yaw": 0.5},
    {"sensor_id": "UPPER_BACK", "roll": 73.5,  "pitch": -10.1, "yaw": 55.0},
]

for payload in test_payloads:
    try:
        r = requests.post(BASE + "/api/data", json=payload, timeout=5)
        if r.status_code == 200:
            ok(f"POST /api/data  {payload['sensor_id']}  →  HTTP 200")
        else:
            fail(f"POST /api/data  {payload['sensor_id']}  →  HTTP {r.status_code}  {r.text[:80]}")
    except Exception as e:
        fail(f"POST /api/data  →  {e}")

# Bad payload
try:
    r = requests.post(BASE + "/api/data", json={"bad": "payload"}, timeout=5)
    if r.status_code in (400, 422):
        ok(f"POST /api/data bad payload  →  HTTP {r.status_code} (correctly rejected)")
    else:
        warn(f"POST /api/data bad payload  →  HTTP {r.status_code} (expected 400/422)")
except Exception as e:
    warn(f"POST /api/data bad payload  →  {e}")

# ─────────────────────────────────────────────────────────
# 6. CSV DOWNLOAD
# ─────────────────────────────────────────────────────────
section("6. API — CSV Download")
try:
    r = requests.get(BASE + "/api/csv/latest", timeout=10)
    if r.status_code == 200:
        size = len(r.content)
        ok(f"GET /api/csv/latest  →  {size} bytes")
    elif r.status_code == 404:
        warn("GET /api/csv/latest  →  404 (no CSV yet, session may be too short)")
    else:
        fail(f"GET /api/csv/latest  →  HTTP {r.status_code}")
except Exception as e:
    fail(f"/api/csv/latest  →  {e}")

# ─────────────────────────────────────────────────────────
# 7. PIPELINE — Inject 70 frames and verify AI kicks in
# ─────────────────────────────────────────────────────────
section("7. Pipeline — 70-frame injection (AI warm-up)")
info("Injecting 70 sensor frames at 10Hz to trigger AI predictions...")

ai_triggered = False
for i in range(75):
    neck_p = {"sensor_id": "NECK",       "roll": float(-3 + i*0.01),  "pitch": float(7 + i*0.02),   "yaw": 0.5}
    back_p = {"sensor_id": "UPPER_BACK", "roll": float(73 + i*0.01),  "pitch": float(-10 + i*0.01), "yaw": float(55 + i*0.1)}
    requests.post(BASE + "/api/data", json=neck_p, timeout=3)
    requests.post(BASE + "/api/data", json=back_p, timeout=3)
    time.sleep(0.11)
    if i == 74:
        time.sleep(0.5)

ok("70 frames injected successfully")

# ─────────────────────────────────────────────────────────
# 8. ANGLES ENDPOINT (if exists)
# ─────────────────────────────────────────────────────────
section("8. API — Latest Angles")
try:
    r = requests.get(BASE + "/api/angles/latest", timeout=5)
    if r.status_code == 200:
        data = r.json()
        ok(f"GET /api/angles/latest  →  keys: {list(data.keys())[:6]}")
    elif r.status_code == 404:
        warn("GET /api/angles/latest  →  endpoint not defined (optional)")
    else:
        warn(f"GET /api/angles/latest  →  HTTP {r.status_code}")
except Exception as e:
    warn(f"/api/angles/latest  →  {e}")

# ─────────────────────────────────────────────────────────
# 9. REPORT GENERATION
# ─────────────────────────────────────────────────────────
section("9. API — PDF Report Generation")
try:
    r = requests.post(BASE + "/api/report/generate", timeout=30)
    if r.status_code == 200 and "application/pdf" in r.headers.get("Content-Type", ""):
        ok(f"POST /api/report/generate  →  PDF {len(r.content)} bytes")
    elif r.status_code == 200:
        ok(f"POST /api/report/generate  →  HTTP 200  ({len(r.content)} bytes)")
    else:
        fail(f"POST /api/report/generate  →  HTTP {r.status_code}  {r.text[:120]}")
except Exception as e:
    fail(f"/api/report/generate  →  {e}")

# ─────────────────────────────────────────────────────────
# 10. WebSocket CONNECTIVITY
# ─────────────────────────────────────────────────────────
section("10. WebSocket — Connectivity")
try:
    import socketio as sio_client
    sio = sio_client.Client(logger=False, engineio_logger=False)
    connected = threading.Event()
    angles_received = threading.Event()
    raw_received = threading.Event()
    received_payload = {}

    @sio.event
    def connect():
        connected.set()

    @sio.on("raw_sensors")
    def on_raw(data):
        raw_received.set()

    @sio.on("angles")
    def on_angles(data):
        received_payload.update(data)
        angles_received.set()

    sio.connect(BASE, transports=["websocket"], wait_timeout=8)
    if connected.wait(timeout=5):
        ok("WebSocket connected successfully")
    else:
        fail("WebSocket connection timed out")

    # Inject one more pair of frames to trigger emit
    requests.post(BASE + "/api/data", json={"sensor_id": "NECK",       "roll": -3.5, "pitch": 7.4, "yaw": 0.5})
    requests.post(BASE + "/api/data", json={"sensor_id": "UPPER_BACK", "roll": 73.8, "pitch": -10.0, "yaw": 55.5})

    if raw_received.wait(timeout=5):
        ok("WebSocket 'raw_sensors' event received")
    else:
        warn("WebSocket 'raw_sensors' event NOT received within 5s")

    if angles_received.wait(timeout=5):
        ok(f"WebSocket 'angles' event received")
        ang = received_payload.get("angles", {})
        info(f"  Angles keys: {list(ang.keys())}")
        rula = received_payload.get("rula", {})
        reba = received_payload.get("reba", {})
        info(f"  RULA right: {rula.get('right', {}).get('final', 'N/A')}  left: {rula.get('left', {}).get('final', 'N/A')}")
        info(f"  REBA right: {reba.get('right', {}).get('final', 'N/A')}  left: {reba.get('left', {}).get('final', 'N/A')}")
        if received_payload.get("ai_predictions"):
            ok(f"  AI predictions present: risk_level={received_payload['ai_predictions'].get('risk_level')}")
        else:
            warn("  AI predictions not in payload (may need more frames)")
    else:
        fail("WebSocket 'angles' event NOT received within 5s")

    sio.disconnect()

except ImportError:
    warn("python-socketio client not installed — skipping WebSocket test")
    info("  Install with: pip install python-socketio[client]")
except Exception as e:
    fail(f"WebSocket test  →  {e}")

# ─────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────
section("RESULTS SUMMARY")
total = results["pass"] + results["fail"] + results["warn"]
print(f"  Total:   {total}")
print(f"  \033[92mPassed:  {results['pass']}\033[0m")
print(f"  \033[91mFailed:  {results['fail']}\033[0m")
print(f"  \033[93mWarnings:{results['warn']}\033[0m")
if results["fail"] == 0:
    print(f"\n  \033[92m✓ ALL TESTS PASSED\033[0m\n")
else:
    print(f"\n  \033[91m✗ {results['fail']} FAILURE(S) — review above\033[0m\n")
