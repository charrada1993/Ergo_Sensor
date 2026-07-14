# -*- coding: utf-8 -*-
"""
test_firebase.py - Test suite for FirebaseListener and FirebaseStorage
Run with: .venv_ergo/Scripts/python.exe test_firebase.py
"""
import os
import sys
import time
import json
import unittest
from unittest.mock import MagicMock, patch, call

# ──────────────────────────────────────────────────────────────────────────────
# Colour helpers (cross-platform)
# ──────────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):  print(f"  {GREEN}[PASS]{RESET} {msg}")
def fail(msg):print(f"  {RED}[FAIL]{RESET} {msg}")
def info(msg):print(f"  {CYAN}[INFO]{RESET} {msg}")
def warn(msg):print(f"  {YELLOW}[WARN]{RESET} {msg}")
def sep(title=""):
    line = "-" * 60
    if title:
        print(f"\n{BOLD}{CYAN}{line}{RESET}")
        print(f"{BOLD}{CYAN}  {title}{RESET}")
        print(f"{BOLD}{CYAN}{line}{RESET}")
    else:
        print(f"{CYAN}{line}{RESET}")

PASS = FAIL = 0


def record(passed, label):
    global PASS, FAIL
    if passed:
        ok(label)
        PASS += 1
    else:
        fail(label)
        FAIL += 1


# ══════════════════════════════════════════════════════════════════════════════
# 1. FirebaseStorage Tests (REST-based, no credentials required)
# ══════════════════════════════════════════════════════════════════════════════
sep("1. FirebaseStorage — Unit Tests (mocked HTTP)")

from firebase_storage import FirebaseStorage

# ── 1a. URL construction ──────────────────────────────────────────────────────
# _get_url simply appends .json to the path; dot->underscore happens in callers
storage = FirebaseStorage("https://ergo-a4b30-default-rtdb.firebaseio.com/")
expected_url = "https://ergo-a4b30-default-rtdb.firebaseio.com/reports/test.pdf.json"
actual_url   = storage._get_url("reports/test.pdf")
record(actual_url == expected_url,
       f"_get_url appends .json to path (no sanitisation): {actual_url}")

# ── 1b. put_file success ──────────────────────────────────────────────────────
with patch("firebase_storage.requests.put") as mock_put:
    mock_put.return_value = MagicMock(status_code=200)
    result = storage.put_file("reports", "hello.pdf", b"PDF content here")
    record(result is True, "put_file returns True on HTTP 200")
    record(mock_put.called, "put_file calls requests.put")
    called_url = mock_put.call_args[0][0]
    record("hello_pdf" in called_url, f"put_file URL contains sanitised key: {called_url}")

# ── 1c. put_file failure ──────────────────────────────────────────────────────
with patch("firebase_storage.requests.put") as mock_put:
    mock_put.return_value = MagicMock(status_code=500, text="server error")
    result = storage.put_file("reports", "bad.pdf", b"data")
    record(result is False, "put_file returns False on HTTP 500")

# ── 1d. put_file network error ────────────────────────────────────────────────
with patch("firebase_storage.requests.put", side_effect=ConnectionError("timeout")):
    result = storage.put_file("reports", "err.pdf", b"data")
    record(result is False, "put_file returns False on network exception")

# ── 1e. get_file success ──────────────────────────────────────────────────────
import base64
raw_bytes = b"fake PDF bytes 12345"
b64_data  = base64.b64encode(raw_bytes).decode()
with patch("firebase_storage.requests.get") as mock_get:
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"filename": "test.pdf", "data": b64_data, "timestamp": time.time()}
    )
    result = storage.get_file("reports", "test.pdf")
    record(result == raw_bytes, f"get_file returns correct decoded bytes ({len(result)} bytes)")

# ── 1f. get_file not found ────────────────────────────────────────────────────
with patch("firebase_storage.requests.get") as mock_get:
    mock_get.return_value = MagicMock(status_code=200, json=lambda: None)
    result = storage.get_file("reports", "missing.pdf")
    record(result is None, "get_file returns None when data is null")

# ── 1g. delete_file success ───────────────────────────────────────────────────
with patch("firebase_storage.requests.delete") as mock_del:
    mock_del.return_value = MagicMock(status_code=200)
    result = storage.delete_file("reports", "old.pdf")
    record(result is True, "delete_file returns True on HTTP 200")

# ── 1h. delete_file failure ───────────────────────────────────────────────────
with patch("firebase_storage.requests.delete") as mock_del:
    mock_del.return_value = MagicMock(status_code=404)
    result = storage.delete_file("reports", "ghost.pdf")
    record(result is False, "delete_file returns False on HTTP 404")

# ── 1i. list_files ────────────────────────────────────────────────────────────
fake_listing = {
    "a": {"filename": "report_2026_b.pdf", "data": "xxx", "timestamp": 1000},
    "b": {"filename": "report_2026_a.pdf", "data": "yyy", "timestamp": 900},
}
with patch("firebase_storage.requests.get") as mock_get:
    mock_get.return_value = MagicMock(status_code=200, json=lambda: fake_listing)
    files = storage.list_files("reports")
    record(isinstance(files, list) and len(files) == 2, f"list_files returns list of {len(files)} items")
    record(files[0] >= files[1], f"list_files is reverse-sorted: {files}")

# ── 1j. list_files empty ─────────────────────────────────────────────────────
with patch("firebase_storage.requests.get") as mock_get:
    mock_get.return_value = MagicMock(status_code=200, json=lambda: None)
    files = storage.list_files("reports")
    record(files == [], "list_files returns [] when no data")

# ── 1k. get_files_metadata ────────────────────────────────────────────────────
with patch("firebase_storage.requests.get") as mock_get:
    mock_get.return_value = MagicMock(status_code=200, json=lambda: fake_listing)
    meta = storage.get_files_metadata("reports")
    record("report_2026_b.pdf" in meta, "get_files_metadata returns correct filename key")
    record("timestamp" in meta.get("report_2026_b.pdf", {}), "get_files_metadata includes timestamp")
    record("size" in meta.get("report_2026_b.pdf", {}), "get_files_metadata includes size estimate")

# ── 1l. base64 round-trip integrity ──────────────────────────────────────────
import io
test_payload = b"\x00\x01\x02" * 100 + b"PDF DATA"
with patch("firebase_storage.requests.put") as mock_put, \
     patch("firebase_storage.requests.get") as mock_get:
    captured = {}
    def capture_put(url, json=None, timeout=None):
        captured['payload'] = json
        return MagicMock(status_code=200)
    mock_put.side_effect = capture_put
    storage.put_file("test", "binary.bin", test_payload)

    def return_captured(url, timeout=None):
        return MagicMock(status_code=200, json=lambda: captured['payload'])
    mock_get.side_effect = return_captured
    retrieved = storage.get_file("test", "binary.bin")
    record(retrieved == test_payload, f"Base64 round-trip integrity OK ({len(test_payload)} bytes)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. FirebaseListener Tests (mocked firebase_admin)
# ══════════════════════════════════════════════════════════════════════════════
sep("2. FirebaseListener — Unit Tests (mocked firebase_admin)")

# Patch firebase_admin before importing FirebaseListener so it doesn't
# actually connect to anything
with patch.dict(sys.modules, {
    'firebase_admin':             MagicMock(),
    'firebase_admin.credentials': MagicMock(),
    'firebase_admin.db':          MagicMock(),
}):
    import importlib
    import firebase_listener as fl_module
    importlib.reload(fl_module)          # reload with mocked deps
    FirebaseListener = fl_module.FirebaseListener

    mock_processor = MagicMock()
    mock_processor.process_incoming = MagicMock()

    # ── 2a. No credentials → does not start ──────────────────────────────────
    listener = FirebaseListener(mock_processor)
    with patch.dict(os.environ, {}, clear=True):
        if "FIREBASE_CREDS_JSON" in os.environ:
            del os.environ["FIREBASE_CREDS_JSON"]
        listener.start("non_existent_creds.json", "https://example.com")
    record(listener.running is False,
           "start() does not run when no credentials found")

    # ── 2b. Credentials via env var ───────────────────────────────────────────
    fake_creds_json = json.dumps({
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "key123",
        "private_key": "-----BEGIN RSA PRIVATE KEY-----\nFAKE\n-----END RSA PRIVATE KEY-----\n",
        "client_email": "test@test-project.iam.gserviceaccount.com",
        "client_id": "123",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    })
    listener2 = FirebaseListener(mock_processor)
    with patch.dict(os.environ, {"FIREBASE_CREDS_JSON": fake_creds_json}), \
         patch.object(fl_module.threading, "Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        listener2.start("dummy.json", "https://example.firebaseio.com/")
    record(listener2.running is True,
           "start() sets running=True when FIREBASE_CREDS_JSON is set")
    record(mock_thread.called, "start() spawns a background thread")

    # ── 2c. _forward_sensor: valid data forwarded ─────────────────────────────
    listener3 = FirebaseListener(mock_processor)
    mock_processor.reset_mock()
    data = {"roll": 10.0, "pitch": 20.0, "yaw": 30.0, "timestamp": 1000.0}
    listener3._forward_sensor("NECK", data)
    record(mock_processor.process_incoming.call_count == 1,
           "_forward_sensor calls process_incoming for valid data")
    args = mock_processor.process_incoming.call_args[0]
    record(args[0] == "NECK" and args[1] == 10.0 and args[2] == 20.0 and args[3] == 30.0,
           f"_forward_sensor passes correct (sensor_id, roll, pitch, yaw): {args[:4]}")

    # ── 2d. _forward_sensor: missing field not forwarded ──────────────────────
    mock_processor.reset_mock()
    listener3._forward_sensor("NECK", {"roll": 5.0, "pitch": 10.0})  # missing yaw
    record(mock_processor.process_incoming.call_count == 0,
           "_forward_sensor ignores data with missing yaw")

    # ── 2e. _forward_sensor: timestamp deduplication ──────────────────────────
    mock_processor.reset_mock()
    listener4 = FirebaseListener(mock_processor)
    data = {"roll": 1.0, "pitch": 2.0, "yaw": 3.0, "timestamp": 500.0}
    listener4._forward_sensor("UPPER_BACK", data)
    listener4._forward_sensor("UPPER_BACK", data)   # same timestamp → duplicate
    record(mock_processor.process_incoming.call_count == 1,
           "_forward_sensor deduplicates identical timestamps")

    # ── 2f. _forward_sensor: newer timestamp IS forwarded ────────────────────
    mock_processor.reset_mock()
    data2 = {"roll": 1.0, "pitch": 2.0, "yaw": 3.0, "timestamp": 501.0}
    listener4._forward_sensor("UPPER_BACK", data2)
    record(mock_processor.process_incoming.call_count == 1,
           "_forward_sensor forwards data with newer timestamp")

    # ── 2g. _process_snapshot: full dict snapshot ─────────────────────────────
    mock_processor.reset_mock()
    listener5 = FirebaseListener(mock_processor)
    snapshot = {
        "NECK":       {"roll": 5.0, "pitch": 10.0, "yaw": 15.0, "timestamp": 1.0},
        "UPPER_BACK": {"roll": 6.0, "pitch": 11.0, "yaw": 16.0, "timestamp": 2.0},
        "BAD_SENSOR": {"no_roll": True},  # should be skipped
    }
    listener5._process_snapshot(snapshot)
    record(mock_processor.process_incoming.call_count == 2,
           f"_process_snapshot forwards only valid sensors (got {mock_processor.process_incoming.call_count}/2)")

    # ── 2h. _process_snapshot: non-dict input ────────────────────────────────
    mock_processor.reset_mock()
    listener5._process_snapshot("invalid_data")
    record(mock_processor.process_incoming.call_count == 0,
           "_process_snapshot safely ignores non-dict input")

    # ── 2i. _process_snapshot: caches sensor data ────────────────────────────
    listener6 = FirebaseListener(mock_processor)
    listener6._process_snapshot({
        "R_BICEPS": {"roll": 1.0, "pitch": 2.0, "yaw": 3.0, "timestamp": 99.0}
    })
    record("R_BICEPS" in listener6._sensor_cache,
           "_process_snapshot populates _sensor_cache for processed sensors")

    # ── 2j. stop() sets running to False ─────────────────────────────────────
    listener7 = FirebaseListener(mock_processor)
    listener7.running = True
    listener7.thread = MagicMock()
    listener7.stop()
    record(listener7.running is False, "stop() sets running=False")
    record(listener7.thread.join.called, "stop() calls thread.join()")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Live connectivity test (no credentials needed — public REST endpoint)
# ══════════════════════════════════════════════════════════════════════════════
sep("3. FirebaseStorage — Live REST Connectivity Test")

import requests as req

STORAGE_URL = "https://ergo-a4b30-default-rtdb.firebaseio.com/"
storage_live = FirebaseStorage(STORAGE_URL)  # use original class

info(f"Target: {STORAGE_URL}")

# ── 3a. Reachability ─────────────────────────────────────────────────────────
try:
    r = req.get(STORAGE_URL + ".json", timeout=5)
    reachable = r.status_code in (200, 401, 403)
    record(reachable, f"Firebase REST endpoint reachable (HTTP {r.status_code})")
    if r.status_code in (401, 403):
        warn("Firebase security rules require authentication — live write/read tests skipped.")
        LIVE_AUTH = False
    else:
        LIVE_AUTH = True
except Exception as e:
    record(False, f"Firebase REST endpoint NOT reachable: {e}")
    LIVE_AUTH = False

# ── 3b. Live put + get + delete cycle (only if rules allow) ──────────────────
if LIVE_AUTH:
    test_path   = "ergo_test"
    test_file   = f"test_{int(time.time())}.txt"
    test_bytes  = b"Ergo Sensor Firebase Storage test payload"

    put_ok = storage_live.put_file(test_path, test_file, test_bytes)
    record(put_ok, f"live put_file: uploaded '{test_file}'")

    if put_ok:
        time.sleep(1)
        got_bytes = storage_live.get_file(test_path, test_file)
        record(got_bytes == test_bytes, f"live get_file: retrieved {len(got_bytes) if got_bytes else 0} bytes (expected {len(test_bytes)})")

        files = storage_live.list_files(test_path)
        record(test_file in files, f"live list_files: '{test_file}' appears in listing")

        del_ok = storage_live.delete_file(test_path, test_file)
        record(del_ok, f"live delete_file: deleted '{test_file}'")
    else:
        warn("Skipping get/list/delete — put failed (likely Firebase rules block anonymous writes).")
else:
    warn("Live write tests skipped (authentication required or endpoint unreachable).")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
sep()
total = PASS + FAIL
print(f"\n{BOLD}Results: {GREEN}{PASS} passed{RESET}{BOLD}, {RED}{FAIL} failed{RESET}{BOLD} / {total} total{RESET}")
if FAIL == 0:
    print(f"\n{GREEN}{BOLD}✓ All tests passed!{RESET}")
else:
    print(f"\n{RED}{BOLD}✗ {FAIL} test(s) failed.{RESET}")
    sys.exit(1)
