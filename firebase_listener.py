import threading
import time
import os
import json
import firebase_admin
from firebase_admin import credentials, db

# ── Suppress internal firebase_admin SSE thread crash (NoneType.read) ──────
_orig_excepthook = threading.excepthook

def _firebase_excepthook(args):
    if args.exc_type is AttributeError and 'read' in str(args.exc_value):
        return
    _orig_excepthook(args)

threading.excepthook = _firebase_excepthook


class FirebaseListener:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.thread = None
        self.running = False
        self._sensor_cache = {}
        # Track last forwarded timestamp per sensor to deduplicate
        self._last_forwarded_ts = {}

    def start(self, cred_path, database_url):
        """Initialize Firebase and start listening in a background thread."""
        env_creds = os.environ.get("FIREBASE_CREDS_JSON")
        if env_creds:
            try:
                cred_dict = json.loads(env_creds)
                cred = credentials.Certificate(cred_dict)
                print("[OK] Firebase initialized using FIREBASE_CREDS_JSON")
            except Exception as e:
                print(f"[ERROR] Failed to parse FIREBASE_CREDS_JSON: {e}")
                return
        elif os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            print(f"[OK] Firebase initialized using {cred_path}")
        else:
            print(f"[ERROR] Firebase credentials not found at {cred_path} and FIREBASE_CREDS_JSON is missing.")
            return

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {'databaseURL': database_url})
        self.running = True
        self.thread = threading.Thread(target=self._listen)
        self.thread.daemon = True
        self.thread.start()
        print("Firebase listener started")

    def _forward_sensor(self, sensor_id, data):
        """Forward sensor data to the data processor with timestamp deduplication."""
        if not isinstance(data, dict):
            return
        roll      = data.get('roll')
        pitch     = data.get('pitch')
        yaw       = data.get('yaw')
        timestamp = data.get('timestamp', time.time())
        if roll is None or pitch is None or yaw is None:
            return

        # Deduplicate: skip if this exact timestamp was already forwarded
        last_ts = self._last_forwarded_ts.get(sensor_id, 0)
        if float(timestamp) <= last_ts:
            return
        self._last_forwarded_ts[sensor_id] = float(timestamp)

        self.data_processor.process_incoming(
            sensor_id, float(roll), float(pitch), float(yaw), float(timestamp)
        )

    def _process_snapshot(self, snapshot_data):
        """Process a full /sensor_data snapshot (dict of sensor_id -> reading)."""
        if not isinstance(snapshot_data, dict):
            return
        for sensor_id, sdata in snapshot_data.items():
            if isinstance(sdata, dict) and 'roll' in sdata and 'pitch' in sdata and 'yaw' in sdata:
                self._forward_sensor(sensor_id, sdata)
                self._sensor_cache[sensor_id] = dict(sdata)

    def _listen(self):
        """Listen for real-time data under /sensor_data/ with auto-reconnect."""
        SILENCE_TIMEOUT = 30  # seconds without any event before reconnect

        while self.running:
            ref = db.reference('/sensor_data')
            self._last_event_time = time.time()
            listener = None

            def stream_handler(message):
                self._last_event_time = time.time()
                try:
                    path = message.path
                    data = message.data

                    if data is None:
                        return

                    parts = [p for p in path.split('/') if p]

                    # ── Full snapshot: /  ─────────────────────────────────
                    # Render gets data this way on initial connect AND on
                    # reconnect. Process it — deduplication prevents replaying.
                    if len(parts) == 0:
                        print("[Firebase] Full snapshot received. Processing...")
                        self._process_snapshot(data)

                    # ── Single sensor node: /SENSOR_ID  ──────────────────
                    elif len(parts) == 1:
                        sensor_id = parts[0]
                        if isinstance(data, dict) and 'roll' in data and 'pitch' in data and 'yaw' in data:
                            self._forward_sensor(sensor_id, data)
                            self._sensor_cache[sensor_id] = dict(data)

                    # ── Single field: /SENSOR_ID/field  ──────────────────
                    elif len(parts) == 2:
                        sensor_id = parts[0]
                        field     = parts[1]
                        if sensor_id not in self._sensor_cache:
                            self._sensor_cache[sensor_id] = {}
                        if isinstance(data, dict):
                            if 'roll' in data and 'pitch' in data and 'yaw' in data:
                                self._forward_sensor(sensor_id, data)
                                self._sensor_cache[sensor_id] = dict(data)
                        else:
                            self._sensor_cache[sensor_id][field] = data
                            cached = self._sensor_cache[sensor_id]
                            if all(k in cached for k in ('roll', 'pitch', 'yaw')):
                                self._forward_sensor(sensor_id, cached)

                    # ── Deep field: /SENSOR_ID/subkey/field  ─────────────
                    elif len(parts) >= 3:
                        sensor_id = parts[0]
                        field     = parts[-1]
                        if sensor_id not in self._sensor_cache:
                            self._sensor_cache[sensor_id] = {}
                        if not isinstance(data, dict):
                            self._sensor_cache[sensor_id][field] = data
                            cached = self._sensor_cache[sensor_id]
                            if all(k in cached for k in ('roll', 'pitch', 'yaw')):
                                self._forward_sensor(sensor_id, cached)

                except Exception as e:
                    print(f"[Firebase ERROR] stream_handler: {e}")

            try:
                try:
                    listener = ref.listen(stream_handler, raise_on_error=False)
                except TypeError:
                    listener = ref.listen(stream_handler)
                print("[Firebase] Stream started.")

                # Watchdog: reconnect if no events arrive for SILENCE_TIMEOUT
                while self.running:
                    time.sleep(5)
                    silence = time.time() - self._last_event_time
                    if silence > SILENCE_TIMEOUT:
                        print(f"[Firebase] No events for {silence:.0f}s — reconnecting...")
                        break

            except Exception as e:
                print(f"[Firebase ERROR] Listener crashed: {e}")

            finally:
                if listener:
                    try:
                        listener.close()
                    except Exception:
                        pass

            if self.running:
                print("[Firebase] Restarting listener in 3s...")
                time.sleep(3)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)