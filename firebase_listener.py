import threading
import time
import os
import json
import firebase_admin
from firebase_admin import credentials, db

# ── Suppress internal firebase_admin SSE thread crash (NoneType.read) ──────
_orig_excepthook = threading.excepthook

def _firebase_excepthook(args):
    # Swallow the AttributeError that firebase_admin throws when the SSE
    # connection drops (AttributeError: 'NoneType' object has no attribute 'read')
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
        if not isinstance(data, dict):
            return
        roll  = data.get('roll')
        pitch = data.get('pitch')
        yaw   = data.get('yaw')
        timestamp = data.get('timestamp', time.time())
        if roll is not None and pitch is not None and yaw is not None:
            self.data_processor.process_incoming(
                sensor_id, float(roll), float(pitch), float(yaw), float(timestamp)
            )

    def _process_data_node(self, sensor_id, data):
        """Recursively process data to find sensor readings."""
        if not isinstance(data, dict):
            return
            
        # Is this a reading itself?
        if 'roll' in data and 'pitch' in data and 'yaw' in data:
            self._forward_sensor(sensor_id, data)
            return
            
        # Otherwise, it might be a dict of push IDs
        for key, value in data.items():
            if isinstance(value, dict):
                if 'roll' in value and 'pitch' in value and 'yaw' in value:
                    self._forward_sensor(sensor_id, value)
                else:
                    self._process_data_node(sensor_id, value)

    def _listen(self):
        """Listen for real-time data under /sensor_data/ with auto-reconnect."""
        SILENCE_TIMEOUT = 30  # seconds with no events before reconnect

        while self.running:
            ref = db.reference('/sensor_data')
            self._initial_load = True
            self._last_event_time = time.time()
            listener = None

            def stream_handler(message):
                self._last_event_time = time.time()
                try:
                    event = message.event_type
                    path  = message.path
                    data  = message.data

                    # Ignore the very first full-history snapshot
                    if self._initial_load and path == '/':
                        self._initial_load = False
                        print("[Firebase] Initial snapshot skipped. Listening for live data...")
                        return
                    self._initial_load = False

                    if data is None:
                        return

                    parts = [p for p in path.split('/') if p]

                    if len(parts) == 0:
                        if isinstance(data, dict):
                            for sid, sdata in data.items():
                                self._process_data_node(sid, sdata)

                    elif len(parts) >= 1:
                        sensor_id = parts[0]

                        if len(parts) == 1:
                            self._process_data_node(sensor_id, data)

                        elif len(parts) == 2:
                            if isinstance(data, dict):
                                self._process_data_node(sensor_id, data)
                            else:
                                field = parts[1]
                                if sensor_id not in self._sensor_cache:
                                    self._sensor_cache[sensor_id] = {}
                                self._sensor_cache[sensor_id][field] = data
                                cached = self._sensor_cache[sensor_id]
                                if all(k in cached for k in ('roll', 'pitch', 'yaw')):
                                    self._forward_sensor(sensor_id, cached)

                        elif len(parts) >= 3:
                            # /SENSOR/push_id/field
                            sensor_id = parts[0]
                            field = parts[-1]
                            if not isinstance(data, dict):
                                if sensor_id not in self._sensor_cache:
                                    self._sensor_cache[sensor_id] = {}
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

                # Watchdog: check for silence and reconnect if needed
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