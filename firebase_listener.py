import threading
import time
import os
import json
import firebase_admin
from firebase_admin import credentials, db

class FirebaseListener:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.thread = None
        self.running = False

    def start(self, cred_path, database_url):
        """Initialize Firebase and start listening in a background thread."""
        # --- Handle credentials from ENV or FILE ---
        env_creds = os.environ.get("FIREBASE_CREDS_JSON")
        if env_creds:
            try:
                # credentials.Certificate can take a dict
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

        firebase_admin.initialize_app(cred, {'databaseURL': database_url})
        self.running = True
        self.thread = threading.Thread(target=self._listen)
        self.thread.daemon = True
        self.thread.start()
        print("Firebase listener started")

    def _listen(self):
        """Listen for new data under /sensor_data/."""
        ref = db.reference('/sensor_data')
        # Use stream to get real-time updates
        def stream_handler(message):
            if message.event_type == 'put' and message.data:
                # New data added
                path_parts = message.path.split('/')
                if len(path_parts) >= 2:
                    sensor_id = path_parts[1].upper()  # Convert to UPPERCASE to match config
                    data = message.data
                    # The data is a dict with keys: roll, pitch, yaw, timestamp
                    roll = data.get('roll')
                    pitch = data.get('pitch')
                    yaw = data.get('yaw')
                    timestamp = data.get('timestamp', time.time())
                    if roll is not None and pitch is not None and yaw is not None:
                        print(f"Firebase update from {sensor_id}")
                        self.data_processor.process_incoming(sensor_id, roll, pitch, yaw, timestamp)

        ref.listen(stream_handler)
        # Keep thread alive
        while self.running:
            time.sleep(1)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)