import threading
import time
import firebase_admin
from firebase_admin import credentials, db

class FirebaseListener:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.thread = None
        self.running = False

    def start(self, cred_path, database_url):
        """Initialize Firebase and start listening in a background thread."""
        cred = credentials.Certificate(cred_path)
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
                    sensor_id = path_parts[1]  # e.g., /NECK/...
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