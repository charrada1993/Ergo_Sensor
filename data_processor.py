import time
import os
from threading import Lock
from angle_math import compute_joint_angles
from risk_engine import RiskEngine
from csv_logger import CSVLogger
from rula_engine import RULAEngine
from reba_engine import REBAEngine

class DataProcessor:
    def __init__(self, config, socketio):
        self.config = config
        self.socketio = socketio
        self.lock = Lock()
        self.sensor_buffer = {}       # sensor_id -> (roll, pitch, yaw, sensor_timestamp)
        self.last_seen = {}            # sensor_id -> server time when last data received
        self.risk_engine = RiskEngine(config)
        self.csv_logger = CSVLogger(config)
        self.rula_engine = RULAEngine(config)   # RULA engine instance
        self.reba_engine = REBAEngine(config)   # REBA engine instance

    def process_incoming(self, sensor_id, roll, pitch, yaw, sensor_timestamp):
        with self.lock:
            print(f"Received from {sensor_id}: roll={roll}, pitch={pitch}, yaw={yaw}")
            # Update last seen with server time
            self.last_seen[sensor_id] = time.time()
            self.sensor_buffer[sensor_id] = (roll, pitch, yaw, sensor_timestamp)

        # Build a dictionary of all available sensors (ignore missing)
        data = {}
        for sid in self.config.EXPECTED_SENSORS:
            if sid in self.sensor_buffer:
                r, p, y, ts = self.sensor_buffer[sid]
                data[sid] = (r, p, y)

        print(f"[DEBUG] Current sensor data: {data}")

        # Only compute if at least two sensors are present
        if len(data) >= 2:
            angles = compute_joint_angles(data)
            print(f"[DEBUG] Computed angles: {angles}")

            if angles:  # only proceed if angles dict is not empty
                # Determine current time (use max timestamp from buffer)
                if self.sensor_buffer:
                    current_time = max(ts for _,_,_,ts in self.sensor_buffer.values())
                else:
                    current_time = time.time()

                # Update risk engine and compute MSD risk
                self.risk_engine.update(angles, current_time)
                risk = self.risk_engine.compute_risk()

                # --- RULA calculation ---
                # Compute trunk angle from UPPER_BACK pitch (if available)
                trunk_angle = 0
                if 'UPPER_BACK' in data:
                    trunk_angle = abs(data['UPPER_BACK'][1])  # pitch

                # Right side RULA
                rula_right = None
                if ('R_Shoulder' in angles and 'R_Elbow' in angles and 'R_Wrist' in angles):
                    rula_right = self.rula_engine.compute_side(
                        shoulder_angle=angles['R_Shoulder'],
                        elbow_angle=angles['R_Elbow'],
                        wrist_angle=angles['R_Wrist'],
                        neck_angle=angles.get('Neck', 0),
                        trunk_angle=trunk_angle
                    )

                # Left side RULA
                rula_left = None
                if ('L_Shoulder' in angles and 'L_Elbow' in angles and 'L_Wrist' in angles):
                    rula_left = self.rula_engine.compute_side(
                        shoulder_angle=angles['L_Shoulder'],
                        elbow_angle=angles['L_Elbow'],
                        wrist_angle=angles['L_Wrist'],
                        neck_angle=angles.get('Neck', 0),
                        trunk_angle=trunk_angle
                    )

                # --- Additional data for trunk and legs ---
                trunk_angle_value = trunk_angle
                legs_score_value = self.config.REBA_LEGS_SCORE  # default from config

                # Log to CSV including RULA scores
                self.csv_logger.log(angles, risk, rula={'right': rula_right, 'left': rula_left})

                # Emit via WebSocket including angles, MSD risk, RULA, and new fields
                self.socketio.emit('angles', {
                    'angles': angles,
                    'risk': risk,
                    'rula': {
                        'right': rula_right,
                        'left': rula_left
                    },
                    'trunk_angle': trunk_angle_value,
                    'legs_score': legs_score_value
                })
                print(f"Emitted angles: {angles}")
                print(f"RULA right: {rula_right}, left: {rula_left}")
                print(f"Trunk angle: {trunk_angle_value}, Legs score: {legs_score_value}")

    def get_sensor_status(self):
        """Return list of sensors with last seen and online status."""
        with self.lock:
            now = time.time()
            status = []
            for sid in self.config.EXPECTED_SENSORS:
                last = self.last_seen.get(sid)
                if last:
                    # Consider online if last seen within 2 * post interval (plus margin)
                    online = (now - last) < (2 * self.config.POST_INTERVAL_MS / 1000.0 + 1)
                else:
                    online = False
                status.append({
                    'sensor_id': sid,
                    'last_seen': last,
                    'online': online
                })
            return status

    def get_current_log_filename(self):
        """Return the basename of the current CSV log file, or None."""
        with self.lock:
            if self.csv_logger and self.csv_logger.filename:
                return os.path.basename(self.csv_logger.filename)
            return None