import time
import os
from threading import Lock
from collections import deque
from angle_math import compute_joint_angles, set_reference
from risk_engine import RiskEngine
from csv_logger import CSVLogger
from rula_engine import RULAEngine
from reba_engine import REBAEngine
from feature_extractor import FeatureExtractor


class DataProcessor:
    def __init__(self, config, socketio, ai_models=None):
        self.config    = config
        self.socketio  = socketio
        self.ai_models = ai_models
        self.lock      = Lock()

        self.sensor_buffer = {}
        self.last_seen = {}

        self.angle_window = deque(maxlen=60)
        self.risk_window  = deque(maxlen=60)
        self.rula_l_window = deque(maxlen=60)
        self.rula_r_window = deque(maxlen=60)
        self.reba_l_window = deque(maxlen=60)
        self.reba_r_window = deque(maxlen=60)

        # Initialise feature_extractor immediately if ai_models is provided
        if ai_models is not None and hasattr(ai_models, 'meta'):
            self.feature_extractor = FeatureExtractor(ai_models.meta)
        else:
            self.feature_extractor = None

        self.risk_engine  = RiskEngine(config)
        self.csv_logger   = CSVLogger(config)
        self.rula_engine  = RULAEngine(config)
        self.reba_engine  = REBAEngine(config)

    def set_ai_models(self, ai_models):
        self.ai_models = ai_models
        from feature_extractor import FeatureExtractor
        self.feature_extractor = FeatureExtractor(ai_models.meta) if hasattr(ai_models, 'meta') else None

    def calibrate(self):
        with self.lock:
            for sid, (r, p, y, ts) in self.sensor_buffer.items():
                set_reference(sid, r, p, y)
        print("[CALIBRATION] Reference orientations stored.")
        return True

    def process_incoming(self, sensor_id, roll, pitch, yaw, sensor_timestamp):
        if sensor_id:
            sensor_id = sensor_id.upper()
        with self.lock:
            print(f"[RECV] {sensor_id}: roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}")
            self.last_seen[sensor_id] = time.time()
            self.sensor_buffer[sensor_id] = (roll, pitch, yaw, sensor_timestamp)

            # --- Build raw sensor data dict (all sensors in buffer) ---
            raw_data = {
                sid: {
                    'roll': self.sensor_buffer[sid][0],
                    'pitch': self.sensor_buffer[sid][1],
                    'yaw': self.sensor_buffer[sid][2]
                }
                for sid in self.sensor_buffer
            }
            print(f"[DEBUG] Raw sensor buffer: {raw_data}")

            # --- Emit raw sensor data to dashboard (for debugging, always) ---
            self.socketio.emit('raw_sensors', raw_data)
            print("[EMIT] raw_sensors event sent.")

            # --- Build data dict using EXPECTED_SENSORS (or all sensors if EXPECTED_SENSORS empty) ---
            expected = getattr(self.config, 'EXPECTED_SENSORS', [])
            if expected:
                sensor_ids = expected
            else:
                sensor_ids = list(self.sensor_buffer.keys())

            data = {}
            for sid in sensor_ids:
                if sid in self.sensor_buffer:
                    data[sid] = self.sensor_buffer[sid][:3]

            print(f"[DEBUG] Filtered sensor snapshot (by EXPECTED_SENSORS): {data}")

            # Need at least two sensors to compute joint angles
            if len(data) < 2:
                print(f"[DEBUG] Only {len(data)} sensor(s) available. Need >=2 for joint angles. Waiting for more sensors...")
                return

            # --- Compute joint angles ---
            angles = compute_joint_angles(data)
            print(f"[DEBUG] Angles: {angles}")
            if not angles:
                return

            self.angle_window.append(angles)

            current_time = max(ts for _, _, _, ts in self.sensor_buffer.values()) if self.sensor_buffer else time.time()

            # --- Risk engine ---
            self.risk_engine.update(angles, current_time)
            risk = self.risk_engine.compute_risk()
            if risk and 'global' in risk:
                self.risk_window.append(risk['global'])

            # --- RULA & REBA ---
            trunk_angle = abs(data.get('UPPER_BACK', (0,0,0))[1]) if 'UPPER_BACK' in data else 0

            rula_right = None
            if all(k in angles for k in ('R_Shoulder', 'R_Elbow', 'R_Wrist')):
                rula_right = self.rula_engine.compute_side(
                    shoulder_flexion=angles['R_Shoulder'],
                    elbow_flexion=angles['R_Elbow'],
                    wrist_flexion=angles['R_Wrist'],
                    neck_flexion=angles.get('Neck', 0),
                    trunk_flexion=trunk_angle,
                    shoulder_abduction=angles.get('R_Shoulder_Abduction', 0),
                    wrist_deviation=angles.get('R_Wrist_Roll', 0),
                    wrist_pronation=angles.get('R_Wrist_Yaw', 0),
                    neck_lateral=angles.get('Neck_Roll', 0),
                    neck_rotation=angles.get('Neck_Yaw', 0),
                    trunk_lateral=angles.get('Trunk_Roll', 0),
                    trunk_rotation=angles.get('Trunk_Yaw', 0),
                    elbow_working_across=angles.get('R_Elbow_Roll', 0) > 15,
                )

            rula_left = None
            if all(k in angles for k in ('L_Shoulder', 'L_Elbow', 'L_Wrist')):
                rula_left = self.rula_engine.compute_side(
                    shoulder_flexion=angles['L_Shoulder'],
                    elbow_flexion=angles['L_Elbow'],
                    wrist_flexion=angles['L_Wrist'],
                    neck_flexion=angles.get('Neck', 0),
                    trunk_flexion=trunk_angle,
                    shoulder_abduction=angles.get('L_Shoulder_Abduction', 0),
                    wrist_deviation=angles.get('L_Wrist_Roll', 0),
                    wrist_pronation=angles.get('L_Wrist_Yaw', 0),
                    neck_lateral=angles.get('Neck_Roll', 0),
                    neck_rotation=angles.get('Neck_Yaw', 0),
                    trunk_lateral=angles.get('Trunk_Roll', 0),
                    trunk_rotation=angles.get('Trunk_Yaw', 0),
                    elbow_working_across=angles.get('L_Elbow_Roll', 0) > 15,
                )

            reba_right = None
            if all(k in angles for k in ('R_Shoulder', 'R_Elbow', 'R_Wrist')):
                reba_right = self.reba_engine.compute_side(
                    trunk_flexion=trunk_angle,
                    neck_flexion=angles.get('Neck', 0),
                    upper_arm_flexion=angles['R_Shoulder'],
                    forearm_flexion=angles['R_Elbow'],
                    wrist_flexion=angles['R_Wrist'],
                    trunk_lateral=angles.get('Trunk_Roll', 0),
                    trunk_rotation=angles.get('Trunk_Yaw', 0),
                    neck_lateral=angles.get('Neck_Roll', 0),
                    neck_rotation=angles.get('Neck_Yaw', 0),
                    upper_arm_abduction=angles.get('R_Shoulder_Abduction', 0),
                    forearm_lateral=angles.get('R_Elbow_Roll', 0),
                    wrist_deviation=angles.get('R_Wrist_Roll', 0),
                    wrist_pronation=angles.get('R_Wrist_Yaw', 0),
                )

            reba_left = None
            if all(k in angles for k in ('L_Shoulder', 'L_Elbow', 'L_Wrist')):
                reba_left = self.reba_engine.compute_side(
                    trunk_flexion=trunk_angle,
                    neck_flexion=angles.get('Neck', 0),
                    upper_arm_flexion=angles['L_Shoulder'],
                    forearm_flexion=angles['L_Elbow'],
                    wrist_flexion=angles['L_Wrist'],
                    trunk_lateral=angles.get('Trunk_Roll', 0),
                    trunk_rotation=angles.get('Trunk_Yaw', 0),
                    neck_lateral=angles.get('Neck_Roll', 0),
                    neck_rotation=angles.get('Neck_Yaw', 0),
                    upper_arm_abduction=angles.get('L_Shoulder_Abduction', 0),
                    forearm_lateral=angles.get('L_Elbow_Roll', 0),
                    wrist_deviation=angles.get('L_Wrist_Roll', 0),
                    wrist_pronation=angles.get('L_Wrist_Yaw', 0),
                )

            self.rula_l_window.append(rula_left['final'] if rula_left else 0.0)
            self.rula_r_window.append(rula_right['final'] if rula_right else 0.0)
            self.reba_l_window.append(reba_left['final'] if reba_left else 0.0)
            self.reba_r_window.append(reba_right['final'] if reba_right else 0.0)

            # --- AI predictions ---
            ai_pred = None
            windows_ready = (
                self.ai_models is not None
                and getattr(self.ai_models, 'ready', True)
                and self.feature_extractor is not None
                and len(self.angle_window) >= 60
                and len(self.risk_window)  >= 60
            )
            if windows_ready:
                try:
                    features_dict = self.feature_extractor.extract(
                        self.angle_window,
                        self.risk_window,
                        self.rula_l_window,
                        self.rula_r_window,
                        self.reba_l_window,
                        self.reba_r_window,
                        current_time
                    )
                
                    ai_pred = self.ai_models.predict(features_dict)
                    print(f"[AI] {ai_pred}")
                except Exception as e:
                    print(f"[AI ERROR] {e}")

            # --- CSV logging ---
            self.csv_logger.log(
                angles,
                rula={'right': rula_right, 'left': rula_left},
                reba={'right': reba_right, 'left': reba_left},
                features=features_dict if windows_ready else None,
                ai_pred=ai_pred
            )

            # --- Build WebSocket payload ---
            payload = {
                'angles':      angles,
                'risk':        risk,
                'rula':        {'right': rula_right, 'left': rula_left},
                'reba':        {'right': reba_right, 'left': reba_left},
                'trunk_angle': trunk_angle,
                'legs_score':  self.config.REBA_LEGS_SCORE,
            }
            if ai_pred is not None:
                payload['ai_predictions'] = ai_pred

            self.socketio.emit('angles', payload)
            print("[EMIT] angles payload sent.")

    def get_sensor_status(self):
        with self.lock:
            now    = time.time()
            cutoff = 2 * self.config.POST_INTERVAL_MS / 1000.0 + 1
            return [
                {
                    'sensor_id': sid,
                    'last_seen': self.last_seen.get(sid),
                    'online':    (now - self.last_seen[sid]) < cutoff
                                 if sid in self.last_seen else False,
                }
                for sid in self.config.EXPECTED_SENSORS
            ]

    def get_current_log_filename(self):
        with self.lock:
            if self.csv_logger and self.csv_logger.filename:
                return os.path.basename(self.csv_logger.filename)
            return None