import time
from collections import deque
import numpy as np

class RiskEngine:
    def __init__(self, config):
        self.config = config
        # Store recent angles for risk calculations (e.g., last 60 seconds)
        self.angle_history = deque(maxlen=600)  # 600 samples = 60s at 10Hz
        self.start_time = time.time()

    def update(self, angles, timestamp):
        """Add new angles to history."""
        self.angle_history.append((timestamp, angles))

    def compute_risk(self):
        """Compute R1-R5 and global risk based on current history."""
        if len(self.angle_history) < 10:
            return None  # Not enough data

        # Extract angles into numpy arrays for each joint over time
        # For simplicity, we use the most recent angles for some components
        latest = self.angle_history[-1][1]

        # R1: Static posture risk (example: based on neck angle)
        # Assume static if angle variation < threshold over last minute
        neck_angles = [a['Neck'] for ts, a in self.angle_history if 'Neck' in a]
        if neck_angles:
            neck_std = np.std(neck_angles[-60:])  # last 60 samples (6s at 10Hz)
            static = neck_std < 2.0  # threshold 2 degrees
        else:
            static = False

        if static:
            # CWD = time spent static (cumulative)
            # Here we approximate CWD as time since start if static
            cwd = time.time() - self.start_time
        else:
            cwd = 0
        R1 = cwd / self.config.CWD if self.config.CWD > 0 else 0
        R1 = min(R1, 1.0)

        # R2: Repetition risk (example: based on right elbow angle changes)
        if 'R_Elbow' in latest:
            # Count peaks in elbow angle over last minute
            elbow_angles = [a['R_Elbow'] for ts, a in self.angle_history if 'R_Elbow' in a]
            # Simple peak count (placeholder)
            cf = len(elbow_angles) / 60  # frequency per second, convert to per minute
            cf *= 60  # per minute
            R2 = (cf - self.config.PF) / (self.config.MF - self.config.PF)
            R2 = np.clip(R2, 0, 1)
        else:
            R2 = 0

        # R3: Postural angle risk (example: peak neck angle)
        if 'Neck' in latest:
            neck_angle = latest['Neck']
            R3 = (neck_angle - self.config.MDR) / (self.config.E - self.config.MDR)
            R3 = np.clip(R3, 0, 1)
        else:
            R3 = 0

        # R4: Effort risk (from Borg scale, here we simulate from muscle activity proxy)
        # For now, use a placeholder based on angle magnitude
        if 'R_Elbow' in latest:
            effort = latest['R_Elbow'] / 90.0  # normalize
            R4 = (effort - self.config.PWL/10) / (self.config.MVC/10 - self.config.PWL/10)
            R4 = np.clip(R4, 0, 1)
        else:
            R4 = 0

        # R5: Recovery risk (time since last high exertion)
        # Simplified: if current effort low, recovery increases
        if R4 < 0.3:
            # recovery time accumulating
            # actual implementation would track periods
            ar = 60  # placeholder
        else:
            ar = 0
        R5 = (self.config.RR - ar) / self.config.RR if self.config.RR > 0 else 0
        R5 = np.clip(R5, 0, 1)

        # Global risk
        weights = self.config.RISK_WEIGHTS
        R_global = weights[0]*R1 + weights[1]*R2 + weights[2]*R3 + weights[3]*R4 + weights[4]*R5

        # Classification
        if R_global <= 0.2:
            level = "Low"
        elif R_global <= 0.4:
            level = "Moderate"
        elif R_global <= 0.6:
            level = "High"
        else:
            level = "Very High"

        return {
            'R1': R1,
            'R2': R2,
            'R3': R3,
            'R4': R4,
            'R5': R5,
            'global': R_global,
            'level': level
        }