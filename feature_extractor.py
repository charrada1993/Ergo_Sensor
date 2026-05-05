"""
feature_extractor.py — v2.0
Maps real-time angle_window → 38 features expected by the new LightGBM models.

Feature list (must match files/02_train_models.py FEATURES_BASIC):
  Angles aggregated (max L/R): neck trunk shoulder elbow wrist hip knee
  Bilateral angles: r_shoulder l_shoulder r_elbow l_elbow r_wrist l_wrist r_hip l_hip r_knee l_knee
  Velocities: *_vel  (abs diff of last 2 frames)
  Durations:  *_duration (frames in bad posture over last 30 frames)
  Frequencies: *_freq (velocity peaks over last 30 frames)
"""

import numpy as np
from collections import deque


# Bad-posture thresholds (must match 01_generate_dataset.py THRESHOLDS)
_THRESHOLDS = {
    'neck':     20,
    'trunk':    20,
    'shoulder': 40,
    'elbow':    80,
    'wrist':    15,
    'hip':      30,
    'knee':     50,
}


class FeatureExtractor:
    def __init__(self, metadata):
        self.feature_cols = metadata.get('feature_cols', [])

    # ─────────────────────────────────────────────────────────────
    def extract(self, angle_window, risk_window,
                rula_l_window, rula_r_window,
                reba_l_window, reba_r_window,
                current_time):
        """
        Build the 38-feature dict from the last N frames of the angle_window.

        angle_window  : deque of dicts like {'Neck': 18.5, 'R_Shoulder': 45.2, ...}
        All other windows are kept for backward compat but not used here.
        """

        # ── Helper: extract time-series for one joint name ─────────
        def series(key_candidates):
            """Return list[float] from angle_window, trying keys in order."""
            for k in key_candidates:
                vals = [f.get(k, None) for f in angle_window]
                vals = [v for v in vals if v is not None]
                if vals:
                    return vals
            return [0.0]

        # ── Latest frame bilateral values ─────────────────────────
        last = angle_window[-1] if angle_window else {}

        def lat(keys):
            for k in keys:
                v = last.get(k)
                if v is not None:
                    return abs(float(v))
            return 0.0

        # Map sensor angle names → bilateral features
        r_shoulder = lat(['R_Shoulder', 'R_Shoulder_Flexion'])
        l_shoulder = lat(['L_Shoulder', 'L_Shoulder_Flexion'])
        r_elbow    = lat(['R_Elbow',    'R_Elbow_Flexion'])
        l_elbow    = lat(['L_Elbow',    'L_Elbow_Flexion'])
        r_wrist    = lat(['R_Wrist',    'R_Wrist_Deviation', 'R_Wrist_Flexion'])
        l_wrist    = lat(['L_Wrist',    'L_Wrist_Deviation', 'L_Wrist_Flexion'])
        r_hip      = lat(['R_Hip',      'R_Hip_Flexion'])
        l_hip      = lat(['L_Hip',      'L_Hip_Flexion'])
        r_knee     = lat(['R_Knee',     'R_Knee_Flexion'])
        l_knee     = lat(['L_Knee',     'L_Knee_Flexion'])
        neck       = lat(['Neck',       'Neck_Flexion'])
        trunk      = lat(['Trunk',      'Trunk_Flexion'])

        # Aggregated (max bilateral)
        shoulder = max(r_shoulder, l_shoulder)
        elbow    = max(r_elbow,    l_elbow)
        wrist    = max(r_wrist,    l_wrist)
        hip      = max(r_hip,      l_hip)
        knee     = max(r_knee,     l_knee)

        # Joint dict for velocity / duration / freq computation
        joint_current = {
            'neck':     neck,
            'trunk':    trunk,
            'shoulder': shoulder,
            'elbow':    elbow,
            'wrist':    wrist,
            'hip':      hip,
            'knee':     knee,
        }

        # ── Time-series per joint for vel/dur/freq ─────────────────
        WINDOW = 30  # frames

        def joint_series(joint):
            """Pull the right series from angle_window for each joint."""
            lookup = {
                'neck':     ['Neck',      'Neck_Flexion'],
                'trunk':    ['Trunk',     'Trunk_Flexion'],
                'shoulder': ['R_Shoulder','L_Shoulder'],
                'elbow':    ['R_Elbow',   'L_Elbow'],
                'wrist':    ['R_Wrist',   'L_Wrist'],
                'hip':      ['R_Hip',     'L_Hip'],
                'knee':     ['R_Knee',    'L_Knee'],
            }
            keys = lookup.get(joint, [joint])
            vals = []
            for f in angle_window:
                v = None
                for k in keys:
                    if k in f:
                        v = abs(float(f[k]))
                        break
                vals.append(v if v is not None else 0.0)
            return vals

        features = {}

        # ── Aggregated angles ──────────────────────────────────────
        features['neck']     = neck
        features['trunk']    = trunk
        features['shoulder'] = shoulder
        features['elbow']    = elbow
        features['wrist']    = wrist
        features['hip']      = hip
        features['knee']     = knee

        # ── Bilateral angles ───────────────────────────────────────
        features['r_shoulder'] = r_shoulder
        features['l_shoulder'] = l_shoulder
        features['r_elbow']    = r_elbow
        features['l_elbow']    = l_elbow
        features['r_wrist']    = r_wrist
        features['l_wrist']    = l_wrist
        features['r_hip']      = r_hip
        features['l_hip']      = l_hip
        features['r_knee']     = r_knee
        features['l_knee']     = l_knee

        # ── Velocities, durations, frequencies ────────────────────
        for j in ['neck', 'trunk', 'shoulder', 'elbow', 'wrist', 'hip', 'knee']:
            s = joint_series(j)
            thr = _THRESHOLDS.get(j, 20)

            # velocity: abs diff between last two frames
            if len(s) >= 2:
                vel = abs(s[-1] - s[-2])
            else:
                vel = 0.0
            features[f'{j}_vel'] = vel

            # duration: # frames > threshold in last WINDOW frames
            recent = s[-WINDOW:] if len(s) >= WINDOW else s
            features[f'{j}_duration'] = float(sum(1 for v in recent if v > thr))

            # frequency: # velocity peaks > 75th percentile in last WINDOW
            if len(recent) >= 2:
                vels   = [abs(recent[i] - recent[i-1]) for i in range(1, len(recent))]
                q75    = float(np.percentile(vels, 75)) if vels else 0.0
                freq   = sum(1 for v in vels if v > q75)
            else:
                freq = 0
            features[f'{j}_freq'] = float(freq)

        # ── Also pass global risk score for convenience ────────────
        features['global_risk_score'] = list(risk_window)[-1] if risk_window else 0.0

        # ── Fill missing expected features with 0 ─────────────────
        for col in self.feature_cols:
            if col not in features:
                features[col] = 0.0

        return features
