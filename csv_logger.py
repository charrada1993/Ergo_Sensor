import csv
import os
from datetime import datetime


class CSVLogger:
    def __init__(self, config):
        self.config   = config
        self.filename = None
        self.file     = None
        self.writer   = None
        self._open_file()

    def _open_file(self):
        os.makedirs(self.config.CSV_DIR, exist_ok=True)
        timestamp     = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.filename = os.path.join(self.config.CSV_DIR, f'session_{timestamp}.csv')
        self.file     = open(self.filename, 'w', newline='', encoding='utf-8')
        self.writer   = csv.writer(self.file)

        # ── Full header ───────────────────────────────────────────────────────
        header = [
            'Timestamp',

            # ── Raw joint angles ──────────────────────────────────────────────
            'Neck_Flexion_deg',
            'Neck_Lateral_deg',
            'Neck_Rotation_deg',
            'Trunk_Flexion_deg',
            'Trunk_Lateral_deg',
            'Trunk_Rotation_deg',
            'R_Shoulder_Flexion_deg',
            'R_Shoulder_Abduction_deg',
            'L_Shoulder_Flexion_deg',
            'L_Shoulder_Abduction_deg',
            'R_Elbow_Flexion_deg',
            'R_Elbow_Lateral_deg',
            'L_Elbow_Flexion_deg',
            'L_Elbow_Lateral_deg',
            'R_Wrist_Flexion_deg',
            'R_Wrist_Deviation_deg',
            'R_Wrist_Pronation_deg',
            'L_Wrist_Flexion_deg',
            'L_Wrist_Deviation_deg',
            'L_Wrist_Pronation_deg',
            'R_Thigh_Flexion_deg',
            'L_Thigh_Flexion_deg',
            'R_Knee_Flexion_deg',
            'L_Knee_Flexion_deg',

            # ── RULA Right ────────────────────────────────────────────────────
            'RULA_R_Shoulder_Flexion_deg',
            'RULA_R_Shoulder_Abduction_deg',
            'RULA_R_Upper_Arm_Score',
            'RULA_R_Elbow_Flexion_deg',
            'RULA_R_Elbow_Working_Across',
            'RULA_R_Forearm_Score',
            'RULA_R_Wrist_Flexion_deg',
            'RULA_R_Wrist_Deviation_deg',
            'RULA_R_Wrist_Pronation_deg',
            'RULA_R_Wrist_Score',
            'RULA_R_Wrist_Twist',
            'RULA_R_Score_A',
            'RULA_R_Neck_Flexion_deg',
            'RULA_R_Neck_Lateral_deg',
            'RULA_R_Neck_Rotation_deg',
            'RULA_R_Neck_Score',
            'RULA_R_Trunk_Flexion_deg',
            'RULA_R_Trunk_Lateral_deg',
            'RULA_R_Trunk_Rotation_deg',
            'RULA_R_Trunk_Score',
            'RULA_R_Legs_Score',
            'RULA_R_Score_B',
            'RULA_R_Muscle_Score',
            'RULA_R_Load_Score',
            'RULA_R_Score_C',
            'RULA_R_Score_D',
            'RULA_R_Final',
            'RULA_R_Action',

            # ── RULA Left ─────────────────────────────────────────────────────
            'RULA_L_Shoulder_Flexion_deg',
            'RULA_L_Shoulder_Abduction_deg',
            'RULA_L_Upper_Arm_Score',
            'RULA_L_Elbow_Flexion_deg',
            'RULA_L_Elbow_Working_Across',
            'RULA_L_Forearm_Score',
            'RULA_L_Wrist_Flexion_deg',
            'RULA_L_Wrist_Deviation_deg',
            'RULA_L_Wrist_Pronation_deg',
            'RULA_L_Wrist_Score',
            'RULA_L_Wrist_Twist',
            'RULA_L_Score_A',
            'RULA_L_Neck_Flexion_deg',
            'RULA_L_Neck_Lateral_deg',
            'RULA_L_Neck_Rotation_deg',
            'RULA_L_Neck_Score',
            'RULA_L_Trunk_Flexion_deg',
            'RULA_L_Trunk_Lateral_deg',
            'RULA_L_Trunk_Rotation_deg',
            'RULA_L_Trunk_Score',
            'RULA_L_Legs_Score',
            'RULA_L_Score_B',
            'RULA_L_Muscle_Score',
            'RULA_L_Load_Score',
            'RULA_L_Score_C',
            'RULA_L_Score_D',
            'RULA_L_Final',
            'RULA_L_Action',

            # ── REBA Right ────────────────────────────────────────────────────
            'REBA_R_Trunk_Flexion_deg',
            'REBA_R_Trunk_Lateral_deg',
            'REBA_R_Trunk_Rotation_deg',
            'REBA_R_Trunk_Score',
            'REBA_R_Neck_Flexion_deg',
            'REBA_R_Neck_Lateral_deg',
            'REBA_R_Neck_Rotation_deg',
            'REBA_R_Neck_Score',
            'REBA_R_Legs_Score',
            'REBA_R_Table_A',
            'REBA_R_Load_Score',
            'REBA_R_Score_A',
            'REBA_R_Upper_Arm_Flexion_deg',
            'REBA_R_Upper_Arm_Abduction_deg',
            'REBA_R_Upper_Arm_Score',
            'REBA_R_Forearm_Flexion_deg',
            'REBA_R_Forearm_Lateral_deg',
            'REBA_R_Forearm_Score',
            'REBA_R_Wrist_Flexion_deg',
            'REBA_R_Wrist_Deviation_deg',
            'REBA_R_Wrist_Pronation_deg',
            'REBA_R_Wrist_Score',
            'REBA_R_Table_B',
            'REBA_R_Coupling_Score',
            'REBA_R_Score_B',
            'REBA_R_Score_C',
            'REBA_R_Activity_Score',
            'REBA_R_Final',
            'REBA_R_Action',

            # ── REBA Left ─────────────────────────────────────────────────────
            'REBA_L_Trunk_Flexion_deg',
            'REBA_L_Trunk_Lateral_deg',
            'REBA_L_Trunk_Rotation_deg',
            'REBA_L_Trunk_Score',
            'REBA_L_Neck_Flexion_deg',
            'REBA_L_Neck_Lateral_deg',
            'REBA_L_Neck_Rotation_deg',
            'REBA_L_Neck_Score',
            'REBA_L_Legs_Score',
            'REBA_L_Table_A',
            'REBA_L_Load_Score',
            'REBA_L_Score_A',
            'REBA_L_Upper_Arm_Flexion_deg',
            'REBA_L_Upper_Arm_Abduction_deg',
            'REBA_L_Upper_Arm_Score',
            'REBA_L_Forearm_Flexion_deg',
            'REBA_L_Forearm_Lateral_deg',
            'REBA_L_Forearm_Score',
            'REBA_L_Wrist_Flexion_deg',
            'REBA_L_Wrist_Deviation_deg',
            'REBA_L_Wrist_Pronation_deg',
            'REBA_L_Wrist_Score',
            'REBA_L_Table_B',
            'REBA_L_Coupling_Score',
            'REBA_L_Score_B',
            'REBA_L_Score_C',
            'REBA_L_Activity_Score',
            'REBA_L_Final',
            'REBA_L_Action',
        ]

        self.writer.writerow(header)
        self.file.flush()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _g(d, key, default=''):
        """Safe getter from dict; returns '' if None or missing."""
        if d is None:
            return default
        v = d.get(key)
        return default if v is None else v

    @staticmethod
    def _r(v, ndigits=4):
        """Round if numeric, else return as-is."""
        try:
            return round(float(v), ndigits)
        except (TypeError, ValueError):
            return v

    # ── Main log method ───────────────────────────────────────────────────────

    def log(self, angles, rula=None, reba=None):
        """
        Write one row.

        angles : dict of joint angles from angle_math.compute_joint_angles()
        rula   : {'right': dict|None, 'left': dict|None}
        reba   : {'right': dict|None, 'left': dict|None}
        """
        if not self.writer:
            return

        g  = self._g
        r  = self._r
        rr = rula.get('right') if rula else None
        rl = rula.get('left')  if rula else None
        br = reba.get('right') if reba else None
        bl = reba.get('left')  if reba else None

        row = [
            datetime.now().isoformat(),

            # ── Raw joint angles ──────────────────────────────────────────────
            r(g(angles, 'Neck')),
            r(g(angles, 'Neck_Roll')),
            r(g(angles, 'Neck_Yaw')),
            r(g(angles, 'Trunk_Pitch')),
            r(g(angles, 'Trunk_Roll')),
            r(g(angles, 'Trunk_Yaw')),
            r(g(angles, 'R_Shoulder')),
            r(g(angles, 'R_Shoulder_Abduction')),
            r(g(angles, 'L_Shoulder')),
            r(g(angles, 'L_Shoulder_Abduction')),
            r(g(angles, 'R_Elbow')),
            r(g(angles, 'R_Elbow_Roll')),
            r(g(angles, 'L_Elbow')),
            r(g(angles, 'L_Elbow_Roll')),
            r(g(angles, 'R_Wrist')),
            r(g(angles, 'R_Wrist_Roll')),
            r(g(angles, 'R_Wrist_Yaw')),
            r(g(angles, 'L_Wrist')),
            r(g(angles, 'L_Wrist_Roll')),
            r(g(angles, 'L_Wrist_Yaw')),
            r(g(angles, 'R_Thigh')),
            r(g(angles, 'L_Thigh')),
            r(g(angles, 'R_Knee')),
            r(g(angles, 'L_Knee')),

            # ── RULA Right ────────────────────────────────────────────────────
            r(g(rr, 'shoulder_flexion')),
            r(g(rr, 'shoulder_abduction')),
            g(rr, 'upper_arm_score'),
            r(g(rr, 'elbow_flexion')),
            g(rr, 'elbow_working_across'),
            g(rr, 'forearm_score'),
            r(g(rr, 'wrist_flexion')),
            r(g(rr, 'wrist_deviation')),
            r(g(rr, 'wrist_pronation')),
            g(rr, 'wrist_score'),
            g(rr, 'wrist_twist'),
            g(rr, 'score_a'),
            r(g(rr, 'neck_flexion')),
            r(g(rr, 'neck_lateral')),
            r(g(rr, 'neck_rotation')),
            g(rr, 'neck_score'),
            r(g(rr, 'trunk_flexion')),
            r(g(rr, 'trunk_lateral')),
            r(g(rr, 'trunk_rotation')),
            g(rr, 'trunk_score'),
            g(rr, 'legs_score'),
            g(rr, 'score_b'),
            g(rr, 'muscle'),
            g(rr, 'load'),
            g(rr, 'score_c'),
            g(rr, 'score_d'),
            g(rr, 'final'),
            g(rr, 'action'),

            # ── RULA Left ─────────────────────────────────────────────────────
            r(g(rl, 'shoulder_flexion')),
            r(g(rl, 'shoulder_abduction')),
            g(rl, 'upper_arm_score'),
            r(g(rl, 'elbow_flexion')),
            g(rl, 'elbow_working_across'),
            g(rl, 'forearm_score'),
            r(g(rl, 'wrist_flexion')),
            r(g(rl, 'wrist_deviation')),
            r(g(rl, 'wrist_pronation')),
            g(rl, 'wrist_score'),
            g(rl, 'wrist_twist'),
            g(rl, 'score_a'),
            r(g(rl, 'neck_flexion')),
            r(g(rl, 'neck_lateral')),
            r(g(rl, 'neck_rotation')),
            g(rl, 'neck_score'),
            r(g(rl, 'trunk_flexion')),
            r(g(rl, 'trunk_lateral')),
            r(g(rl, 'trunk_rotation')),
            g(rl, 'trunk_score'),
            g(rl, 'legs_score'),
            g(rl, 'score_b'),
            g(rl, 'muscle'),
            g(rl, 'load'),
            g(rl, 'score_c'),
            g(rl, 'score_d'),
            g(rl, 'final'),
            g(rl, 'action'),

            # ── REBA Right ────────────────────────────────────────────────────
            r(g(br, 'trunk_flexion')),
            r(g(br, 'trunk_lateral')),
            r(g(br, 'trunk_rotation')),
            g(br, 'trunk_score'),
            r(g(br, 'neck_flexion')),
            r(g(br, 'neck_lateral')),
            r(g(br, 'neck_rotation')),
            g(br, 'neck_score'),
            g(br, 'legs_score'),
            g(br, 'table_a'),
            g(br, 'load'),
            g(br, 'score_a'),
            r(g(br, 'upper_arm_flexion')),
            r(g(br, 'upper_arm_abduction')),
            g(br, 'upper_arm_score'),
            r(g(br, 'forearm_flexion')),
            r(g(br, 'forearm_lateral')),
            g(br, 'forearm_score'),
            r(g(br, 'wrist_flexion')),
            r(g(br, 'wrist_deviation')),
            r(g(br, 'wrist_pronation')),
            g(br, 'wrist_score'),
            g(br, 'table_b'),
            g(br, 'coupling'),
            g(br, 'score_b'),
            g(br, 'score_c'),
            g(br, 'activity'),
            g(br, 'final'),
            g(br, 'action'),

            # ── REBA Left ─────────────────────────────────────────────────────
            r(g(bl, 'trunk_flexion')),
            r(g(bl, 'trunk_lateral')),
            r(g(bl, 'trunk_rotation')),
            g(bl, 'trunk_score'),
            r(g(bl, 'neck_flexion')),
            r(g(bl, 'neck_lateral')),
            r(g(bl, 'neck_rotation')),
            g(bl, 'neck_score'),
            g(bl, 'legs_score'),
            g(bl, 'table_a'),
            g(bl, 'load'),
            g(bl, 'score_a'),
            r(g(bl, 'upper_arm_flexion')),
            r(g(bl, 'upper_arm_abduction')),
            g(bl, 'upper_arm_score'),
            r(g(bl, 'forearm_flexion')),
            r(g(bl, 'forearm_lateral')),
            g(bl, 'forearm_score'),
            r(g(bl, 'wrist_flexion')),
            r(g(bl, 'wrist_deviation')),
            r(g(bl, 'wrist_pronation')),
            g(bl, 'wrist_score'),
            g(bl, 'table_b'),
            g(bl, 'coupling'),
            g(bl, 'score_b'),
            g(bl, 'score_c'),
            g(bl, 'activity'),
            g(bl, 'final'),
            g(bl, 'action'),
        ]

        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()