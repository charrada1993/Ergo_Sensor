"""
REBA (Rapid Entire Body Assessment) calculation module.
Scores computed per the REBA PDF specification.

Group A  →  Trunk + Neck + Legs          →  Table A score  (+load)  →  Score A
Group B  →  Upper Arm + Forearm + Wrist  →  Table B score  (+coupling) →  Score B
Final    →  Table C lookup (Score A × Score B)  + activity
"""


class REBAEngine:
    def __init__(self, config):
        self.config          = config
        self.load_score      = getattr(config, 'REBA_LOAD_SCORE',     0)  # 0,1,2
        self.coupling_score  = getattr(config, 'REBA_COUPLING_SCORE', 0)  # 0,1,2,3
        self.activity_score  = getattr(config, 'REBA_ACTIVITY_SCORE', 0)  # 0,1
        self.legs_score      = getattr(config, 'REBA_LEGS_SCORE',     1)  # 1-4

    # ──────────────────────────────────────────────────────────────────────────
    # Group A: Trunk, Neck, Legs
    # ──────────────────────────────────────────────────────────────────────────

    def trunk_score(self, flexion, lateral=0.0, rotation=0.0):
        """
        Trunk flexion score (Group A).
        flexion: trunk pitch (raw UPPER_BACK sensor, calibrated). Degrees.
        lateral: trunk roll, degrees.
        rotation: trunk yaw, degrees.
        """
        if flexion < 0:           # extension
            s = 2
        elif flexion <= 20:
            s = 2
        elif flexion <= 60:
            s = 3
        else:
            s = 4
        if abs(lateral) > 15 or abs(rotation) > 15:
            s += 1
        return s

    def neck_score(self, flexion, lateral=0.0, rotation=0.0):
        """
        Neck flexion score (Group A).
        flexion: neck pitch diff (head − trunk). Degrees.
        """
        if flexion <= 20:
            s = 1
        else:
            s = 2
        if abs(lateral) > 15 or abs(rotation) > 15:
            s += 1
        return s

    # ──────────────────────────────────────────────────────────────────────────
    # Group B: Upper Arm, Forearm, Wrist
    # ──────────────────────────────────────────────────────────────────────────

    def upper_arm_score(self, flexion, abduction=0.0, shoulder_raised=False, leaning=False):
        """
        Upper arm / shoulder score (Group B).
        flexion: shoulder pitch diff (arm − trunk). Degrees.
        abduction: shoulder roll diff (arm − trunk). Degrees.
        """
        a = abs(flexion)
        if flexion < 0:          # extension
            s = 2
        elif a <= 20:
            s = 1
        elif a <= 45:
            s = 2
        elif a <= 90:
            s = 3
        else:
            s = 4
        if abs(abduction) > 15:
            s += 1
        if shoulder_raised:
            s += 1
        if leaning:
            s -= 1
        return max(s, 1)

    def forearm_score(self, flexion, lateral_deviation=0.0):
        """
        Forearm / elbow score (Group B).
        flexion: absolute elbow internal angle (|forearm pitch − biceps pitch|). Degrees.
        lateral_deviation: |forearm roll − biceps roll|. Degrees.
        """
        if 60 <= flexion <= 100:
            s = 1
        else:
            s = 2
        if abs(lateral_deviation) > 15:
            s += 1
        return s

    def wrist_score(self, flexion, deviation=0.0, pronation=0.0):
        """
        Wrist score (Group B).
        flexion: wrist pitch diff (hand − forearm). Degrees.
        deviation: wrist roll diff (hand − forearm). Degrees.
        pronation: wrist yaw diff (hand − forearm). Degrees.
        """
        a = abs(flexion)
        if a <= 15:
            s = 1
        else:
            s = 2
        if abs(deviation) > 15 or abs(pronation) > 15:
            s += 1
        return s

    # ──────────────────────────────────────────────────────────────────────────
    # REBA lookup tables
    # ──────────────────────────────────────────────────────────────────────────

    # Table A:  [neck 1-3][trunk 1-5][legs 1-4]
    _table_a = [
        # neck = 1
        [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6], [3, 5, 6, 7], [4, 6, 7, 8]],
        # neck = 2
        [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
        # neck = 3
        [[3, 3, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],
    ]

    # Table B:  [upper_arm 1-6][forearm 1-2][wrist 1-3]
    _table_b = [
        [[1, 2, 2], [1, 2, 3]],
        [[1, 2, 3], [2, 3, 4]],
        [[3, 4, 5], [4, 5, 5]],
        [[4, 5, 5], [5, 6, 7]],
        [[6, 7, 8], [7, 8, 8]],
        [[7, 8, 8], [8, 9, 9]],
    ]

    # Table C:  [score_a 1-12][score_b 1-12]
    _table_c = [
        [1,  1,  1,  2,  3,  3,  4,  5,  6,  7,  7,  7],
        [1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  7,  8],
        [2,  3,  3,  3,  4,  5,  6,  7,  7,  8,  8,  8],
        [3,  4,  4,  4,  5,  6,  7,  8,  8,  9,  9,  9],
        [4,  4,  4,  5,  6,  7,  8,  8,  9,  9,  9,  9],
        [6,  6,  6,  7,  8,  8,  9,  9, 10, 10, 10, 10],
        [7,  7,  7,  8,  9,  9,  9, 10, 10, 11, 11, 11],
        [8,  8,  8,  9, 10, 10, 10, 10, 10, 11, 11, 11],
        [9,  9,  9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
        [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
        [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
        [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    ]

    # ──────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────────

    def compute_side(self,
                     trunk_flexion,
                     neck_flexion,
                     upper_arm_flexion,
                     forearm_flexion,
                     wrist_flexion,
                     trunk_lateral=0.0,
                     trunk_rotation=0.0,
                     neck_lateral=0.0,
                     neck_rotation=0.0,
                     upper_arm_abduction=0.0,
                     shoulder_raised=False,
                     arm_leaning=False,
                     forearm_lateral=0.0,
                     wrist_deviation=0.0,
                     wrist_pronation=0.0,
                     legs_score=None,
                     load_score=None,
                     coupling_score=None,
                     activity_score=None):
        """
        Compute REBA score for one side.

        Parameters
        ----------
        trunk_flexion     : trunk pitch (raw UPPER_BACK), degrees
        neck_flexion      : neck pitch diff (head − trunk), degrees
        upper_arm_flexion : shoulder pitch diff (arm − trunk), degrees
        forearm_flexion   : abs elbow internal angle (|forearm − biceps pitch|), degrees
        wrist_flexion     : wrist pitch diff (hand − forearm), degrees
        trunk_lateral     : trunk roll, degrees
        trunk_rotation    : trunk yaw, degrees
        neck_lateral      : neck roll diff, degrees
        neck_rotation     : neck yaw diff, degrees
        upper_arm_abduction: shoulder roll diff (arm − trunk), degrees
        shoulder_raised   : bool
        arm_leaning       : bool
        forearm_lateral   : abs(forearm roll − biceps roll), degrees
        wrist_deviation   : wrist roll diff (hand − forearm), degrees
        wrist_pronation   : wrist yaw diff (hand − forearm), degrees
        legs_score        : 1-4  (1=standing bilateral, 2=unilateral, 3=knees 30-60°, 4=knees>60°)
        load_score        : 0 (<5 kg), 1 (5-10 kg), 2 (>10 kg or shock)
        coupling_score    : 0 (good), 1 (fair), 2 (poor), 3 (unacceptable)
        activity_score    : 0 or 1

        Returns
        -------
        dict with all intermediate and final scores.
        """
        load     = load_score     if load_score     is not None else self.load_score
        coupling = coupling_score if coupling_score is not None else self.coupling_score
        activity = activity_score if activity_score is not None else self.activity_score
        legs     = legs_score     if legs_score     is not None else self.legs_score

        # ── Group A ───────────────────────────────────────────────────────────
        t_raw = self.trunk_score(trunk_flexion, trunk_lateral, trunk_rotation)
        n_raw = self.neck_score(neck_flexion,   neck_lateral,  neck_rotation)
        l     = max(1, min(legs, 4))

        idx_n = min(max(n_raw, 1), 3) - 1
        idx_t = min(max(t_raw, 1), 5) - 1
        idx_l = l - 1

        table_a_val = self._table_a[idx_n][idx_t][idx_l]
        score_a     = table_a_val + load

        # ── Group B ───────────────────────────────────────────────────────────
        ua_raw = self.upper_arm_score(upper_arm_flexion, upper_arm_abduction,
                                      shoulder_raised, arm_leaning)
        fa_raw = self.forearm_score(forearm_flexion, forearm_lateral)
        w_raw  = self.wrist_score(wrist_flexion, wrist_deviation, wrist_pronation)

        idx_ua = min(max(ua_raw, 1), 6) - 1
        idx_fa = min(max(fa_raw, 1), 2) - 1
        idx_w  = min(max(w_raw,  1), 3) - 1

        table_b_val = self._table_b[idx_ua][idx_fa][idx_w]
        score_b     = table_b_val + coupling

        # ── Table C (Final) ───────────────────────────────────────────────────
        idx_sa = min(max(score_a, 1), 12) - 1
        idx_sb = min(max(score_b, 1), 12) - 1
        score_c = self._table_c[idx_sa][idx_sb]
        final   = score_c + activity

        # ── Action level ──────────────────────────────────────────────────────
        if final <= 1:
            action = "Negligible"
        elif final <= 3:
            action = "Low"
        elif final <= 7:
            action = "Medium"
        elif final <= 10:
            action = "High"
        else:
            action = "Very High"

        return {
            # Raw sub-scores
            'trunk_score':      t_raw,
            'neck_score':       n_raw,
            'legs_score':       legs,
            'upper_arm_score':  ua_raw,
            'forearm_score':    fa_raw,
            'wrist_score':      w_raw,
            # Intermediate totals
            'table_a':  table_a_val,
            'score_a':  score_a,
            'table_b':  table_b_val,
            'score_b':  score_b,
            'score_c':  score_c,
            # Adjustments
            'load':     load,
            'coupling': coupling,
            'activity': activity,
            # Input angles (for CSV logging)
            'trunk_flexion':       round(trunk_flexion,     2),
            'trunk_lateral':       round(trunk_lateral,     2),
            'trunk_rotation':      round(trunk_rotation,    2),
            'neck_flexion':        round(neck_flexion,      2),
            'neck_lateral':        round(neck_lateral,      2),
            'neck_rotation':       round(neck_rotation,     2),
            'upper_arm_flexion':   round(upper_arm_flexion, 2),
            'upper_arm_abduction': round(upper_arm_abduction, 2),
            'forearm_flexion':     round(forearm_flexion,   2),
            'forearm_lateral':     round(forearm_lateral,   2),
            'wrist_flexion':       round(wrist_flexion,     2),
            'wrist_deviation':     round(wrist_deviation,   2),
            'wrist_pronation':     round(wrist_pronation,   2),
            # Final
            'final':    final,
            'action':   action,
        }