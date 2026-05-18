"""
RULA (Rapid Upper Limb Assessment) calculation module.
Scores computed per the RULA PDF specification.

Group A  →  Upper Arm + Forearm + Wrist  →  Score A  (+muscle +load)  →  Score C
Group B  →  Neck + Trunk + Legs          →  Score B  (+muscle +load)  →  Score D
Final    →  Table C lookup (Score C × Score D)
"""


class RULAEngine:
    def __init__(self, config):
        self.config       = config
        self.legs_score   = getattr(config, 'RULA_LEGS_SCORE',   1)   # 1=supported, 2=not

    # ──────────────────────────────────────────────────────────────────────────
    # Group A component scores
    # ──────────────────────────────────────────────────────────────────────────

    def upper_arm_score(self, flexion, abduction=0.0, shoulder_raised=False, leaning=False):
        """
        Shoulder flexion score (Group A).
        flexion: shoulder flexion angle (degrees). Negative = extension.
        abduction: shoulder abduction angle (degrees).
        """
        a = abs(flexion)
        if flexion < 0:          # any extension
            s = 2
        elif a < 20:
            s = 1
        elif a < 45:
            s = 2
        elif a < 90:
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

    def forearm_score(self, flexion, working_across=False):
        """
        Elbow flexion score (Group A).
        flexion: elbow flexion angle (degrees, always positive internal angle).
        working_across: forearm working across midline or out to side.
        """
        if 60 <= flexion <= 100:
            s = 1
        else:
            s = 2
        if working_across:
            s += 1
        return s

    def wrist_score(self, flexion, deviation=0.0):
        """
        Wrist flexion/extension score (Group A).
        flexion: wrist flexion angle in degrees (negative = extension).
        deviation: radial/ulnar deviation (roll difference) in degrees.
        """
        a = abs(flexion)
        if a < 15:
            s = 1
        elif a < 30:
            s = 2
        else:
            s = 3
        if abs(deviation) > 15:
            s += 1
        return s

    # ──────────────────────────────────────────────────────────────────────────
    # Group B component scores
    # ──────────────────────────────────────────────────────────────────────────

    def neck_score(self, flexion, lateral=0.0, rotation=0.0):
        """
        Neck flexion score (Group B).
        flexion: neck pitch difference (head pitch − trunk pitch).
        """
        if flexion < 0:
            s = 4   # extension
        elif flexion < 10:
            s = 1
        elif flexion < 20:
            s = 2
        else:
            s = 3
        if abs(lateral) > 15 or abs(rotation) > 15:
            s += 1
        return s

    def trunk_score(self, flexion, lateral=0.0, rotation=0.0):
        """
        Trunk flexion score (Group B).
        flexion: trunk pitch (raw UPPER_BACK pitch).
        """
        if flexion < 0:
            s = 1   # slight extension
        elif flexion < 10:
            s = 1
        elif flexion < 20:
            s = 2
        elif flexion < 60:
            s = 3
        else:
            s = 4
        if abs(lateral) > 15 or abs(rotation) > 15:
            s += 1
        return s

    # ──────────────────────────────────────────────────────────────────────────
    # RULA lookup tables
    # ──────────────────────────────────────────────────────────────────────────

    # Table A:  [upper_arm 1-6][forearm 1-3][wrist 1-4][wrist_twist 1-2]
    _table_a = [
        # Upper Arm = 1
        [[[1, 2], [2, 2], [2, 3], [3, 3]],
         [[2, 2], [2, 2], [3, 3], [3, 3]],
         [[2, 3], [3, 3], [3, 3], [4, 4]]],
        # Upper Arm = 2
        [[[2, 2], [2, 2], [3, 3], [3, 3]],
         [[3, 3], [3, 3], [3, 3], [4, 4]],
         [[3, 4], [4, 4], [4, 4], [5, 5]]],
        # Upper Arm = 3
        [[[3, 3], [3, 3], [4, 4], [4, 4]],
         [[4, 4], [4, 4], [4, 4], [5, 5]],
         [[4, 4], [4, 4], [4, 5], [5, 5]]],
        # Upper Arm = 4
        [[[4, 4], [4, 4], [4, 4], [5, 5]],
         [[5, 5], [5, 5], [5, 5], [6, 6]],
         [[5, 5], [5, 5], [5, 6], [6, 6]]],
        # Upper Arm = 5
        [[[5, 5], [5, 5], [5, 5], [6, 6]],
         [[6, 6], [6, 6], [6, 6], [7, 7]],
         [[6, 6], [6, 6], [6, 7], [7, 7]]],
        # Upper Arm = 6
        [[[7, 7], [7, 7], [7, 7], [8, 8]],
         [[8, 8], [8, 8], [8, 8], [9, 9]],
         [[8, 8], [8, 8], [8, 9], [9, 9]]],
    ]

    # Table B:  [neck 1-6][trunk 1-6][legs 1-2]
    _table_b = [
        [[1, 3], [2, 3], [3, 4], [5, 5], [6, 6], [7, 7]],
        [[2, 3], [2, 3], [4, 5], [5, 5], [6, 7], [7, 7]],
        [[3, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 7]],
        [[5, 5], [5, 6], [6, 7], [7, 7], [7, 7], [8, 8]],
        [[7, 7], [7, 7], [7, 7], [7, 8], [8, 8], [8, 8]],
        [[8, 8], [8, 8], [8, 8], [8, 9], [9, 9], [9, 9]],
    ]

    # Table C:  [score_c 1-8][score_d 1-8]
    _table_c = [
        [1, 2, 3, 3, 4, 5, 5, 5],
        [2, 2, 3, 4, 4, 5, 5, 5],
        [3, 3, 3, 4, 4, 5, 6, 6],
        [3, 3, 3, 4, 5, 6, 6, 6],
        [4, 4, 4, 5, 6, 7, 7, 7],
        [4, 4, 5, 6, 6, 7, 7, 7],
        [5, 5, 6, 6, 7, 7, 7, 7],
        [5, 5, 6, 7, 7, 7, 7, 7],
    ]

    # ──────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────────

    def compute_side(self,
                     shoulder_flexion,
                     elbow_flexion,
                     wrist_flexion,
                     neck_flexion,
                     trunk_flexion,
                     shoulder_abduction=0.0,
                     elbow_working_across=False,
                     wrist_deviation=0.0,
                     wrist_pronation=0.0,
                     neck_lateral=0.0,
                     neck_rotation=0.0,
                     trunk_lateral=0.0,
                     trunk_rotation=0.0,
                     shoulder_raised=False,
                     arm_leaning=False,
                     wrist_twist=1,
                     legs_score=None):
        """
        Compute RULA score for one side.

        Parameters
        ----------
        shoulder_flexion  : shoulder pitch diff (arm − trunk), degrees
        elbow_flexion     : absolute elbow internal angle (|forearm − biceps pitch|), degrees
        wrist_flexion     : wrist pitch diff (hand − forearm), degrees
        neck_flexion      : neck pitch diff (head − trunk), degrees
        trunk_flexion     : trunk pitch (raw UPPER_BACK), degrees
        shoulder_abduction: shoulder roll diff (arm − trunk), degrees
        elbow_working_across: bool – forearm crosses midline (roll diff > 15°)
        wrist_deviation   : wrist roll diff (hand − forearm), degrees
        wrist_pronation   : wrist yaw diff (hand − forearm), degrees  [informational]
        neck_lateral      : neck roll diff, degrees
        neck_rotation     : neck yaw diff, degrees
        trunk_lateral     : trunk roll, degrees
        trunk_rotation    : trunk yaw, degrees
        shoulder_raised   : bool
        arm_leaning       : bool – arm supported / leaning
        wrist_twist       : 1 (mid-range) or 2 (near end of range)
        legs_score        : 1 (both legs supported), 2 (unsupported/single leg)

        Returns
        -------
        dict with all intermediate and final scores.
        """
        legs   = legs_score   if legs_score   is not None else self.legs_score

        # ── Group A ───────────────────────────────────────────────────────────
        ua_raw = self.upper_arm_score(shoulder_flexion, shoulder_abduction,
                                      shoulder_raised, arm_leaning)
        fa_raw = self.forearm_score(elbow_flexion, elbow_working_across)
        w_raw  = self.wrist_score(wrist_flexion, wrist_deviation)
        tw     = max(1, min(wrist_twist, 2))

        idx_ua = min(max(ua_raw, 1), 6) - 1
        idx_fa = min(max(fa_raw, 1), 3) - 1
        idx_w  = min(max(w_raw,  1), 4) - 1
        idx_tw = tw - 1

        score_a = self._table_a[idx_ua][idx_fa][idx_w][idx_tw]
        score_c = score_a

        # ── Group B ───────────────────────────────────────────────────────────
        n_raw = self.neck_score(neck_flexion, neck_lateral, neck_rotation)
        t_raw = self.trunk_score(trunk_flexion, trunk_lateral, trunk_rotation)
        l     = max(1, min(legs, 2))

        idx_n = min(max(n_raw, 1), 6) - 1
        idx_t = min(max(t_raw, 1), 6) - 1
        idx_l = l - 1

        score_b = self._table_b[idx_n][idx_t][idx_l]
        score_d = score_b

        # ── Table C (Final) ───────────────────────────────────────────────────
        idx_sc = min(max(score_c, 1), 8) - 1
        idx_sd = min(max(score_d, 1), 8) - 1
        final  = self._table_c[idx_sc][idx_sd]

        # ── Action level ──────────────────────────────────────────────────────
        if final <= 2:
            action = "Acceptable"
        elif final <= 4:
            action = "Investigate further"
        elif final <= 6:
            action = "Investigate and change soon"
        else:
            action = "Investigate and change immediately"

        return {
            # Raw sub-scores
            'upper_arm_score':  ua_raw,
            'forearm_score':    fa_raw,
            'wrist_score':      w_raw,
            'wrist_twist':      tw,
            'neck_score':       n_raw,
            'trunk_score':      t_raw,
            'legs_score':       legs,
            # Intermediate totals
            'score_a':  score_a,
            'score_b':  score_b,
            'score_c':  score_c,
            'score_d':  score_d,

            # Input angles (for CSV logging)
            'shoulder_flexion':    round(shoulder_flexion, 2),
            'shoulder_abduction':  round(shoulder_abduction, 2),
            'elbow_flexion':       round(elbow_flexion, 2),
            'elbow_working_across': int(elbow_working_across),
            'wrist_flexion':       round(wrist_flexion, 2),
            'wrist_deviation':     round(wrist_deviation, 2),
            'wrist_pronation':     round(wrist_pronation, 2),
            'neck_flexion':        round(neck_flexion, 2),
            'neck_lateral':        round(neck_lateral, 2),
            'neck_rotation':       round(neck_rotation, 2),
            'trunk_flexion':       round(trunk_flexion, 2),
            'trunk_lateral':       round(trunk_lateral, 2),
            'trunk_rotation':      round(trunk_rotation, 2),
            # Final
            'final':    final,
            'action':   action,
        }