"""
RULA (Rapid Upper Limb Assessment) calculation module.
Based on standard ergonomic guidelines and the provided PDF.
"""

class RULAEngine:
    def __init__(self, config):
        self.config = config
        # Load scores from config (can be overridden per call)
        self.load_score = getattr(config, 'RULA_LOAD_SCORE', 0)        # 0,1,2
        self.muscle_score = getattr(config, 'RULA_MUSCLE_SCORE', 0)   # 0 or 1
        self.legs_score = getattr(config, 'RULA_LEGS_SCORE', 1)       # 1 (supported) or 2

    def upper_arm_score(self, angle):
        """Score for upper arm elevation (shoulder flexion)."""
        if angle < 20:
            return 1
        elif angle < 45:
            return 2
        elif angle < 90:
            return 3
        else:
            return 4

    def forearm_score(self, angle):
        """Score for elbow flexion."""
        if 60 <= angle <= 100:
            return 1
        else:
            return 2

    def wrist_score(self, angle):
        """Score for wrist flexion/extension."""
        if angle < 15:
            return 1
        elif angle < 30:
            return 2
        else:
            return 3

    # Wrist twist is not measured; assume twist = 1 (neutral) for now.
    # Combined wrist+twist table:
    _wrist_twist_table = {
        (1,1): 1, (1,2): 2,
        (2,1): 2, (2,2): 3,
        (3,1): 3, (3,2): 4
    }

    # Table A (upper arm & forearm vs wrist+twist)
    # Indexed by (upper_arm, forearm) -> list of 4 scores (for wrist+twist 1..4)
    _table_a = {
        (1,1): [1,2,2,3],
        (1,2): [2,3,3,4],
        (2,1): [2,3,3,4],
        (2,2): [3,4,4,5],
        (3,1): [3,4,4,5],
        (3,2): [4,5,5,6],
        (4,1): [4,5,5,6],
        (4,2): [5,6,6,7]
    }

    def neck_score(self, angle):
        """Score for neck flexion."""
        if angle < 10:
            return 1
        elif angle < 20:
            return 2
        elif angle < 60:
            return 3
        else:
            return 4

    def trunk_score(self, angle):
        """Score for trunk flexion (from vertical)."""
        if angle < 0:  # slight extension? treat as 0
            angle = 0
        if angle < 10:
            return 1
        elif angle < 20:
            return 2
        elif angle < 60:
            return 3
        else:
            return 4

    # Table B (neck & trunk) -> score B
    _table_b = {
        (1,1): 1, (1,2): 2, (1,3): 3, (1,4): 4,
        (2,1): 2, (2,2): 2, (2,3): 3, (2,4): 4,
        (3,1): 3, (3,2): 3, (3,3): 3, (3,4): 4,
        (4,1): 4, (4,2): 4, (4,3): 4, (4,4): 4
    }

    # Table C (score A vs score B + muscle/force) -> final score (1-7)
    # Indexed by (score_a, score_b) with muscle/force adjustments:
    # muscle/force score 0..3 added to score_b before lookup? Actually standard:
    # After obtaining score B, add muscle use score (0-1) and force/load score (0-3) to get score C.
    # Then table C uses score A and score C to give final.
    # We'll implement a direct mapping from (score_a, score_c) to final.
    _table_c = {
        (1,1):1, (1,2):2, (1,3):3, (1,4):3, (1,5):4, (1,6):4, (1,7):5,
        (2,1):2, (2,2):2, (2,3):3, (2,4):4, (2,5):4, (2,6):5, (2,7):5,
        (3,1):3, (3,2):3, (3,3):3, (3,4):4, (3,5):4, (3,6):5, (3,7):6,
        (4,1):3, (4,2):3, (4,3):3, (4,4):4, (4,5):5, (4,6):6, (4,7):6,
        (5,1):4, (5,2):4, (5,3):4, (5,4):5, (5,5):6, (5,6):6, (5,7):7,
        (6,1):4, (6,2):4, (6,3):5, (6,4):5, (6,5):6, (6,6):7, (6,7):7,
        (7,1):5, (7,2):5, (7,3):6, (7,4):6, (7,5):7, (7,6):7, (7,7):7
    }

    def compute_side(self, shoulder_angle, elbow_angle, wrist_angle,
                     neck_angle, trunk_angle,
                     load_score=None, muscle_score=None, legs_score=None,
                     wrist_twist=1):
        """
        Compute RULA score for one side.
        shoulder_angle: upper arm elevation (deg)
        elbow_angle: elbow flexion (deg)
        wrist_angle: wrist flexion (deg)
        neck_angle: neck flexion (deg)
        trunk_angle: trunk flexion from vertical (deg)
        load_score: 0,1,2 (force/load)
        muscle_score: 0 or 1 (static/repeated)
        legs_score: 1 (supported) or 2 (unsupported)
        wrist_twist: 1 (neutral) or 2 (twisted)
        Returns dict with intermediate and final scores.
        """
        # Use instance defaults if not provided
        load = load_score if load_score is not None else self.load_score
        muscle = muscle_score if muscle_score is not None else self.muscle_score
        legs = legs_score if legs_score is not None else self.legs_score

        # Group A
        ua = self.upper_arm_score(shoulder_angle)
        fa = self.forearm_score(elbow_angle)
        w = self.wrist_score(wrist_angle)
        # Combine wrist and twist
        wt = self._wrist_twist_table.get((w, wrist_twist), w)  # fallback
        # Lookup table A
        score_a = self._table_a.get((ua, fa), [1,1,1,1])[wt-1]  # wt is 1-4

        # Group B
        n = self.neck_score(neck_angle)
        t = self.trunk_score(trunk_angle)
        # Table B
        score_b = self._table_b.get((n, t), 1)
        # Add muscle and load to score B
        score_c = score_b + muscle + load
        # Clamp score_c to 1-7 (though max 4+1+2=7)
        score_c = min(max(score_c, 1), 7)

        # Table C
        final_score = self._table_c.get((score_a, score_c), 1)

        # Action level
        if final_score <= 2:
            action = "Acceptable"
        elif final_score <= 4:
            action = "Investigate further"
        elif final_score <= 6:
            action = "Investigate and change soon"
        else:
            action = "Investigate and change immediately"

        return {
            'upper_arm': ua,
            'forearm': fa,
            'wrist': w,
            'wrist_twist': wrist_twist,
            'score_a': score_a,
            'neck': n,
            'trunk': t,
            'legs': legs,
            'score_b': score_b,
            'score_c': score_c,
            'final': final_score,
            'action': action
        }