"""
REBA (Rapid Entire Body Assessment) calculation module.
Based on the provided guide.
"""

class REBAEngine:
    def __init__(self, config):
        self.config = config
        # Load default values from config
        self.load_score = getattr(config, 'REBA_LOAD_SCORE', 0)          # 0,1,2
        self.coupling_score = getattr(config, 'REBA_COUPLING_SCORE', 0)  # 0,1,2
        self.activity_score = getattr(config, 'REBA_ACTIVITY_SCORE', 0)  # 0 or 1
        self.legs_score = getattr(config, 'REBA_LEGS_SCORE', 1)          # 1-4

    # Group A: Trunk, Neck, Legs
    def trunk_score(self, angle):
        """Score for trunk flexion/extension."""
        if angle < 0:  # extension
            return 2
        elif angle <= 20:
            return 2
        elif angle <= 60:
            return 3
        else:
            return 4

    def neck_score(self, angle):
        """Score for neck flexion/extension."""
        if angle <= 20:
            return 1
        else:
            return 2

    # Group B: Upper arm, Forearm, Wrist
    def upper_arm_score(self, angle):
        """Score for upper arm elevation."""
        if angle <= 20:
            return 1
        elif angle <= 45:
            return 2
        elif angle <= 90:
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
        if angle <= 15:
            return 1
        else:
            return 2

    # Table A (Trunk vs Neck) – from guide
    _table_a = {
        (1,1): 1, (1,2): 2, (1,3): 3, (1,4): 4,
        (2,1): 2, (2,2): 2, (2,3): 3, (2,4): 4,
        (3,1): 3, (3,2): 3, (3,3): 3, (3,4): 4,
        (4,1): 4, (4,2): 4, (4,3): 4, (4,4): 4
    }

    # Table B (Upper arm vs Forearm) – from guide
    _table_b = {
        (1,1): 1, (1,2): 2, (1,3): 3, (1,4): 4,
        (2,1): 1, (2,2): 2, (2,3): 3, (2,4): 4,
        (3,1): 2, (3,2): 3, (3,3): 4, (3,4): 5,
        (4,1): 3, (4,2): 4, (4,3): 5, (4,4): 6
    }

    # Table C (Score A vs Score B) – final REBA before adjustments
    _table_c = [
        [1,1,1,2,3,3,4,4,5,6,7,7,7],
        [1,2,2,3,4,4,5,5,6,7,8,8,8],
        [2,3,3,3,4,5,6,6,7,8,9,9,9],
        [3,4,4,4,5,6,7,7,8,9,10,10,10],
        [4,4,4,5,6,7,8,8,9,10,11,11,11],
        [6,6,6,7,8,8,9,9,10,11,12,12,12],
        [7,7,7,8,9,9,10,10,11,12,13,13,13]
    ]

    def compute_side(self, trunk_angle, neck_angle, upper_arm_angle, forearm_angle, wrist_angle,
                     load_score=None, coupling_score=None, activity_score=None, legs_score=None):
        """
        Compute REBA score for one side.
        trunk_angle: trunk flexion (deg)
        neck_angle: neck flexion (deg)
        upper_arm_angle: shoulder flexion (deg)
        forearm_angle: elbow flexion (deg)
        wrist_angle: wrist flexion (deg)
        Returns dict with intermediate and final scores.
        """
        # Use provided or defaults
        load = load_score if load_score is not None else self.load_score
        coupling = coupling_score if coupling_score is not None else self.coupling_score
        activity = activity_score if activity_score is not None else self.activity_score
        legs = legs_score if legs_score is not None else self.legs_score

        # Group A
        t = self.trunk_score(trunk_angle)
        n = self.neck_score(neck_angle)
        # Table A
        score_a_pre = self._table_a.get((t, n), 1)
        # Add legs
        score_a = score_a_pre + legs

        # Group B
        ua = self.upper_arm_score(upper_arm_angle)
        fa = self.forearm_score(forearm_angle)
        # Table B
        score_b_pre = self._table_b.get((ua, fa), 1)
        # Add wrist
        w = self.wrist_score(wrist_angle)
        score_b = score_b_pre + w

        # Table C: score_a (1-12) and score_b (1-12) index into _table_c
        # Clamp to valid range
        sa = max(1, min(score_a, 12))
        sb = max(1, min(score_b, 12))
        # Table C rows: score_a index (0-based) from 1 to 7? Wait our table has 7 rows for score_a 1-7? Actually score_a can go up to 12, but table only has 7 rows. Need to map.
        # In REBA, score A ranges from 1 to 12, score B from 1 to 12. Table C is given for A=1..7 and B=1..12. For A>7, it's extrapolated? The guide shows table with A=1..7 and B=1..12. For A>7, we can use row 7? Typically, score A is capped at 12 but table C only goes to 7. We'll use row = min(sa, 7) - 1.
        row = min(sa, 7) - 1
        col = sb - 1
        final_pre = self._table_c[row][col]

        # Add adjustments
        final_score = final_pre + load + coupling + activity

        # Determine action level
        if final_score <= 1:
            action = "Negligible"
        elif final_score <= 3:
            action = "Low"
        elif final_score <= 7:
            action = "Medium"
        elif final_score <= 10:
            action = "High"
        else:
            action = "Very High"

        return {
            'trunk': t,
            'neck': n,
            'legs': legs,
            'score_a': score_a,
            'upper_arm': ua,
            'forearm': fa,
            'wrist': w,
            'score_b': score_b,
            'final': final_score,
            'action': action
        }