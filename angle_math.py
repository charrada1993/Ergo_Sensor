import numpy as np

# Calibration storage: sensor_id -> (roll, pitch, yaw) reference
_calibration = {}

def set_reference(sensor_id, roll, pitch, yaw):
    _calibration[sensor_id] = (roll, pitch, yaw)

def get_reference(sensor_id):
    return _calibration.get(sensor_id, (0.0, 0.0, 0.0))

def relative_angle(current_val, ref_val):
    """Simple differential angle (current minus reference)."""
    return current_val - ref_val

def compute_joint_angles(sensor_data):
    """
    sensor_data: dict {sensor_id: (roll, pitch, yaw)}

    Computes relative joint angles per the RULA/REBA PDF spec.
    All angles are in degrees.

    Returns dict with keys:
        Neck           – neck flexion (pitch diff head vs trunk)
        Neck_Roll      – neck lateral tilt (roll diff)
        Neck_Yaw       – neck rotation (yaw diff)

        Trunk_Pitch    – trunk flexion (raw UPPER_BACK pitch)
        Trunk_Roll     – trunk lateral tilt (raw UPPER_BACK roll)
        Trunk_Yaw      – trunk rotation (raw UPPER_BACK yaw)

        R_Shoulder     – right shoulder flexion (pitch diff arm vs trunk)
        R_Shoulder_Abduction – right shoulder abduction (roll diff)
        L_Shoulder     – left shoulder flexion
        L_Shoulder_Abduction – left shoulder abduction

        R_Elbow        – right elbow flexion (|pitch forearm - pitch biceps|)
        R_Elbow_Roll   – lateral deviation of right forearm (|roll diff|)
        L_Elbow        – left elbow flexion
        L_Elbow_Roll   – lateral deviation of left forearm

        R_Wrist        – right wrist flexion/extension (pitch diff hand vs forearm)
        R_Wrist_Roll   – right wrist radial deviation (roll diff)
        R_Wrist_Yaw    – right wrist pronation/supination (yaw diff)
        L_Wrist        – left wrist flexion/extension
        L_Wrist_Roll   – left wrist deviation
        L_Wrist_Yaw    – left wrist pronation/supination
    """
    angles = {}

    def get(sid):
        """Return (roll, pitch, yaw) for sensor, minus its calibration reference."""
        raw = sensor_data.get(sid)
        if raw is None:
            return None
        ref = get_reference(sid)
        return (raw[0] - ref[0], raw[1] - ref[1], raw[2] - ref[2])

    # ── Trunk (raw UPPER_BACK relative to calibration) ────────────────────────
    trunk = get('UPPER_BACK')
    if trunk:
        t_roll, t_pitch, t_yaw = trunk
        angles['Trunk_Pitch'] = t_pitch
        angles['Trunk_Roll']  = t_roll
        angles['Trunk_Yaw']   = t_yaw
        # Scalar alias used downstream
        angles['Back'] = t_pitch
    else:
        t_roll, t_pitch, t_yaw = 0.0, 0.0, 0.0

    # ── Neck (head relative to trunk) ─────────────────────────────────────────
    neck_raw = get('NECK')
    if neck_raw:
        n_roll, n_pitch, n_yaw = neck_raw
        angles['Neck']      = n_pitch - t_pitch   # flexion
        angles['Neck_Roll'] = n_roll  - t_roll     # lateral tilt
        angles['Neck_Yaw']  = n_yaw   - t_yaw     # rotation

    # ── Right Shoulder ────────────────────────────────────────────────────────
    r_biceps = get('R_BICEPS')
    if r_biceps:
        rb_roll, rb_pitch, rb_yaw = r_biceps
        angles['R_Shoulder']           = rb_pitch - t_pitch   # flexion
        angles['R_Shoulder_Abduction'] = rb_roll  - t_roll    # abduction

    # ── Left Shoulder ─────────────────────────────────────────────────────────
    l_biceps = get('L_BICEPS')
    if l_biceps:
        lb_roll, lb_pitch, lb_yaw = l_biceps
        angles['L_Shoulder']           = lb_pitch - t_pitch
        angles['L_Shoulder_Abduction'] = lb_roll  - t_roll

    # ── Right Elbow ───────────────────────────────────────────────────────────
    r_forearm = get('R_FOREARM')
    if r_forearm:
        rf_roll, rf_pitch, rf_yaw = r_forearm
        if r_biceps:
            angles['R_Elbow']      = abs(rf_pitch - rb_pitch)   # flexion (internal angle)
            angles['R_Elbow_Roll'] = abs(rf_roll  - rb_roll)    # lateral deviation

    # ── Left Elbow ────────────────────────────────────────────────────────────
    l_forearm = get('L_FOREARM')
    if l_forearm:
        lf_roll, lf_pitch, lf_yaw = l_forearm
        if l_biceps:
            angles['L_Elbow']      = abs(lf_pitch - lb_pitch)
            angles['L_Elbow_Roll'] = abs(lf_roll  - lb_roll)

    # ── Right Wrist ───────────────────────────────────────────────────────────
    r_hand = get('R_HAND')
    if r_hand and r_forearm:
        rh_roll, rh_pitch, rh_yaw = r_hand
        angles['R_Wrist']      = rh_pitch - rf_pitch   # flexion/extension
        angles['R_Wrist_Roll'] = rh_roll  - rf_roll    # radial deviation
        angles['R_Wrist_Yaw']  = rh_yaw   - rf_yaw    # pronation/supination

    # ── Left Wrist ────────────────────────────────────────────────────────────
    l_hand = get('L_HAND')
    if l_hand and l_forearm:
        lh_roll, lh_pitch, lh_yaw = l_hand
        angles['L_Wrist']      = lh_pitch - lf_pitch
        angles['L_Wrist_Roll'] = lh_roll  - lf_roll
        angles['L_Wrist_Yaw']  = lh_yaw   - lf_yaw

    # ── Right Thigh & Knee ─────────────────────────────────────────────────────
    r_thigh = get('R_THIGH')
    if r_thigh and trunk:
        rt_roll, rt_pitch, rt_yaw = r_thigh
        angles['R_Thigh']      = rt_pitch - t_pitch
        angles['R_Thigh_Roll'] = rt_roll  - t_roll
        angles['R_Thigh_Yaw']  = rt_yaw   - t_yaw

    r_shank = get('R_SHANK')
    if r_shank and r_thigh:
        rs_roll, rs_pitch, rs_yaw = r_shank
        angles['R_Knee'] = abs(rs_pitch - rt_pitch)

    # ── Left Thigh & Knee ──────────────────────────────────────────────────────
    l_thigh = get('L_THIGH')
    if l_thigh and trunk:
        lt_roll, lt_pitch, lt_yaw = l_thigh
        angles['L_Thigh']      = lt_pitch - t_pitch
        angles['L_Thigh_Roll'] = lt_roll  - t_roll
        angles['L_Thigh_Yaw']  = lt_yaw   - t_yaw

    l_shank = get('L_SHANK')
    if l_shank and l_thigh:
        ls_roll, ls_pitch, ls_yaw = l_shank
        angles['L_Knee'] = abs(ls_pitch - lt_pitch)

    return angles