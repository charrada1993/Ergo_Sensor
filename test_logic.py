"""
Comprehensive backend logic test - run: python test_logic.py
"""
import traceback
import sys

PASS = 0
FAIL = 0

def ok(msg):
    global PASS
    PASS += 1
    print(f"  [OK]   {msg}")

def fail(msg, exc=None):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {msg}")
    if exc:
        traceback.print_exc()

# ─── 1. Config ────────────────────────────────────────────────────────────────
print("\n=== 1. Config ===")
try:
    from config import Config
    assert Config.CSV_DIR == 'csv_data'
    assert Config.REBA_LEGS_SCORE == 1
    assert Config.RULA_LEGS_SCORE == 1
    ok("Config loaded and attributes present")
except Exception as e:
    fail("Config", e)

# ─── 2. angle_math ───────────────────────────────────────────────────────────
print("\n=== 2. angle_math ===")
try:
    from angle_math import compute_joint_angles, set_reference
    sensor_data = {
        'UPPER_BACK': (0.0, 10.0, 0.0),
        'NECK':       (0.0, 15.0, 0.0),
        'R_BICEPS':   (0.0, 30.0, 0.0),
        'R_FOREARM':  (0.0, 80.0, 0.0),
        'R_HAND':     (0.0, 85.0, 0.0),
        'L_BICEPS':   (0.0, 25.0, 0.0),
        'L_FOREARM':  (0.0, 75.0, 0.0),
        'L_HAND':     (0.0, 80.0, 0.0),
    }
    angles = compute_joint_angles(sensor_data)
    assert 'Neck'      in angles, "Neck missing from angles"
    assert 'R_Shoulder' in angles, "R_Shoulder missing"
    assert 'R_Elbow'   in angles, "R_Elbow missing"
    assert 'R_Wrist'   in angles, "R_Wrist missing"
    assert 'L_Shoulder' in angles, "L_Shoulder missing"
    assert 'L_Elbow'   in angles, "L_Elbow missing"
    assert 'L_Wrist'   in angles, "L_Wrist missing"
    assert 'Trunk_Pitch' in angles, "Trunk_Pitch missing"
    # Neck should be head pitch - trunk pitch = 15 - 10 = 5
    assert abs(angles['Neck'] - 5.0) < 0.01, f"Neck angle wrong: {angles['Neck']}"
    ok(f"compute_joint_angles -> {list(angles.keys())}")
except Exception as e:
    fail("angle_math", e)

# ─── 3. REBA engine direct call ───────────────────────────────────────────────
print("\n=== 3. REBA engine ===")
try:
    from reba_engine import REBAEngine
    reba = REBAEngine(Config)
    result = reba.compute_side(
        trunk_flexion=10.0, neck_flexion=5.0,
        upper_arm_flexion=30.0, forearm_flexion=80.0, wrist_flexion=5.0
    )
    assert 'final'  in result, "final missing"
    assert 'action' in result, "action missing"
    assert 'score_a' in result, "score_a missing"
    assert 'score_b' in result, "score_b missing"
    assert isinstance(result['final'], int), "final should be int"
    ok(f"REBA compute_side -> final={result['final']}, action={result['action']}")
except Exception as e:
    fail("REBA engine", e)

# ─── 4. RULA engine direct call ───────────────────────────────────────────────
print("\n=== 4. RULA engine ===")
try:
    from rula_engine import RULAEngine
    rula = RULAEngine(Config)
    result = rula.compute_side(
        shoulder_flexion=30.0, elbow_flexion=80.0, wrist_flexion=5.0,
        neck_flexion=10.0, trunk_flexion=15.0
    )
    assert 'final'  in result, "final missing"
    assert 'action' in result, "action missing"
    assert 'score_c' in result, "score_c missing"
    assert 'score_d' in result, "score_d missing"
    ok(f"RULA compute_side -> final={result['final']}, action={result['action']}")
except Exception as e:
    fail("RULA engine", e)

# ─── 5. DataProcessor call to REBA/RULA (reproduce the bug) ──────────────────
print("\n=== 5. DataProcessor REBA/RULA call signature check ===")
try:
    from reba_engine import REBAEngine
    from rula_engine import RULAEngine
    from angle_math import compute_joint_angles
    reba = REBAEngine(Config)
    rula = RULAEngine(Config)

    sensor_data = {
        'UPPER_BACK': (0.0, 10.0, 0.0),
        'NECK':       (0.0, 15.0, 0.0),
        'R_BICEPS':   (0.0, 30.0, 0.0),
        'R_FOREARM':  (0.0, 80.0, 0.0),
        'R_HAND':     (0.0, 85.0, 0.0),
        'L_BICEPS':   (0.0, 25.0, 0.0),
        'L_FOREARM':  (0.0, 75.0, 0.0),
        'L_HAND':     (0.0, 80.0, 0.0),
    }
    angles = compute_joint_angles(sensor_data)
    trunk_angle = angles.get('Trunk_Pitch', 0.0)
    trunk_roll  = angles.get('Trunk_Roll',  0.0)
    trunk_yaw   = angles.get('Trunk_Yaw',   0.0)

    # --- Simulate what data_processor.py does (OLD WRONG code) ---
    bug_found = False
    try:
        reba.compute_side(
            upper_arm_angle=angles.get('R_Shoulder', 0),
            forearm_angle=angles.get('R_Elbow', 0),
            wrist_angle=angles.get('R_Wrist', 0),
            neck_angle=angles.get('Neck', 0),
            trunk_angle=trunk_angle,
        )
        ok("[WARNING] Old REBA call with wrong kwargs did NOT raise TypeError - unexpected")
    except TypeError as e:
        bug_found = True
        ok(f"Bug confirmed: REBA called with wrong kwargs: {e}")

    try:
        rula.compute_side(
            shoulder_angle=angles.get('R_Shoulder', 0),
            elbow_angle=angles.get('R_Elbow', 0),
            wrist_angle=angles.get('R_Wrist', 0),
            neck_angle=angles.get('Neck', 0),
            trunk_angle=trunk_angle,
        )
        ok("[WARNING] Old RULA call with wrong kwargs did NOT raise TypeError - unexpected")
    except TypeError as e:
        bug_found = True
        ok(f"Bug confirmed: RULA called with wrong kwargs: {e}")

    # --- Now test CORRECT calls ---
    reba_r = reba.compute_side(
        trunk_flexion=trunk_angle,     neck_flexion=angles.get('Neck', 0),
        upper_arm_flexion=angles.get('R_Shoulder', 0),
        forearm_flexion=angles.get('R_Elbow', 0),
        wrist_flexion=angles.get('R_Wrist', 0),
        trunk_lateral=trunk_roll, trunk_rotation=trunk_yaw,
        neck_lateral=angles.get('Neck_Roll', 0), neck_rotation=angles.get('Neck_Yaw', 0),
        upper_arm_abduction=angles.get('R_Shoulder_Abduction', 0),
        forearm_lateral=angles.get('R_Elbow_Roll', 0),
        wrist_deviation=angles.get('R_Wrist_Roll', 0),
        wrist_pronation=angles.get('R_Wrist_Yaw', 0),
    )
    ok(f"REBA correct call -> final={reba_r['final']}")

    rula_r = rula.compute_side(
        shoulder_flexion=angles.get('R_Shoulder', 0),
        elbow_flexion=angles.get('R_Elbow', 0),
        wrist_flexion=angles.get('R_Wrist', 0),
        neck_flexion=angles.get('Neck', 0),
        trunk_flexion=trunk_angle,
        shoulder_abduction=angles.get('R_Shoulder_Abduction', 0),
        wrist_deviation=angles.get('R_Wrist_Roll', 0),
        wrist_pronation=angles.get('R_Wrist_Yaw', 0),
        neck_lateral=angles.get('Neck_Roll', 0),
        neck_rotation=angles.get('Neck_Yaw', 0),
        trunk_lateral=trunk_roll, trunk_rotation=trunk_yaw,
        elbow_working_across=angles.get('R_Elbow_Roll', 0) > 15,
    )
    ok(f"RULA correct call -> final={rula_r['final']}")

except Exception as e:
    fail("DataProcessor REBA/RULA signature check", e)

# ─── 6. RiskEngine ────────────────────────────────────────────────────────────
print("\n=== 6. RiskEngine ===")
try:
    from risk_engine import RiskEngine
    import time
    re = RiskEngine(Config)
    # Feed some data
    angles_sample = {'Neck': 10.0, 'R_Elbow': 80.0, 'R_Shoulder': 30.0}
    for i in range(15):
        re.update(angles_sample, time.time())
    risk = re.compute_risk()
    assert risk is not None, "risk is None after 15 samples"
    assert 'global' in risk, "global missing"
    assert 'level'  in risk, "level missing"
    ok(f"RiskEngine -> global={risk['global']:.3f}, level={risk['level']}")
except Exception as e:
    fail("RiskEngine", e)

# ─── 7. CSVLogger ─────────────────────────────────────────────────────────────
print("\n=== 7. CSVLogger ===")
try:
    from csv_logger import CSVLogger
    logger = CSVLogger(Config)
    angles_sample = {
        'Neck': 10.0, 'R_Shoulder': 30.0, 'R_Elbow': 80.0, 'R_Wrist': 5.0,
        'L_Shoulder': 25.0, 'L_Elbow': 75.0, 'L_Wrist': 3.0
    }
    ok("CSVLogger instantiated")
except Exception as e:
    fail("CSVLogger", e)

# ─── 8. DataProcessor instantiation ──────────────────────────────────────────
print("\n=== 8. DataProcessor ===")
try:
    from data_processor import DataProcessor
    from socket_manager import socketio
    dp = DataProcessor(Config, socketio, None)
    ok("DataProcessor instantiated")
    status = dp.get_sensor_status()
    ok(f"get_sensor_status -> {len(status)} sensors")
    current_log = dp.get_current_log_filename()
    ok(f"get_current_log_filename -> {current_log}")
except Exception as e:
    fail("DataProcessor", e)

# ─── 9. Feature extractor ─────────────────────────────────────────────────────
print("\n=== 9. FeatureExtractor ===")
try:
    from feature_extractor import FeatureExtractor
    meta = {'feature_cols': ['global_risk_score', 'hour_of_day', 'day_of_week']}
    fe = FeatureExtractor(meta)
    import time
    from collections import deque
    aw = deque(maxlen=60)
    rw = deque(maxlen=60)
    rl_w = deque(maxlen=60)
    rr_w = deque(maxlen=60)
    bl_w = deque(maxlen=60)
    br_w = deque(maxlen=60)
    for i in range(60):
        aw.append({'Neck': 10.0, 'R_Shoulder': 30.0})
        rw.append(0.2)
        rl_w.append(3.0)
        rr_w.append(3.0)
        bl_w.append(4.0)
        br_w.append(4.0)
    features = fe.extract(aw, rw, rl_w, rr_w, bl_w, br_w, time.time())
    assert 'global_risk_score' in features
    assert 'hour_of_day' in features
    ok(f"FeatureExtractor -> {list(features.keys())}")
except Exception as e:
    fail("FeatureExtractor", e)

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Results: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
if FAIL > 0:
    sys.exit(1)
