import itertools
from angle_math import compute_joint_angles

sensors = [
    'UPPER_BACK', 'NECK', 
    'R_BICEPS', 'L_BICEPS', 
    'R_FOREARM', 'L_FOREARM', 
    'R_HAND', 'L_HAND'
]

# Baseline zeroed data
data_pool = {s: (0.0, 0.0, 0.0) for s in sensors}

errors_found = False

# Test combinations of sizes 1 to 8
for i in range(1, len(sensors) + 1):
    count = 0
    for combo in itertools.combinations(sensors, i):
        test_data = {s: data_pool[s] for s in combo}
        try:
            angles = compute_joint_angles(test_data)
        except Exception as e:
            print(f"Error for combo {combo}: {e}")
            errors_found = True
        count += 1
    print(f"Tested {count} combinations of size {i}")

if not errors_found:
    print("ALL COMBINATIONS PASSED SUCCESSFULLY.")
else:
    print("ERRORS WERE FOUND.")
