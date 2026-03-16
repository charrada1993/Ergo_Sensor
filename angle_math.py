import numpy as np

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw (in degrees) to a 3x3 rotation matrix.
    Assumes rotation order: Z (yaw), Y (pitch), X (roll).
    """
    r = np.radians(roll)
    p = np.radians(pitch)
    y = np.radians(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])
    # Combined rotation: R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx

def get_segment_vector(roll, pitch, yaw, axis='x'):
    """
    Compute the global direction vector of a segment given its orientation.
    Assumes the sensor's local axis (default x) points along the segment.
    Returns a 3-element numpy array.
    """
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    if axis == 'x':
        local_vec = np.array([1, 0, 0])
    elif axis == 'y':
        local_vec = np.array([0, 1, 0])
    elif axis == 'z':
        local_vec = np.array([0, 0, 1])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    return R @ local_vec

def angle_between_vectors(v1, v2):
    """
    Compute the angle (in degrees) between two vectors using the dot product.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def compute_joint_angles(sensor_data):
    vectors = {}
    for sid, (r, p, y) in sensor_data.items():
        vectors[sid] = get_segment_vector(r, p, y, axis='x')

    angles = {}
    # Neck
    if 'NECK' in vectors and 'UPPER_BACK' in vectors:
        angles['Neck'] = angle_between_vectors(vectors['NECK'], vectors['UPPER_BACK'])
    # Shoulders
    if 'UPPER_BACK' in vectors and 'R_BICEPS' in vectors:
        angles['R_Shoulder'] = angle_between_vectors(vectors['UPPER_BACK'], vectors['R_BICEPS'])
    if 'UPPER_BACK' in vectors and 'L_BICEPS' in vectors:
        angles['L_Shoulder'] = angle_between_vectors(vectors['UPPER_BACK'], vectors['L_BICEPS'])
    # Elbows
    if 'R_BICEPS' in vectors and 'R_FOREARM' in vectors:
        angles['R_Elbow'] = angle_between_vectors(vectors['R_BICEPS'], vectors['R_FOREARM'])
    if 'L_BICEPS' in vectors and 'L_FOREARM' in vectors:
        angles['L_Elbow'] = angle_between_vectors(vectors['L_BICEPS'], vectors['L_FOREARM'])
    # Wrists
    if 'R_FOREARM' in vectors and 'R_HAND' in vectors:
        angles['R_Wrist'] = angle_between_vectors(vectors['R_FOREARM'], vectors['R_HAND'])
    if 'L_FOREARM' in vectors and 'L_HAND' in vectors:
        angles['L_Wrist'] = angle_between_vectors(vectors['L_FOREARM'], vectors['L_HAND'])
    # Back (if needed)
    # ...
    return angles