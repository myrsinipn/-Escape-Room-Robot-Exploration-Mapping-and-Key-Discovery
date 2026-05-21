import numpy as np

DEFAULT_MAX_MARKERS = 20
ROBOT_DIM = 3
# Planar landmarks: (m_x, m_y) world; measurements are range + bearing (body frame).
LM_DIM = 2


def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def wrap_slam_state_angles(mu, max_markers):
    mu = mu.copy()
    mu[2] = wrap_to_pi(mu[2])
    # 6D landmark poses embed roll,pitch,yaw at indices +3...+5 — not used when LM_DIM < 6.
    if LM_DIM < 6:
        return mu
    for i in range(max_markers):
        base = ROBOT_DIM + i * LM_DIM
        mu[base + 3] = wrap_to_pi(mu[base + 3])
        mu[base + 4] = wrap_to_pi(mu[base + 4])
        mu[base + 5] = wrap_to_pi(mu[base + 5])
    return mu


def is_angle_state_index(j, robot_dim=ROBOT_DIM):
    if j == 2:
        return True
    if j < robot_dim:
        return False
    if LM_DIM < 6:
        return False
    local = (j - robot_dim) % LM_DIM
    return local in (3, 4, 5)


def lm_slice(i, robot_dim=ROBOT_DIM):
    start = robot_dim + i * LM_DIM
    return slice(start, start + LM_DIM)


def rot_x(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])


def rot_y(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])


def rot_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])


def euler_zyx_to_R(roll, pitch, yaw):
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def R_to_euler_zyx(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-10
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def so3_log(R):
    tr = np.trace(R)
    cos_angle = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-10:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - angle < 1e-6:
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) >= 1e-12 else np.array([1.0, 0.0, 0.0])
        return angle * axis
    return (angle / (2.0 * np.sin(angle))) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def pose6_to_T(pose6):
    x, y, z, roll, pitch, yaw = pose6
    T = np.eye(4)
    T[:3, :3] = euler_zyx_to_R(roll, pitch, yaw)
    T[:3, 3] = np.array([x, y, z])
    return T


def T_to_pose6(T):
    x, y, z = T[:3, 3]
    roll, pitch, yaw = R_to_euler_zyx(T[:3, :3])
    return np.array([x, y, z, roll, pitch, yaw])


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def pose2d_to_Twb(x, y, theta):
    T = np.eye(4)
    T[:3, :3] = rot_z(theta)
    T[:3, 3] = np.array([x, y, 0.0])
    return T

def quaternion_to_R(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion.flatten()
    return np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),         1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x**2 + y**2)]
    ])

def quaternion_to_euler_zyx(quaternion):
    x, y, z, w = quaternion
    roll  = np.arctan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x**2 + y**2))
    pitch = np.arcsin( 2.0 * (w*y - z*x))
    yaw   = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y**2 + z**2))
    return roll, pitch, yaw
