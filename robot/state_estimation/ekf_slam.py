#!/usr/bin/env python3
"""
EKF-SLAM node — upgraded version.

Merged enhancements from grid_slam:
  1. Replaced the fixed-stride corner method with the distance-invariant 
     Ramer-Douglas-Peucker (RDP) polyline segmentation.
  2. Integrated corner arm-length validation to filter out laser noise spikes.
  3. Bypasses raycasting grid updates entirely during explicit spin-in-place turns
     to prevent wall smearing and preserve map quality.
"""

import math
import threading
import traceback

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray

from sensors.lidar import LidarSensor
from sensors.odometry import OdometrySensor
from perception.scan_preprocessor import ScanPreprocessor
from perception.motion_model import OmniMotionModel


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class EKFLidarSLAM(Node):

    def __init__(
        self,
        lidar: LidarSensor,
        odom: OdometrySensor,
        scan_preprocessor: ScanPreprocessor,
        motion_model: OmniMotionModel,
    ) -> None:
        super().__init__("ekf_lidar_slam")
        self.callback_group = ReentrantCallbackGroup()
        self.lidar = lidar
        self.odom = odom
        self.scan_preprocessor = scan_preprocessor
        self.motion_model = motion_model

        self.total_corner_detections = 0
        self.total_confirmed_corners = 0

        # Protects the occupancy grid arrays (log_odds, cells_observed).
        self.map_lock = threading.Lock()

        # Dedicated lock for the scan stash shared between
        # correction_step (writer) and publish_map_step (reader).
        self._scan_lock = threading.Lock()

        # ── EKF state ────────────────────────────────────────────────
        self.mu = np.zeros((3, 1))
        self.num_landmarks = 0

        self.Sigma = np.diag([
            0.02,
            0.02,
            math.radians(2.0),
        ]) ** 2

        # ── Noise matrices ───────────────────────────────────────────
        self.motion_noise = np.diag([
            0.02,
            0.02,
            math.radians(1.5),
        ]) ** 2

        self.obs_noise = np.diag([
            0.3,
            math.radians(15.0),
        ]) ** 2

        # ── Data association ─────────────────────────────────────────
        self.mahal_threshold = 5.99
        self.new_landmark_min_dist = 0.80
        self.min_landmark_range = 0.25
        self.max_landmark_range = 3.50

        # ── Candidate buffer ─────────────────────────────────────────
        self.candidate_landmarks = []
        self.candidate_match_dist = 0.35
        self.candidate_required_seen = 4

        # ── Feature extraction parameters (Upgraded via Grid SLAM) ───
        self.corner_cluster_gap = 0.20       # m range jump -> new segment
        self.corner_min_seg_pts = 6          # min pts to fit a segment
        self.corner_rdp_eps = 0.04           # m split-and-merge tolerance
        self.corner_min_angle = 1.483        # rad (~85 deg) real corner
        self.corner_max_angle = 1.658        # rad (~95 deg) real corner
        self.corner_min_arm = 0.15           # m min arm length each side of corner
        self.min_feature_separation = 0.70   # m deduplication boundary

        # ── Occupancy grid ───────────────────────────────────────────
        self.map_resolution = 0.05        # 5 cm / cell
        self.map_size_m = 30.0
        self.map_width = int(self.map_size_m / self.map_resolution)    # 600
        self.map_height = int(self.map_size_m / self.map_resolution)   # 600

        self.map_origin_x = -self.map_size_m / 2.0   # -15.0 m
        self.map_origin_y = -self.map_size_m / 2.0   # -15.0 m

        self.log_odds = np.zeros(
            (self.map_height, self.map_width), dtype=np.float32
        )
        self.cells_observed = np.zeros(
            (self.map_height, self.map_width), dtype=bool
        )

        self.log_odds_free = -0.9   # strong enough to clear ghost obstacles in ~2 scans
        self.log_odds_occ  =  1.4   # walls confirmed fast (1 hit → 1.4, well above threshold)
        self.log_odds_min  = -5.0
        self.log_odds_max  = 10.0   # real walls saturate here, need 11 free rays to clear
        self.max_mapping_range = 4.0

        # ── Rotation Freezing Logic ──────────────────────────────────
        # Detected from odometry (not cmd_vel) to avoid the 100ms bridge lag.
        self._rotating = False
        self._rotate_wz_thresh = 0.18   # rad/s — set above drive_wz_cap (0.12) so SLAM only pauses during true pivot (TURN phase)
        self._rotate_vx_thresh = 0.01   # m/s  — below this counts as stationary (smooth controller keeps vx > 0 even when turning)

        # ── Timing ───────────────────────────────────────────────────
        self.prev_odom_timestamp = None

        self._latest_ranges: np.ndarray | None = None
        self._latest_angles: np.ndarray | None = None

        # ── Publishers ───────────────────────────────────────────────
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/slam_pose", 10)
        self.map_pub = self.create_publisher(
            OccupancyGrid, "/slam_map", 1)
        self.landmark_pub = self.create_publisher(
            MarkerArray, "/slam_landmarks", 1)

        # ── Timers ───────────────────────────────────────────────────
        self.pose_timer_period = 0.05        # 20 Hz  — prediction + pose pub
        self.correction_timer_period = 0.20  # 5 Hz   — EKF correction

        self.create_timer(
            self.pose_timer_period,
            self.prediction_and_publish_step,
            callback_group=self.callback_group,
        )
        self.create_timer(
            self.correction_timer_period,
            self.correction_step,
            callback_group=self.callback_group,
        )
        self.create_timer(
            0.5,
            self.publish_map_step,
            callback_group=self.callback_group,
        )
        self.get_logger().info(
            f"EKF LiDAR SLAM ready — "
            f"map {self.map_size_m:.0f}x{self.map_size_m:.0f} m "
            f"({self.map_width}x{self.map_height} cells @ {self.map_resolution} m/cell), "
            f"origin ({self.map_origin_x:.1f}, {self.map_origin_y:.1f})"
        )

    # ================================================================
    # PREDICTION
    # ================================================================

    def prediction_and_publish_step(self) -> None:
        try:
            self.prediction_step()
            self.publish_pose()
        except Exception:
            self.get_logger().error(
                "prediction crashed:\n" + traceback.format_exc()
            )

    def prediction_step(self) -> None:
        vel_data = self.odom.get_velocity()
        if vel_data is None:
            return

        current_time = self.get_clock().now().nanoseconds * 1e-9

        if self.prev_odom_timestamp is None:
            odom_pose = self.odom.get_pose()
            if odom_pose is not None:
                self.mu[0, 0] = float(odom_pose["pose"][0])
                self.mu[1, 0] = float(odom_pose["pose"][1])
                self.mu[2, 0] = float(odom_pose["pose"][2])
            self.prev_odom_timestamp = current_time
            return

        dt = current_time - self.prev_odom_timestamp
        self.prev_odom_timestamp = current_time
        if not (0.0 < dt <= 2):
            return

        vx, vy, omega = vel_data["velocity"]

        max_substep = 0.05
        n_substeps = max(1, int(math.ceil(dt / max_substep)))
        sub_dt = dt / n_substeps

        robot_pose = self.mu[:3, 0].copy()
        F_accum = np.eye(3)

        for _ in range(n_substeps):
            robot_pose, F_sub = self.motion_model.predict(
                robot_pose,
                float(vx), float(vy), float(omega), sub_dt,
                velocities_in_body_frame=True,
            )
            F_accum = F_sub @ F_accum

        self.mu[0, 0] = robot_pose[0]
        self.mu[1, 0] = robot_pose[1]
        self.mu[2, 0] = robot_pose[2]

        state_size = self.mu.shape[0]
        F_full = np.eye(state_size)
        F_full[0:3, 0:3] = F_accum

        R_full = np.zeros((state_size, state_size))
        R_full[0:3, 0:3] = self.motion_noise

        self.Sigma = F_full @ self.Sigma @ F_full.T + R_full

    # ================================================================
    # CORRECTION
    # ================================================================

    def correction_step(self) -> None:
        try:
            raw_scan = self.lidar.get_scan()
            if raw_scan is None:
                self.get_logger().warn(
                    "correction_step: lidar.get_scan() is None — no scan received yet",
                    throttle_duration_sec=4.0,
                )
                return

            processed = self.scan_preprocessor.preprocess(raw_scan)
            ranges = processed["ranges"]
            angles = processed["angles"]
            points = self.scan_preprocessor.polar_to_cartesian(ranges, angles)

            if len(points) < 10:
                self.get_logger().warn(
                    f"correction_step: scan too sparse ({len(points)} points) — skipping"
                )
                return

            observations = self.extract_lidar_landmarks(points, ranges, angles)
            for obs in observations:
                self.process_observation(obs)

            with self._scan_lock:
                self._latest_ranges = ranges.copy()
                self._latest_angles = angles.copy()

            self.publish_landmarks()

        except Exception:
            self.get_logger().error(
                "correction crashed:\n" + traceback.format_exc()
            )

    # ================================================================
    # FEATURE EXTRACTION (Upgraded RDP + Arm Checking)
    # ================================================================

    def _rdp(self, pts, i0, j0):
        """Ramer-Douglas-Peucker iterative polyline segmentation."""
        keep = [i0, j0]
        stack = [(i0, j0)]
        while stack:
            i, j = stack.pop()
            if j <= i + 1:
                continue
            p0 = pts[i]
            p1 = pts[j]
            seg = p1 - p0
            L = math.hypot(float(seg[0]), float(seg[1]))
            sub = pts[i + 1:j]
            if L < 1e-6:
                d = np.hypot(sub[:, 0] - p0[0], sub[:, 1] - p0[1])
            else:
                d = np.abs(seg[0] * (p0[1] - sub[:, 1]) - seg[1] * (p0[0] - sub[:, 0])) / L
            k = int(np.argmax(d))
            if d[k] > self.corner_rdp_eps:
                ki = i + 1 + k
                keep.append(ki)
                stack.append((i, ki))
                stack.append((ki, j))
        return sorted(set(keep))

    def _seg_corners(self, seg, ranges_seg, angles_seg, out_obs):
        """Processes polyline segments, extracts safe corners with arm validation."""
        if len(seg) < self.corner_min_seg_pts:
            return
        pts = np.asarray(seg, dtype=np.float32)
        idx = self._rdp(pts, 0, len(pts) - 1)
        for k in range(1, len(idx) - 1):
            a = pts[idx[k - 1]]
            b = pts[idx[k]]
            c = pts[idx[k + 1]]
            
            # Distance-invariant structural check (Arm Verification)
            arm1 = math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))
            arm2 = math.hypot(float(c[0] - b[0]), float(c[1] - b[1]))
            if arm1 < self.corner_min_arm or arm2 < self.corner_min_arm:
                continue
                
            v1 = a - b
            v2 = c - b
            cross = float(v1[0] * v2[1] - v1[1] * v2[0])
            dot = float(v1[0] * v2[0] + v1[1] * v2[1])
            ang = abs(math.atan2(cross, dot))
            
            if self.corner_min_angle <= ang <= self.corner_max_angle:
                orig_idx = idx[k]
                r = float(ranges_seg[orig_idx])
                phi = float(angles_seg[orig_idx])
                
                self.total_corner_detections += 1
                out_obs.append({
                    "range": r,
                    "bearing": wrap_angle(phi),
                    "type": "corner",
                })

    def extract_lidar_landmarks(self, points, ranges, angles) -> list:
        observations = []
        seg = []
        ranges_seg = []
        angles_seg = []
        prev_r = None
        
        for i in range(len(ranges)):
            r = float(ranges[i])
            if np.isfinite(r) and r > 0.05:
                if prev_r is not None and abs(r - prev_r) > self.corner_cluster_gap:
                    self._seg_corners(seg, ranges_seg, angles_seg, observations)
                    seg, ranges_seg, angles_seg = [], [], []
                seg.append(points[i])
                ranges_seg.append(ranges[i])
                angles_seg.append(angles[i])
                prev_r = r
            else:
                self._seg_corners(seg, ranges_seg, angles_seg, observations)
                seg, ranges_seg, angles_seg = [], [], []
                prev_r = None
                
        self._seg_corners(seg, ranges_seg, angles_seg, observations)
        return self.remove_duplicate_observations(observations)

    def remove_duplicate_observations(self, observations: list) -> list:
        kept = []
        for obs in observations:
            ox = obs["range"] * math.cos(obs["bearing"])
            oy = obs["range"] * math.sin(obs["bearing"])
            too_close = any(
                math.hypot(ox - k["range"] * math.cos(k["bearing"]),
                           oy - k["range"] * math.sin(k["bearing"]))
                < self.min_feature_separation
                for k in kept
            )
            if not too_close:
                kept.append(obs)
        return kept

    # ================================================================
    # EKF CORRECTION
    # ================================================================

    def process_observation(self, obs: dict) -> None:
        z = np.array([[obs["range"]], [obs["bearing"]]])
        if not (self.min_landmark_range < z[0, 0] < self.max_landmark_range):
            return
        if self.num_landmarks == 0:
            self.add_to_candidate_buffer(obs)
            return

        best_landmark_idx = None
        best_distance_sq = float("inf")
        best_H = best_innovation = best_S = None

        for landmark_idx in range(self.num_landmarks):
            z_hat, H = self.observation_model(landmark_idx)
            innovation = z - z_hat
            innovation[1, 0] = wrap_angle(innovation[1, 0])
            S = H @ self.Sigma @ H.T + self.obs_noise
            try:
                d_sq = float(innovation.T @ np.linalg.inv(S) @ innovation)
            except np.linalg.LinAlgError:
                continue
            if d_sq < best_distance_sq:
                best_distance_sq = d_sq
                best_landmark_idx = landmark_idx
                best_H = H
                best_innovation = innovation
                best_S = S

        good_match = (
            best_landmark_idx is not None
            and best_distance_sq < self.mahal_threshold
            and abs(best_innovation[0, 0]) < 0.5
            and abs(best_innovation[1, 0]) < math.radians(20.0)
        )
        no_match = (
            best_landmark_idx is None
            or best_distance_sq >= self.mahal_threshold
        )
        if good_match:
            self.ekf_correct(best_H, best_innovation, best_S)
        elif no_match:
            self.add_to_candidate_buffer(obs)

    def observation_model(self, landmark_idx: int):
        x = self.mu[0, 0]
        y = self.mu[1, 0]
        theta = self.mu[2, 0]
        lm_idx = 3 + 2 * landmark_idx
        mx = self.mu[lm_idx, 0]
        my = self.mu[lm_idx + 1, 0]
        dx = mx - x
        dy = my - y
        q = max(dx**2 + dy**2, 1e-8)
        sqrt_q = math.sqrt(q)
        z_hat = np.array([
            [sqrt_q],
            [wrap_angle(math.atan2(dy, dx) - theta)],
        ])
        state_size = self.mu.shape[0]
        H = np.zeros((2, state_size))
        H[0, 0] = -dx / sqrt_q;  H[0, 1] = -dy / sqrt_q;  H[0, 2] = 0.0
        H[1, 0] =  dy / q;       H[1, 1] = -dx / q;        H[1, 2] = -1.0
        H[0, lm_idx] =  dx / sqrt_q;  H[0, lm_idx + 1] =  dy / sqrt_q
        H[1, lm_idx] = -dy / q;       H[1, lm_idx + 1] =  dx / q
        return z_hat, H

    def ekf_correct(self, H, innovation, S) -> None:
        kalman_gain = self.Sigma @ H.T @ np.linalg.inv(S)
        correction = kalman_gain @ innovation
        if np.linalg.norm(correction[:2]) > 0.05:
            self.get_logger().warn(
                f"Discarding large xy correction ({np.linalg.norm(correction[:2])*100:.1f}cm)!")
            return
        if abs(correction[2]) > math.radians(3.0):
            self.get_logger().warn(
                f"Discarding large theta correction ({math.degrees(correction[2]):.1f}deg)!")
            return

        self.mu = self.mu + correction
        self.mu[2, 0] = wrap_angle(self.mu[2, 0])
        state_size = self.mu.shape[0]
        self.Sigma = (np.eye(state_size) - kalman_gain @ H) @ self.Sigma

    # ================================================================
    # CANDIDATE BUFFER
    # ================================================================

    def add_to_candidate_buffer(self, obs: dict) -> None:
        world_pos = self.observation_to_world_coords(obs)
        for k in range(self.num_landmarks):
            lm_idx = 3 + 2 * k
            lm_pos = self.mu[lm_idx:lm_idx + 2, 0]
            if np.linalg.norm(world_pos - lm_pos) < self.new_landmark_min_dist:
                return
        for candidate in self.candidate_landmarks:
            if np.linalg.norm(world_pos - candidate["position"]) < self.candidate_match_dist:
                n = candidate["seen"]
                new_pos = (candidate["position"] * n + world_pos) / (n + 1)
                drift = np.linalg.norm(new_pos - candidate["position"])
                candidate["variance"] = 0.7 * candidate["variance"] + 0.3 * drift
                candidate["position"] = new_pos
                candidate["seen"] += 1
                STABLE_THRESHOLD = 0.08
                if (candidate["seen"] >= self.candidate_required_seen or
                        (candidate["seen"] >= 3 and candidate["variance"] < STABLE_THRESHOLD)):
                    self.add_new_landmark(candidate["position"])
                    self.candidate_landmarks = [
                        c for c in self.candidate_landmarks if c is not candidate
                    ]
                return
        self.candidate_landmarks.append({
            "position": world_pos,
            "seen": 1,
            "type": obs["type"],
            "variance": 0.0,
            "prev_position": world_pos.copy(),
        })

    def observation_to_world_coords(self, obs: dict) -> np.ndarray:
        r, b = obs["range"], obs["bearing"]
        x = self.mu[0, 0]
        y = self.mu[1, 0]
        theta = self.mu[2, 0]
        world_angle = theta + b
        return np.array([
            x + r * math.cos(world_angle),
            y + r * math.sin(world_angle),
        ])

    def add_new_landmark(self, world_position: np.ndarray) -> None:
        mx, my = float(world_position[0]), float(world_position[1])
        old_size = self.mu.shape[0]
        new_size = old_size + 2
        new_mu = np.zeros((new_size, 1))
        new_mu[:old_size] = self.mu
        new_mu[old_size, 0] = mx
        new_mu[old_size + 1, 0] = my
        new_Sigma = np.zeros((new_size, new_size))
        new_Sigma[:old_size, :old_size] = self.Sigma

        new_Sigma[old_size,     old_size]     = 0.28 ** 2
        new_Sigma[old_size + 1, old_size + 1] = 0.28 ** 2

        self.mu = new_mu
        self.Sigma = new_Sigma
        self.num_landmarks += 1
        self.total_confirmed_corners += 1

    # ================================================================
    # OCCUPANCY GRID
    # ================================================================

    def update_occupancy_grid(self, ranges: np.ndarray, angles: np.ndarray) -> None:
        robot_x = self.mu[0, 0]
        robot_y = self.mu[1, 0]
        robot_theta = self.mu[2, 0]

        robot_cell = self.world_to_grid_cell(robot_x, robot_y)
        if robot_cell is None:
            return
        x0, y0 = robot_cell

        sub_ranges = ranges[::2]   # every 2nd ray — 2° spacing, <0.07 m gap at 2 m range
        sub_angles = angles[::2]   # full [::1] is safer but heavier on the Pi

        with self.map_lock:
            for r, phi in zip(sub_ranges, sub_angles):
                is_hit = True
                if r > self.max_mapping_range:
                    r = self.max_mapping_range
                    is_hit = False
                if r < self.scan_preprocessor.min_range:
                    continue

                global_angle = robot_theta + float(phi)
                end_x = robot_x + r * math.cos(global_angle)
                end_y = robot_y + r * math.sin(global_angle)
                end_cell = self.world_to_grid_cell(end_x, end_y)
                if end_cell is None:
                    continue
                x1, y1 = end_cell

                num_pts = max(abs(x1 - x0), abs(y1 - y0)) + 1
                if num_pts <= 1:
                    continue

                cx_array = np.linspace(x0, x1, num_pts, endpoint=True).astype(np.int32)
                cy_array = np.linspace(y0, y1, num_pts, endpoint=True).astype(np.int32)

                valid = (
                    (cx_array >= 0) & (cx_array < self.map_width) &
                    (cy_array >= 0) & (cy_array < self.map_height)
                )
                cx_array = cx_array[valid]
                cy_array = cy_array[valid]

                if len(cx_array) == 0:
                    continue

                self.log_odds[cy_array[:-1], cx_array[:-1]] = np.clip(
                    self.log_odds[cy_array[:-1], cx_array[:-1]] + self.log_odds_free,
                    self.log_odds_min, self.log_odds_max,
                )
                self.cells_observed[cy_array[:-1], cx_array[:-1]] = True

                if is_hit:
                    tx, ty = cx_array[-1], cy_array[-1]
                    self.log_odds[ty, tx] = np.clip(
                        self.log_odds[ty, tx] + self.log_odds_occ,
                        self.log_odds_min, self.log_odds_max,
                    )
                    self.cells_observed[ty, tx] = True

    def world_to_grid_cell(self, x: float, y: float):
        cx = int((x - self.map_origin_x) / self.map_resolution)
        cy = int((y - self.map_origin_y) / self.map_resolution)
        if not self.cell_in_bounds(cx, cy):
            return None
        return cx, cy

    def cell_in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.map_width and 0 <= cy < self.map_height

    # ================================================================
    # PUBLISHERS
    # ================================================================

    def publish_pose(self) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        x = self.mu[0, 0]
        y = self.mu[1, 0]
        theta = self.mu[2, 0]
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(theta / 2.0)
        cov = np.zeros(36)
        cov[0]  = self.Sigma[0, 0];  cov[1]  = self.Sigma[0, 1];  cov[5]  = self.Sigma[0, 2]
        cov[6]  = self.Sigma[1, 0];  cov[7]  = self.Sigma[1, 1];  cov[11] = self.Sigma[1, 2]
        cov[30] = self.Sigma[2, 0];  cov[31] = self.Sigma[2, 1];  cov[35] = self.Sigma[2, 2]
        msg.pose.covariance = cov.tolist()
        self.pose_pub.publish(msg)

    def publish_landmarks(self) -> None:
        if self.num_landmarks == 0:
            return
        marker_array = MarkerArray()
        for k in range(self.num_landmarks):
            lm_idx = 3 + 2 * k
            mx = self.mu[lm_idx, 0]
            my = self.mu[lm_idx + 1, 0]
            m = Marker()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = "map"
            m.ns = "ekf_landmarks"
            m.id = k
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(mx)
            m.pose.position.y = float(my)
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.20
            m.color.r = 1.0;  m.color.g = 0.0
            m.color.b = 0.0;  m.color.a = 1.0
            marker_array.markers.append(m)
        self.landmark_pub.publish(marker_array)

    def publish_map_step(self) -> None:
        try:
            with self._scan_lock:
                ranges = self._latest_ranges
                angles = self._latest_angles

            # Detect spinning in place directly from odometry — no bridge lag.
            vel_data = self.odom.get_velocity()
            if vel_data is not None:
                vx, vy, omega = vel_data["velocity"]
                speed = math.hypot(float(vx), float(vy))
                spinning = (abs(float(omega)) > self._rotate_wz_thresh and
                            speed < self._rotate_vx_thresh)
                if spinning != self._rotating:
                    self._rotating = spinning
                    self.get_logger().info(
                        f'SLAM Grid Mapping: {"PAUSED (spinning)" if spinning else "ACTIVE"}'
                    )

            if ranges is not None and not self._rotating:
                self.update_occupancy_grid(ranges, angles)

            self.publish_map()

        except Exception:
            self.get_logger().error(
                "publish_map_step crashed:\n" + traceback.format_exc()
            )

    def publish_map(self) -> None:
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.info.origin.orientation.w = 1.0

        with self.map_lock:
            log_odds_safe = np.clip(self.log_odds, -10.0, 10.0).astype(np.float32)
            cells_obs = self.cells_observed.copy()

        prob_occupied = 1.0 - 1.0 / (1.0 + np.exp(log_odds_safe))
        grid_int = np.clip(
            np.round(100.0 * prob_occupied), 0, 100
        ).astype(np.int8)
        grid_int[~cells_obs] = -1

        msg.data = grid_int.flatten().tolist()
        self.map_pub.publish(msg)

    # ================================================================
    # PUBLIC ACCESSORS
    # ================================================================

    @property
    def pose(self) -> np.ndarray:
        return self.mu[:3, 0].copy()

    @property
    def occupancy_grid(self) -> np.ndarray:
        with self.map_lock:
            return self.log_odds.copy()

    @property
    def known_cells(self) -> np.ndarray:
        with self.map_lock:
            return self.cells_observed.copy()