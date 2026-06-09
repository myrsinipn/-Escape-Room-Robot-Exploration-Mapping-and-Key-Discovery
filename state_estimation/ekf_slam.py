#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class EKFLidarSLAM(Node):
    """
    EKF-SLAM από την αρχή με LiDAR landmarks.

    State:
        mu = [x, y, theta, m1x, m1y, m2x, m2y, ...]^T

    Prediction:
        από /odom, με vx, vy, wz

    Correction:
        από LiDAR landmarks:
            - jump points / depth discontinuities
            - local corner-like points

    Mapping:
        occupancy grid από τα LiDAR rays
    """

    def __init__(self):
        super().__init__('ekf_lidar_slam')

        # --------------------------------------------------
        # Topics
        # --------------------------------------------------
        self.scan_topic = '/scan'
        self.odom_topic = '/odom'

        self.pose_topic = '/slam_pose'
        self.map_topic = '/slam_map'
        self.landmark_topic = '/slam_landmarks'

        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 30)

        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            self.pose_topic,
            10
        )

        self.map_pub = self.create_publisher(
            OccupancyGrid,
            self.map_topic,
            1
        )

        self.landmark_pub = self.create_publisher(
            MarkerArray,
            self.landmark_topic,
            1
        )

        # --------------------------------------------------
        # EKF state
        # --------------------------------------------------
        self.mu = np.zeros((3, 1))  # [x, y, theta]
        self.Sigma = np.diag([0.02, 0.02, np.deg2rad(2.0)]) ** 2

        self.num_landmarks = 0

        # Motion noise R_t
        self.motion_noise = np.diag([
            0.03,               # x noise
            0.03,               # y noise
            np.deg2rad(2.0)     # theta noise
        ]) ** 2

        # Observation noise Q_t for z = [range, bearing]
        self.obs_noise = np.diag([
            0.10,               # range noise in meters
            np.deg2rad(5.0)     # bearing noise
        ]) ** 2

        # --------------------------------------------------
        # Data association
        # --------------------------------------------------
        self.mahalanobis_threshold = 5.99
        # 5.99 περίπου 95% confidence για 2D measurement

        self.new_landmark_min_distance = 0.30
        self.max_landmark_range = 3.50
        self.min_landmark_range = 0.25

        # Δεν προσθέτουμε αμέσως κάθε feature ως landmark.
        # Πρώτα το κρατάμε σαν candidate.
        self.candidate_landmarks = []
        self.candidate_match_distance = 0.35
        self.candidate_required_seen = 2

        # --------------------------------------------------
        # LiDAR feature extraction parameters
        # --------------------------------------------------
        self.jump_threshold = 0.35
        self.min_jump_range = 0.25
        self.max_jump_range = 3.50

        self.corner_angle_min_deg = 40.0
        self.corner_angle_max_deg = 140.0
        self.corner_stride = 4
        self.corner_min_range = 0.30
        self.corner_max_range = 3.00

        self.min_feature_separation = 0.25

        # --------------------------------------------------
        # Occupancy grid
        # --------------------------------------------------
        self.map_resolution = 0.05      # 5 cm / cell
        self.map_size_m = 12.0          # 12m x 12m
        self.map_width = int(self.map_size_m / self.map_resolution)
        self.map_height = int(self.map_size_m / self.map_resolution)

        self.map_origin_x = -self.map_size_m / 2.0
        self.map_origin_y = -self.map_size_m / 2.0

        self.log_odds = np.zeros((self.map_height, self.map_width), dtype=np.float32)
        self.known = np.zeros((self.map_height, self.map_width), dtype=bool)

        self.log_free = -0.35
        self.log_occ = 0.85
        self.log_min = -5.0
        self.log_max = 5.0

        self.max_mapping_range = 4.0

        # --------------------------------------------------
        # Time
        # --------------------------------------------------
        self.prev_odom_time = None

        self.get_logger().info('EKF LiDAR SLAM started.')
        self.get_logger().info(f'Subscribing: {self.odom_topic}, {self.scan_topic}')
        self.get_logger().info(f'Publishing: {self.pose_topic}, {self.map_topic}, {self.landmark_topic}')

    # ======================================================
    # Prediction step
    # ======================================================

    def odom_callback(self, msg: Odometry):
        current_time = stamp_to_sec(msg.header.stamp)

        if self.prev_odom_time is None:
            self.prev_odom_time = current_time
            return

        dt = current_time - self.prev_odom_time
        self.prev_odom_time = current_time

        if dt <= 0.0 or dt > 1.0:
            return

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        wz = msg.twist.twist.angular.z

        self.predict(vx, vy, wz, dt)
        self.publish_pose()

    def predict(self, vx, vy, wz, dt):
        """
        Prediction:
            x_new = x + (vx cosθ - vy sinθ) dt
            y_new = y + (vx sinθ + vy cosθ) dt
            θ_new = θ + wz dt

        Τα landmarks δεν αλλάζουν στο prediction.
        """

        x = self.mu[0, 0]
        y = self.mu[1, 0]
        theta = self.mu[2, 0]

        dx = (vx * math.cos(theta) - vy * math.sin(theta)) * dt
        dy = (vx * math.sin(theta) + vy * math.cos(theta)) * dt
        dtheta = wz * dt

        self.mu[0, 0] = x + dx
        self.mu[1, 0] = y + dy
        self.mu[2, 0] = normalize_angle(theta + dtheta)

        n = self.mu.shape[0]
        F = np.eye(n)

        # Jacobian ως προς θ
        F[0, 2] = (-vx * math.sin(theta) - vy * math.cos(theta)) * dt
        F[1, 2] = ( vx * math.cos(theta) - vy * math.sin(theta)) * dt

        R_big = np.zeros((n, n))
        R_big[0:3, 0:3] = self.motion_noise

        self.Sigma = F @ self.Sigma @ F.T + R_big

    # ======================================================
    # LiDAR scan callback
    # ======================================================

    def scan_callback(self, msg: LaserScan):
        points, ranges, angles = self.scan_to_points(msg)

        if len(points) < 10:
            return

        observations = self.extract_lidar_landmarks(points, ranges, angles)

        for obs in observations:
            self.process_observation(obs)

        self.update_occupancy_grid(msg)
        self.publish_pose()
        self.publish_landmarks()
        self.publish_map()

    # ======================================================
    # LiDAR preprocessing
    # ======================================================

    def scan_to_points(self, scan: LaserScan):
        points = []
        ranges = []
        angles = []

        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r):
                continue

            if r < scan.range_min or r > scan.range_max:
                continue

            angle = scan.angle_min + i * scan.angle_increment

            x = r * math.cos(angle)
            y = r * math.sin(angle)

            points.append([x, y])
            ranges.append(r)
            angles.append(angle)

        return np.array(points), np.array(ranges), np.array(angles)

    # ======================================================
    # Landmark extraction from LiDAR
    # ======================================================

    def extract_lidar_landmarks(self, points, ranges, angles):
        """
        Εξάγει candidate landmarks από LiDAR.

        Χρησιμοποιεί:
            1. jump points:
                απότομη αλλαγή απόστασης |r_i - r_{i-1}|

            2. local corner-like points:
                τοπική αλλαγή κατεύθυνσης στο point cloud

        Επιστρέφει λίστα από observations:
            obs = {
                "range": r,
                "bearing": phi,
                "type": "jump" ή "corner"
            }
        """

        observations = []

        # -----------------------------
        # A. Jump landmarks
        # -----------------------------
        for i in range(1, len(ranges)):
            r_prev = ranges[i - 1]
            r_curr = ranges[i]

            if abs(r_curr - r_prev) > self.jump_threshold:
                # Παίρνουμε το πιο κοντινό από τα δύο σημεία ως edge point
                idx = i if r_curr < r_prev else i - 1

                r = ranges[idx]
                phi = angles[idx]

                if self.min_jump_range < r < self.max_jump_range:
                    observations.append({
                        "range": float(r),
                        "bearing": float(normalize_angle(phi)),
                        "type": "jump"
                    })

        # -----------------------------
        # B. Corner-like landmarks
        # -----------------------------
        s = self.corner_stride

        for i in range(s, len(points) - s):
            p_prev = points[i - s]
            p = points[i]
            p_next = points[i + s]

            v1 = p_prev - p
            v2 = p_next - p

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 < 1e-6 or norm2 < 1e-6:
                continue

            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)

            angle_deg = math.degrees(math.acos(cos_angle))

            r = ranges[i]
            phi = angles[i]

            if self.corner_min_range < r < self.corner_max_range:
                if self.corner_angle_min_deg < angle_deg < self.corner_angle_max_deg:
                    observations.append({
                        "range": float(r),
                        "bearing": float(normalize_angle(phi)),
                        "type": "corner"
                    })

        observations = self.filter_duplicate_observations(observations)

        return observations

    def filter_duplicate_observations(self, observations):
        """
        Αφαιρεί landmarks που είναι σχεδόν στο ίδιο σημείο στο robot frame.
        """

        filtered = []

        for obs in observations:
            r = obs["range"]
            b = obs["bearing"]

            px = r * math.cos(b)
            py = r * math.sin(b)

            too_close = False

            for kept in filtered:
                kr = kept["range"]
                kb = kept["bearing"]

                kx = kr * math.cos(kb)
                ky = kr * math.sin(kb)

                if math.hypot(px - kx, py - ky) < self.min_feature_separation:
                    too_close = True
                    break

            if not too_close:
                filtered.append(obs)

        return filtered

    # ======================================================
    # EKF correction and data association
    # ======================================================

    def process_observation(self, obs):
        z = np.array([
            [obs["range"]],
            [obs["bearing"]]
        ])

        if z[0, 0] < self.min_landmark_range or z[0, 0] > self.max_landmark_range:
            return

        if self.num_landmarks == 0:
            self.handle_new_candidate(obs)
            return

        best_idx = None
        best_d2 = float('inf')
        best_H = None
        best_innovation = None
        best_S = None

        for landmark_idx in range(self.num_landmarks):
            z_hat, H = self.expected_observation_and_jacobian(landmark_idx)

            innovation = z - z_hat
            innovation[1, 0] = normalize_angle(innovation[1, 0])

            S = H @ self.Sigma @ H.T + self.obs_noise

            try:
                d2 = float(innovation.T @ np.linalg.inv(S) @ innovation)
            except np.linalg.LinAlgError:
                continue

            if d2 < best_d2:
                best_d2 = d2
                best_idx = landmark_idx
                best_H = H
                best_innovation = innovation
                best_S = S

        if best_idx is not None and best_d2 < self.mahalanobis_threshold:
            self.correct(best_H, best_innovation, best_S)
        else:
            self.handle_new_candidate(obs)

    def expected_observation_and_jacobian(self, landmark_idx):
        """
        Observation model:
            z = h(mu) = [range, bearing]

        Για landmark m_i = [mx, my]:

            dx = mx - x
            dy = my - y

            range = sqrt(dx^2 + dy^2)
            bearing = atan2(dy, dx) - theta
        """

        x = self.mu[0, 0]
        y = self.mu[1, 0]
        theta = self.mu[2, 0]

        lm_start = 3 + 2 * landmark_idx

        mx = self.mu[lm_start, 0]
        my = self.mu[lm_start + 1, 0]

        dx = mx - x
        dy = my - y

        q = dx ** 2 + dy ** 2

        if q < 1e-8:
            q = 1e-8

        sqrt_q = math.sqrt(q)

        z_hat = np.array([
            [sqrt_q],
            [normalize_angle(math.atan2(dy, dx) - theta)]
        ])

        n = self.mu.shape[0]
        H = np.zeros((2, n))

        # Derivatives ως προς robot pose x, y, theta
        H[0, 0] = -dx / sqrt_q
        H[0, 1] = -dy / sqrt_q
        H[0, 2] = 0.0

        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1.0

        # Derivatives ως προς landmark mx, my
        H[0, lm_start] = dx / sqrt_q
        H[0, lm_start + 1] = dy / sqrt_q

        H[1, lm_start] = -dy / q
        H[1, lm_start + 1] = dx / q

        return z_hat, H

    def correct(self, H, innovation, S):
        """
        EKF correction:
            K = Sigma H^T (H Sigma H^T + Q)^-1
            mu = mu + K innovation
            Sigma = (I - K H) Sigma
        """

        K = self.Sigma @ H.T @ np.linalg.inv(S)

        self.mu = self.mu + K @ innovation
        self.mu[2, 0] = normalize_angle(self.mu[2, 0])

        n = self.mu.shape[0]
        I = np.eye(n)

        self.Sigma = (I - K @ H) @ self.Sigma

    # ======================================================
    # New landmark handling
    # ======================================================

    def handle_new_candidate(self, obs):
        """
        Δεν προσθέτουμε αμέσως κάθε παρατήρηση στο EKF state.
        Την κρατάμε ως candidate και αν την ξαναδούμε κοντά,
        τότε τη βάζουμε ως πραγματικό landmark.
        """

        pos_world = self.observation_to_world(obs)

        # Αν είναι πολύ κοντά σε υπάρχον landmark, μην το προσθέσεις
        for k in range(self.num_landmarks):
            lm_start = 3 + 2 * k
            lm_pos = self.mu[lm_start:lm_start + 2, 0]

            if np.linalg.norm(pos_world - lm_pos) < self.new_landmark_min_distance:
                return

        # Match με ήδη provisional candidate
        for cand in self.candidate_landmarks:
            if np.linalg.norm(pos_world - cand["position"]) < self.candidate_match_distance:
                cand["position"] = 0.5 * cand["position"] + 0.5 * pos_world
                cand["seen"] += 1

                if cand["seen"] >= self.candidate_required_seen:
                    self.add_landmark(cand["position"])
                    self.candidate_landmarks.remove(cand)

                return

        # Νέο candidate
        self.candidate_landmarks.append({
            "position": pos_world,
            "seen": 1,
            "type": obs["type"]
        })

    def observation_to_world(self, obs):
        """
        Μετατρέπει observation [range, bearing] από robot frame σε world frame.
        """

        r = obs["range"]
        b = obs["bearing"]

        x = self.mu[0, 0]
        y = self.mu[1, 0]
        theta = self.mu[2, 0]

        global_angle = theta + b

        mx = x + r * math.cos(global_angle)
        my = y + r * math.sin(global_angle)

        return np.array([mx, my])

    def add_landmark(self, position):
        """
        Προσθήκη νέου landmark στο EKF state.
        """

        mx, my = position

        old_n = self.mu.shape[0]
        new_n = old_n + 2

        new_mu = np.zeros((new_n, 1))
        new_mu[:old_n, :] = self.mu
        new_mu[old_n, 0] = mx
        new_mu[old_n + 1, 0] = my

        new_Sigma = np.zeros((new_n, new_n))
        new_Sigma[:old_n, :old_n] = self.Sigma

        # Αρχική αβεβαιότητα νέου landmark
        new_Sigma[old_n, old_n] = 0.50 ** 2
        new_Sigma[old_n + 1, old_n + 1] = 0.50 ** 2

        self.mu = new_mu
        self.Sigma = new_Sigma
        self.num_landmarks += 1

        self.get_logger().info(
            f'Added landmark #{self.num_landmarks}: ({mx:.2f}, {my:.2f})'
        )

    # ======================================================
    # Occupancy grid mapping
    # ======================================================

    def update_occupancy_grid(self, scan: LaserScan):
        """
        Απλό inverse sensor model:
            cells πάνω στην ακτίνα μέχρι το hit -> free
            τελικό cell -> occupied
        """

        robot_x = self.mu[0, 0]
        robot_y = self.mu[1, 0]
        robot_theta = self.mu[2, 0]

        robot_cell = self.world_to_cell(robot_x, robot_y)

        if robot_cell is None:
            return

        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r):
                continue

            if r < scan.range_min:
                continue

            hit = True

            if r > self.max_mapping_range:
                r = self.max_mapping_range
                hit = False

            angle = scan.angle_min + i * scan.angle_increment
            global_angle = robot_theta + angle

            end_x = robot_x + r * math.cos(global_angle)
            end_y = robot_y + r * math.sin(global_angle)

            end_cell = self.world_to_cell(end_x, end_y)

            if end_cell is None:
                continue

            cells = self.bresenham(robot_cell[0], robot_cell[1], end_cell[0], end_cell[1])

            if len(cells) == 0:
                continue

            # Free cells, εκτός από το τελευταίο
            for cx, cy in cells[:-1]:
                if self.valid_cell(cx, cy):
                    self.log_odds[cy, cx] += self.log_free
                    self.log_odds[cy, cx] = np.clip(
                        self.log_odds[cy, cx],
                        self.log_min,
                        self.log_max
                    )
                    self.known[cy, cx] = True

            # Occupied τελικό cell
            if hit:
                cx, cy = cells[-1]
                if self.valid_cell(cx, cy):
                    self.log_odds[cy, cx] += self.log_occ
                    self.log_odds[cy, cx] = np.clip(
                        self.log_odds[cy, cx],
                        self.log_min,
                        self.log_max
                    )
                    self.known[cy, cx] = True

    def world_to_cell(self, x, y):
        cx = int((x - self.map_origin_x) / self.map_resolution)
        cy = int((y - self.map_origin_y) / self.map_resolution)

        if not self.valid_cell(cx, cy):
            return None

        return cx, cy

    def valid_cell(self, cx, cy):
        return 0 <= cx < self.map_width and 0 <= cy < self.map_height

    def bresenham(self, x0, y0, x1, y1):
        """
        Bresenham line algorithm για grid ray tracing.
        """

        cells = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy

        x = x0
        y = y0

        while True:
            cells.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x += sx

            if e2 < dx:
                err += dx
                y += sy

        return cells

    # ======================================================
    # Publishers
    # ======================================================

    def publish_pose(self):
        msg = PoseWithCovarianceStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        x = self.mu[0, 0]
        y = self.mu[1, 0]
        theta = self.mu[2, 0]

        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0

        msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(theta / 2.0)

        cov = np.zeros(36)

        cov[0] = self.Sigma[0, 0]
        cov[1] = self.Sigma[0, 1]
        cov[5] = self.Sigma[0, 2]

        cov[6] = self.Sigma[1, 0]
        cov[7] = self.Sigma[1, 1]
        cov[11] = self.Sigma[1, 2]

        cov[30] = self.Sigma[2, 0]
        cov[31] = self.Sigma[2, 1]
        cov[35] = self.Sigma[2, 2]

        msg.pose.covariance = cov.tolist()

        self.pose_pub.publish(msg)

    def publish_landmarks(self):
        marker_array = MarkerArray()

        # Clear old markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        for k in range(self.num_landmarks):
            lm_start = 3 + 2 * k

            mx = self.mu[lm_start, 0]
            my = self.mu[lm_start + 1, 0]

            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'map'

            marker.ns = 'ekf_landmarks'
            marker.id = k
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(mx)
            marker.pose.position.y = float(my)
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.12
            marker.scale.y = 0.12
            marker.scale.z = 0.12

            marker.color.r = 1.0
            marker.color.g = 0.1
            marker.color.b = 0.1
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.landmark_pub.publish(marker_array)

    def publish_map(self):
        msg = OccupancyGrid()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height

        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        data = []

        for y in range(self.map_height):
            for x in range(self.map_width):
                if not self.known[y, x]:
                    data.append(-1)
                else:
                    p = 1.0 - 1.0 / (1.0 + math.exp(self.log_odds[y, x]))
                    occ = int(round(100.0 * p))
                    occ = max(0, min(100, occ))
                    data.append(occ)

        msg.data = data

        self.map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = EKFLidarSLAM()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()