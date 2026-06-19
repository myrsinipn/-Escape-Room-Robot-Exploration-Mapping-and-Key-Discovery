#!/usr/bin/env python3
"""
EKF-SLAM node — readable version.

Called from main.py which injects the sensor/perception objects and
adds this node to the shared MultiThreadedExecutor.

Algorithm structure
-------------------
Every 0.1 s the timer fires and runs two steps in order:

  1. PREDICTION  — uses the latest odometry velocity + OmniMotionModel
                   to propagate the robot pose forward in time.

  2. CORRECTION  — uses the latest LiDAR scan (cleaned by ScanPreprocessor)
                   to extract geometric landmarks and update the full EKF state.

State vector
------------
    mu = [ x, y, theta,  m1x, m1y,  m2x, m2y,  ... ]^T

    The first 3 elements are the robot pose.
    Every pair after that is one landmark (world-frame x, y).

Covariance matrix
-----------------
    Sigma is (3+2N) x (3+2N) where N = number of confirmed landmarks.
    It grows as new landmarks are added.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray

from sensors.lidar    import LidarSensor
from sensors.odometry import OdometrySensor
from perception.scan_preprocessor import ScanPreprocessor
from perception.motion_model       import OmniMotionModel

import traceback


def wrap_angle(angle: float) -> float:
    """Keep an angle inside (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class EKFLidarSLAM(Node):
    """
    EKF-SLAM node.

    Receives sensor objects from main.py (dependency injection).
    Does not create its own subscribers — it polls the sensor objects
    each timer tick instead, which keeps all ROS plumbing in one place.
    """

    def __init__(
        self,
        lidar: LidarSensor,
        odom: OdometrySensor,
        scan_preprocessor: ScanPreprocessor,
        motion_model: OmniMotionModel,
    ) -> None:
        super().__init__("ekf_lidar_slam")

        # ----------------------------------------------------------------
        # Injected collaborators
        # ----------------------------------------------------------------
        self.lidar            = lidar
        self.odom             = odom
        self.scan_preprocessor = scan_preprocessor
        self.motion_model     = motion_model
        # Debug counters
        self.total_corner_detections = 0
        self.total_confirmed_corners = 0

        # ----------------------------------------------------------------
        # EKF state
        # ----------------------------------------------------------------

        # mu: column vector, starts as [x=0, y=0, theta=0]
        # grows by 2 each time a new landmark is confirmed
        # Αλλαγή: αρχικοποίηση mu από odom
        #self.mu_initialized = False  # προσθήκη flag
        self.mu = np.zeros((3, 1))
        odom_pose = self.odom.get_pose()        # <-- ΠΡΟΣΑΡΜΟΓΗ στο δικό σου API
        if odom_pose is not None:
            self.mu[0, 0] = float(odom_pose["pose"][0])
            self.mu[1, 0] = float(odom_pose["pose"][1])
            self.mu[2, 0] = float(odom_pose["pose"][2])

        self.num_landmarks = 0

        # Sigma: initial robot pose uncertainty (small — we start at origin)
        self.Sigma = np.diag([
            0.02,                   # x std = 2 cm
            0.02,                   # y std = 2 cm
            math.radians(2.0),      # theta std = 2 deg
        ]) ** 2

        # ----------------------------------------------------------------
        # Noise matrices (tuning knobs)
        # ----------------------------------------------------------------

        # R — motion noise added to the robot pose block each prediction step
        self.motion_noise = np.diag([
            0.03,                   # x noise std = 3 cm
            0.03,                   # y noise std = 3 cm
            math.radians(2.0),      # theta noise std = 2 deg
        ]) ** 2

        # Q — observation noise for a single measurement z = [range, bearing]
        self.obs_noise = np.diag([
            0.10,                   # range noise std = 10 cm
            math.radians(5.0),      # bearing noise std = 5 deg
        ]) ** 2

        # ----------------------------------------------------------------
        # Data association parameters
        # ----------------------------------------------------------------

        # Mahalanobis distance threshold for matching an observation to a
        # known landmark.  5.99 = 95% confidence gate for a 2D measurement.
        self.mahal_threshold = 5.99

        # Don't add a new landmark if one already exists closer than this
        self.new_landmark_min_dist = 0.80   # metres

        # Ignore observations outside this range window
        self.min_landmark_range = 0.25      # metres
        self.max_landmark_range = 3.50      # metres

        # ----------------------------------------------------------------
        # Candidate landmark buffer
        #
        # We don't trust a single sighting.  A feature must be seen at
        # least `candidate_required_seen` times within
        # `candidate_match_dist` metres before it enters the EKF state.
        # ----------------------------------------------------------------
        self.candidate_landmarks    = []    # list of dicts
        self.candidate_match_dist   = 0.5  # metres — same feature?
        self.candidate_required_seen = 5    # how many sightings needed

        # ----------------------------------------------------------------
        # LiDAR feature extraction parameters
        # ----------------------------------------------------------------

        # Jump landmarks: sudden depth discontinuity between adjacent rays
        self.jump_threshold  = 0.35   # metres
        self.min_jump_range  = 0.25
        self.max_jump_range  = 3.50

        # Corner landmarks: local change of direction in the point cloud
        self.corner_angle_min_deg = 85.0
        self.corner_angle_max_deg = 95.0
        self.corner_stride        = 8     # rays to look ahead/behind
        self.corner_min_range     = 0.30
        self.corner_max_range     = 3.00

        # Minimum pixel separation between two kept features (robot frame)
        self.min_feature_separation = 0.70  # metres

        # ----------------------------------------------------------------
        # Occupancy grid parameters
        # ----------------------------------------------------------------
        self.map_resolution = 0.05       # 5 cm per cell
        self.map_size_m     = 12.0       # covers 12 m x 12 m
        self.map_width      = int(self.map_size_m / self.map_resolution)   # 240
        self.map_height     = int(self.map_size_m / self.map_resolution)   # 240

        # World coordinate of cell (0, 0)
        self.map_origin_x = -self.map_size_m / 2.0   # -6.0 m
        self.map_origin_y = -self.map_size_m / 2.0   # -6.0 m

        # Log-odds grid: negative = free, positive = occupied, 0 = unknown
        self.log_odds = np.zeros((self.map_height, self.map_width), dtype=np.float32)

        # Track which cells have ever been updated (for the -1 / unknown output)
        self.cells_observed = np.zeros((self.map_height, self.map_width), dtype=bool)

        # Log-odds increments per ray update
        self.log_odds_free = -0.35   # a free ray vote
        self.log_odds_occ  =  0.85   # an occupied endpoint vote
        self.log_odds_min  = -5.0    # clamp — fully free
        self.log_odds_max  =  5.0    # clamp — fully occupied

        # Don't trace rays beyond this distance (performance + reliability)
        self.max_mapping_range = 4.0  # metres

        # ----------------------------------------------------------------
        # Timing
        # ----------------------------------------------------------------
        self.prev_odom_timestamp = None   # wall-clock float, set on first tick

        # ----------------------------------------------------------------
        # ROS publishers
        # ----------------------------------------------------------------
        self.pose_pub     = self.create_publisher(PoseWithCovarianceStamped, "/slam_pose", 10)
        self.map_pub      = self.create_publisher(OccupancyGrid,             "/slam_map",  1)
        self.landmark_pub = self.create_publisher(MarkerArray,               "/slam_landmarks", 1)

        # ----------------------------------------------------------------
        # Main timer — drives the predict → correct loop at 10 Hz
        # ----------------------------------------------------------------
        # Fast pose publishing for smooth RViz motion
        # 0.05 s = 20 Hz
        self.pose_timer_period = 0.05

        # Slower EKF correction because LiDAR + landmark + map update is expensive
        # 0.20 s = 5 Hz. If the Raspberry can handle it, try 0.10 for 10 Hz.
        self.correction_timer_period = 0.20

        # Map publishing throttle.
        # correction at 5 Hz and _map_every = 3 gives about 1.7 Hz map publishing.
        self._map_tick = 0
        self._map_every = 3

        self.create_timer(self.pose_timer_period, self.prediction_and_publish_step)
        self.create_timer(self.correction_timer_period, self.correction_step)

        self.get_logger().info("EKF LiDAR SLAM ready.")

    # ====================================================================
    # TOP-LEVEL UPDATE  (called every 0.1 s by the timer)
    # ====================================================================

    #def update(self) -> None:
    #   self.prediction_step()
    #   self.correction_step()
    #   self.publish_pose()
    #   self._map_tick += 1

    def prediction_and_publish_step(self) -> None:
        """
        Fast loop for smooth RViz motion.

        Runs prediction and publishes /slam_pose frequently,
        without waiting for the expensive LiDAR correction/map update.
        """
        try:
            self.prediction_step()
            self.publish_pose()
        except Exception:
            self.get_logger().error("prediction crashed:\n" + traceback.format_exc())

    # ====================================================================
    # STEP 1 — PREDICTION
    #
    # Moves the robot pose estimate forward using odometry velocity.
    # Landmarks are not touched — they don't move.
    #
    # mu_pred = f(mu, u)           — nonlinear motion model
    # Sigma_pred = F Sigma F^T + R — covariance propagation
    # ====================================================================

    def prediction_step(self) -> None:

        vel_data = self.odom.get_velocity()
        if vel_data is None:
            return   # no odometry yet


        # Use node wall time so prediction can run smoothly at the timer rate,
        # even if odometry messages arrive less frequently.
        current_time = self.get_clock().now().nanoseconds * 1e-9


        vx, vy, omega = vel_data["velocity"]

        # -- State update --
        # OmniMotionModel computes the new pose and the 3x3 Jacobian F
        # using the omni-drive kinematics:
        #   x_new     = x + (vx cos(theta) - vy sin(theta)) * dt
        #   y_new     = y + (vx sin(theta) + vy cos(theta)) * dt
        #   theta_new = theta + omega * dt
        robot_pose_before = self.mu[:3, 0].copy()

        new_robot_pose, F_robot = self.motion_model.predict(
            robot_pose_before,
            float(vx), float(vy), float(omega), dt,
            velocities_in_body_frame=True,
        )

        # Write the predicted pose back into the state vector
        self.mu[0, 0] = new_robot_pose[0]   # x
        self.mu[1, 0] = new_robot_pose[1]   # y
        self.mu[2, 0] = new_robot_pose[2]   # theta

        # -- Covariance update --
        # The full Jacobian F_full is (3+2N) x (3+2N).
        # Only the 3x3 robot-pose block is non-identity; landmarks don't move.
        state_size = self.mu.shape[0]

        F_full = np.eye(state_size)
        F_full[0:3, 0:3] = F_robot          # robot pose block

        # Motion noise R only affects the robot pose block
        R_full = np.zeros((state_size, state_size))
        R_full[0:3, 0:3] = self.motion_noise

        # EKF covariance prediction:  Sigma = F Sigma F^T + R
        self.Sigma = F_full @ self.Sigma @ F_full.T + R_full


    # ====================================================================
    # STEP 2 — CORRECTION
    #
    # For each observed landmark:
    #   1. Try to match it to a known landmark (data association)
    #   2. If matched: run EKF update  (mu, Sigma corrected)
    #   3. If no match: add to candidate buffer; promote if seen enough
    #
    # After landmark updates, refresh the occupancy grid.
    # ====================================================================

    def correction_step(self) -> None:
        try:
            raw_scan = self.lidar.get_scan()
            if raw_scan is None:
                return   # no scan yet

            # Clean and smooth the scan via ScanPreprocessor
            processed = self.scan_preprocessor.preprocess(raw_scan)
            ranges = processed["ranges"]
            angles = processed["angles"]

            # Convert to Cartesian points (needed for corner detection)
            points = self.scan_preprocessor.polar_to_cartesian(ranges, angles)

            if len(points) < 10:
                return   # scan too sparse to be useful

            # Extract geometric features from the scan
            observations = self.extract_lidar_landmarks(points, ranges, angles)

            # Run EKF correction for each observation
            for obs in observations:
                self.process_observation(obs)

            # Update the occupancy grid from the same scan
            self.update_occupancy_grid(ranges, angles)

            # Always publish landmarks at correction rate
            self.publish_landmarks()

            # Publish map less often because it is expensive
            self._map_tick += 1
            if self._map_tick % self._map_every == 0:
                self.publish_map()
        except Exception:
            self.get_logger().error("correction crashed:\n" + traceback.format_exc())


    # ====================================================================
    # LIDAR FEATURE EXTRACTION
    #
    # Produces a list of observations:
    #   { "range": float, "bearing": float (rad), "type": "jump"|"corner" }
    #
    # Two feature types:
    #   A) Jump points — large depth discontinuity between adjacent rays
    #   B) Corner-like points — local curvature change in the point cloud
    # ====================================================================

    def extract_lidar_landmarks(self, points, ranges, angles) -> list:
        observations = []

        # # ---- A) Jump points ----------------------------------------
        # # When |r[i] - r[i-1]| > threshold, the nearer endpoint is a
        # # geometric edge (e.g. corner of a wall or box).
        # for i in range(1, len(ranges)):
        #     depth_jump = abs(ranges[i] - ranges[i - 1])
        #     if depth_jump > self.jump_threshold:
        #         # Take the closer point as the feature
        #         idx = i if ranges[i] < ranges[i - 1] else i - 1
        #         r   = float(ranges[idx])
        #         phi = float(angles[idx])
        #         if self.min_jump_range < r < self.max_jump_range:
        #             observations.append({
        #                 "range":   r,
        #                 "bearing": wrap_angle(phi),
        #                 "type":    "jump",
        #             })

        # ---- B) Corner-like points ----------------------------------
        # At point i, look at the vectors to the points stride steps
        # before and after.  If the angle between those two vectors falls
        # in the expected range, point i sits at a concave or convex corner.
        stride = self.corner_stride
        for i in range(stride, len(points) - stride):
            vec_before = points[i - stride] - points[i]   # vector to earlier point
            vec_after  = points[i + stride] - points[i]   # vector to later point

            len_before = np.linalg.norm(vec_before)
            len_after  = np.linalg.norm(vec_after)
            if len_before < 1e-6 or len_after < 1e-6:
                continue

            # Angle between the two vectors
            cos_angle = np.dot(vec_before, vec_after) / (len_before * len_after)
            cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            angle_deg = math.degrees(math.acos(cos_angle))

            r   = float(ranges[i])
            phi = float(angles[i])

            if (self.corner_min_range < r < self.corner_max_range and
                    self.corner_angle_min_deg < angle_deg < self.corner_angle_max_deg):

                self.total_corner_detections += 1

                #self.get_logger().info(
                #   f"CORNER DETECTED | "
                #    f"range={r:.2f} m | "
                #     f"bearing={math.degrees(phi):.1f} deg | "
                #     f"angle={angle_deg:.1f} deg"
                # )

                observations.append({
                    "range": r,
                    "bearing": wrap_angle(phi),
                    "type": "corner",
                })

        # Remove features that are too close to each other in robot frame
        observations = self.remove_duplicate_observations(observations)
        return observations

    def remove_duplicate_observations(self, observations: list) -> list:
        """
        If two observations land within `min_feature_separation` of each
        other in the robot frame, keep only the first one.
        """
        kept = []
        for obs in observations:
            # Convert polar -> Cartesian in robot frame
            ox = obs["range"] * math.cos(obs["bearing"])
            oy = obs["range"] * math.sin(obs["bearing"])

            too_close = False
            for kept_obs in kept:
                kx = kept_obs["range"] * math.cos(kept_obs["bearing"])
                ky = kept_obs["range"] * math.sin(kept_obs["bearing"])
                if math.hypot(ox - kx, oy - ky) < self.min_feature_separation:
                    too_close = True
                    break

            if not too_close:
                kept.append(obs)
        return kept

    # ====================================================================
    # EKF CORRECTION — data association + update
    #
    # For one observation z = [range, bearing]:
    #
    #   1. Compute expected observation z_hat and Jacobian H for every
    #      known landmark.
    #   2. Compute Mahalanobis distance d² to each.
    #   3. If best match is below threshold → EKF update with that landmark.
    #   4. Otherwise → treat as a potential new landmark (candidate buffer).
    # ====================================================================

    def process_observation(self, obs: dict) -> None:

        z = np.array([[obs["range"]], [obs["bearing"]]])

        # Ignore if out of useful range
        if not (self.min_landmark_range < z[0, 0] < self.max_landmark_range):
            return

        # No landmarks yet — skip straight to candidate
        if self.num_landmarks == 0:
            self.add_to_candidate_buffer(obs)
            return

        # --- Find the best matching landmark via Mahalanobis distance ----
        best_landmark_idx  = None
        best_distance_sq   = float("inf")
        best_H             = None
        best_innovation    = None
        best_S             = None

        for landmark_idx in range(self.num_landmarks):

            # Expected observation and measurement Jacobian for this landmark
            z_hat, H = self.observation_model(landmark_idx)

            # Innovation: difference between what we see and what we expect
            innovation = z - z_hat
            innovation[1, 0] = wrap_angle(innovation[1, 0])  # wrap bearing

            # Innovation covariance S = H Sigma H^T + Q
            S = H @ self.Sigma @ H.T + self.obs_noise

            # Mahalanobis distance  d² = innovation^T  S⁻¹  innovation
            try:
                d_squared = float(innovation.T @ np.linalg.inv(S) @ innovation)
            except np.linalg.LinAlgError:
                continue

            if d_squared < best_distance_sq:
                best_distance_sq  = d_squared
                best_landmark_idx = landmark_idx
                best_H            = H
                best_innovation   = innovation
                best_S            = S

        # --- Decision: update or new candidate ---------------------------
        if best_landmark_idx is not None and best_distance_sq < self.mahal_threshold:
            # Good match → correct the state
            self.ekf_correct(best_H, best_innovation, best_S)
        else:
            # No match → might be a new landmark
            self.add_to_candidate_buffer(obs)

    def observation_model(self, landmark_idx: int):
        """
        Compute expected observation and its Jacobian for landmark k.

        Given robot pose (x, y, theta) and landmark position (mx, my):

            dx = mx - x
            dy = my - y

            expected_range   = sqrt(dx² + dy²)
            expected_bearing = atan2(dy, dx) - theta

        The Jacobian H is (2 x state_size) — partial derivatives of
        [range, bearing] with respect to each state variable.
        """
        x     = self.mu[0, 0]
        y     = self.mu[1, 0]
        theta = self.mu[2, 0]

        # Index where this landmark's (mx, my) lives in mu
        lm_idx = 3 + 2 * landmark_idx
        mx = self.mu[lm_idx,     0]
        my = self.mu[lm_idx + 1, 0]

        dx = mx - x
        dy = my - y
        q  = max(dx**2 + dy**2, 1e-8)   # squared distance (clamped for safety)
        sqrt_q = math.sqrt(q)

        # Expected measurement
        z_hat = np.array([
            [sqrt_q],
            [wrap_angle(math.atan2(dy, dx) - theta)],
        ])

        # Jacobian H — (2 x state_size)
        state_size = self.mu.shape[0]
        H = np.zeros((2, state_size))

        # d(range) / d(robot x, y, theta)
        H[0, 0] = -dx / sqrt_q
        H[0, 1] = -dy / sqrt_q
        H[0, 2] =  0.0

        # d(bearing) / d(robot x, y, theta)
        H[1, 0] =  dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1.0

        # d(range) / d(landmark mx, my)
        H[0, lm_idx]     =  dx / sqrt_q
        H[0, lm_idx + 1] =  dy / sqrt_q

        # d(bearing) / d(landmark mx, my)
        H[1, lm_idx]     = -dy / q
        H[1, lm_idx + 1] =  dx / q

        return z_hat, H

    def ekf_correct(self, H, innovation, S) -> None:
        """
        Standard EKF measurement update:

            K     = Sigma H^T S⁻¹          (Kalman gain)
            mu    = mu + K * innovation     (state update)
            Sigma = (I - K H) Sigma         (covariance update)
        """
        kalman_gain = self.Sigma @ H.T @ np.linalg.inv(S)

        self.mu = self.mu + kalman_gain @ innovation
        self.mu[2, 0] = wrap_angle(self.mu[2, 0])   # keep theta in (-pi, pi]

        state_size  = self.mu.shape[0]
        self.Sigma  = (np.eye(state_size) - kalman_gain @ H) @ self.Sigma

    # ====================================================================
    # CANDIDATE LANDMARK BUFFER
    #
    # A new feature must be seen at least `candidate_required_seen` times
    # before we trust it enough to add to the EKF state.  This avoids
    # bloating the state with spurious detections.
    # ====================================================================

    def add_to_candidate_buffer(self, obs: dict) -> None:

        # Convert observation to world frame using current pose estimate
        world_pos = self.observation_to_world_coords(obs)

        # Reject if it's too close to an already-confirmed landmark
        for k in range(self.num_landmarks):
            lm_idx  = 3 + 2 * k
            lm_pos  = self.mu[lm_idx:lm_idx + 2, 0]
            if np.linalg.norm(world_pos - lm_pos) < self.new_landmark_min_dist:
                return

        # Check if this matches an existing candidate
        for candidate in self.candidate_landmarks:
            if np.linalg.norm(world_pos - candidate["position"]) < self.candidate_match_dist:
                # Refine position estimate with running average
                candidate["position"] = 0.5 * candidate["position"] + 0.5 * world_pos
                candidate["seen"] += 1

                # Seen enough times — promote to confirmed landmark
                if candidate["seen"] >= self.candidate_required_seen:
                    self.add_new_landmark(candidate["position"])
                    self.candidate_landmarks = [c for c in self.candidate_landmarks if c is not candidate]
                    return

        # No matching candidate — start a new one
        self.candidate_landmarks.append({
            "position": world_pos,
            "seen":     1,
            "type":     obs["type"],
        })

    def observation_to_world_coords(self, obs: dict) -> np.ndarray:
        """
        Convert a polar observation (range, bearing) in the robot frame
        to a Cartesian position in the world frame.

            world_angle = theta + bearing
            mx = x + range * cos(world_angle)
            my = y + range * sin(world_angle)
        """
        r, b  = obs["range"], obs["bearing"]
        x     = self.mu[0, 0]
        y     = self.mu[1, 0]
        theta = self.mu[2, 0]

        world_angle = theta + b
        return np.array([
            x + r * math.cos(world_angle),
            y + r * math.sin(world_angle),
        ])

    def add_new_landmark(self, world_position: np.ndarray) -> None:
        """
        Append a new landmark to the EKF state vector and covariance matrix.

        mu grows from (3+2N, 1) to (3+2(N+1), 1).
        Sigma grows from (3+2N, 3+2N) to (3+2(N+1), 3+2(N+1)).
        The new landmark's initial uncertainty is set to 0.5 m std-dev.
        Cross-correlations with existing state are initialised to zero.
        """
        mx, my = float(world_position[0]), float(world_position[1])

        old_size = self.mu.shape[0]
        new_size = old_size + 2

        # Extend mu
        new_mu = np.zeros((new_size, 1))
        new_mu[:old_size] = self.mu
        new_mu[old_size,     0] = mx
        new_mu[old_size + 1, 0] = my

        # Extend Sigma (new off-diagonal blocks are zero = uncorrelated initially)
        new_Sigma = np.zeros((new_size, new_size))
        new_Sigma[:old_size, :old_size] = self.Sigma
        new_Sigma[old_size,     old_size]     = 0.50 ** 2   # 0.5 m std-dev
        new_Sigma[old_size + 1, old_size + 1] = 0.50 ** 2

        self.mu    = new_mu
        self.Sigma = new_Sigma
        self.num_landmarks += 1
        self.total_confirmed_corners += 1


        self.get_logger().info(
            f"New landmark #{self.num_landmarks} at ({mx:.2f}, {my:.2f})"
        )

    # ====================================================================
    # OCCUPANCY GRID UPDATE
    #
    # Inverse sensor model — for each LiDAR ray:
    #   - All cells along the ray (up to the hit) are marked FREE
    #   - The endpoint cell is marked OCCUPIED
    #
    # Updates are additive in log-odds space and clamped to [min, max].
    # ====================================================================

    def update_occupancy_grid(self, ranges: np.ndarray, angles: np.ndarray) -> None:

        robot_x     = self.mu[0, 0]
        robot_y     = self.mu[1, 0]
        robot_theta = self.mu[2, 0]

        robot_cell = self.world_to_grid_cell(robot_x, robot_y)
        if robot_cell is None:
            return   # robot is outside the map

        for r, phi in zip(ranges, angles):

            # Rays that exceed max range don't give us an occupied endpoint
            is_hit = True
            if r > self.max_mapping_range:
                r      = self.max_mapping_range
                is_hit = False

            if r < self.scan_preprocessor.min_range:
                continue   # too close — unreliable

            # Endpoint in world frame
            global_angle = robot_theta + float(phi)
            end_x = robot_x + r * math.cos(global_angle)
            end_y = robot_y + r * math.sin(global_angle)

            end_cell = self.world_to_grid_cell(end_x, end_y)
            if end_cell is None:
                continue   # endpoint outside map

            # Trace the ray through the grid
            ray_cells = self.bresenham_line(robot_cell[0], robot_cell[1],
                                            end_cell[0],   end_cell[1])

            # All cells except the last one are free space
            for cx, cy in ray_cells[:-1]:
                if self.cell_in_bounds(cx, cy):
                    self.log_odds[cy, cx] = np.clip(
                        self.log_odds[cy, cx] + self.log_odds_free,
                        self.log_odds_min, self.log_odds_max)
                    self.cells_observed[cy, cx] = True

            # The last cell is occupied (only if the ray actually hit something)
            if is_hit:
                cx, cy = ray_cells[-1]
                if self.cell_in_bounds(cx, cy):
                    self.log_odds[cy, cx] = np.clip(
                        self.log_odds[cy, cx] + self.log_odds_occ,
                        self.log_odds_min, self.log_odds_max)
                    self.cells_observed[cy, cx] = True

    def world_to_grid_cell(self, x: float, y: float):
        """Convert world coordinates (metres) to grid cell indices (cx, cy)."""
        cx = int((x - self.map_origin_x) / self.map_resolution)
        cy = int((y - self.map_origin_y) / self.map_resolution)
        if not self.cell_in_bounds(cx, cy):
            return None
        return cx, cy

    def cell_in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.map_width and 0 <= cy < self.map_height

    def bresenham_line(self, x0, y0, x1, y1) -> list:
        """
        Bresenham's line algorithm — returns all grid cells that the
        straight line from (x0,y0) to (x1,y1) passes through.
        Used to trace each LiDAR ray through the grid efficiently.
        """
        cells = []
        dx = abs(x1 - x0);  dy = abs(y1 - y0)
        step_x = 1 if x0 < x1 else -1
        step_y = 1 if y0 < y1 else -1
        error  = dx - dy
        x, y   = x0, y0

        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * error
            if e2 > -dy:  error -= dy;  x += step_x
            if e2 <  dx:  error += dx;  y += step_y

        return cells

    # ====================================================================
    # PUBLISHERS
    # ====================================================================

    def publish_pose(self) -> None:
        """Publish current robot pose estimate with covariance."""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        x     = self.mu[0, 0]
        y     = self.mu[1, 0]
        theta = self.mu[2, 0]

        msg.pose.pose.position.x    = float(x)
        msg.pose.pose.position.y    = float(y)
        msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(theta / 2.0)

        # ROS covariance is a flat 36-element array (6x6: x,y,z,roll,pitch,yaw)
        # We only fill the x, y, yaw relevant entries
        cov = np.zeros(36)
        cov[0]  = self.Sigma[0, 0];  cov[1]  = self.Sigma[0, 1];  cov[5]  = self.Sigma[0, 2]
        cov[6]  = self.Sigma[1, 0];  cov[7]  = self.Sigma[1, 1];  cov[11] = self.Sigma[1, 2]
        cov[30] = self.Sigma[2, 0];  cov[31] = self.Sigma[2, 1];  cov[35] = self.Sigma[2, 2]
        msg.pose.covariance = cov.tolist()

        self.pose_pub.publish(msg)

    def publish_landmarks(self) -> None:
        """Publish confirmed EKF landmarks as red spheres for RViz."""
        marker_array = MarkerArray()

        # First marker clears all old ones from RViz
        clear = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for k in range(self.num_landmarks):
            lm_idx = 3 + 2 * k
            mx = self.mu[lm_idx,     0]
            my = self.mu[lm_idx + 1, 0]

            m = Marker()
            m.header.stamp    = self.get_clock().now().to_msg()
            m.header.frame_id = "map"
            m.ns              = "ekf_landmarks"
            m.id              = k
            m.type            = Marker.SPHERE
            m.action          = Marker.ADD
            m.pose.position.x = float(mx)
            m.pose.position.y = float(my)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.12
            m.color.r = 1.0;  m.color.g = 0.1;  m.color.b = 0.1;  m.color.a = 1.0
            marker_array.markers.append(m)

        self.landmark_pub.publish(marker_array)

    def publish_map(self) -> None:
        """Convert log-odds grid to ROS OccupancyGrid and publish."""
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.info.resolution       = self.map_resolution
        msg.info.width            = self.map_width
        msg.info.height           = self.map_height
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.info.origin.orientation.w = 1.0

        # Vectorised conversion — processes all 57,600 cells in one numpy pass
        # instead of a Python loop with per-cell math.exp calls.
        #
        # Step 1: convert log-odds -> probability for every cell at once
        prob_occupied = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))   # shape (H, W)

        # Step 2: scale to 0-100 integer and flatten in row-major order
        grid_int = np.clip(np.round(100.0 * prob_occupied), 0, 100).astype(np.int8)

        # Step 3: overwrite unobserved cells with -1 (ROS convention for unknown)
        grid_int[~self.cells_observed] = -1

        msg.data = grid_int.flatten().tolist()

        self.map_pub.publish(msg)
    def print_debug_summary(self):
        self.get_logger().info("=================================")
        self.get_logger().info("EKF DEBUG SUMMARY")
        self.get_logger().info(
            f"Corner detections: {self.total_corner_detections}"
        )
        self.get_logger().info(
            f"Confirmed landmarks: {self.total_confirmed_corners}"
        )
        self.get_logger().info(
            f"Current landmarks in EKF: {self.num_landmarks}"
        )
        self.get_logger().info("=================================")

    # ====================================================================
    # PUBLIC ACCESSORS  (used by frontier planner, task manager, etc.)
    # ====================================================================

    @property
    def pose(self) -> np.ndarray:
        """Current best-estimate robot pose as [x, y, theta]."""
        return self.mu[:3, 0].copy()

    @property
    def occupancy_grid(self) -> np.ndarray:
        """Log-odds grid (map_height x map_width, float32). Positive = occupied."""
        return self.log_odds

    @property
    def known_cells(self) -> np.ndarray:
        """Boolean mask — True where a cell has been observed at least once."""
        return self.cells_observed