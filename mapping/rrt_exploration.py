#!/usr/bin/env python3
"""
RRT frontier exploration + waypoint follower for B07 Escape Room Robot.

Minimal, thread-safe version.
  - Planner timer (slow) builds an RRT path toward the nearest frontier.
  - Control timer (fast) follows the current path.
Both run on separate threads (MultiThreadedExecutor + ReentrantCallbackGroup),
so all shared navigation state is protected by a single lock and the follower
works on a consistent snapshot.

If the robot SPINS IN PLACE forever: your odom/motion-model yaw sign is likely
inverted relative to cmd_vel. Flip `self.yaw_sign` below from +1.0 to -1.0.
"""

from __future__ import annotations

import math
import threading
import random
import traceback
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class MapSnapshot:
    grid: np.ndarray
    safe: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int


class RRTExplorer(Node):

    def __init__(
        self,
        slam,
        lidar=None,
        preprocessor=None,
        step_size: float = 0.50,
        max_iterations: int = 1000,
        robot_radius_cells: int = 5,   # 5 × 0.05 m = 0.25 m clearance from any marked obstacle
        min_known_cells: int = 800,
        min_plan_interval: float = 2.0,
        slam_map_topic: str = "/slam_map",
        cmd_vel_topic: str = "/cmd_vel",
    ) -> None:
        super().__init__("rrt_frontier_explorer")
        self.cb_group = ReentrantCallbackGroup()

        self.slam = slam
        self.lidar = lidar
        self.preprocessor = preprocessor

        # ---- RRT params ----
        self.step_size = float(step_size)
        self.max_iterations = int(max_iterations)
        self.robot_radius_cells = int(robot_radius_cells)
        self.min_known_cells = int(min_known_cells)
        self.min_plan_interval = float(min_plan_interval)

        # ---- control params ----
        self.control_period = 0.05
        self.planner_period = 3.0
        self.wp_tol = 0.35           # waypoint acceptance radius (m)
        self.goal_tolerance = 0.40   # final-goal acceptance radius (m)
        self.max_speed  = 0.05       # m/s — forward speed during DRIVE
        self.turn_speed = 0.0        # m/s — pure pivot during TURN; triggers SLAM pause (abs(vx)<0.04)
        self.k_yaw      = 0.6        # proportional gain: heading_err → wz
        self.max_wz     = 0.35       # rad/s cap during TURN
        self.drive_wz_cap = self.max_wz          # full wz during DRIVE — fast recovery, no orbit risk
        self.turn_enter = math.radians(90.0)  # re-enter TURN only if facing backward
        self.turn_exit  = math.radians(6.0)   # stop turning once tightly aligned
        self.yaw_sign   = 1.0        # flip to -1.0 if robot spins the wrong way

        # ---- two-phase controller state ----
        self._turning  = True   # start in TURN mode for first waypoint
        self._prev_wp  = -1     # detect waypoint advances to reset phase

        # ---- shared state (guarded by _lock) ----
        self._lock = threading.RLock()
        self.current_path: List[np.ndarray] = []
        self.waypoint_index = 0
        self.active_goal: Optional[np.ndarray] = None
        self.exploration_complete = False
        self._need_replan = True
        self._last_plan_time = -1e9
        self._planning = False

        # ---- pubs ----
        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.path_pub = self.create_publisher(Path, "/exploration_path", 1)
        self.goal_pub = self.create_publisher(Marker, "/rrt_goal", 1)
        self._wp_pub  = self.create_publisher(PointStamped, "/exploration_current_wp", 1)

        self._last_sent_wz: float = 0.0
        self._last_sent_vx: float = 0.0

        # ---- timers ----
        self.create_timer(self.planner_period, self._plan_cb, callback_group=self.cb_group)
        self.create_timer(self.control_period, self._ctrl_cb, callback_group=self.cb_group)

        self.get_logger().info("RRT frontier explorer initialized (minimal).")

    # ====================================================================
    # TIMERS
    # ====================================================================
    def _plan_cb(self) -> None:
        try:
            self.plan_if_needed()
        except Exception:
            self.get_logger().error("planner crashed:\n" + traceback.format_exc())
            self.stop_robot()

    def _ctrl_cb(self) -> None:
        try:
            self.follow_path()
        except Exception:
            self.get_logger().error("follower crashed:\n" + traceback.format_exc())
            self.stop_robot()

    # ====================================================================
    # PLANNER
    # ====================================================================
    def plan_if_needed(self) -> None:
        with self._lock:
            if self.exploration_complete:
                return

        pose = self.get_robot_pose()
        if pose is None:
            return
        snap = self.get_map_snapshot()
        if snap is None:
            return

        known = int(np.count_nonzero(snap.grid >= 0))
        if known < self.min_known_cells:
            self.get_logger().info(
                f"Waiting for map context. Discovered cells: {known}/{self.min_known_cells}",
                throttle_duration_sec=3.0)
            self.stop_robot()
            return

        # Only cancel the current path if there is a genuine obstacle blocking it.
        # Do NOT cancel just because the robot hasn't moved yet — that caused
        # constant replanning that prevented the follower from ever executing.
        with self._lock:
            have_path = bool(self.current_path)
            need_replan = self._need_replan
        if have_path and not need_replan:
            diverged = self._robot_diverged_from_path(pose[:2])
            if diverged or self.path_blocked_by_obstacle(snap, pose):
                reason = "drifted away from path" if diverged else "path blocked by obstacle"
                self.get_logger().warn(f"Replanning: {reason}.")
                with self._lock:
                    self.current_path = []
                    self.waypoint_index = 0
                    self.active_goal = None
                    self._need_replan = True
                self.stop_robot()
            else:
                return   # path still valid, let follower execute it

        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_plan_time < self.min_plan_interval:
            return

        # simple non-blocking guard against re-entrant planning
        with self._lock:
            if self._planning:
                return
            self._planning = True
        try:
            self._last_plan_time = now
            with self._lock:
                self._need_replan = False
            path = self.build_path(snap, pose[:2], heading=float(pose[2]))
            if path is None or len(path) < 2:
                with self._lock:
                    self.current_path = []
                    self.waypoint_index = 0
                    self.active_goal = None
                    self._need_replan = True   # try again on next planner tick
                self.get_logger().warn(
                    f"No reachable frontier found (known={known} cells). "
                    "Will retry when map updates.",
                    throttle_duration_sec=4.0,
                )
                self.stop_robot()
                self.publish_path([])
                return
            self.set_path(path)
        finally:
            with self._lock:
                self._planning = False

    def build_path(self, snap: MapSnapshot, start_xy, heading: float = 0.0) -> Optional[List[np.ndarray]]:
        start_xy = np.array(start_xy, dtype=float)
        if not self.point_safe(start_xy, snap):
            repl = self.nearest_safe(start_xy, snap)
            if repl is None:
                return None
            start_xy = repl

        nodes: List[np.ndarray] = [start_xy]
        parents: List[int] = [-1]
        frontiers: List[Tuple[float, float, int]] = []
        h, w = snap.height, snap.width

        for _ in range(self.max_iterations):
            if random.random() < 0.30 and frontiers:
                f = frontiers[random.randint(0, len(frontiers) - 1)]
                xr = f[0] + random.uniform(-self.step_size, self.step_size)
                yr = f[1] + random.uniform(-self.step_size, self.step_size)
            else:
                th = random.uniform(-math.pi, math.pi)
                rad = 8.0 * math.sqrt(random.random())
                xr = start_xy[0] + rad * math.cos(th)
                yr = start_xy[1] + rad * math.sin(th)

            bi = self.nearest_node(nodes, xr, yr)
            nx, ny = nodes[bi]
            ang = math.atan2(yr - ny, xr - nx)
            x1 = nx + self.step_size * math.cos(ang)
            y1 = ny + self.step_size * math.sin(ang)
            new = np.array([x1, y1])

            if not self.point_safe(new, snap) or not self.line_safe(nodes[bi], new, snap):
                continue
            nodes.append(new)
            parents.append(bi)

            gx = int((x1 - snap.origin_x) / snap.resolution)
            gy = int((y1 - snap.origin_y) / snap.resolution)
            if 1 <= gx < w - 1 and 1 <= gy < h - 1:
                if np.any(snap.grid[gy - 1:gy + 2, gx - 1:gx + 2] == -1):
                    frontiers.append((x1, y1, len(nodes) - 1))

        if not frontiers:
            return None
        # Cost = distance + turn penalty: a 180° turn costs ~0.6 m equivalent
        # so the planner prefers frontiers roughly in front of the robot.
        def _frontier_cost(f):
            dist = math.hypot(f[0] - start_xy[0], f[1] - start_xy[1])
            bearing = math.atan2(f[1] - start_xy[1], f[0] - start_xy[0])
            turn = abs(wrap_angle(bearing - heading))   # 0 .. π
            return dist + (turn / math.pi) * 0.6
        best = min(frontiers, key=_frontier_cost)
        return self.reconstruct(nodes, parents, best[2])

    def nearest_node(self, nodes, sx, sy) -> int:
        best_i, best_d = 0, float("inf")
        for i, n in enumerate(nodes):
            d = (n[0] - sx) ** 2 + (n[1] - sy) ** 2
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def reconstruct(self, nodes, parents, gi) -> List[np.ndarray]:
        path, idx = [], gi
        while idx >= 0:
            path.append(np.array(nodes[idx], dtype=float))
            idx = parents[idx]
        path.reverse()
        return path

    def _robot_diverged_from_path(self, robot_xy, max_deviation: float = 1.0) -> bool:
        """True when robot is farther than max_deviation from every remaining waypoint.

        This catches stale plans: robot drifted away (obstacle backup, SLAM jump)
        but path_blocked_by_obstacle returns False because the cells are still free.
        """
        with self._lock:
            path = self.current_path
            wp = self.waypoint_index
        if not path:
            return False
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
        for pt in path[wp:]:
            if math.hypot(pt[0] - rx, pt[1] - ry) < max_deviation:
                return False   # close enough to at least one remaining waypoint
        self.get_logger().warn(
            f"Robot at ({rx:.2f},{ry:.2f}) is >{max_deviation:.1f}m from all remaining waypoints.",
            throttle_duration_sec=2.0,
        )
        return True

    def path_blocked_by_obstacle(self, snap: MapSnapshot, pose) -> bool:
        """True if a newly-observed occupied cell falls on any remaining path segment.

        Only checks grid >= 50 (actual walls), NOT the inflation buffer.
        Paths are planned using the full safe mask so they start inflation-free;
        we only need to replan when a genuine new wall appears on the path itself.
        Checking inflation would trigger constant replanning as walls solidify.
        """
        with self._lock:
            path = self.current_path
            wp = self.waypoint_index
        if not path or wp >= len(path):
            return False
        remaining = [pose[:2]] + [p for p in path[wp:]]
        for a, b in zip(remaining[:-1], remaining[1:]):
            dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            steps = max(int(dist / (snap.resolution * 0.5)), 1)
            for k in range(steps + 1):
                t = k / steps
                cx = int((a[0] + t * dx - snap.origin_x) / snap.resolution)
                cy = int((a[1] + t * dy - snap.origin_y) / snap.resolution)
                if 0 <= cx < snap.width and 0 <= cy < snap.height:
                    if snap.grid[cy, cx] >= 50:
                        return True
        return False

    def path_is_valid(self, snap: MapSnapshot, pose) -> bool:
        with self._lock:
            path = self.current_path
            wp = self.waypoint_index
        if not path:
            return False
        remaining = [pose[:2]] + path[wp:]
        if len(remaining) < 2:
            return False
        for a, b in zip(remaining[:-1], remaining[1:]):
            if not self.line_safe(np.array(a), np.array(b), snap):
                return False
        return True

    def set_path(self, path: List[np.ndarray]) -> None:
        with self._lock:
            self.current_path = [np.array(p, dtype=float) for p in path]
            self.waypoint_index = 1 if len(self.current_path) > 1 else 0
            self.active_goal = self.current_path[-1].copy()
            pub_path = list(self.current_path)
            pub_goal = self.active_goal.copy()
        self.get_logger().info(
            f"[plan] new goal=({pub_goal[0]:.2f},{pub_goal[1]:.2f}) "
            f"waypoints={len(pub_path)}",
            throttle_duration_sec=1.0)
        self.publish_path(pub_path)
        self.publish_goal(pub_goal)

    # ====================================================================
    # FOLLOWER  (map-based planning → no obstacle avoidance needed here)
    # ====================================================================
    def follow_path(self) -> None:
        pose = self.get_robot_pose()
        if pose is None:
            return
        rx, ry, rth = float(pose[0]), float(pose[1]), float(pose[2])

        with self._lock:
            path = self.current_path
            goal = self.active_goal
            wp   = self.waypoint_index
            done = self.exploration_complete

        if done or not path or goal is None:
            self.stop_robot()
            return

        # reached final goal?
        if math.hypot(goal[0] - rx, goal[1] - ry) < self.goal_tolerance:
            self.stop_robot()
            with self._lock:
                self.current_path = []
                self.waypoint_index = 0
                self.active_goal = None
                self._need_replan = True
            return

        # advance past waypoints already within wp_tol
        while wp < len(path) and math.hypot(path[wp][0] - rx, path[wp][1] - ry) < self.wp_tol:
            wp += 1
        with self._lock:
            if self.current_path is path:
                self.waypoint_index = wp

        if wp >= len(path):
            self.stop_robot()
            with self._lock:
                self._need_replan = True
            return

        # publish current target waypoint for debug_viz
        _wps = PointStamped()
        _wps.header.stamp = self.get_clock().now().to_msg()
        _wps.header.frame_id = "map"
        _wps.point.x = float(path[wp][0])
        _wps.point.y = float(path[wp][1])
        self._wp_pub.publish(_wps)

        # ------------------------------------------------------------------
        # Two-phase waypoint follower
        #
        # Phase 1 – TURN IN PLACE:  rotate until facing the waypoint within
        #           align_threshold.  vx=0 so the robot does not arc away.
        # Phase 2 – DRIVE STRAIGHT: go forward at max_speed with only a
        #           gentle heading correction (small wz cap) so the line
        #           from robot to waypoint stays straight and dist strictly
        #           decreases.
        #
        # Running vx and wz simultaneously at large errors caused the robot
        # to orbit the waypoint (circular arc) instead of approaching it.
        # ------------------------------------------------------------------
        tx, ty = float(path[wp][0]), float(path[wp][1])
        dist        = math.hypot(tx - rx, ty - ry)
        heading_err = wrap_angle(math.atan2(ty - ry, tx - rx) - rth)

        # Reset to TURN mode whenever the active waypoint changes.
        if wp != self._prev_wp:
            self._turning = True
            self._prev_wp = wp

        # Hysteresis: enter TURN at >18°, exit TURN at <6°.
        # This prevents EKF theta corrections (≤3°) from constantly flipping phases.
        if self._turning:
            if abs(heading_err) < self.turn_exit:
                self._turning = False
        else:
            if abs(heading_err) > self.turn_enter:
                self._turning = True

        if self._turning:
            # TURN: rotate toward waypoint.
            # vx=0.04 keeps SLAM active (pause threshold is abs(vx) < 0.04).
            vx    = self.turn_speed
            wz    = clamp(self.yaw_sign * self.k_yaw * heading_err, -self.max_wz, self.max_wz)
            phase = "TURN"
        else:
            # DRIVE: go straight, tiny heading correction only.
            vx    = self.max_speed
            wz    = clamp(self.yaw_sign * self.k_yaw * heading_err, -self.drive_wz_cap, self.drive_wz_cap)
            phase = "DRIVE"

        self.get_logger().info(
            f"[{phase}] wp={wp}/{len(path)-1} "
            f"target=({tx:.2f},{ty:.2f}) dist={dist:.2f}m "
            f"yaw_err={math.degrees(heading_err):.0f}deg "
            f"vx={vx:.3f} wz={wz:.3f}",
            throttle_duration_sec=1.0)

        cmd = Twist()
        cmd.linear.x  = float(vx)
        cmd.angular.z = float(wz)
        self._last_sent_vx = cmd.linear.x
        self._last_sent_wz = cmd.angular.z
        self._cmd_pub.publish(cmd)

    # ----------------------------------------------------------------------
    # ====================================================================
    # MAP / GEOMETRY (reads conventions straight from SLAM)
    # ====================================================================
    def get_map_snapshot(self) -> Optional[MapSnapshot]:
        try:
            log_odds = np.asarray(self.slam.occupancy_grid, dtype=np.float32)
            observed = np.asarray(self.slam.known_cells, dtype=bool)
        except Exception:
            return None
        h, w = log_odds.shape
        grid = np.full((h, w), -1, dtype=np.int16)
        grid[observed & (log_odds > 0.5)] = 100   # requires net positive hit evidence
        grid[observed & (log_odds <= 0.5)] = 0
        occ = grid >= 50
        inflated = self.inflate(occ, self.robot_radius_cells)
        safe = (grid == 0) & (~inflated)

        # read resolution/origin from SLAM if available (no more hardcoding)
        res = float(getattr(self.slam, "map_resolution", 0.05))
        ox = float(getattr(self.slam, "map_origin_x", -15.0))
        oy = float(getattr(self.slam, "map_origin_y", -15.0))
        return MapSnapshot(grid, safe, res, ox, oy, w, h)

    def inflate(self, mask: np.ndarray, r: int) -> np.ndarray:
        if r <= 0:
            return mask.copy()
        H, W = mask.shape
        out = mask.copy()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx > r * r:
                    continue
                dy0, dy1 = max(0, dy), H + min(0, dy)
                dx0, dx1 = max(0, dx), W + min(0, dx)
                sy0, sy1 = max(0, -dy), H - max(0, dy)
                sx0, sx1 = max(0, -dx), W - max(0, dx)
                out[dy0:dy1, dx0:dx1] |= mask[sy0:sy1, sx0:sx1]
        return out

    def point_safe(self, p, snap: MapSnapshot) -> bool:
        cx = int((p[0] - snap.origin_x) / snap.resolution)
        cy = int((p[1] - snap.origin_y) / snap.resolution)
        if 0 <= cx < snap.width and 0 <= cy < snap.height:
            return bool(snap.safe[cy, cx])
        return False

    def line_safe(self, a, b, snap: MapSnapshot) -> bool:
        dx, dy = b[0] - a[0], b[1] - a[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return True
        # Half-cell steps so diagonal segments never skip a cell.
        # A diagonal crosses a cell every √2×res ≈ 0.07 m; sampling at res/2 = 0.025 m
        # guarantees every cell on the line is visited at least once.
        steps = max(int(dist / (snap.resolution * 0.5)), 1)
        for k in range(steps + 1):
            t = k / steps
            cx = int((a[0] + t * dx - snap.origin_x) / snap.resolution)
            cy = int((a[1] + t * dy - snap.origin_y) / snap.resolution)
            if not (0 <= cx < snap.width and 0 <= cy < snap.height) or not snap.safe[cy, cx]:
                return False
        return True

    def nearest_safe(self, p, snap: MapSnapshot) -> Optional[np.ndarray]:
        cx0 = int((p[0] - snap.origin_x) / snap.resolution)
        cy0 = int((p[1] - snap.origin_y) / snap.resolution)
        if not (0 <= cx0 < snap.width and 0 <= cy0 < snap.height):
            return None
        for r in range(1, 60):   # search up to 3 m (was 1 m) to survive SLAM drift
            x0, x1 = max(0, cx0 - r), min(snap.width - 1, cx0 + r)
            y0, y1 = max(0, cy0 - r), min(snap.height - 1, cy0 + r)
            win = snap.safe[y0:y1 + 1, x0:x1 + 1]
            if np.any(win):
                best_p, best_d = None, float("inf")
                for ly, lx in np.argwhere(win):
                    wx = snap.origin_x + (x0 + lx + 0.5) * snap.resolution
                    wy = snap.origin_y + (y0 + ly + 0.5) * snap.resolution
                    d = math.hypot(wx - p[0], wy - p[1])
                    if d < best_d:
                        best_d, best_p = d, np.array([wx, wy])
                return best_p
        return None

    def get_robot_pose(self) -> Optional[np.ndarray]:
        try:
            pose = np.asarray(self.slam.pose, dtype=float)
            if pose.shape[0] >= 3 and np.all(np.isfinite(pose[:3])):
                return pose[:3].copy()
        except Exception:
            pass
        return None

    def block_door_in_costmap(self, left, right) -> None:
        """Stub: door blocking not implemented in this explorer version."""
        self.get_logger().info(f"[door] block requested: {left} / {right} (stub)")

    def unblock_door(self, left, right) -> None:
        """Stub: door unblocking not implemented in this explorer version."""
        self.get_logger().info(f"[door] unblock requested: {left} / {right} (stub)")

    def stop_robot(self) -> None:
        self._last_sent_vx = 0.0
        self._last_sent_wz = 0.0
        self._cmd_pub.publish(Twist())

    # ====================================================================
    # VIZ
    # ====================================================================
    def publish_path(self, path) -> None:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for p in path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(p[0])
            ps.pose.position.y = float(p[1])
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def publish_goal(self, goal) -> None:
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.ns = "rrt_frontier_goal"
        m.id = 0
        if goal is None:
            m.action = Marker.DELETE
            self.goal_pub.publish(m)
            return
        m.action = Marker.ADD
        m.type = Marker.SPHERE
        m.pose.position.x = float(goal[0])
        m.pose.position.y = float(goal[1])
        m.pose.position.z = 0.05
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.8, 1.0, 1.0
        self.goal_pub.publish(m)