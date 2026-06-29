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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_dilation

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Twist, PoseStamped
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
        # Keep exploration goals local, but outside the robot/goal-tolerance
        # footprint. The preferred radius prevents the nearest RRT node from
        # repeatedly becoming a goal almost underneath the robot.
        self.min_frontier_dist = 0.75
        self.preferred_frontier_dist = 1.0
        self.max_frontier_dist = 1.35

        # ---- control params ----
        self.control_period = 0.05
        self.planner_period = 3.0
        self.goal_tolerance = 0.40   # accept goal when within this radius (m)
        self.max_speed  = 0.05       # m/s — forward speed when aligned with goal
        self.k_yaw      = 0.6        # proportional gain: heading_err → wz
        self.max_wz     = 0.35       # rad/s — angular velocity cap
        self.yaw_sign   = 1.0        # flip to -1.0 if robot spins the wrong way

        # ---- LiDAR safety (forward obstacle detection) ----
        self.safe_dist        = 0.30   # m — stop if obstacle closer than this
        self.forward_sector   = 40.0   # degrees — sector checked (±40° from forward)
        self._obstacle_timeout = 3.0   # s — how long to wait before escape spin
        self._obstacle_since: Optional[float] = None
        self._escaping = False
        self._escape_yaw_target: Optional[float] = None

        # ---- shared state (guarded by _lock) ----
        self._lock = threading.RLock()
        self.current_path: List[np.ndarray] = []
        self.waypoint_index = 0
        self.active_goal: Optional[np.ndarray] = None
        self.exploration_complete = False
        self._need_replan = True
        self._last_plan_time = -1e9
        self._planning = False

        # Semantic navigation state. Locked doors are virtual obstacles laid
        # over the SLAM grid. A priority goal temporarily pre-empts frontier
        # exploration (for example, revisiting a door after collecting its key).
        self._locked_doors: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._priority_goal: Optional[np.ndarray] = None
        self._priority_label: Optional[str] = None
        self._priority_queue: List[Tuple[str, np.ndarray]] = []
        self._navigation_revision = 0

        # ---- pubs ----
        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.path_pub = self.create_publisher(Path, "/exploration_path", 1)
        self.goal_pub = self.create_publisher(Marker, "/rrt_goal", 1)

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
        # RViz-only mode: build an RRT path from current pose and publish it for
        # visualisation.  The robot is driven by the reactive LiDAR controller in
        # follow_path() and does NOT follow these paths.
        pose = self.get_robot_pose()
        if pose is None:
            return
        snap = self.get_map_snapshot()
        if snap is None:
            return

        known = int(np.count_nonzero(snap.grid >= 0))
        if known < self.min_known_cells:
            self.get_logger().info(
                f"Waiting for map context ({known}/{self.min_known_cells} cells).",
                throttle_duration_sec=3.0)
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_plan_time < self.min_plan_interval:
            return

        with self._lock:
            if self._planning:
                return
            self._planning = True
        try:
            self._last_plan_time = now
            path = self.build_path(snap, pose[:2], heading=float(pose[2]))
            if path and len(path) >= 2:
                with self._lock:
                    self.current_path = path
                    self.active_goal = path[-1].copy()
                self.publish_path(path)
                self.publish_goal(path[-1])
                self.get_logger().info(
                    f"[RViz] RRT path: {len(path)} nodes → "
                    f"goal=({path[-1][0]:.2f},{path[-1][1]:.2f})",
                    throttle_duration_sec=2.0)
            else:
                self.publish_path([])
                self.publish_goal(None)
                self.get_logger().info(
                    "No frontier found yet — map too small or fully explored.",
                    throttle_duration_sec=4.0)
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
                rad = self.max_frontier_dist * math.sqrt(random.random())
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
                    fdist = math.hypot(x1 - start_xy[0], y1 - start_xy[1])
                    if self.min_frontier_dist <= fdist <= self.max_frontier_dist:
                        frontiers.append((x1, y1, len(nodes) - 1))

        if not frontiers:
            return None

        # Prefer a short goal around 1 m away instead of always selecting the
        # nearest admissible node. Heading is only a small tie-breaker.
        def _frontier_cost(f):
            dist = math.hypot(f[0] - start_xy[0], f[1] - start_xy[1])
            bearing = math.atan2(f[1] - start_xy[1], f[0] - start_xy[0])
            turn = abs(wrap_angle(bearing - heading))
            return abs(dist - self.preferred_frontier_dist) + (turn / math.pi) * 0.25

        # Get a fresh snapshot to validate against — planning takes time and
        # the map may have changed since `snap` was taken at the start of planning.
        fresh = self.get_map_snapshot() or snap

        # Iterate frontiers cheapest-first; skip any whose goal cell is now
        # occupied or inside the inflation buffer in the fresh map.
        for f in sorted(frontiers, key=_frontier_cost):
            goal_pt = np.array([f[0], f[1]])
            if self.point_safe(goal_pt, fresh):
                return self.reconstruct(nodes, parents, f[2])

        return None   # every candidate frontier is now unsafe

    def nearest_node(self, nodes, sx, sy) -> int:
        pts = np.asarray(nodes)
        dx = pts[:, 0] - sx
        dy = pts[:, 1] - sy
        return int(np.argmin(dx * dx + dy * dy))

    def reconstruct(self, nodes, parents, gi) -> List[np.ndarray]:
        path, idx = [], gi
        while idx >= 0:
            path.append(np.array(nodes[idx], dtype=float))
            idx = parents[idx]
        path.reverse()
        return path

    def set_path(self, path: List[np.ndarray]) -> None:
        with self._lock:
            self.current_path = [np.array(p, dtype=float) for p in path]
            self.waypoint_index = len(self.current_path) - 1
            self.active_goal = self.current_path[-1].copy()
            pub_path = list(self.current_path)
            pub_goal = self.active_goal.copy()
        self._escaping = False
        self._obstacle_since = None
        self._escape_yaw_target = None
        self.get_logger().info(
            f"[plan] new goal=({pub_goal[0]:.2f},{pub_goal[1]:.2f}) "
            f"via {len(pub_path)} waypoints",
            throttle_duration_sec=1.0)
        self.publish_path(pub_path)
        self.publish_goal(pub_goal)

    # ====================================================================
    # FOLLOWER  — smooth safe-LiDAR drive to goal
    #
    # No two-phase TURN/DRIVE — the robot always drives forward with speed
    # scaled by cos(heading_err) and steers proportionally toward the goal.
    # LiDAR stops forward motion if an obstacle appears; escape spin
    # triggers after 3 s of blocking.
    # ====================================================================
    def _forward_clearance(self) -> float:
        """Minimum LiDAR range in the forward sector (±forward_sector degrees)."""
        if self.lidar is None or self.preprocessor is None:
            return float("inf")
        raw = self.lidar.get_scan()
        if raw is None:
            return float("inf")
        processed = self.preprocessor.preprocess(raw)
        return self.preprocessor.get_sector_min(
            processed, -self.forward_sector, self.forward_sector
        )

    def follow_path(self) -> None:
        # ── Pure reactive LiDAR controller ────────────────────────────────────
        # Robot roams freely: drive forward, steer away from obstacles.
        # RRT paths are published to RViz only — they do NOT control motion.
        if self.lidar is None or self.preprocessor is None:
            self.stop_robot()
            return

        raw = self.lidar.get_scan()
        if raw is None:
            return

        scan = self.preprocessor.preprocess(raw)

        front = self.preprocessor.get_sector_min(scan, -self.forward_sector, self.forward_sector)

        if front > self.safe_dist:
            vx = self.max_speed
            wz = 0.0
        else:
            # Obstacle ahead — compare left vs right clearance and turn toward clearer side
            left  = self.preprocessor.get_sector_min(scan,  self.forward_sector, 120.0)
            right = self.preprocessor.get_sector_min(scan, -120.0, -self.forward_sector)
            vx = 0.0
            wz = (self.max_wz if left > right else -self.max_wz) * self.yaw_sign

        self.get_logger().info(
            f"[LIDAR] front={front:.2f}m vx={vx:.3f} wz={wz:.3f}",
            throttle_duration_sec=1.0)

        cmd = Twist()
        cmd.linear.x  = float(vx)
        cmd.angular.z = float(wz)
        self._last_sent_vx = float(vx)
        self._last_sent_wz = float(wz)
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

        # Add semantic locked doors without modifying SLAM's sensor map. Door
        # endpoints are rasterized every snapshot, so unlock is immediate and
        # does not leave permanent occupied cells behind.
        with self._lock:
            locked_doors = [
                (left.copy(), right.copy())
                for left, right in self._locked_doors.values()
            ]
        for left, right in locked_doors:
            self._rasterize_segment(
                occ, left, right, res=float(getattr(self.slam, "map_resolution", 0.05)),
                ox=float(getattr(self.slam, "map_origin_x", -15.0)),
                oy=float(getattr(self.slam, "map_origin_y", -15.0)),
            )
        inflated = self.inflate(occ, self.robot_radius_cells)
        safe = (grid == 0) & (~inflated)

        # read resolution/origin from SLAM if available (no more hardcoding)
        res = float(getattr(self.slam, "map_resolution", 0.05))
        ox = float(getattr(self.slam, "map_origin_x", -15.0))
        oy = float(getattr(self.slam, "map_origin_y", -15.0))
        return MapSnapshot(grid, safe, res, ox, oy, w, h)

    @staticmethod
    def _rasterize_segment(mask, a, b, res: float, ox: float, oy: float) -> None:
        """Mark every grid cell crossed by world-space segment ``a`` -> ``b``."""
        dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
        distance = math.hypot(dx, dy)
        steps = max(int(math.ceil(distance / max(res * 0.5, 1e-6))), 1)
        height, width = mask.shape
        for i in range(steps + 1):
            t = i / steps
            cx = int((float(a[0]) + t * dx - ox) / res)
            cy = int((float(a[1]) + t * dy - oy) / res)
            if 0 <= cx < width and 0 <= cy < height:
                mask[cy, cx] = True

    def inflate(self, mask: np.ndarray, r: int) -> np.ndarray:
        if r <= 0:
            return mask.copy()
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        struct = (x * x + y * y <= r * r)
        return binary_dilation(mask, structure=struct)

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

    def block_door_in_costmap(self, left, right, door_id: int = 0) -> None:
        """Insert a detected locked doorway as a persistent virtual obstacle."""
        left_arr = np.asarray(left, dtype=float)[:2]
        right_arr = np.asarray(right, dtype=float)[:2]
        with self._lock:
            self._locked_doors[int(door_id)] = (left_arr, right_arr)
            self._navigation_revision += 1
            # Cancel a frontier path that may have been planned through it.
            if self._priority_goal is None:
                self.current_path = []
                self.active_goal = None
                self._need_replan = True
        self.get_logger().info(
            f"[door {door_id}] LOCKED: added to semantic costmap"
        )

    def unblock_door(self, left=None, right=None, door_id: int = 0) -> None:
        """Remove a doorway's virtual obstacle after its key is collected."""
        with self._lock:
            self._locked_doors.pop(int(door_id), None)
            self._navigation_revision += 1
            self.current_path = []
            self.active_goal = None
            self._need_replan = True
        self.get_logger().info(
            f"[door {door_id}] UNLOCKED: removed from semantic costmap"
        )

    def navigate_to_door(self, door_id: int, center) -> None:
        """Pre-empt exploration and revisit an unlocked door immediately."""
        goal = np.asarray(center, dtype=float)[:2]
        label = f"door {door_id}"
        start_now = False
        with self._lock:
            self.exploration_complete = False
            labels = [queued_label for queued_label, _ in self._priority_queue]
            if self._priority_label == label or label in labels:
                return
            if self._priority_goal is None:
                self._priority_goal = goal
                self._priority_label = label
                self.current_path = []
                self.active_goal = None
                self._need_replan = False
                self._navigation_revision += 1
                start_now = True
            else:
                self._priority_queue.append((label, goal))
        if start_now:
            self.stop_robot()
            self.get_logger().info(
                f"Key collected — prioritizing door {door_id} at "
                f"({goal[0]:.2f}, {goal[1]:.2f})"
            )
        else:
            self.get_logger().info(f"Queued {label} behind the current key task")

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
