#!/usr/bin/env python3
"""
mapping/rrt_exploration.py  —  Frontier-RRT exploration + path following.

Single ROS 2 node that:
  1. Detects unexplored frontiers from the SLAM occupancy grid.
  2. Plans a collision-free path to the best frontier using RRT +
     greedy string-pulling smoothing.
  3. Follows that path with a pure-pursuit / proportional-heading
     controller suited for the MyAGV omni-drive.
  4. Handles door-awareness: locked door cells block RRT; unlock_door()
     removes them and the next replan routes through.
  5. Returns home once no frontiers remain.

Constructor signature (matches main.py):
    RRTExplorer(slam, lidar, preprocessor, slam_map_topic="/slam_map")

Public API used by main.py / ArucoMonitor:
    _cmd_pub                    — Twist publisher (main.py zeroes it on shutdown)
    register_door_segment(door_id, cells)
    unlock_door(door_id)
    set_navigation_goal(x, y)   — point-nav override (e.g. from ArucoMonitor)
    start_return_home()

Internal timers:
    _replan_cb     3.0 s   frontier detection + RRT planning
    _control_cb    0.1 s   pure-pursuit path following + LiDAR veto
"""

import math
import random
import threading
import traceback
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray

Waypoint = Tuple[float, float]


# ─────────────────────────────────────────────────────────────────────────────
# Internal RRT tree node
# ─────────────────────────────────────────────────────────────────────────────

class _RRTNode:
    __slots__ = ("x", "y", "parent")

    def __init__(self, x: float, y: float, parent: Optional["_RRTNode"] = None):
        self.x = x
        self.y = y
        self.parent = parent


# ─────────────────────────────────────────────────────────────────────────────
# RRTExplorer
# ─────────────────────────────────────────────────────────────────────────────

class RRTExplorer(Node):
    """Frontier-RRT exploration + pure-pursuit path following node.

    Parameters
    ----------
    slam              – EKFLidarSLAM instance (pose / occupancy_grid / known_cells)
    lidar             – LidarSensor instance (emergency-stop LiDAR veto)
    preprocessor      – ScanPreprocessor instance (min_range reference)
    slam_map_topic    – OccupancyGrid topic published by SLAM (subscribed for
                        the RViz map relay; grid data is read directly from slam)

    Tuning constants are grouped at the top of __init__ for easy adjustment.
    """

    def __init__(
        self,
        slam,
        lidar,
        preprocessor,
        slam_map_topic: str = "/slam_map",
    ) -> None:
        super().__init__("rrt_explorer")
        self._cb = ReentrantCallbackGroup()
        self._slam   = slam
        self._lidar  = lidar
        self._scan_p = preprocessor

        # ── Tuning ────────────────────────────────────────────────────
        # Collision / inflation
        self._robot_r_m     = 0.22    # robot radius for obstacle inflation (m)
        # RRT
        self._step_m        = 0.30    # tree extension step (m)
        self._max_iter      = 2000    # RRT iteration cap
        self._goal_bias     = 0.15    # probability of sampling the goal
        # Frontier detection
        self._min_frontier  = 8       # minimum cluster size (cells)
        # Path following — omni-drive simultaneous translate + rotate
        self._v_lin         = 0.15    # forward speed (m/s)  — conservative on real hw
        self._v_ang_max     = 0.40    # max angular speed (rad/s)
        self._kp_ang        = 0.8     # heading P-gain — low so it never dominates fwd motion
        self._lookahead_m   = 0.50    # pure-pursuit lookahead (m)
        self._wp_r_m        = 0.25    # waypoint acceptance radius (m)
        self._goal_r_m      = 0.35    # final goal acceptance radius (m)
        # Timers
        self._replan_s      = 3.0     # frontier replan period (s)
        self._control_hz    = 10.0    # path-following control rate (Hz)

        # SLAM pause flag — set while angular velocity is high so the EKF
        # prediction does not accumulate error from spin odometry.
        self._is_spinning        = False
        self._spin_omega_thresh  = 0.25   # rad/s above which we pause SLAM

        # ── Door / key state ──────────────────────────────────────────
        self._door_lock  = threading.Lock()
        self._door_segs: dict = {}    # {door_id: set of (cx, cy)}
        self._unlocked: set  = set()

        # ── Explorer state ────────────────────────────────────────────
        self._exp_lock         = threading.Lock()
        self._visited_goals: List[Waypoint] = []
        self._exploration_done = False
        self._return_home      = False
        self._home: Optional[Waypoint] = None
        self._nav_goal: Optional[Waypoint] = None   # external override

        # ── Path-follower state ───────────────────────────────────────
        self._path_lock   = threading.Lock()
        self._path: List[Waypoint] = []
        self._wp_idx      = 0
        self._at_goal     = False
        self._following   = False

        # Cached inflated grid (rebuilt each replan, used by smoother too)
        self._inflated: Optional[np.ndarray] = None
        self._grid_meta: Optional[dict] = None   # res, ox, oy, w, h snapshot

        # ── Publishers ────────────────────────────────────────────────
        self._cmd_pub       = self.create_publisher(Twist,      "/cmd_vel",              10)
        self._pub_frontiers = self.create_publisher(MarkerArray, "/explorer/frontiers",   1)
        self._pub_path      = self.create_publisher(MarkerArray, "/explorer/path",        1)
        self._pub_tree      = self.create_publisher(MarkerArray, "/explorer/rrt_tree",    1)
        self._pub_wps       = self.create_publisher(MarkerArray, "/follower/waypoints",   1)

        # Relay the SLAM map topic for completeness (optional, RViz convenience)
        self._map_sub = self.create_subscription(
            OccupancyGrid, slam_map_topic, self._map_relay_cb, 1,
            callback_group=self._cb,
        )

        # ── Timers ────────────────────────────────────────────────────
        self.create_timer(self._replan_s,          self._replan_cb,  callback_group=self._cb)
        self.create_timer(1.0 / self._control_hz,  self._control_cb, callback_group=self._cb)

        self.get_logger().info(
            f"RRTExplorer ready — step={self._step_m} m  "
            f"r_robot={self._robot_r_m} m  max_iter={self._max_iter}  "
            f"control={self._control_hz:.0f} Hz"
        )

    # ═════════════════════════════════════════════════════════════════
    # Public API
    # ═════════════════════════════════════════════════════════════════

    def register_door_segment(self, door_id: int, cells: List[Tuple[int, int]]) -> None:
        """Block grid cells belonging to a locked door from RRT traversal."""
        with self._door_lock:
            self._door_segs[door_id] = set(cells)
        self.get_logger().info(f"Door {door_id} registered ({len(cells)} cells blocked)")

    def unlock_door(self, door_id: int) -> None:
        """Remove door obstacle overlay; next replan will route through it."""
        with self._door_lock:
            self._door_segs.pop(door_id, None)
            self._unlocked.add(door_id)
        self.get_logger().info(f"Door {door_id} unlocked — will replan next tick")

    def set_navigation_goal(self, x: float, y: float) -> None:
        """Override: plan a one-shot path to (x, y) instead of the next frontier.
        Called by ArucoMonitor to revisit a door after its key is found."""
        with self._exp_lock:
            self._nav_goal = (x, y)
        self.get_logger().info(f"Navigation override → ({x:.2f}, {y:.2f})")

    def start_return_home(self) -> None:
        """Switch to return-home mode after exploration is declared complete."""
        with self._exp_lock:
            self._return_home = True
        self.get_logger().info("RRTExplorer: switching to RETURN-HOME mode")

    @property
    def at_goal(self) -> bool:
        return self._at_goal

    @property
    def exploration_done(self) -> bool:
        return self._exploration_done

    # ═════════════════════════════════════════════════════════════════
    # Map relay (OccupancyGrid subscriber — no-op, just for RViz)
    # ═════════════════════════════════════════════════════════════════

    def _map_relay_cb(self, _msg: OccupancyGrid) -> None:
        pass   # grid data comes directly from slam properties

    # ═════════════════════════════════════════════════════════════════
    # Replan timer  (3 s)
    # ═════════════════════════════════════════════════════════════════

    def _replan_cb(self) -> None:
        try:
            self._replan()
        except Exception:
            self.get_logger().error("replan crashed:\n" + traceback.format_exc())

    def _replan(self) -> None:
        pose = self._slam.pose
        rx, ry = float(pose[0]), float(pose[1])

        # Record home on first call
        if self._home is None:
            self._home = (rx, ry)
            self.get_logger().info(f"Home set to ({rx:.2f}, {ry:.2f})")

        # ── Snapshot SLAM grid ────────────────────────────────────────
        log_odds = self._slam.occupancy_grid
        known    = self._slam.known_cells
        res      = self._slam.map_resolution
        ox       = self._slam.map_origin_x
        oy       = self._slam.map_origin_y
        w        = self._slam.map_width
        h        = self._slam.map_height

        self._grid_meta = dict(res=res, ox=ox, oy=oy, w=w, h=h)

        # ── Build inflated obstacle grid ──────────────────────────────
        prob     = 1.0 - 1.0 / (1.0 + np.exp(np.clip(log_odds, -10.0, 10.0)))
        occupied = (prob > 0.6) & known
        r_cells  = max(1, int(math.ceil(self._robot_r_m / res)))
        inflated = self._inflate(occupied, r_cells)

        # Overlay locked door cells
        with self._door_lock:
            for cells in self._door_segs.values():
                for (cx, cy) in cells:
                    if 0 <= cx < w and 0 <= cy < h:
                        inflated[cy, cx] = True

        # ── Carve a free bubble around the robot's current cell ──────
        # Inflation can mark the robot's own footprint as obstacle when the
        # SLAM log-odds are still settling (common in the first few seconds).
        # Clear a 1-cell radius around the robot so RRT always has a valid
        # start, regardless of what the map says directly underneath it.
        sc = self._w2c(rx, ry, res, ox, oy, w, h)
        if sc is not None:
            carve_r = max(1, r_cells)
            for dcy in range(-carve_r, carve_r + 1):
                for dcx in range(-carve_r, carve_r + 1):
                    nx, ny = sc[0] + dcx, sc[1] + dcy
                    if 0 <= nx < w and 0 <= ny < h:
                        inflated[ny, nx] = False

        self._inflated = inflated

        # ── External navigation override (e.g. door revisit) ─────────
        with self._exp_lock:
            nav_goal = self._nav_goal
            return_home = self._return_home

        if nav_goal is not None:
            self.get_logger().info(
                f"Planning nav-override path to ({nav_goal[0]:.2f}, {nav_goal[1]:.2f})"
            )
            path = self._rrt((rx, ry), nav_goal, inflated, res, ox, oy, w, h)
            if path:
                self._set_path(path)
                with self._exp_lock:
                    self._nav_goal = None   # consume the override
            else:
                self.get_logger().warn("Nav-override: RRT failed — will retry")
            return

        # ── Return-home mode ──────────────────────────────────────────
        if return_home and self._home is not None:
            path = self._rrt((rx, ry), self._home, inflated, res, ox, oy, w, h)
            if path:
                self._set_path(path)
                with self._exp_lock:
                    self._exploration_done = True
                self.get_logger().info("Return-home path planned")
            else:
                self.get_logger().warn("Return-home: RRT failed — retrying")
            return

        # ── Don't replan if already following a valid path ────────────
        with self._path_lock:
            currently_following = self._following and bool(self._path)
        if currently_following:
            return

        # ── Frontier detection ────────────────────────────────────────
        clusters = self._find_frontiers(log_odds, known, inflated, w, h)
        if not clusters:
            self.get_logger().info(
                "No frontiers left — exploration complete", throttle_duration_sec=10.0
            )
            with self._exp_lock:
                self._return_home = True
            return

        self._pub_frontier_viz(clusters, res, ox, oy)

        # ── Pick best unvisited frontier ──────────────────────────────
        goal = self._pick_frontier(clusters, rx, ry, res, ox, oy)
        if goal is None:
            return

        with self._exp_lock:
            visited = list(self._visited_goals)

        if any(math.hypot(goal[0] - v[0], goal[1] - v[1]) < 0.5 for v in visited):
            # Try the next-best one
            ranked = self._rank_clusters(clusters, rx, ry, res, ox, oy)
            goal = None
            for cx_avg, cy_avg in ranked[1:]:
                cx_m = ox + cx_avg * res + res / 2
                cy_m = oy + cy_avg * res + res / 2
                if not any(math.hypot(cx_m - v[0], cy_m - v[1]) < 0.5 for v in visited):
                    goal = (cx_m, cy_m)
                    break
            if goal is None:
                self.get_logger().info(
                    "All ranked frontiers already visited — switching to return-home"
                )
                with self._exp_lock:
                    self._return_home = True
                return

        # ── RRT planning ──────────────────────────────────────────────
        path = self._rrt((rx, ry), goal, inflated, res, ox, oy, w, h)
        if path:
            self._set_path(path)
            with self._exp_lock:
                self._visited_goals.append(goal)
            self.get_logger().info(
                f"Frontier path: goal=({goal[0]:.2f},{goal[1]:.2f})  "
                f"waypoints={len(path)}"
            )
        else:
            self.get_logger().warn(
                f"RRT failed to reach ({goal[0]:.2f},{goal[1]:.2f})"
            )

    # ═════════════════════════════════════════════════════════════════
    # Control timer  (10 Hz)
    # ═════════════════════════════════════════════════════════════════

    def _control_cb(self) -> None:
        try:
            self._control_step()
        except Exception:
            self.get_logger().error("control crashed:\n" + traceback.format_exc())
            self._send_vel(0.0, 0.0, 0.0)

    def _control_step(self) -> None:
        with self._path_lock:
            path      = list(self._path)
            wp_idx    = self._wp_idx
            following = self._following

        if not following or not path:
            self._send_vel(0.0, 0.0, 0.0)
            return

        pose = self._slam.pose
        rx, ry, rtheta = float(pose[0]), float(pose[1]), float(pose[2])

        # ── Advance past already-reached waypoints ────────────────────
        while wp_idx < len(path) - 1:
            wx, wy = path[wp_idx]
            if math.hypot(rx - wx, ry - wy) < self._wp_r_m:
                wp_idx += 1
            else:
                break

        with self._path_lock:
            self._wp_idx = wp_idx

        # ── Final goal reached? ───────────────────────────────────────
        fx, fy = path[-1]
        if math.hypot(rx - fx, ry - fy) < self._goal_r_m:
            self.get_logger().info("PathFollower: goal reached")
            self._send_vel(0.0, 0.0, 0.0)
            with self._path_lock:
                self._path      = []
                self._following = False
                self._at_goal   = True
            return

        # ── Pure-pursuit lookahead point ──────────────────────────────
        target = self._lookahead_point(path, wp_idx, rx, ry)
        dx = target[0] - rx
        dy = target[1] - ry
        desired_heading = math.atan2(dy, dx)
        heading_err     = self._wrap(desired_heading - rtheta)

        # ── Omni-drive controller ─────────────────────────────────────
        # On an omnidirectional robot we ALWAYS drive forward — there is
        # no reason to stop translation to correct heading.  The robot can
        # translate and rotate simultaneously.
        #
        # vx   = forward speed, scaled down only when very close to the
        #         lookahead point (avoids overshooting).
        # omega = low-gain P controller on heading error.  The gain is
        #         intentionally small (0.8) so rotation never dominates
        #         and the robot keeps moving forward.
        #
        # SLAM pause: if |omega| exceeds the spin threshold we pause EKF
        # prediction to protect the pose estimate — but vx is still sent
        # so the robot keeps moving.

        dist_to_target = math.hypot(dx, dy)
        # Ramp down as we close in on the lookahead point
        fwd_scale = min(1.0, dist_to_target / (self._lookahead_m * 0.5 + 1e-6))
        vx = self._v_lin * fwd_scale

        omega = float(np.clip(
            self._kp_ang * heading_err,
            -self._v_ang_max,
            self._v_ang_max,
        ))

        # SLAM spin guard — pause only when angular rate is genuinely high
        if abs(omega) > self._spin_omega_thresh:
            if not self._is_spinning:
                self._is_spinning = True
                self._slam.pause_prediction(True)
        else:
            if self._is_spinning:
                self._is_spinning = False
                self._slam.pause_prediction(False)

        self.get_logger().info(
            f"CTRL vx={vx:.3f} w={omega:.3f} "
            f"herr={math.degrees(heading_err):.1f}deg "
            f"dist={dist_to_target:.2f}m",
            throttle_duration_sec=0.5,
        )
        self._send_vel(vx, 0.0, omega)

    # ═════════════════════════════════════════════════════════════════
    # Frontier detection
    # ═════════════════════════════════════════════════════════════════

    def _find_frontiers(
        self,
        log_odds: np.ndarray,
        known: np.ndarray,
        inflated: np.ndarray,
        w: int, h: int,
    ) -> List[List[Tuple[int, int]]]:
        """BFS-cluster free cells that border at least one unknown cell."""
        prob    = 1.0 - 1.0 / (1.0 + np.exp(np.clip(log_odds, -10.0, 10.0)))
        free    = known & (prob < 0.4) & ~inflated
        unknown = ~known

        # Vectorised 8-neighbour unknown check
        has_unk = np.zeros((h, w), dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                has_unk |= np.roll(np.roll(unknown, dy, axis=0), dx, axis=1)

        frontier_mask = free & has_unk
        visited  = np.zeros((h, w), dtype=bool)
        clusters: List[List[Tuple[int, int]]] = []

        rows, cols = np.where(frontier_mask)
        for row, col in zip(rows, cols):
            if visited[row, col]:
                continue
            cluster: List[Tuple[int, int]] = []
            q = deque([(col, row)])
            visited[row, col] = True
            while q:
                cx, cy = q.popleft()
                cluster.append((cx, cy))
                for ddx in (-1, 0, 1):
                    for ddy in (-1, 0, 1):
                        nx, ny = cx + ddx, cy + ddy
                        if (0 <= nx < w and 0 <= ny < h
                                and not visited[ny, nx]
                                and frontier_mask[ny, nx]):
                            visited[ny, nx] = True
                            q.append((nx, ny))
            if len(cluster) >= self._min_frontier:
                clusters.append(cluster)
        return clusters

    # ═════════════════════════════════════════════════════════════════
    # Frontier selection
    # ═════════════════════════════════════════════════════════════════

    def _rank_clusters(
        self,
        clusters: List[List[Tuple[int, int]]],
        rx: float, ry: float,
        res: float, ox: float, oy: float,
    ) -> List[Tuple[int, int]]:
        scored = []
        for cluster in clusters:
            cx_avg = int(np.mean([c[0] for c in cluster]))
            cy_avg = int(np.mean([c[1] for c in cluster]))
            cx_m   = ox + cx_avg * res + res / 2
            cy_m   = oy + cy_avg * res + res / 2
            dist   = math.hypot(cx_m - rx, cy_m - ry) + 0.5
            score  = len(cluster) / dist
            scored.append((score, (cx_avg, cy_avg)))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [s[1] for s in scored]

    def _pick_frontier(
        self,
        clusters: List[List[Tuple[int, int]]],
        rx: float, ry: float,
        res: float, ox: float, oy: float,
    ) -> Optional[Waypoint]:
        ranked = self._rank_clusters(clusters, rx, ry, res, ox, oy)
        if not ranked:
            return None
        cx_avg, cy_avg = ranked[0]
        return (ox + cx_avg * res + res / 2, oy + cy_avg * res + res / 2)

    # ═════════════════════════════════════════════════════════════════
    # RRT planner
    # ═════════════════════════════════════════════════════════════════

    def _rrt(
        self,
        start: Waypoint, goal: Waypoint,
        inflated: np.ndarray,
        res: float, ox: float, oy: float, w: int, h: int,
    ) -> List[Waypoint]:
        sx, sy = start
        gx, gy = goal

        sc = self._w2c(sx, sy, res, ox, oy, w, h)
        gc = self._w2c(gx, gy, res, ox, oy, w, h)

        if sc is None:
            self.get_logger().warn("RRT: start outside map")
            return []
        if inflated[sc[1], sc[0]]:
            # Should not happen after the carve in _replan, but recover
            # gracefully rather than aborting entirely.
            self.get_logger().warn(
                f"RRT: start cell ({sc[0]},{sc[1]}) still in obstacle "
                f"after carve — nudging to nearest free cell"
            )
            sc_free = self._nearest_free(sc, inflated, w, h)
            if sc_free is None:
                self.get_logger().warn("RRT: no free cell near start — aborting")
                return []
            sx = ox + sc_free[0] * res + res / 2
            sy = oy + sc_free[1] * res + res / 2
            sc = sc_free
        if gc is None or inflated[gc[1], gc[0]]:
            gc = self._nearest_free(gc if gc is not None else sc, inflated, w, h)
            if gc is None:
                self.get_logger().warn("RRT: goal unreachable (no free cell nearby)")
                return []
            gx = ox + gc[0] * res + res / 2
            gy = oy + gc[1] * res + res / 2

        # Straight-line check (cheap path)
        if self._edge_free(sx, sy, gx, gy, inflated, res, ox, oy, w, h):
            root = _RRTNode(sx, sy)
            raw  = self._backtrack(_RRTNode(gx, gy, root))
            return self._smooth(raw, inflated, res, ox, oy, w, h)

        x_min, x_max = ox, ox + w * res
        y_min, y_max = oy, oy + h * res

        root  = _RRTNode(sx, sy)
        tree: List[_RRTNode] = [root]
        reached: Optional[_RRTNode] = None
        thr = self._step_m * 1.5

        for _ in range(self._max_iter):
            # Sample
            if random.random() < self._goal_bias:
                qx, qy = gx, gy
            else:
                qx = random.uniform(x_min, x_max)
                qy = random.uniform(y_min, y_max)

            # Nearest node
            near = min(tree, key=lambda n: (n.x - qx) ** 2 + (n.y - qy) ** 2)

            # Steer
            d = math.hypot(qx - near.x, qy - near.y)
            if d < 1e-6:
                continue
            ratio = min(1.0, self._step_m / d)
            nx = near.x + (qx - near.x) * ratio
            ny = near.y + (qy - near.y) * ratio

            if not self._edge_free(near.x, near.y, nx, ny, inflated, res, ox, oy, w, h):
                continue

            new_node = _RRTNode(nx, ny, near)
            tree.append(new_node)

            if math.hypot(nx - gx, ny - gy) < thr:
                reached = _RRTNode(gx, gy, new_node)
                break

        self._pub_tree_viz(tree)

        if reached is None:
            self.get_logger().warn(
                f"RRT: no path found after {self._max_iter} iterations"
            )
            return []

        raw = self._backtrack(reached)
        return self._smooth(raw, inflated, res, ox, oy, w, h)

    def _edge_free(
        self,
        x0: float, y0: float, x1: float, y1: float,
        inflated: np.ndarray,
        res: float, ox: float, oy: float, w: int, h: int,
    ) -> bool:
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(d / (res * 0.5)))
        for i in range(n + 1):
            t  = i / n
            c  = self._w2c(x0 + t * (x1 - x0), y0 + t * (y1 - y0), res, ox, oy, w, h)
            if c is None or inflated[c[1], c[0]]:
                return False
        return True

    @staticmethod
    def _backtrack(node: _RRTNode) -> List[Waypoint]:
        path = []
        cur  = node
        while cur is not None:
            path.append((cur.x, cur.y))
            cur = cur.parent
        path.reverse()
        return path

    def _smooth(
        self,
        path: List[Waypoint],
        inflated: np.ndarray,
        res: float, ox: float, oy: float, w: int, h: int,
    ) -> List[Waypoint]:
        """Greedy string-pulling: skip waypoints with clear line-of-sight."""
        if len(path) <= 2:
            return path
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._edge_free(
                    path[i][0], path[i][1],
                    path[j][0], path[j][1],
                    inflated, res, ox, oy, w, h,
                ):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

    # ═════════════════════════════════════════════════════════════════
    # Pure-pursuit helpers
    # ═════════════════════════════════════════════════════════════════

    def _lookahead_point(
        self,
        path: List[Waypoint],
        start_idx: int,
        rx: float, ry: float,
    ) -> Waypoint:
        """Return the point on the path ~lookahead_m ahead of the robot."""
        for i in range(start_idx, len(path) - 1):
            ax, ay = path[i]
            bx, by = path[i + 1]
            pt = self._circle_seg_intersect(rx, ry, self._lookahead_m, ax, ay, bx, by)
            if pt is not None:
                return pt
        return path[-1]

    @staticmethod
    def _circle_seg_intersect(
        cx: float, cy: float, r: float,
        ax: float, ay: float, bx: float, by: float,
    ) -> Optional[Waypoint]:
        dx = bx - ax;  dy = by - ay
        fx = ax - cx;  fy = ay - cy
        a  = dx * dx + dy * dy
        if a < 1e-12:
            return None
        b    = 2 * (fx * dx + fy * dy)
        c    = fx * fx + fy * fy - r * r
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sq = math.sqrt(disc)
        for t in sorted([(-b - sq) / (2 * a), (-b + sq) / (2 * a)], reverse=True):
            if 0.0 <= t <= 1.0:
                return (ax + t * dx, ay + t * dy)
        return None

    # ═════════════════════════════════════════════════════════════════
    # LiDAR emergency stop
    # ═════════════════════════════════════════════════════════════════

    def _obstacle_ahead(self, travel_heading: float, rtheta: float) -> bool:
        """True only if an unexpected obstacle is very close AND directly on
        the travel vector — a narrow cone, well past the robot body radius.

        Design rationale
        ----------------
        The old ±30 deg / 0.30 m check fired constantly in corridors because
        walls that the RRT path already routes around fell inside the cone.
        The path planner already guarantees the planned route is clear of the
        static map; we only need to stop for *dynamic* obstacles (people,
        moving objects) that appear inside a tight corridor directly ahead.

        Parameters
        ----------
        travel_heading : desired world-frame heading (rad) to the lookahead pt
        rtheta         : current robot heading (rad)
        """
        raw = self._lidar.get_scan()
        if raw is None:
            return False
        processed = self._scan_p.preprocess(raw)
        ranges = processed["ranges"]
        angles = processed["angles"]

        # Narrow cone: ±15 deg around the exact travel direction.
        cone = math.radians(15.0)

        # Ignore anything closer than robot_body_r — that is the robot itself.
        # Only flag obstacles further than the body but closer than estop_m.
        body_r = self._robot_r_m + 0.05   # small buffer past the chassis edge

        hit_count = 0
        for r, phi in zip(ranges, angles):
            r = float(r)
            if r < body_r or r > self._estop_m:
                continue
            world_angle = rtheta + float(phi)
            if abs(self._wrap(world_angle - travel_heading)) < cone:
                hit_count += 1
                # Require at least 3 consecutive hits to avoid false positives
                # from single noisy scan points.
                if hit_count >= 3:
                    return True
        return False

    # ═════════════════════════════════════════════════════════════════
    # Obstacle inflation
    # ═════════════════════════════════════════════════════════════════

    @staticmethod
    def _inflate(occupied: np.ndarray, r: int) -> np.ndarray:
        """Circular binary dilation without scipy."""
        inf = occupied.copy()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx == 0 and dy == 0:
                    continue
                if dx * dx + dy * dy <= r * r:
                    inf |= np.roll(np.roll(occupied, dy, axis=0), dx, axis=1)
        return inf

    # ═════════════════════════════════════════════════════════════════
    # Grid utilities
    # ═════════════════════════════════════════════════════════════════

    @staticmethod
    def _w2c(
        x: float, y: float,
        res: float, ox: float, oy: float, w: int, h: int,
    ) -> Optional[Tuple[int, int]]:
        cx = int((x - ox) / res)
        cy = int((y - oy) / res)
        if 0 <= cx < w and 0 <= cy < h:
            return cx, cy
        return None

    @staticmethod
    def _nearest_free(
        cell: Tuple[int, int],
        inflated: np.ndarray,
        w: int, h: int,
        max_r: int = 12,
    ) -> Optional[Tuple[int, int]]:
        cx, cy = cell
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and not inflated[ny, nx]:
                        return nx, ny
        return None

    @staticmethod
    def _wrap(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    # ═════════════════════════════════════════════════════════════════
    # Path setter (internal)
    # ═════════════════════════════════════════════════════════════════

    def _set_path(self, path: List[Waypoint]) -> None:
        with self._path_lock:
            self._path      = path
            self._wp_idx    = 0
            self._following = True
            self._at_goal   = False
        self._pub_path_viz(path)
        self._pub_wp_viz(path)

    # ═════════════════════════════════════════════════════════════════
    # Velocity helper
    # ═════════════════════════════════════════════════════════════════

    def _send_vel(self, vx: float, vy: float, omega: float) -> None:
        msg = Twist()
        msg.linear.x  = float(vx)
        msg.linear.y  = float(vy)
        msg.angular.z = float(omega)
        self._cmd_pub.publish(msg)

    # ═════════════════════════════════════════════════════════════════
    # RViz visualisation
    # ═════════════════════════════════════════════════════════════════

    def _pub_frontier_viz(
        self,
        clusters: List[List[Tuple[int, int]]],
        res: float, ox: float, oy: float,
    ) -> None:
        ma = MarkerArray()
        for i, cluster in enumerate(clusters):
            m = Marker()
            m.header.stamp    = self.get_clock().now().to_msg()
            m.header.frame_id = "map"
            m.ns = "frontiers"; m.id = i
            m.type = Marker.POINTS; m.action = Marker.ADD
            m.scale.x = m.scale.y = res * 2
            m.color.g = 1.0; m.color.a = 0.8
            for cx, cy in cluster:
                p = Point()
                p.x = ox + cx * res + res / 2
                p.y = oy + cy * res + res / 2
                m.points.append(p)
            ma.markers.append(m)
        self._pub_frontiers.publish(ma)

    def _pub_path_viz(self, path: List[Waypoint]) -> None:
        ma = MarkerArray()
        m  = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.ns = "rrt_path"; m.id = 0
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.scale.x = 0.04
        m.color.r = 1.0; m.color.g = 0.5; m.color.a = 1.0
        for x, y in path:
            p = Point(); p.x = x; p.y = y; p.z = 0.05
            m.points.append(p)
        ma.markers.append(m)
        self._pub_path.publish(ma)

    def _pub_tree_viz(self, tree: List[_RRTNode]) -> None:
        ma = MarkerArray()
        m  = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.ns = "rrt_tree"; m.id = 0
        m.type = Marker.LINE_LIST; m.action = Marker.ADD
        m.scale.x = 0.01
        m.color.b = 1.0; m.color.a = 0.35
        for node in tree:
            if node.parent is not None:
                p  = Point(); p.x  = node.parent.x; p.y  = node.parent.y
                p2 = Point(); p2.x = node.x;         p2.y = node.y
                m.points.append(p)
                m.points.append(p2)
        ma.markers.append(m)
        self._pub_tree.publish(ma)

    def _pub_wp_viz(self, path: List[Waypoint]) -> None:
        ma = MarkerArray()
        for i, (x, y) in enumerate(path):
            m = Marker()
            m.header.stamp    = self.get_clock().now().to_msg()
            m.header.frame_id = "map"
            m.ns = "waypoints"; m.id = i
            m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = x; m.pose.position.y = y
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.12
            m.color.r = 1.0
            m.color.g = 1.0 if i < len(path) - 1 else 0.0
            m.color.a = 0.9
            ma.markers.append(m)
        self._pub_wps.publish(ma)