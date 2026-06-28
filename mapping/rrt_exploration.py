#!/usr/bin/env python3
"""RRT frontier explorer + integrated turn-then-drive follower (minimal)."""

import math
import random
import time as _time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from scipy.ndimage import binary_dilation

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist


# Cell classification
FREE_MAX     = 30    # 0..FREE_MAX -> free
UNKNOWN_VAL  = -1    # -1 -> frontier; >FREE_MAX -> obstacle

# Follower tuning
MAX_SPEED     = 0.15
TURN_THRESH   = 0.40   # start turning above this heading error (rad)
ALIGN_THRESH  = 0.25   # stop turning below this (rad)
K_V           = 0.80
K_YAW         = 0.40
MAX_WZ        = 0.40
CLOSE_WZ      = 0.50
WP_TOL        = 0.10   # waypoint reached tolerance (m) — exact small-file value
STRAY_DIST    = 0.9    # m; if robot strays farther than this from its target
                       # waypoint, the committed path is stale (SLAM jumped) and
                       # is abandoned so the next tick replans from current pose
GOAL_TIMEOUT  = 80.0
STOP_DIST     = 0.20
STOP_CONE_DEG = 35.0
BACKUP_DIST   = 0.10
BACKUP_SPEED  = 0.06
LIDAR_OFFSET_DEG = 180
CONTROL_HZ    = 20.0


@dataclass
class TreeNode:
    position: Tuple[float, float]
    parent:   Optional[int]


class RRTExplorer(Node):
    """Plans RRT paths over /slam_map and drives them itself."""

    def __init__(self, slam, lidar, preprocessor,
                 step_size: float = 0.5,
                 max_iterations: int = 600,
                 frontier_cluster_radius: float = 0.6,
                 robot_radius_cells: int = 6,
                 sampling_padding_cells: int = 30,
                 completion_rounds: int = 3,
                 slam_map_topic: str = "/slam_map",
                 min_plan_interval: float = 1.5,
                 min_known_cells: int = 200,
                 blacklist_radius: float = 0.5):
        super().__init__("rrt_explorer")

        self.slam, self._lidar, self._prep = slam, lidar, preprocessor

        self.step_size = step_size
        self.max_iterations = max_iterations
        self.frontier_cluster_radius = frontier_cluster_radius
        self.robot_radius_cells = robot_radius_cells
        self.sampling_padding_cells = sampling_padding_cells
        self.completion_rounds = completion_rounds
        self._min_plan_interval = min_plan_interval
        self._min_known_cells = min_known_cells
        self._blacklist_radius = blacklist_radius

        # SLAM map + working snapshots
        self._slam_grid: Optional[np.ndarray] = None
        self._map_info: Optional[dict] = None
        self._map_lock = threading.Lock()
        self._grid_snapshot: Optional[np.ndarray] = None
        self._inflated: Optional[np.ndarray] = None
        self._snap_info: Optional[dict] = None

        # RRT + exploration state
        self.tree_nodes: list = []
        self.frontier_candidates: list = []
        self._no_frontier_rounds = 0
        self._exploration_done = False
        self._returning_home = False
        self._returned_home = False
        self._path_executing = False
        self._destroyed = False
        self._last_plan_time = 0.0
        self._current_goal: Optional[Tuple[float, float]] = None
        self._blacklisted_frontiers: list = []
        self._start_pose: Optional[Tuple[float, float]] = None

        # Follower state
        self._follow_path: List[Tuple[float, float]] = []
        self._follow_index = 1
        self._follow_lock = threading.Lock()
        self._turning = False
        self._goal_time: Optional[float] = None
        self._backing = False
        self._backup_from: Optional[Tuple[float, float]] = None

        plan_cb = MutuallyExclusiveCallbackGroup()
        ctrl_cb = MutuallyExclusiveCallbackGroup()

        self.create_subscription(OccupancyGrid, slam_map_topic,
                                 self._map_callback, 1, callback_group=plan_cb)
        self.path_pub = self.create_publisher(Path, "/exploration_path", 10)
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.path_follower = None  # API compat

        self.create_timer(0.5, self._tick, callback_group=plan_cb)
        self.create_timer(1.0 / CONTROL_HZ, self._control_loop, callback_group=ctrl_cb)

        self.get_logger().info(
            f"[RRT] ready — map={slam_map_topic} step={step_size} "
            f"max_iter={max_iterations} robot_radius_cells={robot_radius_cells}")

    # ---------------------------------------------------------------- lifecycle
    def destroy_node(self):
        self._destroyed = True
        try:
            self._cmd_pub.publish(Twist())
        except Exception:
            pass
        super().destroy_node()

    def notify_path_done(self, reason: str = "completed"):
        self._on_path_done(reason)

    def _map_callback(self, msg: OccupancyGrid):
        info = {"resolution": msg.info.resolution,
                "origin_x": msg.info.origin.position.x,
                "origin_y": msg.info.origin.position.y,
                "width": msg.info.width, "height": msg.info.height}
        grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        with self._map_lock:
            self._slam_grid, self._map_info = grid, info

    # ---------------------------------------------------------------- coords
    def _world_to_cell(self, x, y):
        i = self._snap_info
        return (int((x - i["origin_x"]) / i["resolution"]),
                int((y - i["origin_y"]) / i["resolution"]))

    def _cell_to_world(self, cx, cy):
        i = self._snap_info
        return (i["origin_x"] + cx * i["resolution"], i["origin_y"] + cy * i["resolution"])

    def _in_bounds(self, cx, cy):
        i = self._snap_info
        return 0 <= cx < i["width"] and 0 <= cy < i["height"]

    def _classify_cell(self, cx, cy):
        if not self._in_bounds(cx, cy):
            return "out"
        rx, ry, _ = self.slam.pose
        rcx, rcy = self._world_to_cell(rx, ry)
        if abs(cx - rcx) <= self.robot_radius_cells and abs(cy - rcy) <= self.robot_radius_cells:
            return "free"
        if self._inflated[cy, cx]:
            return "obstacle"
        raw = int(self._grid_snapshot[cy, cx])
        if raw == UNKNOWN_VAL:
            return "frontier"
        return "free" if raw <= FREE_MAX else "obstacle"

    # ---------------------------------------------------------------- explore
    def explore_step(self):
        if self._exploration_done and not self._returned_home:
            self._returned_home = self._returning_home = True
            return self._plan_return_home()
        if self._exploration_done:
            return None, None

        self._reset_tree()
        if not self._grow_tree():
            self._no_frontier_rounds += 1
            if self._no_frontier_rounds >= self.completion_rounds:
                self._exploration_done = True
                self.get_logger().info("[RRT] Exploration complete.")
            return None, None

        self._no_frontier_rounds = 0
        best = self._select_best_frontier()
        if best is None:
            self._no_frontier_rounds += 1
            return None, None
        return best["position"], self._backtrack_path(best["parent"], best["position"])

    def _check_edge_return(self, c0, c1):
        for cx, cy in self._bresenham_line(c0[0], c0[1], c1[0], c1[1]):
            if self._classify_cell(cx, cy) == "obstacle":
                return False
        return True

    def _plan_return_home(self):
        home = self._start_pose if self._start_pose is not None else (0.0, 0.0)
        x, y, _ = self.slam.pose
        self.tree_nodes = [TreeNode((float(x), float(y)), None)]
        for _ in range(self.max_iterations * 2):
            sample = home if random.random() < 0.5 else self._random_sample()
            ni = self._find_nearest(sample)
            nearest = self.tree_nodes[ni].position
            new_pt = self._steer(nearest, sample)
            if not self._check_edge_return(self._world_to_cell(*nearest),
                                           self._world_to_cell(*new_pt)):
                continue
            idx = len(self.tree_nodes)
            self.tree_nodes.append(TreeNode(new_pt, ni))
            if math.hypot(new_pt[0] - home[0], new_pt[1] - home[1]) < self.step_size:
                return home, self._backtrack_path(idx, home)
        self._returned_home = self._returning_home = False
        return None, None

    # ---------------------------------------------------------------- tree
    def _reset_tree(self):
        x, y, _ = self.slam.pose
        self.tree_nodes = [TreeNode((float(x), float(y)), None)]
        self.frontier_candidates = []

    def _grow_tree(self):
        grid = self._grid_snapshot
        if int(np.sum(grid != UNKNOWN_VAL)) == 0:
            return False
        rx, ry, _ = self.slam.pose
        rcx, rcy = self._world_to_cell(rx, ry)
        if not self._in_bounds(rcx, rcy):
            self.get_logger().error("[RRT] robot cell out of map.")
            return False
        if self._classify_cell(rcx, rcy) == "obstacle":
            self.get_logger().error("[RRT] robot cell is obstacle — root blocked.")
            return False

        x0, x1, y0, y1 = self._sampling_bounds()
        for _ in range(self.max_iterations):
            sample = (random.uniform(x0, x1), random.uniform(y0, y1))
            ni = self._find_nearest(sample)
            nearest = self.tree_nodes[ni].position
            new_pt = self._steer(nearest, sample)
            res = self._check_edge(self._world_to_cell(*nearest),
                                   self._world_to_cell(*new_pt))
            if res == "free":
                self.tree_nodes.append(TreeNode(new_pt, ni))
            elif res == "frontier":
                self.frontier_candidates.append({"position": new_pt, "parent": ni})
        return len(self.frontier_candidates) > 0

    def _inflate_obstacles(self, radius):
        if self._grid_snapshot is None:
            return np.zeros((1, 1), dtype=bool)
        mask = (self._grid_snapshot > FREE_MAX) & (self._grid_snapshot != UNKNOWN_VAL)
        if radius <= 0:
            return mask
        struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        return binary_dilation(mask, structure=struct)

    # ---------------------------------------------------------------- doors
    def _paint_door(self, p1, p2, width_m, value):
        with self._map_lock:
            if self._slam_grid is None or self._map_info is None:
                return
            info, grid = self._map_info, self._slam_grid
            res = info["resolution"]
            h, w = grid.shape
            rc = max(1, int(width_m / res))

            def w2c(x, y):
                return (int((x - info["origin_x"]) / res), int((y - info["origin_y"]) / res))

            c0, c1 = w2c(*p1), w2c(*p2)
            for cx, cy in self._bresenham_line(c0[0], c0[1], c1[0], c1[1]):
                grid[max(0, cy - rc):min(h, cy + rc + 1),
                     max(0, cx - rc):min(w, cx + rc + 1)] = value

    def block_door_in_costmap(self, p1, p2, width_m: float = 0.3):
        self._paint_door(p1, p2, width_m, 100)
        self.get_logger().info(f"[RRT] Door blocked {p1} -> {p2}")

    def unblock_door(self, p1, p2, width_m: float = 0.3):
        self._paint_door(p1, p2, width_m, -1)
        self.get_logger().info(f"[RRT] Door unblocked {p1} -> {p2}")

    # ---------------------------------------------------------------- frontier
    def _is_blacklisted(self, pos):
        return any(math.hypot(pos[0] - bx, pos[1] - by) < self._blacklist_radius
                   for bx, by in self._blacklisted_frontiers)

    def _select_best_frontier(self):
        rx, ry, _ = self.slam.pose
        robot = np.array([rx, ry])
        positions = np.array([c["position"] for c in self.frontier_candidates])
        best_score, best_idx = -1.0, None
        for i in range(len(self.frontier_candidates)):
            pos = positions[i]
            if self._is_blacklisted(pos):
                continue
            dist = float(np.linalg.norm(pos - robot))
            count = float(np.sum(np.linalg.norm(positions - pos, axis=1)
                                 < self.frontier_cluster_radius))
            score = count / (1.0 + 0.4 * dist)
            if score > best_score:
                best_idx, best_score = i, score
        if best_idx is None:
            return None
        return self.frontier_candidates[best_idx]

    # ---------------------------------------------------------------- rrt helpers
    def _random_sample(self):
        x0, x1, y0, y1 = self._sampling_bounds()
        return (random.uniform(x0, x1), random.uniform(y0, y1))

    def _sampling_bounds(self):
        info = self._snap_info
        known = self._grid_snapshot != UNKNOWN_VAL
        if not known.any() or int(np.sum(known)) < self._min_known_cells:
            rx, ry, _ = self.slam.pose
            r = 3.0
            return (rx - r, rx + r, ry - r, ry + r)
        ys, xs = np.where(known)
        pad = self.sampling_padding_cells
        xmin = max(0, int(xs.min()) - pad); xmax = min(info["width"], int(xs.max()) + pad)
        ymin = max(0, int(ys.min()) - pad); ymax = min(info["height"], int(ys.max()) + pad)
        x0, y0 = self._cell_to_world(xmin, ymin)
        x1, y1 = self._cell_to_world(xmax, ymax)
        return x0, x1, y0, y1

    def _find_nearest(self, sample):
        positions = np.array([n.position for n in self.tree_nodes])
        return int(np.argmin(np.linalg.norm(positions - np.array(sample), axis=1)))

    def _steer(self, p0, p1):
        p0, p1 = np.array(p0), np.array(p1)
        d = np.linalg.norm(p1 - p0)
        if d < 1e-6:
            return (float(p0[0]), float(p0[1]))
        p = p0 + (p1 - p0) / d * min(self.step_size, d)
        return (float(p[0]), float(p[1]))

    def _check_edge(self, c0, c1):
        cells = self._bresenham_line(c0[0], c0[1], c1[0], c1[1])
        for cx, cy in cells[:-1]:
            if self._classify_cell(cx, cy) == "obstacle":
                return None
        terminal = self._classify_cell(*cells[-1])
        return terminal if terminal in ("free", "frontier") else None

    @staticmethod
    def _bresenham_line(x0, y0, x1, y1):
        cells = []
        dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
        dy = abs(y1 - y0); sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy: err -= dy; x += sx
            if e2 < dx:  err += dx; y += sy
        return cells

    def _backtrack_path(self, parent, leaf):
        path = [leaf]
        idx = parent
        while idx is not None:
            path.append(self.tree_nodes[idx].position)
            idx = self.tree_nodes[idx].parent
        path.reverse()
        return path

    # ---------------------------------------------------------------- tick (plan)
    def _tick(self):
        if self._destroyed or self._path_executing or self._returned_home:
            return
        with self._map_lock:
            if self._slam_grid is None:
                return
            self._grid_snapshot = self._slam_grid.copy()
            self._snap_info = dict(self._map_info)

        now = _time.monotonic()
        if now - self._last_plan_time < self._min_plan_interval:
            return
        self._last_plan_time = now

        try:
            if int(np.sum(self._grid_snapshot != UNKNOWN_VAL)) < self._min_known_cells:
                return
            self._inflated = self._inflate_obstacles(self.robot_radius_cells)

            goal, path = self.explore_step()
            if goal is None or path is None:
                return
            self._current_goal = goal

            if self._start_pose is None and not self._returning_home:
                rx, ry, _ = self.slam.pose
                self._start_pose = (float(rx), float(ry))

            msg = Path()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            for x, y in path:
                ps = PoseStamped()
                ps.header = msg.header
                ps.pose.position.x = float(x)
                ps.pose.position.y = float(y)
                msg.poses.append(ps)
            if self._destroyed:
                return
            self.path_pub.publish(msg)
            self._start_following(path)

            phase = "RETURN" if self._returning_home else "EXPLORE"
            self.get_logger().info(
                f"[RRT] [{phase}] following {len(path)} wpts -> "
                f"({goal[0]:.2f}, {goal[1]:.2f})")
        except Exception:
            import traceback
            self.get_logger().error("[RRT] _tick crashed:\n" + traceback.format_exc())

    # ---------------------------------------------------------------- follower
    def _start_following(self, path):
        with self._follow_lock:
            self._follow_path = [(float(x), float(y)) for x, y in path]
            self._follow_index = 1
        self._turning = False
        self._goal_time = _time.monotonic()
        self._path_executing = True

    def _on_path_done(self, reason):
        if reason == "path_timeout" and self._current_goal is not None:
            self._blacklisted_frontiers.append(self._current_goal)
        self.get_logger().info(f"[RRT] path done ({reason})")
        with self._follow_lock:
            self._follow_path = []
            self._follow_index = 1
        self._cmd_pub.publish(Twist())
        self._turning = False
        self._goal_time = None
        self._backing = False
        self._backup_from = None
        self._path_executing = False

    def _cell_occ_latest(self, x, y):
        with self._map_lock:
            if self._slam_grid is None or self._map_info is None:
                return 0
            info = self._map_info
            gx = int((x - info["origin_x"]) / info["resolution"])
            gy = int((y - info["origin_y"]) / info["resolution"])
            if 0 <= gx < info["width"] and 0 <= gy < info["height"]:
                return int(self._slam_grid[gy, gx])
        return 0

    def _path_blocked(self):
        with self._follow_lock:
            if not self._follow_path:
                return False
            path, idx = list(self._follow_path), self._follow_index
        with self._map_lock:
            if self._slam_grid is None or self._map_info is None:
                return False
            res = self._map_info["resolution"]
        for i in range(max(1, idx), len(path)):
            x0, y0 = path[i - 1]; x1, y1 = path[i]
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / (res * 0.5)))
            for k in range(n + 1):
                t = k / n
                if self._cell_occ_latest(x0 + t * (x1 - x0), y0 + t * (y1 - y0)) >= 50:
                    return True
        return False

    def _front_blocked(self):
        scan = self._lidar.get_scan()
        if scan is None:
            return False
        p = self._prep.preprocess(scan)
        return self._sector_min(p, -STOP_CONE_DEG, STOP_CONE_DEG) < STOP_DIST

    def _cone_mins(self):
        scan = self._lidar.get_scan()
        if scan is None:
            return float("inf"), float("inf")
        p = self._prep.preprocess(scan)
        front = self._sector_min(p, -STOP_CONE_DEG, STOP_CONE_DEG)
        rear = min(self._sector_min(p, 180.0 - STOP_CONE_DEG, 180.0),
                   self._sector_min(p, -180.0, -180.0 + STOP_CONE_DEG))
        return front, rear

    def _sector_min(self, processed, a_min, a_max):
        off = LIDAR_OFFSET_DEG
        return self._prep.get_sector_min(processed, a_min + off, a_max + off)

    def _control_loop(self):
        if self._destroyed:
            return

        # Backup maneuver (reverse, then replan)
        if self._backing:
            pose = self.slam.pose
            if self._backup_from is None:
                self._backing = False
                self._path_executing = False
                return
            moved = math.hypot(pose[0] - self._backup_from[0], pose[1] - self._backup_from[1])
            _f, rear = self._cone_mins()
            if moved >= BACKUP_DIST or rear < STOP_DIST:
                self._cmd_pub.publish(Twist())
                self._backing = False
                self._backup_from = None
                self._path_executing = False
                return
            cmd = Twist(); cmd.linear.x = -BACKUP_SPEED
            self._cmd_pub.publish(cmd)
            return

        if not self._path_executing:
            return
        with self._follow_lock:
            if not self._follow_path:
                return

        if self._path_blocked():
            self.get_logger().warn("[FOLLOW] path crosses occupied cell; replanning.",
                                   throttle_duration_sec=2.0)
            self._on_path_done("replan")
            return

        if self._goal_time is not None and _time.monotonic() - self._goal_time > GOAL_TIMEOUT:
            self.get_logger().warn("[FOLLOW] timeout; blacklisting.", throttle_duration_sec=2.0)
            self._on_path_done("path_timeout")
            return

        pose = self.slam.pose
        rx, ry, rth = float(pose[0]), float(pose[1]), float(pose[2])

        # Exact small-file follower: drive toward the next UNREACHED waypoint,
        # node by node. No lookahead / no shortcutting, so the robot tracks the
        # planned path itself instead of cutting across it.
        with self._follow_lock:
            while self._follow_index < len(self._follow_path):
                tx, ty = self._follow_path[self._follow_index]
                if math.hypot(tx - rx, ty - ry) < WP_TOL:
                    self._follow_index += 1
                else:
                    break
            if self._follow_index >= len(self._follow_path):
                done, target = True, None
            else:
                done, target = False, self._follow_path[self._follow_index]

        if done:
            self.get_logger().info("[FOLLOW] reached goal.")
            self._on_path_done("completed")
            return

        tx, ty = target
        ex, ey = tx - rx, ty - ry
        dist = math.hypot(ex, ey)

        # Re-anchor guard: a healthy path starts at the robot, so the target is
        # always within ~step_size. If it's much farther, the SLAM pose jumped
        # and this path is stale — abandon it and replan from the current pose.
        if dist > STRAY_DIST:
            self.get_logger().warn(
                "[FOLLOW] robot far from path (SLAM jump?); replanning.",
                throttle_duration_sec=2.0)
            self._on_path_done("replan")
            return

        yaw_err = _wrap_angle(math.atan2(ey, ex) - rth)

        if self._turning:
            if abs(yaw_err) < ALIGN_THRESH:
                self._turning = False
        elif abs(yaw_err) > TURN_THRESH:
            self._turning = True

        cmd = Twist()
        if self._turning:
            wz = _clamp(K_YAW * yaw_err, -MAX_WZ, MAX_WZ)
            if self._front_blocked():
                wz = _clamp(wz, -CLOSE_WZ, CLOSE_WZ)
            cmd.angular.z = float(wz)
        else:
            if self._front_blocked():
                self.get_logger().warn("[FOLLOW] obstacle ahead; backing up.",
                                       throttle_duration_sec=2.0)
                self._start_backup()
                return
            cmd.linear.x = float(min(MAX_SPEED, K_V * dist))
        self._cmd_pub.publish(cmd)

    def _start_backup(self):
        self._cmd_pub.publish(Twist())
        if self._current_goal is not None:
            self._blacklisted_frontiers.append(self._current_goal)
        with self._follow_lock:
            self._follow_path = []
            self._follow_index = 1
        self._turning = False
        self._goal_time = None
        pose = self.slam.pose
        if pose is not None:
            self._backup_from = (float(pose[0]), float(pose[1]))
            self._backing = True
            self._path_executing = True
        else:
            self._backup_from = None
            self._backing = False
            self._path_executing = False


def _wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))