#!/usr/bin/env python3
"""
RRT-based frontier explorer  +  integrated path follower.
============================================================

This node both PLANS exploration paths (RRT over the EKF-SLAM
OccupancyGrid) and FOLLOWS them itself — no separate PathFollower node,
no notify_path_done() wiring across nodes.

How planning and following are coordinated
------------------------------------------
  Two timers, one shared flag (_path_executing):

    _tick()          @ 0.5 s   — PLANNING.  Runs only when _path_executing
                                 is False.  Plans a path, publishes it to
                                 /exploration_path (for RViz), stores it
                                 internally, and sets _path_executing=True.

    _control_loop()  @ 20 Hz   — FOLLOWING.  Runs only when _path_executing
                                 is True.  Drives the robot waypoint by
                                 waypoint.  When the path finishes (reached
                                 or stuck) it calls _on_path_done(reason),
                                 which clears _path_executing so the next
                                 _tick() can plan again.

  Because _on_path_done() is an internal method, the old cross-node
  notify_path_done() contract is preserved exactly: a stuck path still
  blacklists its goal, a completed path still frees the planner.

Following logic (simplified per request)
----------------------------------------
  Motion model: DIFFERENTIAL-STYLE (no lateral strafe)
    linear.x  = forward only (proportional to distance to waypoint)
    linear.y  = always 0.0
    angular.z = heading correction

  Obstacle avoidance: EMERGENCY tier ONLY.
    If something is closer than EMERGENCY_FRONT_DIST in front, stop all
    translation and spin in place toward the more open side until clear.
    The SLOW and NORMAL "nudge" tiers are intentionally removed — the RRT
    path is already collision-checked, and keeping only EMERGENCY makes
    the follow behaviour easy to reason about while debugging.

Everything about the RRT planner itself (tree growth, frontier scoring,
blacklist, return-to-origin, cell classification) is UNCHANGED from the
standalone version.

TUNING GUIDE
════════════════════════════════════════════════════════════════════════

── Cell classification thresholds ───────────────────────────────────
  FREE_MAX      (default 30)   cells 0..FREE_MAX are free/traversable
  OBSTACLE_MIN  (default 31)   cells OBSTACLE_MIN..100 are obstacle
  UNKNOWN_VAL   (-1)           exactly -1 is a frontier (unseen)

── RRT constructor parameters (unchanged) ───────────────────────────
  step_size, max_iterations, frontier_cluster_radius, robot_radius_cells,
  sampling_padding_cells, completion_rounds, min_plan_interval,
  min_known_cells, blacklist_radius  — see inline docs below.

── Path-following constants (new — top of file) ─────────────────────
  KP_LINEAR / KD_LINEAR  forward PID gains
  KP_ANGULAR             heading correction gain
  MAX_LINEAR / MAX_ANGULAR  velocity caps
  WAYPOINT_TOLERANCE / FINAL_WAYPOINT_TOLERANCE  acceptance radii
  HEADING_GATE_RAD / HEADING_SCALE_MIN  rotate-before-drive gate
  STUCK_TIMEOUT / STUCK_DIST_THRESHOLD  stuck detection
  EMERGENCY_FRONT_DIST / EMERGENCY_TURN_SPEED  emergency tier
  LIDAR_OFFSET_DEG       LiDAR mount rotation (180° here)
  CONTROL_HZ             follow loop rate
"""

import math
import random
import time as _time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, List
import json

import numpy as np
from scipy.ndimage import binary_dilation

from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist


# ======================================================================= #
#  Cell classification thresholds  (RRT planner)
# ======================================================================= #
FREE_MAX     = 30    # 0  .. FREE_MAX       → free
OBSTACLE_MIN = 31    # OBSTACLE_MIN .. 100  → obstacle  (incl. uncertain 31-59)
UNKNOWN_VAL  = -1    # exactly -1           → frontier


# ======================================================================= #
#  Path-following constants  (integrated follower)
# ======================================================================= #

# ── Forward / angular PID ─────────────────────────────────────────────
KP_LINEAR   = 0.50   # proportional gain on distance to waypoint
KI_LINEAR   = 0.00   # integral gain (keep 0 unless persistent drift)
KD_LINEAR   = 0.03   # derivative gain — damps overshoot on approach
KP_ANGULAR  = 1.00   # proportional gain on heading error (rad → rad/s)

# ── Velocity limits ───────────────────────────────────────────────────
MAX_LINEAR  = 0.22   # m/s   — forward speed cap
MAX_ANGULAR = 0.80   # rad/s — angular speed cap

# ── Waypoint advancement ──────────────────────────────────────────────
WAYPOINT_TOLERANCE       = 0.30  # m — intermediate waypoint acceptance radius
FINAL_WAYPOINT_TOLERANCE = 0.20  # m — tighter for the last waypoint

# ── Heading gate (rotate before driving) ──────────────────────────────
HEADING_GATE_RAD  = 0.35   # rad (~20°)
HEADING_SCALE_MIN = 0.10   # vx scale floor during rotation

# ── Stuck detection ───────────────────────────────────────────────────
STUCK_TIMEOUT        = 10.0  # s
STUCK_DIST_THRESHOLD = 0.10  # m — minimum displacement per window

# ── EMERGENCY obstacle tier (only tier kept) ──────────────────────────
EMERGENCY_FRONT_DIST = 0.15  # m   — full stop + spin below this
EMERGENCY_TURN_SPEED = 0.40  # rad/s — spin speed in EMERGENCY

LIDAR_OFFSET_DEG = 180       # LiDAR mounted backwards (same as SafeLidarMotion)

# ── Control loop rate ─────────────────────────────────────────────────
CONTROL_HZ = 20.0


# ======================================================================= #
#  Data classes
# ======================================================================= #

@dataclass
class TreeNode:
    position: Tuple[float, float]
    parent:   Optional[int]


# ======================================================================= #
#  RRTExplorer  (planner + follower)
# ======================================================================= #

class RRTExplorer(Node):
    """
    Autonomous frontier explorer that BOTH plans (RRT over /slam_map)
    and follows the resulting path itself.

    Wiring in main.py
    -----------------
        rrt = RRTExplorer(
            slam=slam, lidar=lidar, preprocessor=preprocessor, ...
        )
        # No PathFollower node needed. No notify_path_done wiring needed.

    Required new constructor args vs the old planner-only version:
        lidar         — LidarSensor (for EMERGENCY avoidance during follow)
        preprocessor  — ScanPreprocessor (sector queries)
    """

    def __init__(
        self,
        slam,
        lidar,
        preprocessor,
        step_size:               float = 0.35,
        max_iterations:          int   = 600,
        frontier_cluster_radius: float = 0.6,
        robot_radius_cells:      int   = 6,
        sampling_padding_cells:  int   = 30,
        completion_rounds:       int   = 3,
        slam_map_topic:          str   = "/slam_map",
        min_plan_interval:       float = 1.5,
        min_known_cells:         int   = 200,
        blacklist_radius:        float = 0.5,
    ):
        super().__init__("rrt_explorer")

        self.slam   = slam
        self._lidar = lidar
        self._prep  = preprocessor

        self.step_size               = step_size
        self.max_iterations          = max_iterations
        self.frontier_cluster_radius = frontier_cluster_radius
        self.robot_radius_cells      = robot_radius_cells
        self.sampling_padding_cells  = sampling_padding_cells
        self.completion_rounds       = completion_rounds
        self._min_plan_interval      = min_plan_interval
        self._min_known_cells        = min_known_cells
        self._blacklist_radius       = blacklist_radius

        # ── SLAM map storage ─────────────────────────────────────────
        self._slam_grid:  Optional[np.ndarray] = None
        self._map_info:   Optional[dict]        = None
        self._map_lock    = threading.Lock()

        # Working snapshots taken at the start of each planning tick
        self._grid_snapshot: Optional[np.ndarray] = None
        self._inflated:      Optional[np.ndarray] = None
        self._snap_info:     Optional[dict]        = None

        # ── RRT state ────────────────────────────────────────────────
        self.tree_nodes:          list = []
        self.frontier_candidates: list = []

        # ── Exploration bookkeeping ──────────────────────────────────
        self._no_frontier_rounds = 0
        self._exploration_done   = False
        self._returning_home     = False
        self._returned_home      = False
        self._path_executing     = False
        self._destroyed          = False
        self._last_plan_time     = 0.0
        self._tick_count         = 0
        self._current_goal: Optional[Tuple[float, float]] = None

        # Blacklist: list of (x, y) world positions of stuck goals
        self._blacklisted_frontiers: list = []

        self._start_pose: Optional[Tuple[float, float]] = None

        # ── FOLLOWER state (new) ─────────────────────────────────────
        self._follow_path:   List[Tuple[float, float]] = []
        self._follow_index:  int   = 0
        self._follow_lock    = threading.Lock()

        self._err_fwd_prev   = 0.0
        self._err_fwd_integ  = 0.0
        self._last_ctrl_t    = _time.monotonic()

        self._stuck_ref_pos:  Optional[Tuple[float, float]] = None
        self._stuck_ref_time: float = _time.monotonic()
        self._last_turn_dir  = 1

        # ── ROS interfaces ───────────────────────────────────────────
        self._map_sub = self.create_subscription(
            OccupancyGrid,
            slam_map_topic,
            self._map_callback,
            1,
        )
        self.path_pub = self.create_publisher(Path, "/exploration_path", 10)
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Kept for API compatibility; no longer required for operation.
        self.path_follower = None

        # Planning timer (slow) and following timer (fast)
        self.create_timer(0.5, self._tick)
        self.create_timer(1.0 / CONTROL_HZ, self._control_loop)

        self.get_logger().info(
            f"[RRT INIT] RRT Explorer + integrated follower ready — "
            f"map topic: {slam_map_topic}, "
            f"FREE_MAX={FREE_MAX} OBSTACLE_MIN={OBSTACLE_MIN}, "
            f"step_size={step_size} max_iter={max_iterations} "
            f"robot_radius_cells={robot_radius_cells}, "
            f"blacklist_radius={blacklist_radius:.2f} m | "
            f"follow: KP_lin={KP_LINEAR} KP_ang={KP_ANGULAR} "
            f"max_lin={MAX_LINEAR} max_ang={MAX_ANGULAR} "
            f"emergency_front={EMERGENCY_FRONT_DIST}m"
        )

    # ================================================================== #
    #  ROS lifecycle
    # ================================================================== #

    def destroy_node(self):
        self._destroyed = True
        # Stop the robot on shutdown
        try:
            self._cmd_pub.publish(Twist())
        except Exception:
            pass
        super().destroy_node()

    def notify_path_done(self, reason: str = "completed"):
        """
        Retained for API compatibility (e.g. ArucoMonitor may call it to
        abort the current path).  Internally the follower calls
        _on_path_done() directly, but routing through here keeps the old
        contract working for any external caller.
        """
        self._on_path_done(reason)

    # ================================================================== #
    #  SLAM map callback
    # ================================================================== #

    def _map_callback(self, msg: OccupancyGrid):
        info = {
            "resolution": msg.info.resolution,
            "origin_x":   msg.info.origin.position.x,
            "origin_y":   msg.info.origin.position.y,
            "width":      msg.info.width,
            "height":     msg.info.height,
        }
        grid = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width
        )
        with self._map_lock:
            self._slam_grid = grid
            self._map_info  = info

    # ================================================================== #
    #  Coordinate helpers
    # ================================================================== #

    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        info = self._snap_info
        cx = int((x - info["origin_x"]) / info["resolution"])
        cy = int((y - info["origin_y"]) / info["resolution"])
        return cx, cy

    def _cell_to_world(self, cx: int, cy: int) -> Tuple[float, float]:
        info = self._snap_info
        x = info["origin_x"] + cx * info["resolution"]
        y = info["origin_y"] + cy * info["resolution"]
        return x, y

    def _in_bounds(self, cx: int, cy: int) -> bool:
        info = self._snap_info
        return 0 <= cx < info["width"] and 0 <= cy < info["height"]

    # ================================================================== #
    #  Cell classification
    # ================================================================== #

    def _classify_cell(self, cx: int, cy: int) -> str:
        if not self._in_bounds(cx, cy):
            return "out"

        # Safety radius around robot chassis → always free
        rx, ry, _ = self.slam.pose
        rcx, rcy = self._world_to_cell(rx, ry)
        if abs(cx - rcx) <= self.robot_radius_cells and abs(cy - rcy) <= self.robot_radius_cells:
            return "free"

        if self._inflated[cy, cx]:
            return "obstacle"

        raw = int(self._grid_snapshot[cy, cx])
        if raw == UNKNOWN_VAL:
            return "frontier"
        if raw <= FREE_MAX:
            return "free"
        return "obstacle"

    # ================================================================== #
    #  Public exploration entry point
    # ================================================================== #

    def explore_step(self):
        # ── Return-to-origin phase ───────────────────────────────────
        if self._exploration_done and not self._returned_home:
            self.get_logger().info(
                "[RRT] Exploration complete — planning return path to origin"
            )
            self._returned_home  = True
            self._returning_home = True
            goal, path = self._plan_return_home()
            return goal, path

        if self._exploration_done:
            return None, None

        # ── Normal exploration ───────────────────────────────────────
        self._reset_tree()
        found = self._grow_tree()

        if not found:
            self._no_frontier_rounds += 1
            self.get_logger().warn(
                f"[BUG ① candidate] No frontiers found — "
                f"round {self._no_frontier_rounds}/{self.completion_rounds}. "
                f"If this keeps happening the robot will stop. "
                f"Check: is /slam_map arriving? Are known cells > {self._min_known_cells}? "
                f"Is FREE_MAX={FREE_MAX} too tight?"
            )
            if self._no_frontier_rounds >= self.completion_rounds:
                self._exploration_done = True
                self.get_logger().info(
                    "[RRT] Exploration complete — all frontiers exhausted."
                )
            return None, None

        self._no_frontier_rounds = 0
        best = self._select_best_frontier()
        if best is None:
            self._no_frontier_rounds += 1
            return None, None

        path = self._backtrack_path(best["parent"], best["position"])
        return best["position"], path

    # ================================================================== #
    #  RETURN TO ORIGIN
    # ================================================================== #

    def _check_edge_return(self, c0: Tuple[int, int],
                            c1: Tuple[int, int]) -> bool:
        cells = self._bresenham_line(c0[0], c0[1], c1[0], c1[1])
        for cx, cy in cells:
            if self._classify_cell(cx, cy) == "obstacle":
                return False
        return True

    def _plan_return_home(self) -> Tuple:
        home = self._start_pose if self._start_pose is not None else (0.0, 0.0)
        self.get_logger().info(f"[RRT RETURN] Return-home target: {home}")

        x, y, _ = self.slam.pose
        self.tree_nodes = [TreeNode(position=(float(x), float(y)), parent=None)]

        for _ in range(self.max_iterations * 2):
            sample = home if random.random() < 0.50 else self._random_sample()

            nearest_idx = self._find_nearest(sample)
            nearest     = self.tree_nodes[nearest_idx].position
            new_point   = self._steer(nearest, sample)

            c0 = self._world_to_cell(*nearest)
            c1 = self._world_to_cell(*new_point)

            if not self._check_edge_return(c0, c1):
                continue

            new_idx = len(self.tree_nodes)
            self.tree_nodes.append(
                TreeNode(position=new_point, parent=nearest_idx)
            )
            if math.hypot(new_point[0] - home[0],
                           new_point[1] - home[1]) < self.step_size:
                path = self._backtrack_path(new_idx, home)
                self.get_logger().info(
                    f"[RRT RETURN] Return path found: {len(path)} waypoints -> {home}"
                )
                return home, path

        self.get_logger().warn(
            f"[BUG ②] Return RRT failed — could not find a collision-free path to {home}. "
            f"Will retry on next tick."
        )
        self._returned_home  = False
        self._returning_home = False
        return None, None

    # ================================================================== #
    #  TREE GROWTH
    # ================================================================== #

    def _reset_tree(self):
        x, y, _ = self.slam.pose
        self.tree_nodes          = [TreeNode(position=(float(x), float(y)), parent=None)]
        self.frontier_candidates = []

    def _grow_tree(self) -> bool:
        info = self._snap_info
        grid = self._grid_snapshot

        known_count   = int(np.sum(grid != UNKNOWN_VAL))
        rx, ry, _     = self.slam.pose
        rcx, rcy      = self._world_to_cell(rx, ry)
        in_bounds      = self._in_bounds(rcx, rcy)
        robot_status   = self._classify_cell(rcx, rcy) if in_bounds else "OUT_OF_MAP"

        # ── BUG DETECTOR: zero known cells ──────────────────────────
        if known_count == 0:
            self.get_logger().error(
                f"[BUG ①] Grid snapshot has 0 known cells. "
                f"/slam_map arrived but every cell is -1 (unknown). "
                f"Check SLAM correction is running and /scan is publishing. "
                f"Robot will stay still."
            )
            return False

        # ── BUG DETECTOR: sparse map ─────────────────────────────────
        if known_count < self._min_known_cells:
            self.get_logger().warn(
                f"[BUG ① candidate] Only {known_count} known cells "
                f"(threshold={self._min_known_cells}). "
                f"Map too sparse — RRT will struggle to find free paths. "
                f"Robot pose: ({rx:.2f}, {ry:.2f}). "
                f"Map: {info['width']}x{info['height']} @ {info['resolution']}m, "
                f"origin ({info['origin_x']:.1f},{info['origin_y']:.1f})."
            )

        # ── BUG DETECTOR: robot outside map ──────────────────────────
        if not in_bounds:
            self.get_logger().error(
                f"[BUG ①] Robot at world ({rx:.2f}, {ry:.2f}) maps to cell "
                f"({rcx},{rcy}) which is OUTSIDE the map "
                f"(map is 0..{info['width']-1} x 0..{info['height']-1}). "
                f"Map origin ({info['origin_x']:.1f},{info['origin_y']:.1f}), "
                f"size {info['width']*info['resolution']:.1f}x{info['height']*info['resolution']:.1f}m. "
                f"Increase map_size_m in EKF or fix odometry offset. "
                f"Robot will stay still."
            )
            return False

        # ── BUG DETECTOR: robot cell occupied ────────────────────────
        if robot_status == "obstacle":
            raw_val = int(self._grid_snapshot[rcy, rcx])
            self.get_logger().error(
                f"[BUG ①③] Robot cell ({rcx},{rcy}) classified as OBSTACLE "
                f"(raw={raw_val}, FREE_MAX={FREE_MAX}, "
                f"inflated={bool(self._inflated[rcy,rcx])}). "
                f"Every RRT branch from root is rejected — robot stays still. "
                f"Fix: raise FREE_MAX, reduce robot_radius_cells, "
                f"or check SLAM for ghost obstacles near robot."
            )
            return False

        if robot_status == "frontier":
            self.get_logger().warn(
                f"[BUG ① candidate] Robot cell ({rcx},{rcy}) is UNKNOWN (frontier). "
                f"SLAM hasn't marked the robot's immediate surroundings as free yet. "
                f"Most branches from root will be rejected. "
                f"Robot may stay still until SLAM fills in nearby cells."
            )

        # ── Map state summary ─────────────────────────────────────────
        free_count     = int(np.sum(
            (grid != UNKNOWN_VAL) & (grid <= FREE_MAX)
        ))
        obs_count      = int(np.sum(
            (grid != UNKNOWN_VAL) & (grid > FREE_MAX)
        ))
        unknown_count  = int(np.sum(grid == UNKNOWN_VAL))
        inflated_count = int(np.sum(self._inflated))

        self.get_logger().info(
            f"[RRT GROW] known={known_count} free={free_count} "
            f"obs={obs_count} unknown={unknown_count} "
            f"inflated={inflated_count} | "
            f"robot=({rx:.2f},{ry:.2f}) cell=({rcx},{rcy}) "
            f"status={robot_status} | "
            f"map {info['width']}x{info['height']} @ {info['resolution']:.3f} m/cell"
        )

        rejected       = 0
        free_added     = 0
        frontier_added = 0
        t0 = _time.monotonic()

        # SPEED FIX: Fetch the sampling boundaries ONCE per tick
        x0_bound, x1_bound, y0_bound, y1_bound = self._sampling_bounds()

        for _ in range(self.max_iterations):
            sample = (random.uniform(x0_bound, x1_bound), random.uniform(y0_bound, y1_bound))

            nearest_idx = self._find_nearest(sample)
            nearest     = self.tree_nodes[nearest_idx].position
            new_point   = self._steer(nearest, sample)

            c0 = self._world_to_cell(*nearest)
            c1 = self._world_to_cell(*new_point)

            result = self._check_edge(c0, c1)

            if result is None:
                rejected += 1
                continue
            elif result == "free":
                self.tree_nodes.append(
                    TreeNode(position=new_point, parent=nearest_idx)
                )
                free_added += 1
            elif result == "frontier":
                self.frontier_candidates.append(
                    {"position": new_point, "parent": nearest_idx}
                )
                frontier_added += 1

        elapsed     = _time.monotonic() - t0
        reject_rate = rejected / self.max_iterations

        # ── BUG DETECTOR: high rejection rate ────────────────────────
        if reject_rate > 0.95:
            self.get_logger().error(
                f"[BUG ①②③] {rejected}/{self.max_iterations} iterations "
                f"rejected ({100*reject_rate:.0f}%). "
                f"Almost no branches passable. Likely causes: "
                f"(a) robot cell unknown/occupied so root is blocked, "
                f"(b) FREE_MAX={FREE_MAX} too tight, "
                f"(c) robot_radius_cells={self.robot_radius_cells} "
                f"inflating all corridors, "
                f"(d) map has no free corridor to any unknown cell."
            )
        elif reject_rate > 0.80:
            self.get_logger().warn(
                f"[BUG ②③ candidate] {rejected}/{self.max_iterations} "
                f"rejected ({100*reject_rate:.0f}%). "
                f"Map may be cluttered or FREE_MAX too tight. "
                f"Paths will be sparse and may clip obstacles."
            )

        # ── BUG DETECTOR: planning too slow ──────────────────────────
        if elapsed > self._min_plan_interval:
            self.get_logger().warn(
                f"[BUG ②] _grow_tree took {elapsed:.2f}s which exceeds "
                f"min_plan_interval={self._min_plan_interval:.1f}s. "
                f"New paths publish faster than robot can follow them. "
                f"Reduce max_iterations or map resolution."
            )

        self.get_logger().info(
            f"[RRT GROW DONE] {elapsed:.2f}s | "
            f"tree={len(self.tree_nodes)} | "
            f"free_added={free_added} frontier_added={frontier_added} "
            f"rejected={rejected} ({100*reject_rate:.0f}%)"
        )

        # ── BUG DETECTOR: zero frontiers ─────────────────────────────
        if frontier_added == 0:
            self.get_logger().warn(
                f"[BUG ① candidate] Zero frontier candidates after "
                f"{self.max_iterations} iterations. "
                f"Causes: (a) map fully explored, "
                f"(b) FREE_MAX={FREE_MAX} too tight blocking corridors, "
                f"(c) step_size={self.step_size}m skipping thin passages, "
                f"(d) robot surrounded by unknown cells with no free path out."
            )

        return len(self.frontier_candidates) > 0

    # ================================================================== #
    #  OBSTACLE INFLATION
    # ================================================================== #

    def _inflate_obstacles(self, radius: int) -> np.ndarray:
        if self._grid_snapshot is None:
            return np.zeros((1, 1), dtype=bool)
        obstacle_mask = (self._grid_snapshot > FREE_MAX) & \
                        (self._grid_snapshot != UNKNOWN_VAL)
        if radius <= 0:
            return obstacle_mask
        struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        return binary_dilation(obstacle_mask, structure=struct)

    # ================================================================== #
    #  DOOR BLOCKING
    # ================================================================== #

    def block_door_in_costmap(self, p1, p2, width_m: float = 0.3):
        with self._map_lock:
            if self._slam_grid is None or self._map_info is None:
                self.get_logger().warn("[RRT] block_door: no SLAM map yet — ignored")
                return
            info = self._map_info
            res  = info["resolution"]

        radius_cells = max(1, int(width_m / res))

        def w2c(x, y):
            cx = int((x - info["origin_x"]) / res)
            cy = int((y - info["origin_y"]) / res)
            return cx, cy

        c0 = w2c(p1[0], p1[1])
        c1 = w2c(p2[0], p2[1])
        cells = self._bresenham_line(c0[0], c0[1], c1[0], c1[1])

        with self._map_lock:
            h, w = self._slam_grid.shape
            for cx, cy in cells:
                xmin = max(0, cx - radius_cells)
                xmax = min(w, cx + radius_cells + 1)
                ymin = max(0, cy - radius_cells)
                ymax = min(h, cy + radius_cells + 1)
                self._slam_grid[ymin:ymax, xmin:xmax] = 100

        self.get_logger().info(f"[RRT] Door blocked {p1} → {p2}")

    def unblock_door(self, p1, p2, width_m: float = 0.3):
        with self._map_lock:
            if self._slam_grid is None or self._map_info is None:
                self.get_logger().warn("[RRT] unblock_door: no SLAM map yet — ignored")
                return
            info = self._map_info
            res  = info["resolution"]

        radius_cells = max(1, int(width_m / res))

        def w2c(x, y):
            cx = int((x - info["origin_x"]) / res)
            cy = int((y - info["origin_y"]) / res)
            return cx, cy

        c0 = w2c(p1[0], p1[1])
        c1 = w2c(p2[0], p2[1])
        cells = self._bresenham_line(c0[0], c0[1], c1[0], c1[1])

        with self._map_lock:
            h, w = self._slam_grid.shape
            for cx, cy in cells:
                xmin = max(0, cx - radius_cells)
                xmax = min(w, cx + radius_cells + 1)
                ymin = max(0, cy - radius_cells)
                ymax = min(h, cy + radius_cells + 1)
                self._slam_grid[ymin:ymax, xmin:xmax] = -1

        self.get_logger().info(f"[RRT] Door unblocked {p1} → {p2}")

    # ================================================================== #
    #  FRONTIER SELECTION
    # ================================================================== #

    def _is_blacklisted(self, pos: np.ndarray) -> bool:
        for bx, by in self._blacklisted_frontiers:
            if math.hypot(pos[0] - bx, pos[1] - by) < self._blacklist_radius:
                return True
        return False

    def _select_best_frontier(self) -> Optional[dict]:
        rx, ry, _ = self.slam.pose
        robot     = np.array([rx, ry])
        positions = np.array([c["position"] for c in self.frontier_candidates])

        best_score        = -1.0
        best_idx          = None
        skipped_blacklist = 0

        for i in range(len(self.frontier_candidates)):
            pos = positions[i]
            if self._is_blacklisted(pos):
                skipped_blacklist += 1
                continue
            dist_to_robot = float(np.linalg.norm(pos - robot))
            d             = np.linalg.norm(positions - pos, axis=1)
            count         = float(np.sum(d < self.frontier_cluster_radius))
            score         = count / (1.0 + 0.4 * dist_to_robot)
            if score > best_score:
                best_idx   = i
                best_score = score

        if skipped_blacklist > 0:
            self.get_logger().info(
                f"[RRT FRONTIER] Skipped {skipped_blacklist} blacklisted candidates"
            )

        if best_idx is None:
            self.get_logger().warn(
                "[BUG ① candidate] All frontier candidates are blacklisted — "
                "no valid frontier found. Consider increasing blacklist_radius "
                "or check if exploration is genuinely complete."
            )
            return None

        chosen = self.frontier_candidates[best_idx]
        cx, cy = chosen["position"]
        dist   = float(np.linalg.norm(np.array([cx, cy]) - robot))

        if dist > 5.0:
            self.get_logger().warn(
                f"[BUG ② candidate] Best frontier is {dist:.2f}m away at "
                f"({cx:.2f},{cy:.2f}). "
                f"Score={best_score:.2f} — robot may skip nearby frontiers. "
                f"Consider tuning frontier_cluster_radius or step_size."
            )

        self.get_logger().info(
            f"[RRT FRONTIER] chosen=({cx:.2f},{cy:.2f}) "
            f"dist={dist:.2f}m score={best_score:.2f} "
            f"blacklisted_zones={len(self._blacklisted_frontiers)}"
        )
        return chosen

    # ================================================================== #
    #  RRT HELPERS
    # ================================================================== #

    def _random_sample(self) -> Tuple[float, float]:
        x0, x1, y0, y1 = self._sampling_bounds()
        return (random.uniform(x0, x1), random.uniform(y0, y1))

    def _sampling_bounds(self) -> Tuple[float, float, float, float]:
        info = self._snap_info
        known = self._grid_snapshot != UNKNOWN_VAL

        known_count = int(np.sum(known))

        if not known.any() or known_count < 15000:
            self.get_logger().warn(
                f"[BUG ① candidate] _sampling_bounds: only {known_count} known cells — "
                f"falling back to full map extent "
                f"({info['width']*info['resolution']:.1f}x"
                f"{info['height']*info['resolution']:.1f}m). "
                f"Most samples will hit unknown/obstacle cells."
            )
            return (
                info["origin_x"],
                info["origin_x"] + info["width"]  * info["resolution"],
                info["origin_y"],
                info["origin_y"] + info["height"] * info["resolution"],
            )

        ys, xs = np.where(known)
        pad    = self.sampling_padding_cells
        xmin   = max(0,              int(xs.min()) - pad)
        xmax   = min(info["width"],  int(xs.max()) + pad)
        ymin   = max(0,              int(ys.min()) - pad)
        ymax   = min(info["height"], int(ys.max()) + pad)

        x0, y0 = self._cell_to_world(xmin, ymin)
        x1, y1 = self._cell_to_world(xmax, ymax)

        area_m2 = (x1 - x0) * (y1 - y0)
        if area_m2 < 0.5:
            self.get_logger().warn(
                f"[BUG ① candidate] Sampling area only {area_m2:.2f} m² "
                f"({x0:.2f}..{x1:.2f}, {y0:.2f}..{y1:.2f}). "
                f"Known map is tiny — increase sampling_padding_cells."
            )

        return x0, x1, y0, y1

    def _find_nearest(self, sample: Tuple[float, float]) -> int:
        positions = np.array([n.position for n in self.tree_nodes])
        dists     = np.linalg.norm(positions - np.array(sample), axis=1)
        return int(np.argmin(dists))

    def _steer(self, p0: Tuple[float, float],
                p1: Tuple[float, float]) -> Tuple[float, float]:
        p0 = np.array(p0)
        p1 = np.array(p1)
        d  = np.linalg.norm(p1 - p0)
        if d < 1e-6:
            return (float(p0[0]), float(p0[1]))
        p = p0 + (p1 - p0) / d * min(self.step_size, d)
        return (float(p[0]), float(p[1]))

    def _check_edge(self, c0: Tuple[int, int],
                    c1: Tuple[int, int]) -> Optional[str]:
        cells = self._bresenham_line(c0[0], c0[1], c1[0], c1[1])
        for cx, cy in cells[:-1]:
            if self._classify_cell(cx, cy) == "obstacle":
                return None
        terminal = self._classify_cell(*cells[-1])
        if terminal == "free":
            return "free"
        if terminal == "frontier":
            return "frontier"
        return None

    @staticmethod
    def _bresenham_line(x0, y0, x1, y1) -> list:
        cells = []
        dx = abs(x1 - x0);  sx = 1 if x0 < x1 else -1
        dy = abs(y1 - y0);  sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:  err -= dy;  x += sx
            if e2 <  dx:  err += dx;  y += sy
        return cells

    def _backtrack_path(self, parent, leaf) -> list:
        path = [leaf]
        idx  = parent
        while idx is not None:
            path.append(self.tree_nodes[idx].position)
            idx = self.tree_nodes[idx].parent
        path.reverse()

        if len(path) < 2:
            self.get_logger().warn(
                f"[BUG ② candidate] Path has only {len(path)} point(s). "
                f"Follower may complete it instantly before robot moves."
            )
        return path

    # ================================================================== #
    #  DEBUG
    # ================================================================== #

    def _save_debug(self, path=None, goal=None):
        try:
            info = self._snap_info or {}
            data = {
                "robot": {
                    "x":     float(self.slam.pose[0]),
                    "y":     float(self.slam.pose[1]),
                    "theta": float(self.slam.pose[2]),
                },
                "tree": [
                    {
                        "x1": float(self.tree_nodes[n.parent].position[0]),
                        "y1": float(self.tree_nodes[n.parent].position[1]),
                        "x2": float(n.position[0]),
                        "y2": float(n.position[1]),
                    }
                    for n in self.tree_nodes if n.parent is not None
                ],
                "path": [[float(x), float(y)] for x, y in path] if path else [],
                "goal": (
                    {"x": float(goal[0]), "y": float(goal[1])} if goal else None
                ),
                "slam_map": {
                    "width":      info.get("width"),
                    "height":     info.get("height"),
                    "resolution": info.get("resolution"),
                    "origin_x":   info.get("origin_x"),
                    "origin_y":   info.get("origin_y"),
                },
                "blacklisted": [
                    {"x": float(bx), "y": float(by)}
                    for bx, by in self._blacklisted_frontiers
                ],
            }
            with open("/tmp/rrt_debug.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"[RRT] _save_debug failed: {e}")

    # ================================================================== #
    #  TICK  (every 0.5 s)  — PLANNING
    # ================================================================== #

    def _tick(self):
        self._tick_count += 1

        if self._destroyed:
            return

        # ── Path still executing → planning paused, follower is driving ─
        if self._path_executing:
            self.get_logger().info(
                f"[RRT TICK #{self._tick_count}] Skipping plan — path still "
                f"executing (follower active). Will plan again when path done."
            )
            return

        if self._returned_home:
            return

        # ── BUG DETECTOR: no map yet ──────────────────────────────────
        with self._map_lock:
            if self._slam_grid is None:
                self.get_logger().warn(
                    f"[BUG ①] TICK #{self._tick_count}: "
                    f"No /slam_map received yet. "
                    f"Check EKF SLAM is running and publishing /slam_map. "
                    f"Robot will stay still."
                )
                return
            self._grid_snapshot = self._slam_grid.copy()
            self._snap_info     = dict(self._map_info)

        # ── Rate limit ────────────────────────────────────────────────
        now = _time.monotonic()
        if now - self._last_plan_time < self._min_plan_interval:
            return
        self._last_plan_time = now

        try:
            # ── Minimum known-cell check ──────────────────────────────
            known_count = int(np.sum(self._grid_snapshot != UNKNOWN_VAL))
            if known_count < self._min_known_cells:
                self.get_logger().info(
                    f"[RRT TICK #{self._tick_count}] Waiting for SLAM map to build "
                    f"({known_count}/{self._min_known_cells} known cells). "
                    f"Robot will stay still."
                )
                return

            # ── Inflate obstacles ──────────────────────────────────────
            self._inflated = self._inflate_obstacles(self.robot_radius_cells)

            if self._exploration_done and not self._returned_home:
                self.get_logger().info(
                    f"[RRT TICK #{self._tick_count}] ===== Planning return path to origin ====="
                )
            else:
                self.get_logger().info(
                    f"[RRT TICK #{self._tick_count}] ===== Planning new exploration path ====="
                )

            goal, path = self.explore_step()

            if goal is None or path is None:
                self.get_logger().warn(
                    f"[RRT TICK #{self._tick_count}] No goal returned — "
                    f"no_frontier_rounds={self._no_frontier_rounds} "
                    f"exploration_done={self._exploration_done}. "
                    f"Robot will stay still this tick."
                )
                return

            self._current_goal = goal

            # Record real start pose on first successful plan
            if self._start_pose is None and not self._returning_home:
                rx, ry, _ = self.slam.pose
                rcx, rcy  = self._world_to_cell(rx, ry)
                raw       = self._grid_snapshot[rcy, rcx]
                inflated  = self._inflated[rcy, rcx]
                self.get_logger().info(
                    f"[RRT] Recording start pose ({rx:.2f}, {ry:.2f}) — "
                    f"robot cell raw={raw} inflated={inflated}"
                )
                self._start_pose = (float(rx), float(ry))

            # ── Publish path to /exploration_path (for RViz) ──────────
            path_msg                 = Path()
            path_msg.header.stamp    = self.get_clock().now().to_msg()
            path_msg.header.frame_id = "map"

            for x, y in path:
                p                 = PoseStamped()
                p.header          = path_msg.header
                p.pose.position.x = float(x)
                p.pose.position.y = float(y)
                path_msg.poses.append(p)

            if self._destroyed:
                return

            self.path_pub.publish(path_msg)

            # ── Hand the path to the INTERNAL follower ────────────────
            self._start_following(path)

            phase = "RETURN HOME" if self._returning_home else "EXPLORE"
            self.get_logger().info(
                f"[RRT TICK #{self._tick_count}] [{phase}] Published + following path: "
                f"{len(path)} waypoints → goal ({goal[0]:.2f}, {goal[1]:.2f}). "
                f"_path_executing=True — planning paused until follow completes."
            )
            self._save_debug(path, goal)

        except Exception:
            import traceback
            self.get_logger().error(
                f"[BUG] _tick #{self._tick_count} crashed:\n" + traceback.format_exc()
            )

    # ================================================================== #
    #  INTEGRATED PATH FOLLOWER
    # ================================================================== #

    def _start_following(self, path: List[Tuple[float, float]]):
        rx, ry, _ = self.slam.pose
        # Prepend the robot's CURRENT position so the path starts where the
        # robot actually is, not where the tree root was at plan time.
        full_path = [(float(rx), float(ry))] + [(float(x), float(y)) for x, y in path]
        with self._follow_lock:
            self._follow_path  = full_path
            self._follow_index = 0
        self._reset_follow_pid()
        self._reset_stuck_detector()
        self._path_executing = True

    def _on_path_done(self, reason: str):
        """
        Called by the follower when the path ends.

        reason:
          "completed"     — robot reached the final waypoint.
          "stuck_timeout" — no progress; blacklist the goal.

        Clears _path_executing so the next _tick() can plan a new path.
        This replaces the old cross-node notify_path_done() contract.
        """
        self.get_logger().info(
            f"[RRT PATH_DONE] _on_path_done(reason='{reason}') — "
            f"last goal was {self._current_goal}"
        )

        if reason == "stuck_timeout" and self._current_goal is not None:
            bx, by = self._current_goal
            self._blacklisted_frontiers.append((bx, by))
            self.get_logger().warn(
                f"[RRT BLACKLIST] Blacklisting frontier ({bx:.2f}, {by:.2f}) — "
                f"total blacklisted: {len(self._blacklisted_frontiers)}"
            )

        # Clear path + stop robot
        with self._follow_lock:
            self._follow_path  = []
            self._follow_index = 0
        self._cmd_pub.publish(Twist())

        # Re-arm planning
        self._path_executing = False

    def _control_loop(self):
        """Fast (20 Hz) follow loop. Active only while _path_executing."""
        if self._destroyed:
            return
        if not self._path_executing:
            return

        with self._follow_lock:
            if not self._follow_path:
                return
            path     = list(self._follow_path)
            wp_index = self._follow_index

        # ── 1. Robot pose from SLAM ───────────────────────────────
        pose = self.slam.pose   # (x, y, theta)
        rx, ry, rtheta = float(pose[0]), float(pose[1]), float(pose[2])

        # ── 2. Guard against exhausted waypoint list ──────────────
        if wp_index >= len(path):
            self._on_path_done("completed")
            return

        is_final  = (wp_index == len(path) - 1)
        tolerance = FINAL_WAYPOINT_TOLERANCE if is_final else WAYPOINT_TOLERANCE
        tx, ty    = path[wp_index]

        dx_world = tx - rx
        dy_world = ty - ry
        dist     = math.hypot(dx_world, dy_world)

        # ── 3. Advance waypoint when close enough ─────────────────
        if dist < tolerance:
            self.get_logger().info(
                f"[FOLLOW] WP {wp_index}/{len(path)-1} reached "
                f"(dist={dist:.3f} m). "
                + ("Path complete!" if is_final else "Next WP.")
            )
            if is_final:
                self._on_path_done("completed")
                return
            with self._follow_lock:
                self._follow_index += 1
                wp_index = self._follow_index
            # Do NOT reset PID between waypoints (avoids derivative spike)
            if wp_index < len(path):
                tx, ty   = path[wp_index]
                dx_world = tx - rx
                dy_world = ty - ry
                dist     = math.hypot(dx_world, dy_world)
                is_final = (wp_index == len(path) - 1)
            else:
                self._on_path_done("completed")
                return

        # ── 4. Stuck detection ────────────────────────────────────
        if self._check_stuck(rx, ry):
            self.get_logger().warn(
                f"[FOLLOW] STUCK at ({rx:.2f},{ry:.2f}) "
                f"after {STUCK_TIMEOUT:.0f}s — blacklisting goal."
            )
            self._on_path_done("stuck_timeout")
            return

        # ── 5. Heading error ──────────────────────────────────────
        desired_heading = math.atan2(dy_world, dx_world)
        heading_err     = _wrap_angle(desired_heading - rtheta)

        # ── 6. Skip waypoints directly behind the robot ───────────
        #cos_t        = math.cos(rtheta)
        #sin_t        = math.sin(rtheta)
        # err_fwd_proj = cos_t * dx_world + sin_t * dy_world

        # if err_fwd_proj < 0.05 and not is_final:
        #     self.get_logger().info(
        #         f"[FOLLOW] WP {wp_index} behind robot "
        #         f"(proj={err_fwd_proj:.2f}m) — skipping."
        #     )
        #     with self._follow_lock:
        #         self._follow_index += 1
        #     return

        # ── 7. Forward error = full Euclidean distance ────────────
        err_fwd = dist

        # ── 8. Forward PID ────────────────────────────────────────
        now = _time.monotonic()
        dt  = now - self._last_ctrl_t
        dt  = dt if 0.0 < dt <= 1.0 else 1.0 / CONTROL_HZ
        self._last_ctrl_t = now

        vx_raw = KP_LINEAR * err_fwd
        self._err_fwd_integ += err_fwd * dt
        vx_raw += KI_LINEAR * self._err_fwd_integ
        vx_raw += KD_LINEAR * (err_fwd - self._err_fwd_prev) / dt
        self._err_fwd_prev = err_fwd

        # ── 9. Heading gate ───────────────────────────────────────
        heading_abs = abs(heading_err)
        if heading_abs > HEADING_GATE_RAD:
            t = min((heading_abs - HEADING_GATE_RAD) / HEADING_GATE_RAD, 1.0)
            linear_scale = max(HEADING_SCALE_MIN, 1.0 - t)
        else:
            linear_scale = 1.0

        vx_cmd = _clamp(vx_raw * linear_scale, 0.0, MAX_LINEAR)

        # ── 10. Angular command ───────────────────────────────────
        wz_cmd = _clamp(KP_ANGULAR * heading_err, -MAX_ANGULAR, MAX_ANGULAR)

        # ── 11. EMERGENCY obstacle override (only tier kept) ──────
        vx_cmd, wz_cmd = self._apply_emergency(vx_cmd, wz_cmd)

        # ── 12. Publish (vy always 0) ─────────────────────────────
        twist           = Twist()
        twist.linear.x  = float(vx_cmd)
        twist.linear.y  = 0.0
        twist.angular.z = float(wz_cmd)
        self._cmd_pub.publish(twist)

        self.get_logger().info(
            f"[FOLLOW] WP {wp_index}/{len(path)-1} dist={dist:.2f}m | "
            f"h_err={math.degrees(heading_err):+.1f}° scale={linear_scale:.2f} | "
            f"vx={vx_cmd:+.3f} vy=0.000 wz={wz_cmd:+.3f}"
        )

    # ── EMERGENCY-only avoidance ──────────────────────────────────────

    def _apply_emergency(self, vx: float, wz: float) -> Tuple[float, float]:
        """
        Single-tier obstacle safety: if the front sector is closer than
        EMERGENCY_FRONT_DIST, stop translating and spin toward the more
        open side.  Otherwise pass the path-following commands through.

        SLOW and NORMAL nudge tiers are intentionally omitted — the RRT
        path is already collision-checked, so during normal following we
        trust the path and only guard against imminent collision.
        """
        scan = self._lidar.get_scan()
        if scan is None:
            self.get_logger().warn(
                "[FOLLOW/Emergency] No LiDAR scan — commands unchanged.",
                throttle_duration_sec=2.0,
            )
            return vx, wz

        processed = self._prep.preprocess(scan)
        front     = self._sector_min(processed, -20, 20)

        if front < EMERGENCY_FRONT_DIST:
            left  = self._sector_min(processed,  20,  70)
            right = self._sector_min(processed, -70, -20)
            if left >= right:
                turn = +EMERGENCY_TURN_SPEED
                self._last_turn_dir = +1
            else:
                turn = -EMERGENCY_TURN_SPEED
                self._last_turn_dir = -1

            self.get_logger().warn(
                f"[FOLLOW/Emergency] front={front:.2f}m < {EMERGENCY_FRONT_DIST:.2f}m — "
                f"HALT + spin {'LEFT' if turn > 0 else 'RIGHT'}."
            )
            return 0.0, turn

        return vx, wz

    def _sector_min(self, processed: dict, a_min: float, a_max: float) -> float:
        """Min range in sector [a_min, a_max] deg, with LiDAR mount offset."""
        off = LIDAR_OFFSET_DEG
        return self._prep.get_sector_min(processed, a_min + off, a_max + off)

    # ── Follower PID / stuck helpers ──────────────────────────────────

    def _reset_follow_pid(self):
        self._err_fwd_prev  = 0.0
        self._err_fwd_integ = 0.0
        self._last_ctrl_t   = _time.monotonic()

    def _reset_stuck_detector(self):
        pose = self.slam.pose
        self._stuck_ref_pos  = (float(pose[0]), float(pose[1]))
        self._stuck_ref_time = _time.monotonic()

    def _check_stuck(self, rx: float, ry: float) -> bool:
        elapsed = _time.monotonic() - self._stuck_ref_time
        if elapsed < STUCK_TIMEOUT:
            return False
        if self._stuck_ref_pos is None:
            self._reset_stuck_detector()
            return False

        moved = math.hypot(rx - self._stuck_ref_pos[0],
                           ry - self._stuck_ref_pos[1])
        if moved < STUCK_DIST_THRESHOLD:
            return True

        self._stuck_ref_pos  = (rx, ry)
        self._stuck_ref_time = _time.monotonic()
        return False


# ======================================================================= #
#  Utility functions
# ======================================================================= #

def _wrap_angle(a: float) -> float:
    """Wrap angle to (-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))