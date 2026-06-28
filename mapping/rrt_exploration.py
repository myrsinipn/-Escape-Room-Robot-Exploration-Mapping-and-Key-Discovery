#!/usr/bin/env python3
"""
mapping/rrt_exploration.py - minimal RRT exploration where the RRT is ALSO the planner.

This is a drop-in replacement for the old slam-object version. It uses the SAME
interface as the two working explorers:
    - SLAM map comes in over the /map topic (nav_msgs/OccupancyGrid)
    - robot pose comes from TF (map -> base_footprint)
    - obstacle safety comes from /scan (sensor_msgs/LaserScan)
    - it drives by publishing /cmd_vel (geometry_msgs/Twist)

It does NOT take a slam / lidar / preprocessor object. Run it as its own node:

    ros2 run <your_package> rrt_exploration
    # or directly:
    python3 mapping/rrt_exploration.py

Behaviour:
  1. take the SLAM /map,
  2. grow ONE RRT (with parent links) from the robot through known free space,
  3. a branch that reaches unknown space marks a frontier,
  4. pick a frontier, EXTRACT the path root->frontier THROUGH THE TREE,
  5. drive the robot along that path ourselves (turn-then-drive), no A*.

So the green /plan is literally a branch of the blue RRT.
"""
import math
import random

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                       HistoryPolicy)

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point, Twist, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from sensor_msgs.msg import LaserScan


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class RRTExplorer(Node):
    def __init__(self, map_topic='/slam_map'):
        super().__init__('rrt_explorer')
        # Topic your SLAM publishes the OccupancyGrid on. Files 1/2 used '/map';
        # EKFLidarSLAM in this stack uses a different name, so it's a parameter.
        self.map_topic = map_topic
        # --- RRT / frontier knobs ---
        self.map_frame = 'map'
        self.robot_frame = 'base_footprint'
        self.step = 0.5              # RRT step length (m); bigger = fewer, cleaner branches
        self.sample_radius = 10.0      # sampling reach (m); how far the tree can extend out
        self.explore_sector = math.radians(120)  # half-angle of the OUTWARD sampling cone;
                                     # samples only within this of the explore direction,
                                     # so the tree grows away from the robot, not behind it
        self.max_iter = 1200          # tree-growth iterations per plan. Kept LOW on
                                      # purpose: plan() is pure Python with an O(N^2)
                                      # nearest-node search, so a big number (5000)
                                      # can block for seconds. 1200 still reaches
                                      # nearby frontiers easily.
        self.min_goal_dist = 0.2    # HARD floor: ONLY accept frontiers >= this far (m);
                                    # closer frontiers are ignored entirely
        self.wall_clear = 0.1       # keep frontiers this far from walls (m)
        self.frontier_min_sep = 0.3  # m; min spacing between kept frontiers (cluster dedup)
        self.inflate_radius = 0.20   # m; grow walls by this so the tree keeps clear of them
        self.inflate_exempt = 0.20   # m; ignore inflation within this radius of the robot, so
                                     # a thin start pocket / corridor can never seal the tree in
        self.blacklist_radius = 0.2  # m; around abandoned goals
        self.goal_timeout = 80.0     # s; abandon a path that takes too long
        # --- sticky-goal exploration (commit to a direction instead of re-deciding
        # a new nearest frontier from each fresh random tree -> stops the wander) ---
        self.goal_bias = 0.3        # fraction of RRT samples aimed at the committed goal
        self.goal_stick_radius = 1.5 # m; a frontier within this of the committed goal
                                     # counts as "same direction still open" -> keep it
        self.committed_goal = None   # (x, y) persisted across plan cycles
        # --- driving knobs (turn-then-drive; this base can't mix lin+ang) ---
        self.max_speed = 0.15        # m/s
        self.turn_thresh = 0.40      # rad; START turning when heading error exceeds this
        self.align_thresh = 0.25    # rad; STOP turning only once within this (hysteresis,
                                     # so it can't chatter turn<->drive and orbit a waypoint)
        self.turning = False         # Schmitt-trigger state for turn-then-drive
        self.k_v = 0.8               # forward gain
        self.k_yaw = 0.4            # turn gain (lowered: WiFi latency made 0.7 overshoot/oscillate)
        self.max_wz = 0.4            # rad/s (lowered: slower turns survive the feedback delay)
        self.close_wz = 0.5          # rad/s; capped turn rate when an obstacle is within
                                     # stop_dist ahead (limits forward drift while turning)
        self.wp_tol = 0.10           # m; waypoint reached tolerance

        # --- lidar emergency stop ---
        self.scan_topic = '/scan'
        self.stop_dist = 0.2      # m; stop if an obstacle is closer than this
                                   # (measured from the LIDAR centre; >= robot radius + margin)
        self.laser_yaw = math.pi   # MUST match grid_slam's laser_yaw (lidar faces backward)
        # Front-cone half-angle. _front_blocked() rotates beam angles by laser_yaw,
        # so this genuinely looks AHEAD.
        self.stop_cone = math.radians(35)   # front cone half-angle

        # --- backup-on-obstacle: reverse this far before abandoning + replanning ---
        self.backup_dist = 0.1      # m; how far to reverse when an obstacle is hit
        self.backup_speed = 0.06    # m/s; reverse speed
        self.backing = False        # currently executing a backup maneuver
        self.backup_from = None     # (x, y) map pose where the backup started

        self.map = None
        self.path = None             # list of (x, y) waypoints, root..goal
        self.path_idx = 1
        self.goal_xy = None
        self.goal_time = None
        self.blacklist = []
        # blacklist a frontier only after this many failed attempts to the SAME one,
        # so a single obstacle e-stop doesn't abandon a whole direction - the planner
        # gets a few tries to route AROUND the obstacle first.
        self.max_goal_attempts = 5
        self._goal_fail_count = 0
        self._goal_fail_xy = None
        # restart exploration after this many consecutive plan() cycles with no
        # usable frontier (e.g. after a door sealed off the current direction):
        # clear the blacklist so abandoned directions become open again.
        self.restart_after_empty = 6
        self._empty_cycles = 0
        self.patrol_blacklist = []   # far points already patrolled (avoid re-picking)
        # key/door: when a door opens, /door_goal gives its centre. We then drive
        # THERE (one-shot, ignoring frontiers) instead of running to the far map.
        # If nothing ever publishes /door_goal this stays idle and does nothing.
        self.key_goal = None
        self.key_goal_tol = 0.6      # m; "reached / passed the door" radius
        self.key_reach = 0.7         # m; beeline to the door ONLY if a tree node is
                                     # within this of it (i.e. a route exists). Else
                                     # explore toward it instead of wedging on a wall.
        self.key_timeout = 25.0      # s; give up chasing the door after this, resume.
        self.key_goal_time = None

        self.scan = None             # latest LaserScan message

        # Two callback groups so the (slow, pure-Python) RRT plan() can NEVER
        # starve the fast control() loop. Under the MultiThreadedExecutor in
        # main.py these run on separate threads, so driving keeps ticking at
        # 20 Hz even while a tree is being rebuilt. A single shared group is the
        # usual reason "the robot just sits there": a multi-second plan blocks
        # control(), no /cmd_vel goes out, and the base stops.
        self.cb_fast = MutuallyExclusiveCallbackGroup()   # control + sensor cbs
        self.cb_plan = MutuallyExclusiveCallbackGroup()   # planning only

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(OccupancyGrid, self.map_topic, self._map_cb, qos,
                                 callback_group=self.cb_fast)

        self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, 10,
                                 callback_group=self.cb_fast)
        self.create_subscription(PointStamped, '/door_goal', self._door_goal_cb, 10,
                                 callback_group=self.cb_fast)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.plan_pub = self.create_publisher(Path, '/plan', 1)
        self.rrt_pub = self.create_publisher(MarkerArray, '/rrt_tree', 1)

        self.create_timer(2.0, self.plan, callback_group=self.cb_plan)
        self.create_timer(0.05, self.control, callback_group=self.cb_fast)
        self.get_logger().info('RRT explorer (RRT = planner) up.')

    def _map_cb(self, msg):
        # While rotating in place the SLAM map jitters - walls shift as new scans get
        # integrated mid-turn and the EKF corrects. Ignore map updates during an
        # in-place turn so planning/shortcutting isn't thrown off by walls that move
        # only transiently; the map refreshes as soon as the turn ends.
        if self.turning:
            return
        self.map = msg

    def _scan_cb(self, msg):
        self.scan = msg

    def _door_goal_cb(self, msg):
        """A door just opened (key seen); head to its centre instead of frontiers."""
        self.key_goal = (msg.point.x, msg.point.y)
        self.key_goal_time = self.get_clock().now()
        self.get_logger().warn(
            f'/door_goal -> heading to opened door '
            f'({self.key_goal[0]:.2f}, {self.key_goal[1]:.2f}).')
        self._stop()      # drop current path/patrol so plan() heads to the door now

    def _goto_point(self, nodes, parents, root, target):
        """Build a path to the RRT node nearest a world `target` (the open door).
        Returns True if a path was set."""
        best_i, best_d = None, 1e18
        for i, (nx, ny) in enumerate(nodes):
            d = math.hypot(nx - target[0], ny - target[1])
            if d < best_d:
                best_d, best_i = d, i
        if best_i is None or best_i == 0:
            return False
        path = self.extract_path(nodes, parents, best_i)
        if len(path) < 2:
            return False
        self.path = path
        self.path_idx = 1
        self.goal_xy = nodes[best_i]
        self.goal_time = self.get_clock().now()
        self.publish_plan(path)
        return True

    def _front_blocked(self):
        """True if any valid lidar beam inside the front cone (+/- stop_cone) is
        closer than stop_dist (i.e. an obstacle is right ahead)."""
        s = self.scan
        if s is None:
            return False                      # no scan yet -> don't block
        n = len(s.ranges)
        for i in range(n):
            # lidar faces backward (laser_yaw=pi): robot-FRONT is at raw angle ±pi,
            # so rotate the beam angle by laser_yaw before the front-cone test.
            ang = wrap(s.angle_min + i * s.angle_increment + self.laser_yaw)
            if abs(ang) > self.stop_cone:
                continue                      # outside the front cone
            r = s.ranges[i]
            if math.isinf(r) or math.isnan(r) or r < s.range_min:
                continue                      # invalid / too-close noise
            if r < self.stop_dist:
                return True
        return False

    def _cone_mins(self):
        """DEBUG: closest valid beam in the front cone and in the rear cone."""
        s = self.scan
        if s is None:
            return float('inf'), float('inf')
        fmn, rmn = float('inf'), float('inf')
        for i in range(len(s.ranges)):
            r = s.ranges[i]
            if math.isinf(r) or math.isnan(r) or r < s.range_min:
                continue
            ang = wrap(s.angle_min + i * s.angle_increment + self.laser_yaw)
            if abs(ang) <= self.stop_cone:
                if r < fmn:
                    fmn = r
            elif abs(ang) >= math.pi - self.stop_cone:
                if r < rmn:
                    rmn = r
        return fmn, rmn

    def robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.robot_frame,
                                                rclpy.time.Time())
            q = t.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return (t.transform.translation.x, t.transform.translation.y, yaw)
        except tf2_ros.TransformException:
            return None

    def _nearest_unknown_dist(self, root):
        """DEBUG: distance (m) from root to the nearest UNKNOWN (-1) map cell, or
        -1 if none. If this is small but 0 frontiers -> the unknown is ringed by
        obstacles (unobservable shadow). If it's > sample_radius -> the tree
        simply can't reach it (raise sample_radius/max_iter)."""
        m = self.map
        if m is None:
            return -1.0
        res = m.info.resolution
        ox = m.info.origin.position.x
        oy = m.info.origin.position.y
        w, h = m.info.width, m.info.height
        grid = np.array(m.data, dtype=np.int16).reshape(h, w)
        ys, xs = np.where(grid == -1)
        if xs.size == 0:
            return -1.0
        wx = ox + (xs + 0.5) * res
        wy = oy + (ys + 0.5) * res
        return float(np.hypot(wx - root[0], wy - root[1]).min())

    # ------------------------------------------------ PLANNING (RRT)
    def plan(self):
        if self.map is None or self.path is not None or self.backing:
            return
        pose = self.robot_pose()
        if pose is None:
            self.get_logger().info('Waiting for robot TF...', throttle_duration_sec=5.0)
            return
        root = (pose[0], pose[1])
        nodes, parents, frontiers = self.rrt_build(root, pose[2])
        self.publish_rrt(nodes, parents, frontiers)

        # KEY-DOOR override: head to the opened door, BUT only beeline when a route
        # actually exists (a tree node within key_reach of it). If the door is
        # unreachable (behind a wall / across unknown), DON'T wedge toward it -
        # fall through to normal exploration to open a way there. A timeout also
        # gives up, so it can never get stuck chasing an unreachable door.
        if self.key_goal is not None:
            kx, ky = self.key_goal
            elapsed = (1e18 if self.key_goal_time is None else
                       (self.get_clock().now() - self.key_goal_time).nanoseconds * 1e-9)
            if math.hypot(kx - root[0], ky - root[1]) <= self.key_goal_tol:
                self.get_logger().warn('Reached opened door; resuming exploration.')
                self.key_goal = None
                self.committed_goal = None
            elif elapsed > self.key_timeout:
                self.get_logger().warn(
                    f'key_goal timeout ({elapsed:.0f}s) -> giving up, exploring.')
                self.key_goal = None
                self.committed_goal = None
            else:
                dmin = min((math.hypot(nx - kx, ny - ky) for (nx, ny) in nodes),
                           default=1e18)
                if dmin <= self.key_reach and \
                        self._goto_point(nodes, parents, root, self.key_goal):
                    return
                # door not reachable yet -> fall through to exploration (toward it).

        cand = [f for f in frontiers if not self._blacklisted((f[0], f[1]))]
        if not cand:
            self._empty_cycles += 1
            # --- DEBUG: why are there 0 frontiers? ---
            tree_reach = max((math.hypot(nx - root[0], ny - root[1])
                              for (nx, ny) in nodes), default=0.0)
            unk = self._nearest_unknown_dist(root)
            self.get_logger().warn(
                f'DBG 0-frontiers: nodes={len(nodes)} tree_reach={tree_reach:.2f}m '
                f'nearest_unknown={unk:.2f}m sample_radius={self.sample_radius} '
                f'raw_frontiers={len(frontiers)}',
                throttle_duration_sec=2.0)
            # Distinguish the failure modes (and recover from over-blacklisting):
            if frontiers and self.blacklist:
                self.get_logger().warn(
                    f'All {len(frontiers)} frontier(s) blacklisted; '
                    'clearing blacklist and retrying.')
                self.blacklist.clear()
            else:
                self.get_logger().info(
                    f'No frontiers detected ({len(frontiers)} found, '
                    'none usable) -- try larger sample_radius/max_iter or '
                    'smaller wall_clear.',
                    throttle_duration_sec=5.0)
            # After N empty cycles in a row, RESTART exploration: wipe the
            # blacklist so previously-abandoned directions are searched again.
            if self._empty_cycles >= self.restart_after_empty:
                self.get_logger().warn(
                    f'No usable frontier {self._empty_cycles}x in a row -> '
                    'RESTART: roaming the known area.')
                self.blacklist.clear()
                self._empty_cycles = 0
                if self._start_patrol(nodes, parents, root):
                    return            # patrolling -> path set, drive there
            self.committed_goal = None
            return

        # HARD distance gate: only frontiers at least min_goal_dist away are usable.
        cand = [f for f in cand
                if math.hypot(f[0] - root[0], f[1] - root[1]) >= self.min_goal_dist]
        if not cand:
            self._empty_cycles += 1
            self.get_logger().info(
                f'No frontier >= {self.min_goal_dist:.2f} m away yet; '
                'waiting for the map to grow.', throttle_duration_sec=5.0)
            if self._empty_cycles >= self.restart_after_empty:
                self.get_logger().warn(
                    f'No usable frontier {self._empty_cycles}x in a row -> '
                    'RESTART: roaming the known area.')
                self.blacklist.clear()
                self._empty_cycles = 0
                if self._start_patrol(nodes, parents, root):
                    return            # patrolling -> path set, drive there
            self.committed_goal = None      # release -> full-circle search next cycle
            return

        # STICKY GOAL: if we already committed to a direction and a frontier is
        # still open near it, keep heading there instead of re-deciding from this
        # fresh random tree (that re-deciding every cycle is what made it wander).
        chosen = None
        if (self.committed_goal is not None
                and not self._blacklisted(self.committed_goal)):
            cgx, cgy = self.committed_goal
            near = [f for f in cand
                    if (f[0] - cgx) ** 2 + (f[1] - cgy) ** 2 < self.goal_stick_radius ** 2
                    and math.hypot(f[0] - root[0], f[1] - root[1]) >= self.min_goal_dist]
            if near:
                chosen = min(near, key=lambda f: (f[0] - cgx) ** 2
                             + (f[1] - cgy) ** 2)
        if chosen is None:
            # fresh choice: nearest usable frontier (all already >= min_goal_dist).
            chosen = min(cand, key=lambda f:
                         math.hypot(f[0] - root[0], f[1] - root[1]))
        fx, fy, bi = chosen
        self.committed_goal = (fx, fy)     # remember the direction for next cycle
        self._empty_cycles = 0             # found a usable frontier -> reset counter

        path = self.extract_path(nodes, parents, bi)   # root .. nodes[bi]
        if len(path) < 2:
            path.append((fx, fy))                       # ensure a target ahead
        # Follow the RRT branch EXACTLY, node by node (no shortcutting).
        self.path = path
        self.path_idx = 1
        self.goal_xy = (fx, fy)
        self.goal_time = self.get_clock().now()
        self.publish_plan(path)
        self.get_logger().info(
            f'RRT path ({len(path)} pts) -> frontier ({fx:.2f}, {fy:.2f})')

    def extract_path(self, nodes, parents, bi):
        idxs, i = [], bi
        while i != -1:
            idxs.append(i)
            i = parents[i]
        idxs.reverse()
        return [nodes[i] for i in idxs]

    def _start_patrol(self, nodes, parents, root):
        """No frontiers left to explore: drive to the FARTHEST reachable RRT node
        (a known free cell) so the robot roams the explored area. Returns True if
        a patrol path was set."""
        best_i, best_d = None, -1.0
        for i, (nx, ny) in enumerate(nodes):
            d = math.hypot(nx - root[0], ny - root[1])
            if d < self.min_goal_dist:
                continue
            if any(math.hypot(nx - bx, ny - by) < self.blacklist_radius
                   for (bx, by) in self.patrol_blacklist):
                continue
            if d > best_d:
                best_d, best_i = d, i
        if best_i is None:
            # everything reachable already patrolled -> wipe and try again next time
            self.patrol_blacklist.clear()
            return False
        nx, ny = nodes[best_i]
        path = self.extract_path(nodes, parents, best_i)
        if len(path) < 2:
            return False
        self.path = path
        self.path_idx = 1
        self.goal_xy = (nx, ny)
        self.goal_time = self.get_clock().now()
        self.patrol_blacklist.append((nx, ny))
        self.publish_plan(path)
        self.get_logger().warn(
            f'PATROL -> far known point ({nx:.2f}, {ny:.2f})')
        return True

    def rrt_build(self, root, yaw):
        m = self.map
        res = m.info.resolution
        ox = m.info.origin.position.x
        oy = m.info.origin.position.y
        w, h = m.info.width, m.info.height
        grid = np.array(m.data, dtype=np.int16).reshape(h, w)
        # Inflate walls so the tree keeps its distance, BUT exempt a disk around
        # the robot so a thin start pocket / corridor can never seal it in. Real
        # walls still block everywhere; only the inflation band is waived nearby.
        r_inf = int(math.ceil(self.inflate_radius / res))
        occ_inflated = self._inflate(grid >= 50, r_inf)
        rxg = int((root[0] - ox) / res); ryg = int((root[1] - oy) / res)
        r_ex = int(math.ceil(self.inflate_exempt / res))

        def cell(x, y):
            gx = int((x - ox) / res); gy = int((y - oy) / res)
            if 0 <= gx < w and 0 <= gy < h:
                if grid[gy, gx] >= 50:
                    return 100                  # real wall: always blocked
                if occ_inflated[gy, gx] and \
                        (gx - rxg) ** 2 + (gy - ryg) ** 2 > r_ex * r_ex:
                    return 100                  # inflation band, away from robot
                return int(grid[gy, gx])        # free (0) or unknown (-1)
            return -1

        nodes = [root]
        parents = [-1]
        frontiers = []                          # (fx, fy, parent_node_index)
        # Outward exploration direction: toward the committed goal (if we have a
        # valid one) else the robot's heading. Samples are drawn in a cone around
        # it so the tree expands AWAY from the robot, not back over explored space.
        cg = None
        if (self.committed_goal is not None
                and not self._blacklisted(self.committed_goal)
                and math.hypot(self.committed_goal[0] - root[0],
                               self.committed_goal[1] - root[1]) >= self.min_goal_dist):
            cg = self.committed_goal
        if cg is not None:
            explore_dir = math.atan2(cg[1] - root[1], cg[0] - root[0])
            sector = self.explore_sector
        else:
            explore_dir = yaw
            sector = math.pi                    # no goal yet -> sample all directions
        for _ in range(self.max_iter):
            if cg is not None and random.random() < self.goal_bias:
                # a fraction of samples aim straight at the committed goal
                xr = cg[0] + random.uniform(-self.step, self.step)
                yr = cg[1] + random.uniform(-self.step, self.step)
            else:
                # outward cone: angle within +/-sector of explore_dir, radius out to
                # sample_radius (sqrt -> uniform area, so it favours reaching far)
                theta = explore_dir + random.uniform(-sector, sector)
                rad = self.sample_radius * math.sqrt(random.random())
                xr = root[0] + rad * math.cos(theta)
                yr = root[1] + rad * math.sin(theta)
            bi, bd = 0, 1e18
            for i, (px, py) in enumerate(nodes):
                d = (px - xr) ** 2 + (py - yr) ** 2
                if d < bd:
                    bd, bi = d, i
            nx, ny = nodes[bi]
            ang = math.atan2(yr - ny, xr - nx)
            x1 = nx + self.step * math.cos(ang)
            y1 = ny + self.step * math.sin(ang)
            hit = self._segment(nx, ny, x1, y1, cell, res)
            if hit == 'blocked':
                continue
            if hit == 'free':
                nodes.append((x1, y1))
                parents.append(bi)
            else:                               # reached unknown -> frontier
                if self._wall_ok(hit[0], hit[1], grid, ox, oy, res, w, h):
                    fx, fy = hit[0], hit[1]
                    # min-separation dedup: collapse a cluster of near-duplicate
                    # frontier hits into a single representative point.
                    if not any((fx - efx) ** 2 + (fy - efy) ** 2
                               < self.frontier_min_sep ** 2
                               for (efx, efy, _ep) in frontiers):
                        frontiers.append((fx, fy, bi))
        return nodes, parents, frontiers

    def _segment(self, x0, y0, x1, y1, cell, res):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / (res * 0.5)))
        for i in range(1, n + 1):
            t = i / n
            x = x0 + t * (x1 - x0); y = y0 + t * (y1 - y0)
            v = cell(x, y)
            if v >= 50:
                return 'blocked'
            if v == -1:
                return (x, y)
        return 'free'

    def _wall_ok(self, fx, fy, grid, ox, oy, res, w, h):
        r = max(1, int(self.wall_clear / res))
        gx = int((fx - ox) / res); gy = int((fy - oy) / res)
        x0, x1 = max(0, gx - r), min(w, gx + r + 1)
        y0, y1 = max(0, gy - r), min(h, gy + r + 1)
        return not bool(np.any(grid[y0:y1, x0:x1] >= 50))

    @staticmethod
    def _inflate(occ, r):
        """Boolean dilation of the occupied mask by a disk of radius r cells.
        Each occupied cell is smeared over its disk-shaped neighbourhood via
        shift-and-OR, so the result is True for every cell within r cells of a
        wall. Pure NumPy (no SciPy), runs once per plan."""
        if r <= 0:
            return occ.copy()
        H, W = occ.shape
        out = occ.copy()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx > r * r:
                    continue                 # keep the disk circular, not square
                dy0, dy1 = max(0, dy), H + min(0, dy)
                dx0, dx1 = max(0, dx), W + min(0, dx)
                sy0, sy1 = max(0, -dy), H - max(0, dy)
                sx0, sx1 = max(0, -dx), W - max(0, dx)
                out[dy0:dy1, dx0:dx1] |= occ[sy0:sy1, sx0:sx1]
        return out

    def _blacklisted(self, f):
        return any(math.hypot(f[0] - bx, f[1] - by) < self.blacklist_radius
                   for (bx, by) in self.blacklist)

    # ------------------------------------------------ DRIVING (follow the path)
    def _cell_occ(self, x, y):
        """Occupancy value at world (x, y) on the latest /map (0 if off-map)."""
        m = self.map
        if m is None:
            return 0
        res = m.info.resolution
        ox = m.info.origin.position.x
        oy = m.info.origin.position.y
        w, h = m.info.width, m.info.height
        gx = int((x - ox) / res)
        gy = int((y - oy) / res)
        if 0 <= gx < w and 0 <= gy < h:
            return m.data[gy * w + gx]
        return 0

    def _path_blocked(self):
        """True if the remaining path now crosses an occupied (>=50) cell - e.g.
        a virtual door painted AFTER the path was committed. Samples each
        remaining segment so a thick wall between waypoints is still caught."""
        if self.map is None or self.path is None:
            return False
        res = self.map.info.resolution
        for i in range(max(1, self.path_idx), len(self.path)):
            x0, y0 = self.path[i - 1]
            x1, y1 = self.path[i]
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / (res * 0.5)))
            for k in range(n + 1):
                t = k / n
                if self._cell_occ(x0 + t * (x1 - x0),
                                  y0 + t * (y1 - y0)) >= 50:
                    return True
        return False

    def control(self):
        # --- active backup maneuver (triggered by an obstacle) ---
        if self.backing:
            pose = self.robot_pose()
            if pose is None:
                return
            moved = math.hypot(pose[0] - self.backup_from[0],
                               pose[1] - self.backup_from[1])
            _fmn, rmn = self._cone_mins()
            if moved >= self.backup_dist or rmn < self.stop_dist:
                self.cmd_pub.publish(Twist())   # backed up enough (or wall behind) -> stop
                self.backing = False
                return                          # path is None -> plan() replans next cycle
            cmd = Twist()
            cmd.linear.x = -self.backup_speed
            self.cmd_pub.publish(cmd)
            return

        if self.path is None:
            return

        # Path-validity guard: a virtual wall (door) may be painted on the map
        # AFTER this path was committed; control() otherwise drives the fixed
        # waypoints blind, and a virtual wall has no lidar return to e-stop on.
        # If the remaining path now crosses an occupied cell, abandon it ->
        # plan() replans around the door next cycle.
        if self._path_blocked():
            self.get_logger().warn('Path now crosses a wall (door?); replanning.',
                                   throttle_duration_sec=2.0)
            self._stop()
            return

        # give up on a path that takes too long
        if (self.goal_time is not None and
                (self.get_clock().now() - self.goal_time).nanoseconds * 1e-9
                > self.goal_timeout):
            self.get_logger().warn('Path timeout; abandoning + blacklisting.')
            if self.goal_xy is not None:
                self.blacklist.append(self.goal_xy)
            self._stop()
            return
        pose = self.robot_pose()
        if pose is None:
            return
        rx, ry, rth = pose
        # advance past any waypoints we've reached
        while self.path_idx < len(self.path):
            tx, ty = self.path[self.path_idx]
            if math.hypot(tx - rx, ty - ry) < self.wp_tol:
                self.path_idx += 1
            else:
                break
        if self.path_idx >= len(self.path):
            self.get_logger().info('Reached frontier.')
            self._stop()                        # done -> replan next cycle
            return
        tx, ty = self.path[self.path_idx]
        ex, ey = tx - rx, ty - ry
        yaw_err = wrap(math.atan2(ey, ex) - rth)
        # Hysteresis: once turning, keep turning until well aligned (< align_thresh);
        # once driving, keep driving until badly misaligned (> turn_thresh). This is
        # what stops the turn<->drive chatter that made it orbit a close waypoint.
        if self.turning:
            if abs(yaw_err) < self.align_thresh:
                self.turning = False
        elif abs(yaw_err) > self.turn_thresh:
            self.turning = True
        # DEBUG: which branch are we in, and what does the cone see?
        _fmn, _rmn = self._cone_mins()
        self.get_logger().info(
            f'[ctrl] {"TURN" if self.turning else "DRIVE"} '
            f'yaw_err={math.degrees(yaw_err):+.0f}deg  front_min={_fmn:.2f} '
            f'rear_min={_rmn:.2f}  stop_dist={self.stop_dist}',
            throttle_duration_sec=0.5)
        cmd = Twist()
        if self.turning:                        # turn OR drive, never both
            wz = max(-self.max_wz, min(self.max_wz, self.k_yaw * yaw_err))
            # Turning-gap safety: the obstacle check only guards the DRIVE branch,
            # and this base drifts forward a little during in-place rotation. So
            # when something is within stop_dist in the front cone, cap the turn
            # rate to close_wz to limit that drift (still rotates away to escape).
            if self._front_blocked():
                wz = max(-self.close_wz, min(self.close_wz, wz))
            cmd.angular.z = wz
        else:
            if self._front_blocked():           # only check when about to drive forward
                self.get_logger().warn(
                    f'Obstacle ahead; backing up {self.backup_dist:.2f} m then replanning.',
                    throttle_duration_sec=2.0)
                # Blacklist the frontier ONLY after max_goal_attempts fails to the
                # SAME one - give the planner a few tries to route around first.
                g = self.goal_xy
                if g is not None:
                    if (self._goal_fail_xy is not None and
                            math.hypot(g[0] - self._goal_fail_xy[0],
                                       g[1] - self._goal_fail_xy[1]) < self.blacklist_radius):
                        self._goal_fail_count += 1
                    else:
                        self._goal_fail_xy = g
                        self._goal_fail_count = 1
                    if self._goal_fail_count >= self.max_goal_attempts:
                        self.blacklist.append(g)
                        self.get_logger().warn(
                            f'frontier ({g[0]:.2f}, {g[1]:.2f}) failed '
                            f'{self.max_goal_attempts}x -> blacklisted.')
                        self._goal_fail_count = 0
                        self._goal_fail_xy = None
                self._start_backup()            # reverse, then plan() replans
                return
            cmd.linear.x = min(self.max_speed, self.k_v * math.hypot(ex, ey))
        self.cmd_pub.publish(cmd)

    def _stop(self):
        self.cmd_pub.publish(Twist())
        self.path = None
        self.goal_xy = None
        self.goal_time = None
        self.turning = False

    def _start_backup(self):
        """Abandon the current path and begin reversing backup_dist metres. The
        control loop drives the reverse and stops once that distance is covered
        (or something enters the rear cone). Then path is None -> plan() replans."""
        self.cmd_pub.publish(Twist())       # halt forward motion first
        self.path = None
        self.goal_xy = None
        self.goal_time = None
        self.turning = False
        pose = self.robot_pose()
        if pose is not None:
            self.backup_from = (pose[0], pose[1])
            self.backing = True
        else:
            self.backing = False            # no pose -> can't measure; just replan

    # ------------------------------------------------ visualisation
    def publish_plan(self, path):
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for (x, y) in path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position = Point(x=float(x), y=float(y), z=0.0)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.plan_pub.publish(msg)

    def publish_rrt(self, nodes, parents, frontiers):
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        line = Marker()
        line.header.frame_id = self.map_frame
        line.header.stamp = now
        line.ns = 'rrt'; line.id = 0
        line.type = Marker.LINE_LIST; line.action = Marker.ADD
        line.scale.x = 0.02
        line.color.r, line.color.g, line.color.b, line.color.a = 0.2, 0.6, 1.0, 0.8
        line.pose.orientation.w = 1.0
        for i, p in enumerate(parents):
            if p == -1:
                continue
            a, b = nodes[p], nodes[i]
            line.points.append(Point(x=float(a[0]), y=float(a[1]), z=0.05))
            line.points.append(Point(x=float(b[0]), y=float(b[1]), z=0.05))
        arr.markers.append(line)
        fr = Marker()
        fr.header.frame_id = self.map_frame
        fr.header.stamp = now
        fr.ns = 'frontiers'; fr.id = 1
        fr.type = Marker.CUBE_LIST; fr.action = Marker.ADD
        fr.scale.x = fr.scale.y = fr.scale.z = 0.18
        fr.color.r, fr.color.g, fr.color.b, fr.color.a = 1.0, 0.1, 0.1, 1.0
        fr.pose.orientation.w = 1.0
        for (fx, fy, _bi) in frontiers:
            fr.points.append(Point(x=float(fx), y=float(fy), z=0.1))
        arr.markers.append(fr)
        self.rrt_pub.publish(arr)


def main():
    rclpy.init()
    node = RRTExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()