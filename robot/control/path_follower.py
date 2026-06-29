#!/usr/bin/env python3
"""
PathFollower
============
Follows the nav_msgs/Path published by RRTExplorer on /exploration_path.

Motion model: DIFFERENTIAL-STYLE (no lateral strafe)
-----------------------------------------------------
  linear.x  = forward only  (proportional to full distance to waypoint)
  linear.y  = always 0.0
  angular.z = heading correction (proportional to heading error)

Control sequence per waypoint
------------------------------
  1. Compute heading error to waypoint.
  2. If |heading_err| > HEADING_GATE_RAD → scale vx down (mostly rotate).
  3. Once aligned → drive forward at full vx while trimming heading with wz.
  4. Within WAYPOINT_TOLERANCE → advance to next waypoint.
  5. If waypoint is behind the robot (err_fwd < 0.05) → skip it.

Key design decision — err_fwd = dist (not a projection)
---------------------------------------------------------
  With no lateral strafe available, the robot's only way to reach a
  waypoint is to rotate toward it and then drive forward.  Using the
  full Euclidean distance as err_fwd means:
    - vx is always meaningful (never collapses to zero due to angle)
    - The heading gate ensures the robot aligns before driving
    - No inconsistency between the PID input and the tolerance check
  Using the body-frame projection (cos·dx + sin·dy) instead causes the
  robot to stall when the waypoint is perpendicular (projection ≈ 0)
  and never advance toward it.

Obstacle avoidance (SafeLidarMotion-compatible, three tiers)
-------------------------------------------------------------
  EMERGENCY  front < EMERGENCY_FRONT_DIST  → stop, spin toward open side
  SLOW       front < SLOW_FRONT_DIST       → scale vx, angular nudge
  NORMAL                                   → pass vx/wz; soft nudge if
                                             sides tight
  linear.y is forced to 0.0 in every tier.

LiDAR mounting
--------------
  Physical LiDAR mounted 180° rotated — all sector angles shifted by
  +LIDAR_OFFSET_DEG, identical to SafeLidarMotion.

PID tuning quick-reference
---------------------------
  KP_LINEAR   — forward acceleration toward waypoint.
                Too high → overshoot.  Too low → sluggish.
                Good range: 0.40–0.60.

  KD_LINEAR   — damps forward overshoot on final approach.
                Keep small: 0.02–0.05.

  KP_ANGULAR  — rotation speed for heading correction.
                Too high → oscillation around heading.
                Too low  → slow alignment, robot misses waypoints.
                Good range: 0.80–1.20.

  MAX_LINEAR  — forward speed cap (m/s). MyAGV max ≈ 0.5; keep ≤ 0.30.
  MAX_ANGULAR — angular speed cap (rad/s). Keep ≤ 0.90.

  HEADING_GATE_RAD  — |heading_err| above which vx is scaled down.
                      Must be reachable despite SLAM noise (~2–5°).
                      Good value: 0.35 rad (~20°).

  HEADING_SCALE_MIN — vx scale floor during rotation.
                      Small but nonzero so robot creeps forward slightly.

  WAYPOINT_TOLERANCE      — distance (m) for intermediate WP acceptance.
  FINAL_WAYPOINT_TOLERANCE— tighter for the last waypoint.

  STUCK_TIMEOUT        — seconds of no progress before blacklisting.
  STUCK_DIST_THRESHOLD — metres per window that counts as progress.
"""

import math
import time as _time
import threading
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Path

from sensors.lidar import LidarSensor
from perception.scan_preprocessor import ScanPreprocessor


# ═══════════════════════════════════════════════════════════════════════ #
#  TUNING CONSTANTS
# ═══════════════════════════════════════════════════════════════════════ #

# ── Path-following PID ────────────────────────────────────────────────
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

# ── Heading gate ──────────────────────────────────────────────────────
# When |heading_err| > HEADING_GATE_RAD, vx is scaled toward
# HEADING_SCALE_MIN so the robot mostly rotates before driving.
# Must be wide enough to be reachable despite SLAM heading noise.
HEADING_GATE_RAD  = 0.35   # rad (~20°)
HEADING_SCALE_MIN = 0.10   # vx scale floor during rotation

# ── Stuck detection ───────────────────────────────────────────────────
STUCK_TIMEOUT        = 10.0  # s
STUCK_DIST_THRESHOLD = 0.10  # m — minimum displacement per window

# ── Obstacle avoidance ────────────────────────────────────────────────
EMERGENCY_FRONT_DIST = 0.15  # m   — full stop + spin
SLOW_FRONT_DIST      = 0.35  # m   — reduce speed + angular nudge
SIDE_SAFE_DIST       = 0.10  # m   — lateral clearance trigger
EMERGENCY_TURN_SPEED = 0.40  # rad/s
SLOW_TURN_NUDGE      = 0.15  # rad/s — nudge added in SLOW mode
AVOIDANCE_SCALE_SLOW = 0.55  # vx scale factor in SLOW mode

LIDAR_OFFSET_DEG = 180       # LiDAR mounted backwards (same as SafeLidarMotion)

# ── Control loop rate ─────────────────────────────────────────────────
CONTROL_HZ = 20.0


# ═══════════════════════════════════════════════════════════════════════ #
#  PathFollower node
# ═══════════════════════════════════════════════════════════════════════ #

class PathFollower(Node):
    """
    Subscribes to /exploration_path and drives the robot along each
    waypoint using only linear.x and angular.z (vy = 0 always).

    Wiring in main.py
    -----------------
        follower = PathFollower(slam=slam_node, lidar=lidar, preprocessor=prep)
        follower.explorer = rrt          # REQUIRED — enables notify_path_done()
        rrt.path_follower = follower     # optional back-reference
    """

    def __init__(
        self,
        slam,
        lidar: LidarSensor,
        preprocessor: ScanPreprocessor,
    ):
        super().__init__("path_follower")

        self.slam     = slam
        self._lidar   = lidar
        self._prep    = preprocessor
        self.explorer = None   # set externally to the RRTExplorer instance

        # ── Path state ────────────────────────────────────────────
        self._path:      List[Tuple[float, float]] = []
        self._wp_index:  int  = 0
        self._path_lock  = threading.Lock()
        self._active     = False

        # ── PID state (forward axis only) ─────────────────────────
        self._err_fwd_prev  = 0.0
        self._err_fwd_integ = 0.0
        self._last_ctrl_t   = _time.monotonic()

        # ── Stuck detection ───────────────────────────────────────
        self._stuck_ref_pos:  Optional[Tuple[float, float]] = None
        self._stuck_ref_time: float = _time.monotonic()

        # ── Avoidance state ───────────────────────────────────────
        self._last_turn_dir = 1   # +1 left / -1 right tiebreak memory

        # ── ROS interfaces ────────────────────────────────────────
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self._path_sub = self.create_subscription(
            Path,
            "/exploration_path",
            self._path_callback,
            10,
        )

        self.create_timer(1.0 / CONTROL_HZ, self._control_loop)

        self.get_logger().info(
            f"[PathFollower] Ready (differential-style: vy=0 always) — "
            f"KP_lin={KP_LINEAR} KP_ang={KP_ANGULAR} "
            f"max_lin={MAX_LINEAR} m/s  max_ang={MAX_ANGULAR} rad/s  "
            f"heading_gate={math.degrees(HEADING_GATE_RAD):.0f}°  "
            f"wp_tol={WAYPOINT_TOLERANCE} m  stuck_t={STUCK_TIMEOUT} s"
        )

    # ══════════════════════════════════════════════════════════════ #
    #  Path subscription
    # ══════════════════════════════════════════════════════════════ #

    def _path_callback(self, msg: Path):
        """Replace current path and reset all state on every new path."""
        waypoints = [
            (float(p.pose.position.x), float(p.pose.position.y))
            for p in msg.poses
        ]
        if not waypoints:
            self.get_logger().warn("[PathFollower] Received empty path — ignoring.")
            return

        with self._path_lock:
            self._path     = waypoints
            self._wp_index = 0
            self._active   = True

        # Reset PID only on a brand-new path, NOT between waypoints
        self._reset_pid()
        self._reset_stuck_detector()

        self.get_logger().info(
            f"[PathFollower] New path: {len(waypoints)} waypoints. "
            f"First=({waypoints[0][0]:.2f},{waypoints[0][1]:.2f})  "
            f"Last=({waypoints[-1][0]:.2f},{waypoints[-1][1]:.2f})"
        )

    # ══════════════════════════════════════════════════════════════ #
    #  Main control loop  (20 Hz)
    # ══════════════════════════════════════════════════════════════ #

    def _control_loop(self):
        with self._path_lock:
            if not self._active or not self._path:
                return
            path     = list(self._path)
            wp_index = self._wp_index

        # ── 1. Robot pose from SLAM ───────────────────────────────
        pose = self.slam.pose   # (x, y, theta)
        rx, ry, rtheta = float(pose[0]), float(pose[1]), float(pose[2])

        # ── 2. Guard against exhausted waypoint list ──────────────
        if wp_index >= len(path):
            self._finish("completed")
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
                f"[PathFollower] WP {wp_index}/{len(path)-1} reached "
                f"(dist={dist:.3f} m). "
                + ("Path complete!" if is_final else "Next WP.")
            )
            if is_final:
                self._finish("completed")
                return
            with self._path_lock:
                self._wp_index += 1
                wp_index = self._wp_index
            # NOTE: do NOT reset PID here — causes derivative spike
            if wp_index < len(path):
                tx, ty   = path[wp_index]
                dx_world = tx - rx
                dy_world = ty - ry
                dist     = math.hypot(dx_world, dy_world)
                is_final = (wp_index == len(path) - 1)
            else:
                self._finish("completed")
                return

        # ── 4. Stuck detection ────────────────────────────────────
        if self._check_stuck(rx, ry):
            self.get_logger().warn(
                f"[PathFollower] STUCK at ({rx:.2f},{ry:.2f}) "
                f"after {STUCK_TIMEOUT:.0f}s — blacklisting goal."
            )
            self._finish("stuck_timeout")
            return

        # ── 5. Heading error ──────────────────────────────────────
        desired_heading = math.atan2(dy_world, dx_world)
        heading_err     = _wrap_angle(desired_heading - rtheta)

        # ── 6. Skip waypoints that are directly behind the robot ──
        #
        # With vy=0, a waypoint behind the robot requires a ~180° turn
        # followed by a forward drive.  For intermediate waypoints this
        # wastes time; skip them and let the robot proceed to the next
        # one which is usually ahead.
        #
        # err_fwd_proj: projection of distance onto robot forward axis.
        # Negative means the waypoint is behind the robot.
        cos_t        = math.cos(rtheta)
        sin_t        = math.sin(rtheta)
        err_fwd_proj = cos_t * dx_world + sin_t * dy_world

        if err_fwd_proj < 0.05 and not is_final:
            self.get_logger().info(
                f"[PathFollower] WP {wp_index} is behind robot "
                f"(proj={err_fwd_proj:.2f}m) — skipping."
            )
            with self._path_lock:
                self._wp_index += 1
            return

        # ── 7. Forward error = full Euclidean distance ────────────
        #
        # With vy=0 the robot must rotate to face the waypoint before
        # driving.  Using the full distance (not the body-frame projection)
        # ensures vx is always meaningful — it never collapses to zero
        # when the robot is perpendicular to the waypoint.
        # The heading gate (step 9) handles alignment before forward motion.
        #
        err_fwd = dist

        # ── 8. PID on forward error ───────────────────────────────
        now = _time.monotonic()
        dt  = now - self._last_ctrl_t
        dt  = dt if 0.0 < dt <= 1.0 else 1.0 / CONTROL_HZ
        self._last_ctrl_t = now

        # Proportional
        vx_raw = KP_LINEAR * err_fwd

        # Integral
        self._err_fwd_integ += err_fwd * dt
        vx_raw += KI_LINEAR * self._err_fwd_integ

        # Derivative (negative because we want to damp approach)
        vx_raw += KD_LINEAR * (err_fwd - self._err_fwd_prev) / dt
        self._err_fwd_prev = err_fwd

        # ── 9. Heading gate — scale vx down when misaligned ───────
        #
        # When |heading_err| > HEADING_GATE_RAD, vx is reduced so the
        # robot mostly rotates before driving forward.
        # HEADING_GATE_RAD=0.35 rad (~20°) is wide enough to be
        # consistently reachable despite SLAM heading noise of ~2–5°.
        #
        heading_abs = abs(heading_err)
        if heading_abs > HEADING_GATE_RAD:
            # Linear interpolation: at gate boundary → scale=1.0
            #                       at 2× gate      → scale=HEADING_SCALE_MIN
            t = min((heading_abs - HEADING_GATE_RAD) / HEADING_GATE_RAD, 1.0)
            linear_scale = max(HEADING_SCALE_MIN, 1.0 - t)
        else:
            linear_scale = 1.0

        vx_cmd = _clamp(vx_raw * linear_scale, 0.0, MAX_LINEAR)
        # Clamp to [0, MAX_LINEAR]: robot should never drive backward
        # toward a waypoint — it should rotate 180° instead.

        # ── 10. vy is always zero ─────────────────────────────────
        vy_cmd = 0.0

        # ── 11. Angular command ───────────────────────────────────
        wz_cmd = _clamp(KP_ANGULAR * heading_err, -MAX_ANGULAR, MAX_ANGULAR)

        # ── 12. Obstacle avoidance override ──────────────────────
        vx_cmd, wz_cmd = self._apply_avoidance(vx_cmd, wz_cmd)

        # ── 13. Publish ───────────────────────────────────────────
        twist           = Twist()
        twist.linear.x  = float(vx_cmd)
        twist.linear.y  = 0.0          # enforced zero always
        twist.angular.z = float(wz_cmd)
        self._cmd_pub.publish(twist)

        self.get_logger().info(
            f"[PathFollower] WP {wp_index}/{len(path)-1} "
            f"dist={dist:.2f}m | "
            f"h_err={math.degrees(heading_err):+.1f}° "
            f"scale={linear_scale:.2f} | "
            f"vx={vx_cmd:+.3f} vy=0.000 wz={wz_cmd:+.3f}"
        )

    # ══════════════════════════════════════════════════════════════ #
    #  Obstacle avoidance  (SafeLidarMotion-style, vy always 0)
    # ══════════════════════════════════════════════════════════════ #

    def _apply_avoidance(
        self,
        vx: float,
        wz: float,
    ) -> Tuple[float, float]:
        """
        Reads LiDAR and overlays SafeLidarMotion-style avoidance.
        Returns modified (vx, wz).
        vy is never a parameter here — it is always 0.0.

        EMERGENCY  front < EMERGENCY_FRONT_DIST
            vx=0, wz=spin toward more open side.

        SLOW       front < SLOW_FRONT_DIST
            vx scaled by AVOIDANCE_SCALE_SLOW.
            Angular nudge added to wz only when it agrees with the
            existing heading PID direction (prevents fight between
            avoidance and path-following angular commands).

        NORMAL
            vx unchanged.
            Half-strength angular nudge when sides are tight.
        """
        scan = self._lidar.get_scan()
        if scan is None:
            self.get_logger().warn(
                "[PathFollower/Avoidance] No LiDAR scan — commands unchanged.",
                throttle_duration_sec=2.0,
            )
            return vx, wz

        processed = self._prep.preprocess(scan)
        sec       = self._get_sectors(processed)
        left_score, right_score = self._score_sides(sec)

        # ── EMERGENCY ─────────────────────────────────────────────
        if sec["front"] < EMERGENCY_FRONT_DIST:
            if left_score >= right_score:
                turn = +EMERGENCY_TURN_SPEED
                self._last_turn_dir = +1
            else:
                turn = -EMERGENCY_TURN_SPEED
                self._last_turn_dir = -1

            self.get_logger().warn(
                f"[PathFollower/Avoidance] EMERGENCY front={sec['front']:.2f}m — "
                f"halting, spinning {'LEFT' if turn > 0 else 'RIGHT'}."
            )
            return 0.0, turn

        # ── SLOW ──────────────────────────────────────────────────
        if sec["front"] < SLOW_FRONT_DIST:
            vx_slow = vx * AVOIDANCE_SCALE_SLOW

            if left_score >= right_score:
                nudge = +SLOW_TURN_NUDGE
                self._last_turn_dir = +1
            else:
                nudge = -SLOW_TURN_NUDGE
                self._last_turn_dir = -1

            # Only add nudge when it agrees with the heading PID direction.
            # If they oppose, keep the PID command clamped — don't fight.
            if math.copysign(1.0, nudge) == math.copysign(1.0, wz):
                wz_slow = _clamp(wz + nudge, -MAX_ANGULAR, MAX_ANGULAR)
            else:
                wz_slow = _clamp(wz, -MAX_ANGULAR, MAX_ANGULAR)

            self.get_logger().info(
                f"[PathFollower/Avoidance] SLOW front={sec['front']:.2f}m — "
                f"vx×{AVOIDANCE_SCALE_SLOW:.2f} "
                f"nudge={'LEFT' if nudge > 0 else 'RIGHT'}."
            )
            return vx_slow, wz_slow

        # ── NORMAL — soft side repulsion ──────────────────────────
        fl_danger = sec["front_left"]  < SIDE_SAFE_DIST
        fr_danger = sec["front_right"] < SIDE_SAFE_DIST
        l_danger  = sec["left"]        < SIDE_SAFE_DIST
        r_danger  = sec["right"]       < SIDE_SAFE_DIST

        if fl_danger or fr_danger or l_danger or r_danger:
            if left_score >= right_score:
                nudge = +SLOW_TURN_NUDGE * 0.5
                self._last_turn_dir = +1
            else:
                nudge = -SLOW_TURN_NUDGE * 0.5
                self._last_turn_dir = -1

            wz_nudged = _clamp(wz + nudge, -MAX_ANGULAR, MAX_ANGULAR)
            self.get_logger().info(
                f"[PathFollower/Avoidance] NORMAL nudge "
                f"FL={sec['front_left']:.2f} FR={sec['front_right']:.2f} "
                f"L={sec['left']:.2f} R={sec['right']:.2f} "
                f"→ {'LEFT' if nudge > 0 else 'RIGHT'}"
            )
            return vx, wz_nudged

        return vx, wz

    # ── LiDAR sector helpers (identical to SafeLidarMotion) ───────

    def _sector_min(self, processed: dict, a_min: float, a_max: float) -> float:
        """Min range in sector [a_min, a_max] degrees, with LiDAR offset."""
        off = LIDAR_OFFSET_DEG
        return self._prep.get_sector_min(processed, a_min + off, a_max + off)

    def _get_sectors(self, processed: dict) -> dict:
        s      = self._sector_min
        back_l = s(processed,  150,  180)
        back_r = s(processed, -180, -150)
        return {
            "front":       s(processed,  -20,   20),
            "front_left":  s(processed,   20,   70),
            "front_right": s(processed,  -70,  -20),
            "left":        s(processed,   70,  110),
            "right":       s(processed, -110,  -70),
            "back_left":   s(processed,  110,  150),
            "back_right":  s(processed, -150, -110),
            "back":        min(back_l, back_r),
        }

    def _score_sides(self, sec: dict) -> Tuple[float, float]:
        """
        Weighted clearance score for left vs right.
        Higher = more open = preferred turn direction.
        Identical weights to SafeLidarMotion.
        """
        left_score = (
            0.50 * sec["front_left"]  +
            0.30 * sec["left"]        +
            0.20 * sec["back_left"]
        )
        right_score = (
            0.50 * sec["front_right"] +
            0.30 * sec["right"]       +
            0.20 * sec["back_right"]
        )
        return left_score, right_score

    # ══════════════════════════════════════════════════════════════ #
    #  Stuck detection
    # ══════════════════════════════════════════════════════════════ #

    def _reset_stuck_detector(self):
        pose = self.slam.pose
        self._stuck_ref_pos  = (float(pose[0]), float(pose[1]))
        self._stuck_ref_time = _time.monotonic()

    def _check_stuck(self, rx: float, ry: float) -> bool:
        """
        True if the robot has not moved STUCK_DIST_THRESHOLD metres
        within the last STUCK_TIMEOUT seconds.
        Slides the reference window forward when progress is made.
        """
        elapsed = _time.monotonic() - self._stuck_ref_time
        if elapsed < STUCK_TIMEOUT:
            return False
        if self._stuck_ref_pos is None:
            self._reset_stuck_detector()
            return False

        moved = math.hypot(rx - self._stuck_ref_pos[0],
                           ry - self._stuck_ref_pos[1])
        if moved < STUCK_DIST_THRESHOLD:
            return True   # genuinely stuck

        # Good progress — slide window forward
        self._stuck_ref_pos  = (rx, ry)
        self._stuck_ref_time = _time.monotonic()
        return False

    # ══════════════════════════════════════════════════════════════ #
    #  Internal helpers
    # ══════════════════════════════════════════════════════════════ #

    def _reset_pid(self):
        """Reset PID state. Call only on new path arrival, not between WPs."""
        self._err_fwd_prev  = 0.0
        self._err_fwd_integ = 0.0
        self._last_ctrl_t   = _time.monotonic()

    def _finish(self, reason: str):
        """
        Stop the robot, clear path state, notify RRTExplorer.
        reason: 'completed' | 'stuck_timeout'
        """
        with self._path_lock:
            self._active   = False
            self._path     = []
            self._wp_index = 0

        self._cmd_pub.publish(Twist())   # zero Twist — stops all motion
        self.get_logger().info(
            f"[PathFollower] Finished — reason='{reason}'. "
            f"Motors stopped. Notifying explorer."
        )
        self._reset_pid()

        if self.explorer is not None:
            self.explorer.notify_path_done(reason)
        else:
            self.get_logger().error(
                "[PathFollower] explorer is None — notify_path_done() NOT called. "
                "RRTExplorer will never replan. "
                "Fix: set follower.explorer = rrt in main.py."
            )

    def stop(self):
        """Immediate emergency stop."""
        self._cmd_pub.publish(Twist())
        with self._path_lock:
            self._active = False


# ═══════════════════════════════════════════════════════════════════════ #
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════ #

def _wrap_angle(a: float) -> float:
    """Wrap angle to (-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))