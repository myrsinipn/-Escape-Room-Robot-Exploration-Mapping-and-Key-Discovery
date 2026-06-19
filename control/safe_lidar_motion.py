#!/usr/bin/env python3
"""
SafeLidarMotion  +  Occupancy Grid Mapping
==========================================
Ένα αρχείο που κάνει και τα δύο:
  - Obstacle avoidance (safe_lidar_motion logic)
  - 2D Occupancy Grid mapping από LiDAR + /odom

Publishes:
  /cmd_vel          → κίνηση robot
  /slam_map         → OccupancyGrid (RViz Map display)

Subscribes:
  /scan             → LidarSensor node
  /odom             → θέση robot (nav_msgs/Odometry)
"""

import math
import os
import sys
from typing import Tuple

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensors.lidar import LidarSensor
from perception.scan_preprocessor import ScanPreprocessor


# ══════════════════════════════════════════════════════════════════
#  Occupancy Grid Mapper  (standalone node — τρέχει παράλληλα)
# ══════════════════════════════════════════════════════════════════

class OccupancyMapper(Node):
    """
    Κάνει subscribe στο /scan και /odom,
    χτίζει έναν 2D log-odds occupancy grid,
    και τον δημοσιεύει στο /slam_map.

    Bresenham ray-casting:
      - cells μέχρι το hit  → free   (log_free)
      - τελικό cell         → occupied (log_occ)
    """

    def __init__(self):
        super().__init__('occupancy_mapper')

        # ── Παράμετροι χάρτη ──────────────────────────────────────
        self.map_resolution  = 0.05      # m / cell  (5 cm)
        self.map_size_m      = 20.0      # m  (20×20 m)
        self.map_width       = int(self.map_size_m / self.map_resolution)
        self.map_height      = int(self.map_size_m / self.map_resolution)
        self.map_origin_x    = -self.map_size_m / 2.0
        self.map_origin_y    = -self.map_size_m / 2.0

        # Log-odds πίνακες
        self.log_odds = np.zeros((self.map_height, self.map_width), dtype=np.float32)
        self.known    = np.zeros((self.map_height, self.map_width), dtype=bool)

        self.log_free = -0.35
        self.log_occ  =  0.85
        self.log_min  = -5.0
        self.log_max  =  5.0

        self.max_mapping_range = 6.0     # m — ακτίνες πέρα από εδώ αγνοούνται

        # ── Pose από /odom ─────────────────────────────────────────
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.robot_theta = 0.0

        # ── Publishers / Subscribers ───────────────────────────────
        self.map_pub = self.create_publisher(OccupancyGrid, '/slam_map', 1)

        self.create_subscription(Odometry,   '/odom',  self._odom_cb,  30)
        self.create_subscription(LaserScan,  '/scan',  self._scan_cb,  10)

        self.get_logger().info(
            f'OccupancyMapper: {self.map_width}×{self.map_height} cells '
            f'@ {self.map_resolution*100:.0f}cm, '
            f'area {self.map_size_m}×{self.map_size_m}m'
        )

    # ── Odom callback ──────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_theta = math.atan2(siny_cosp, cosy_cosp)

    # ── Scan callback ──────────────────────────────────────────────
    def _scan_cb(self, msg):
        """
        Για κάθε ακτίνα:
          1. Υπολόγισε το τελικό σημείο στο world frame
          2. Bresenham από robot cell → end cell
          3. Ενημέρωσε free / occupied
        """
        rx = self.robot_x
        ry = self.robot_y
        rt = self.robot_theta

        robot_cell = self._world_to_cell(rx, ry)
        if robot_cell is None:
            return

        for i, r in enumerate(msg.ranges):
            if math.isnan(r) or math.isinf(r):
                continue
            if r < msg.range_min:
                continue

            hit = r <= self.max_mapping_range
            r_use = r if hit else self.max_mapping_range

            angle_local  = msg.angle_min + i * msg.angle_increment
            angle_global = rt + angle_local

            end_x = rx + r_use * math.cos(angle_global)
            end_y = ry + r_use * math.sin(angle_global)

            end_cell = self._world_to_cell(end_x, end_y)
            if end_cell is None:
                continue

            cells = self._bresenham(
                robot_cell[0], robot_cell[1],
                end_cell[0],   end_cell[1]
            )
            if not cells:
                continue

            # Free cells (όλα εκτός τελευταίου)
            for cx, cy in cells[:-1]:
                if self._valid(cx, cy):
                    self.log_odds[cy, cx] = np.clip(
                        self.log_odds[cy, cx] + self.log_free,
                        self.log_min, self.log_max
                    )
                    self.known[cy, cx] = True

            # Occupied — μόνο αν είχαμε πραγματικό hit
            if hit:
                cx, cy = cells[-1]
                if self._valid(cx, cy):
                    self.log_odds[cy, cx] = np.clip(
                        self.log_odds[cy, cx] + self.log_occ,
                        self.log_min, self.log_max
                    )
                    self.known[cy, cx] = True

        self._publish_map()

    # ── Map publisher ──────────────────────────────────────────────
    def _publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.info.resolution = self.map_resolution
        msg.info.width      = self.map_width
        msg.info.height     = self.map_height
        msg.info.origin.position.x    = self.map_origin_x
        msg.info.origin.position.y    = self.map_origin_y
        msg.info.origin.position.z    = 0.0
        msg.info.origin.orientation.w = 1.0

        data = []
        for y in range(self.map_height):
            for x in range(self.map_width):
                if not self.known[y, x]:
                    data.append(-1)          # unknown
                else:
                    p = 1.0 - 1.0 / (1.0 + math.exp(self.log_odds[y, x]))
                    data.append(int(round(100.0 * p)))

        msg.data = data
        self.map_pub.publish(msg)

    # ── Helpers ────────────────────────────────────────────────────
    def _world_to_cell(self, x, y):
        cx = int((x - self.map_origin_x) / self.map_resolution)
        cy = int((y - self.map_origin_y) / self.map_resolution)
        return (cx, cy) if self._valid(cx, cy) else None

    def _valid(self, cx, cy):
        return 0 <= cx < self.map_width and 0 <= cy < self.map_height

    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        """Bresenham line — επιστρέφει λίστα από (cx, cy)."""
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


# ══════════════════════════════════════════════════════════════════
#  SafeLidarMotion  (obstacle avoidance — αναλλοίωτο)
# ══════════════════════════════════════════════════════════════════

class SafeLidarMotion(Node):

    def __init__(self, lidar: LidarSensor, preprocessor: ScanPreprocessor):
        super().__init__('safe_lidar_motion')

        self._lidar        = lidar
        self._preprocessor = preprocessor

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.lidar_offset_deg = 180

        # Safety distances
        self.emergency_front_distance = 0.4
        self.slow_front_distance      = 0.8
        self.side_safe_distance       = 0.4

        # Speeds
        self.normal_forward_speed = 0.10
        self.slow_forward_speed   = 0.04
        self.base_turn_speed      = 0.35
        self.max_turn_speed       = 1.50
        self.turn_acceleration    = 0.30
        self.small_turn_speed     = 0.18

        self.last_turn_direction = 1
        self.turn_time           = 0.0
        self._last_time          = self.get_clock().now()
        self._debug_counter      = 0

        self.timer = self.create_timer(0.1, self._control_loop)
        self.get_logger().info("SafeLidarMotion node started.")

    # ── helpers ───────────────────────────────────────────────────

    def _s(self, processed: dict, a_min: float, a_max: float) -> float:
        o = self.lidar_offset_deg
        return self._preprocessor.get_sector_min(processed, a_min + o, a_max + o)

    def _get_sectors(self, processed: dict) -> dict:
        back_left  = self._s(processed,  150,  180)
        back_right = self._s(processed, -180, -150)
        return {
            "front":       self._s(processed,  -20,   20),
            "front_left":  self._s(processed,   20,   70),
            "front_right": self._s(processed,  -70,  -20),
            "left":        self._s(processed,   70,  110),
            "right":       self._s(processed, -110,  -70),
            "back_left":   self._s(processed,  110,  150),
            "back_right":  self._s(processed, -150, -110),
            "back":        min(back_left, back_right),
        }

    def _score_sides(self, sec: dict) -> Tuple[float, float]:
        left_score = (
            0.50 * sec["front_left"] +
            0.30 * sec["left"] +
            0.20 * sec["back_left"]
        )
        right_score = (
            0.50 * sec["front_right"] +
            0.30 * sec["right"] +
            0.20 * sec["back_right"]
        )
        return left_score, right_score

    # ── main loop ─────────────────────────────────────────────────

    def _control_loop(self):
        scan = self._lidar.get_scan()
        if scan is None:
            self.get_logger().warn(
                "Waiting for LiDAR scan...",
                throttle_duration_sec=2.0
            )
            return

        processed = self._preprocessor.preprocess(scan)

        now = self.get_clock().now()
        dt  = (now - self._last_time).nanoseconds / 1e9
        self._last_time = now

        sec = self._get_sectors(processed)
        left_score, right_score = self._score_sides(sec)

        self.get_logger().info(
            f"front={sec['front']:.2f}  "
            f"FL={sec['front_left']:.2f}  FR={sec['front_right']:.2f}  "
            f"L={sec['left']:.2f}  R={sec['right']:.2f}  "
            f"BL={sec['back_left']:.2f}  BR={sec['back_right']:.2f}  "
            f"| L_score={left_score:.2f}  R_score={right_score:.2f}"
        )

        cmd = Twist()
        cmd.linear.y = 0.0

        # ── EMERGENCY ─────────────────────────────────────────────
        if sec["front"] < self.emergency_front_distance:
            self.turn_time += dt
            turn_speed = min(
                self.base_turn_speed + self.turn_acceleration * self.turn_time,
                self.max_turn_speed,
            )
            cmd.linear.x = 0.0
            if left_score > right_score:
                cmd.angular.z =  turn_speed
                self.last_turn_direction =  1
            elif right_score > left_score:
                cmd.angular.z = -turn_speed
                self.last_turn_direction = -1
            else:
                cmd.angular.z = turn_speed * self.last_turn_direction
            self.get_logger().warn(
                f"EMERGENCY TURN {'LEFT' if cmd.angular.z > 0 else 'RIGHT'}"
                f"  speed={turn_speed:.2f}"
                f"  scores: L={left_score:.2f} R={right_score:.2f}"
            )

        # ── SLOW ──────────────────────────────────────────────────
        elif sec["front"] < self.slow_front_distance:
            self.turn_time = 0.0
            cmd.linear.x   = self.slow_forward_speed
            if left_score > right_score:
                cmd.angular.z =  self.small_turn_speed
                self.last_turn_direction =  1
            else:
                cmd.angular.z = -self.small_turn_speed
                self.last_turn_direction = -1
            self.get_logger().info(
                f"SLOW MODE  turning {'LEFT' if cmd.angular.z > 0 else 'RIGHT'}"
                f"  scores: L={left_score:.2f} R={right_score:.2f}"
            )

        # ── NORMAL ────────────────────────────────────────────────
        else:
            self.turn_time   = 0.0
            cmd.linear.x     = self.normal_forward_speed

            fl_danger = sec["front_left"]  < self.side_safe_distance
            fr_danger = sec["front_right"] < self.side_safe_distance
            l_danger  = sec["left"]        < self.side_safe_distance
            r_danger  = sec["right"]       < self.side_safe_distance

            if fl_danger or fr_danger or l_danger or r_danger:
                if left_score > right_score:
                    cmd.angular.z =  self.small_turn_speed
                    self.last_turn_direction =  1
                else:
                    cmd.angular.z = -self.small_turn_speed
                    self.last_turn_direction = -1
                self.get_logger().info(
                    f"NORMAL nudge {'LEFT' if cmd.angular.z > 0 else 'RIGHT'}"
                    f"  FL={sec['front_left']:.2f} FR={sec['front_right']:.2f}"
                    f"  scores: L={left_score:.2f} R={right_score:.2f}"
                )
            else:
                cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)


# ══════════════════════════════════════════════════════════════════
#  Main  — τρέχει και τους δύο nodes μαζί
# ══════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)

    # ── Sensors (shared) ──────────────────────────────────────────
    lidar = LidarSensor(topic_name='/scan', min_range=0.10, max_range=8.0)
    preprocessor = ScanPreprocessor(
        min_range=0.10,
        max_range=8.0,
        apply_smoothing=True,
        smoothing_kernel_size=5,
    )

    # ── Nodes ─────────────────────────────────────────────────────
    controller = SafeLidarMotion(lidar, preprocessor)
    mapper     = OccupancyMapper()

    # ── Executor ──────────────────────────────────────────────────
    executor = MultiThreadedExecutor()
    executor.add_node(lidar)
    executor.add_node(controller)
    executor.add_node(mapper)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Σταμάτα το robot πριν κλείσεις
        stop_cmd = Twist()
        controller.cmd_pub.publish(stop_cmd)

        controller.destroy_node()
        mapper.destroy_node()
        lidar.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    