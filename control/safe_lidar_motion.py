#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
import os
import sys
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensors.lidar import LidarSensor
from perception.scan_preprocessor import ScanPreprocessor


class SafeLidarMotion(Node):

    def __init__(self, lidar: LidarSensor, preprocessor: ScanPreprocessor):
        super().__init__('safe_lidar_motion')

        self._lidar       = lidar
        self._preprocessor = preprocessor

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.lidar_offset_deg = 180  

        # Safety distances
        self.emergency_front_distance = 1.0
        self.slow_front_distance      = 2
        self.side_safe_distance       = 1.32

        # Speeds
        self.normal_forward_speed = 0.10
        self.slow_forward_speed   = 0.04
        self.base_turn_speed      = 0.35
        self.max_turn_speed       = 1.50
        self.turn_acceleration    = 0.30
        self.small_turn_speed     = 0.18

        self.last_turn_direction  = 1
        self.turn_time            = 0.0
        self._last_time           = self.get_clock().now()
        self._debug_counter       = 0 

        self.timer = self.create_timer(0.1, self._control_loop)
        self.get_logger().info("SafeLidarMotion node started.")

    # ── helpers ──────────────────────────────────────────────────

    def _s(self, processed: dict, a_min: float, a_max: float) -> float:
        """Sector min with lidar offset applied."""
        o = self.lidar_offset_deg
        return self._preprocessor.get_sector_min(
            processed, a_min + o, a_max + o
        )

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
            cmd.linear.x = self.slow_forward_speed
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
            self.turn_time = 0.0
            cmd.linear.x = self.normal_forward_speed

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


def main(args=None):
    rclpy.init(args=args)

    lidar = LidarSensor(topic_name='/scan', min_range=0.10, max_range=8.0)
    preprocessor = ScanPreprocessor(
        min_range=0.10,
        max_range=8.0,
        apply_smoothing=True,
        smoothing_kernel_size=5,
    )
    controller = SafeLidarMotion(lidar, preprocessor)

    executor = MultiThreadedExecutor()
    executor.add_node(lidar)
    executor.add_node(controller)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd = Twist()
        controller.cmd_pub.publish(stop_cmd)
        controller.destroy_node()
        lidar.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()