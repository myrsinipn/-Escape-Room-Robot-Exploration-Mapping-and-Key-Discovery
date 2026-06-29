
#!/usr/bin/env python3
"""
Debug entry point — Continuous Straight Line Test for EKF-SLAM.

Run this on the robot/server terminal.

Node graph
----------
  lidar   (LidarSensor)       ─┐
  odom    (OdometrySensor)    ─┤
                               │
  straight(StraightCmdTest)  ──┤  Publishes /cmd_vel continuously
  slam    (EKFLidarSLAM)     ──┘  Uses lidar + odom + preprocessor + motion_model
"""

import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist

# Because this file is inside /test, add the project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# ── Sensors ───────────────────────────────────────────────────────────────────
from sensors.lidar    import LidarSensor
from sensors.odometry import OdometrySensor

# ── Perception ────────────────────────────────────────────────────────────────
from perception.scan_preprocessor import ScanPreprocessor
from perception.motion_model      import OmniMotionModel

# ── State estimation ──────────────────────────────────────────────────────────
from state_estimation.ekf_slam import EKFLidarSLAM


# ── Custom Debug Node ─────────────────────────────────────────────────────────

class StraightCmdTest(Node):
    """
    Publishes a continuous straight-line velocity command.
    """
    def __init__(self, speed=0.08):
        super().__init__("straight_cmd_test")

        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.speed = speed

        # Command timer: 20 Hz
        self.timer = self.create_timer(0.05, self.cmd_callback)

        self.get_logger().info(
            f"Straight test started: moving continuously at {self.speed} m/s. "
            "Press Ctrl+C to stop."
        )

    def cmd_callback(self):
        msg = Twist()
        msg.linear.x = self.speed
        self.pub.publish(msg)

    def stop_robot(self):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.linear.y = 0.0
        stop_msg.angular.z = 0.0

        for _ in range(10):
            self.pub.publish(stop_msg)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rclpy.init()

    # ── Sensors ───────────────────────────────────────────────────────────
    lidar = LidarSensor(topic_name="/scan", min_range=0.10, max_range=8.0)
    odom  = OdometrySensor(topic_name="/odom", qos_profile=30)

    # ── Shared perception helpers ─────────────────────────────────────────
    preprocessor = ScanPreprocessor(
        min_range=0.10,
        max_range=8.0,
        apply_smoothing=True,
        smoothing_kernel_size=5,
    )

    motion_model = OmniMotionModel()

    # ── Control node: straight-line command ───────────────────────────────
    straight_test = StraightCmdTest(speed=0.08)
    
    # ── SLAM node ─────────────────────────────────────────────────────────
    slam = EKFLidarSLAM(
        lidar=lidar,
        odom=odom,
        scan_preprocessor=preprocessor,
        motion_model=motion_model,
    )

    # ── Executor setup ────────────────────────────────────────────────────
    executor = MultiThreadedExecutor()

    executor.add_node(lidar)
    executor.add_node(odom)
    executor.add_node(straight_test)
    executor.add_node(slam)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        print("\nStopping robot and cleaning up...")

        straight_test.stop_robot()

        slam.print_debug_summary()

        for node in [slam, straight_test, odom, lidar]:
            node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
