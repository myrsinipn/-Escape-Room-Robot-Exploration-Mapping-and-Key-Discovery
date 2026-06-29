#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import JointState


class SlamWheelAnimator(Node):
    """
    Publishes wheel joint states so RViz can show wheel links.

    Input:
        /slam_pose

    Output:
        /joint_states

    Important:
        It publishes /joint_states at fixed rate, even if /slam_pose is slow.
        This prevents RViz errors like:
        No transform from [fl_wheel_link]
    """

    def __init__(self):
        super().__init__("slam_wheel_animator")

        self.sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/slam_pose",
            self.pose_callback,
            10
        )

        self.pub = self.create_publisher(JointState, "/joint_states", 10)

        self.last_pose = None
        self.prev_pose = None

        self.wheel_angle = 0.0

        # Approximate wheel radius in metres.
        # Tune this if wheel rotation looks too fast/slow.
        self.wheel_radius = 0.04

        # Publish joint states at 30 Hz so robot_state_publisher always has wheel TF.
        self.create_timer(1.0 / 30.0, self.publish_joint_states)

        self.get_logger().info("slam_wheel_animator ready: publishing wheel /joint_states at 30 Hz")

    def pose_callback(self, msg):
        self.last_pose = msg

    def publish_joint_states(self):
        # If we have pose data, update wheel angle from EKF/SLAM displacement.
        if self.last_pose is not None:
            x = self.last_pose.pose.pose.position.x
            y = self.last_pose.pose.pose.position.y

            if self.prev_pose is not None:
                prev_x, prev_y = self.prev_pose

                dx = x - prev_x
                dy = y - prev_y
                ds = math.sqrt(dx * dx + dy * dy)

                # Ignore tiny noise so wheels do not jitter.
                if ds > 0.001:
                    self.wheel_angle += ds / self.wheel_radius

            self.prev_pose = (x, y)

        # Always publish, even before movement.
        # This makes wheel transforms exist immediately in RViz.
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.name = [
            "fl_wheel_joint",
            "fr_wheel_joint",
            "rl_wheel_joint",
            "rr_wheel_joint"
        ]

        # If left/right visually rotate wrong, change signs.
        msg.position = [
            self.wheel_angle,
            -self.wheel_angle,
            self.wheel_angle,
            -self.wheel_angle
        ]

        msg.velocity = []
        msg.effort = []

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = SlamWheelAnimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
