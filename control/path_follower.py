#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Path


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class PathFollower(Node):

    def __init__(self, slam):

        super().__init__("path_follower")

        self.slam = slam

        self.path = []
        self.current_idx = 0

        self.goal_tolerance = 0.15
        self.max_linear = 0.20
        self.max_angular = 1.0

        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

        self.create_subscription(
            Path,
            "/exploration_path",
            self.path_callback,
            1
        )

        self.create_timer(
            0.1,
            self.control_step
        )

        self.get_logger().info("PathFollower ready.")

    def path_callback(self, msg):

        self.path = []

        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            self.path.append((x,y))

        self.current_idx = 0

        self.get_logger().info(
            f"Received path with {len(self.path)} points"
        )


    def control_step(self):
        if len(self.path) == 0:
            return

        robot_x, robot_y, robot_theta = self.slam.pose

        # Skip any waypoints we're already within tolerance of (handles the
        # "first point == current pose" case and short hops in one tick)
        while self.current_idx < len(self.path):
            tx, ty = self.path[self.current_idx]
            dx = tx - robot_x
            dy = ty - robot_y
            distance = math.hypot(dx, dy)

            if distance < self.goal_tolerance:
                self.current_idx += 1
                continue
            break

        if self.current_idx >= len(self.path):
            self.get_logger().info("Path completed")
            self.path = []
            stop = Twist()
            self.cmd_pub.publish(stop)
            return

        desired_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(desired_heading - robot_theta)

        cmd = Twist()
        cmd.angular.z = 1.5 * heading_error
        cmd.angular.z = max(-self.max_angular, min(self.max_angular, cmd.angular.z))

        if abs(heading_error) < 0.5:
            cmd.linear.x = min(self.max_linear, 0.5 * distance)

        self.cmd_pub.publish(cmd)