#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener


def yaw_from_quat(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def quat_from_yaw(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class SlamPoseToMapOdomTF(Node):
    """
    Input:
        /slam_pose = EKF pose in map frame, i.e. map -> base_footprint

    Existing TF:
        odom -> base_footprint

    Output:
        map -> odom

    Final RViz chain:
        map -> odom -> base_footprint -> base_link -> wheels

    Important:
        This version DOES NOT reject jumps.
        It trusts the EKF /slam_pose and only republishes the last correction at 30 Hz.
    """

    def __init__(self):
        super().__init__("slam_pose_to_map_odom_tf")

        self.br = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.last_slam_pose = None
        self.latest_map_to_odom = None

        self.sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/slam_pose",
            self.slam_pose_callback,
            10
        )

        # Republish latest map -> odom at 30 Hz.
        self.timer = self.create_timer(1.0 / 30.0, self.publish_latest_tf)

        self.get_logger().info(
            "slam_pose_to_tf ready: /slam_pose + odom->base_footprint => map->odom"
        )

    def slam_pose_callback(self, msg):
        self.last_slam_pose = msg
        self.compute_map_to_odom()

    def compute_map_to_odom(self):
        if self.last_slam_pose is None:
            return

        try:
            odom_to_base = self.tf_buffer.lookup_transform(
                "odom",
                "base_footprint",
                Time()
            )
        except Exception as e:
            self.get_logger().warn(
                f"Waiting for odom -> base_footprint TF: {e}",
                throttle_duration_sec=2.0
            )
            return

        # EKF pose: map -> base_footprint
        slam_msg = self.last_slam_pose

        x_mb = float(slam_msg.pose.pose.position.x)
        y_mb = float(slam_msg.pose.pose.position.y)
        th_mb = yaw_from_quat(slam_msg.pose.pose.orientation)

        # Raw odom pose: odom -> base_footprint
        x_ob = float(odom_to_base.transform.translation.x)
        y_ob = float(odom_to_base.transform.translation.y)
        th_ob = yaw_from_quat(odom_to_base.transform.rotation)

        # Compute:
        # map -> odom = (map -> base_footprint) * inverse(odom -> base_footprint)
        th_mo = wrap_angle(th_mb - th_ob)

        c = math.cos(th_mo)
        s = math.sin(th_mo)

        rotated_x_ob = c * x_ob - s * y_ob
        rotated_y_ob = s * x_ob + c * y_ob

        x_mo = x_mb - rotated_x_ob
        y_mo = y_mb - rotated_y_ob

        qz, qw = quat_from_yaw(th_mo)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"

        t.transform.translation.x = float(x_mo)
        t.transform.translation.y = float(y_mo)
        t.transform.translation.z = 0.0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)

        self.latest_map_to_odom = t

        self.get_logger().info(
            f"Updated map->odom | x={x_mo:.3f}, y={y_mo:.3f}, yaw={math.degrees(th_mo):.1f} deg",
            throttle_duration_sec=1.0
        )

    def publish_latest_tf(self):
        if self.latest_map_to_odom is None:
            return

        self.latest_map_to_odom.header.stamp = self.get_clock().now().to_msg()
        self.br.sendTransform(self.latest_map_to_odom)


def main():
    rclpy.init()
    node = SlamPoseToMapOdomTF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

