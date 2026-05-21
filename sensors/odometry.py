import time
from typing import Optional, Dict, Any

import numpy as np
import rclpy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf_transformations import euler_from_quaternion


class OdometrySensor(Node):
    """
    ROS2 odometry interface.

    Responsibilities:
    - Subscribe to odometry topic
    - Extract robot pose
    - Extract robot velocities
    - Convert quaternion -> yaw
    - Provide clean API

    """

    def __init__(
        self,
        topic_name: str = "/odom",
        qos_profile: int = 10,
    ) -> None:

        super().__init__("odometry_sensor")

        self.topic_name = topic_name

        # pose
        self._pose: Optional[np.ndarray] = None

        # velocities [vx, vy, omega]
        self._velocity: Optional[np.ndarray] = None

        self._timestamp: Optional[float] = None

        # ROS2 subscriber
        self.subscription = self.create_subscription(
            Odometry,
            self.topic_name,
            self.odometry_callback,
            qos_profile,
        )

        self.get_logger().info(
            f"Odometry subscriber initialized on topic: {self.topic_name}"
        )

    def odometry_callback(self, msg: Odometry) -> None:
        """
        ROS2 odometry callback.
        """

        # position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # orientation quaternion
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # quaternion -> euler
        _, _, yaw = euler_from_quaternion(
            [qx, qy, qz, qw]
        )

        self._pose = np.array(
            [x, y, yaw],
            dtype=np.float32,
        )

        # body velocities
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z

        self._velocity = np.array(
            [vx, vy, omega],
            dtype=np.float32,
        )

        self._timestamp = time.time()

    def get_pose(self) -> Optional[Dict[str, Any]]:
        """
        Returns robot pose.

        Output:
        {
            "pose": np.ndarray([x, y, theta]),
            "timestamp": float,
        }
        """

        if self._pose is None:
            return None

        return {
            "pose": self._pose.copy(),
            "timestamp": self._timestamp,
        }

    def get_velocity(self) -> Optional[Dict[str, Any]]:
        """
        Returns robot velocity.

        Output:
        {
            "velocity": np.ndarray([vx, vy, omega]),
            "timestamp": float,
        }
        """

        if self._velocity is None:
            return None

        return {
            "velocity": self._velocity.copy(),
            "timestamp": self._timestamp,
        }

    def has_odometry(self) -> bool:
        """
        Returns True if odometry has been received.
        """

        return self._pose is not None

    @property
    def pose(self) -> Optional[np.ndarray]:

        if self._pose is None:
            return None

        return self._pose.copy()

    @property
    def velocity(self) -> Optional[np.ndarray]:

        if self._velocity is None:
            return None

        return self._velocity.copy()

    @property
    def timestamp(self) -> Optional[float]:

        return self._timestamp

    def print_odometry_info(self) -> None:

        if not self.has_odometry():

            self.get_logger().warn(
                "No odometry received yet."
            )
            return

        x, y, theta = self._pose
        vx, vy, omega = self._velocity

        self.get_logger().info(
            f"Pose: x={x:.2f}, y={y:.2f}, θ={theta:.2f} | "
            f"Velocity: vx={vx:.2f}, vy={vy:.2f}, ω={omega:.2f}"
        )


def main(args=None):

    rclpy.init(args=args)

    odometry_sensor = OdometrySensor()

    try:

        rclpy.spin(odometry_sensor)

    except KeyboardInterrupt:

        pass

    finally:

        odometry_sensor.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()