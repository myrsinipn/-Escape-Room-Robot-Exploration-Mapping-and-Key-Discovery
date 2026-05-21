import time
from typing import Optional, Dict, Any

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarSensor(Node):
    """
    ROS2 LiDAR interface for /scan topic.

    Responsibilities:
    - Subscribe to LaserScan messages
    - Store latest scan
    - Convert to numpy arrays
    - Provide clean API to the rest of the system

    """

    def __init__(
        self,
        topic_name: str = "/scan",
        min_range: float = 0.05,
        max_range: float = 10.0,
        qos_profile: int = 10,
    ) -> None:

        super().__init__("lidar_sensor")

        self.topic_name = topic_name

        self.min_range = min_range
        self.max_range = max_range

        # latest scan data
        self._ranges: Optional[np.ndarray] = None
        self._angles: Optional[np.ndarray] = None
        self._timestamp: Optional[float] = None

        # metadata
        self._angle_min: Optional[float] = None
        self._angle_max: Optional[float] = None
        self._angle_increment: Optional[float] = None

        # ROS2 subscriber
        self.subscription = self.create_subscription(
            LaserScan,
            self.topic_name,
            self.lidar_callback,
            qos_profile,
        )

        self.get_logger().info(
            f"LiDAR subscriber initialized on topic: {self.topic_name}"
        )

    def lidar_callback(self, msg: LaserScan) -> None:
        """
        Callback for incoming LaserScan messages.
        """

        ranges = np.array(msg.ranges, dtype=np.float32)

        # Replace invalid values
        invalid_mask = np.isnan(ranges) | np.isinf(ranges)

        ranges[invalid_mask] = self.max_range

        # Clip ranges
        ranges = np.clip(
            ranges,
            self.min_range,
            self.max_range,
        )

        # Generate angles
        angles = np.arange(
            msg.angle_min,
            msg.angle_max,
            msg.angle_increment,
            dtype=np.float32,
        )

        # Ensure equal lengths
        min_len = min(len(ranges), len(angles))

        self._ranges = ranges[:min_len]
        self._angles = angles[:min_len]

        self._timestamp = time.time()

        # store metadata
        self._angle_min = msg.angle_min
        self._angle_max = msg.angle_max
        self._angle_increment = msg.angle_increment

    def get_scan(self) -> Optional[Dict[str, Any]]:
        """
        Returns latest scan.

        Output format:
        {
            "ranges": np.ndarray,
            "angles": np.ndarray,
            "timestamp": float,
        }
        """

        if self._ranges is None:
            return None

        return {
            "ranges": self._ranges.copy(),
            "angles": self._angles.copy(),
            "timestamp": self._timestamp,
        }

    def has_scan(self) -> bool:
        """
        Returns True if at least one scan has been received.
        """
        return self._ranges is not None

    @property
    def ranges(self) -> Optional[np.ndarray]:

        if self._ranges is None:
            return None

        return self._ranges.copy()

    @property
    def angles(self) -> Optional[np.ndarray]:

        if self._angles is None:
            return None

        return self._angles.copy()

    @property
    def timestamp(self) -> Optional[float]:

        return self._timestamp

    def print_scan_info(self) -> None:

        if not self.has_scan():

            self.get_logger().warn("No LiDAR scan received yet.")
            return

        self.get_logger().info(
            f"Scan size: {len(self._ranges)} | "
            f"Angle range: [{self._angle_min:.2f}, "
            f"{self._angle_max:.2f}] rad"
        )


def main(args=None):

    rclpy.init(args=args)

    lidar_sensor = LidarSensor()

    try:

        rclpy.spin(lidar_sensor)

    except KeyboardInterrupt:

        pass

    finally:

        lidar_sensor.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":

    main()