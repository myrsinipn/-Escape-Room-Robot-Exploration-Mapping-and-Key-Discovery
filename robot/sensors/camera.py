import time
from typing import Optional, Dict, Any

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraSensor(Node):
    """
    ROS2 camera interface.

    Responsibilities:
    - Subscribe to camera topic
    - Convert ROS image -> OpenCV image
    - Store latest frame
    - Provide clean API

    """

    def __init__(
        self,
        topic_name: str = "/camera/image_raw",
        qos_profile: int = 10,
    ) -> None:

        super().__init__("camera_sensor")

        self.topic_name = topic_name

        self.bridge = CvBridge()

        # latest frame
        self._frame: Optional[np.ndarray] = None
        self._timestamp: Optional[float] = None

        # image metadata
        self._height: Optional[int] = None
        self._width: Optional[int] = None
        self._encoding: Optional[str] = None

        # ROS2 subscriber
        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.image_callback,
            qos_profile,
        )

        self.get_logger().info(
            f"Camera subscriber initialized on topic: {self.topic_name}"
        )

    def image_callback(self, msg: Image) -> None:
        """
        ROS2 image callback.
        Converts ROS Image message -> OpenCV frame.
        """

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )

            self._frame = frame
            self._timestamp = time.time()

            self._height = msg.height
            self._width = msg.width
            self._encoding = msg.encoding

        except Exception as e:

            self.get_logger().error(
                f"Failed to convert image: {str(e)}"
            )

    def get_frame(self) -> Optional[Dict[str, Any]]:
        """
        Returns latest frame.

        Output format:
        {
            "frame": np.ndarray,
            "timestamp": float,
        }
        """

        if self._frame is None:
            return None

        return {
            "frame": self._frame.copy(),
            "timestamp": self._timestamp,
        }

    def has_frame(self) -> bool:
        """
        Returns True if at least one frame has been received.
        """

        return self._frame is not None

    @property
    def frame(self) -> Optional[np.ndarray]:

        if self._frame is None:
            return None

        return self._frame.copy()

    @property
    def timestamp(self) -> Optional[float]:

        return self._timestamp

    def print_camera_info(self) -> None:

        if not self.has_frame():

            self.get_logger().warn(
                "No camera frame received yet."
            )
            return

        self.get_logger().info(
            f"Resolution: {self._width}x{self._height} | "
            f"Encoding: {self._encoding}"
        )

    def show_live_view(
        self,
        window_name: str = "Camera",
    ) -> None:
        """
        Displays live camera feed.
        """

        if self._frame is None:
            return

        cv2.imshow(window_name, self._frame)
        cv2.waitKey(1)


def main(args=None):

    rclpy.init(args=args)

    camera_sensor = CameraSensor()

    try:

        while rclpy.ok():

            rclpy.spin_once(camera_sensor)

            camera_sensor.show_live_view()

    except KeyboardInterrupt:

        pass

    finally:

        cv2.destroyAllWindows()

        camera_sensor.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()