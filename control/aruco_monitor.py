#!/usr/bin/env python3
from typing import Set
import numpy as np
import rclpy
from rclpy.node import Node

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensors.camera import CameraSensor
from perception.aruco_detector import ArucoDetector


class ArucoMonitor(Node):
    """
    Runs ArUco detection independently of motion.
    Logs new markers when first seen, throttles repeat detections.
    Ready to publish detections as ROS2 topic when needed later.
    """

    def __init__(self, camera: CameraSensor, aruco: ArucoDetector):
        super().__init__('aruco_monitor')

        self._camera = camera
        self._aruco  = aruco
        self._seen_markers: Set[int] = set()

        # 5 Hz — camera doesn't need to run as fast as LiDAR
        self.timer = self.create_timer(0.2, self._detection_loop)
        self.get_logger().info("ArucoMonitor node started.")

    def _detection_loop(self):
        frame_data = self._camera.get_frame()
        if frame_data is None:
            return

        detections = self._aruco.detect(frame_data["frame"])
        if not detections:
            return

        for det in detections:
            marker_id = det["marker_id"]
            tvec      = det["tvec"]
            distance  = float(np.linalg.norm(tvec))

            if marker_id not in self._seen_markers:
                self._seen_markers.add(marker_id)
                self.get_logger().info(
                    f"*** NEW MARKER  ID={marker_id} "
                    f"distance={distance:.2f}m "
                    f"tvec=[{tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f}] ***"
                )
            else:
                self.get_logger().info(
                    f"Marker ID={marker_id} visible  distance={distance:.2f}m",
                    throttle_duration_sec=2.0,
                )

    def seen_markers(self) -> Set[int]:
        """Public API for other nodes to query discovered markers."""
        return self._seen_markers.copy()
    def print_summary(self) -> None:
        if not self._seen_markers:
            self.get_logger().info("No ArUco markers were detected during this run.")
            return

        self.get_logger().info(
            f"\n"
            f"╔══════════════════════════════════╗\n"
            f"║       ARUCO MARKERS FOUND        ║\n"
            f"╠══════════════════════════════════╣\n"
            f"║  Total: {len(self._seen_markers):<26}║\n"
            f"║  IDs:   {str(sorted(self._seen_markers)):<26}║\n"
            f"╚══════════════════════════════════╝"
        )