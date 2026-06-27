#!/usr/bin/env python3
from typing import Set
import numpy as np
import rclpy
from rclpy.node import Node
import os
import sys

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
from perception.door_localizer import DoorLocalizer
from perception.door_registry import DoorRegistry
from sensors.camera import CameraSensor
from perception.aruco_detector import ArucoDetector


class ArucoMonitor(Node):
    def __init__(
        self,
        camera: CameraSensor,
        aruco: ArucoDetector,
        slam,
        explorer
    ):
        super().__init__("aruco_monitor")
        self._camera = camera
        self._aruco = aruco
        self.slam = slam
        self.explorer = explorer
        self._seen_markers: Set[int] = set()
        self.localizer = DoorLocalizer()
        self.doors = DoorRegistry(
            door_marker_pairs={
                (5, 6): 1,
            },
            key_to_door={
                10: 1,
            }
        )

        # --- RViz visualization publisher ---
        self._marker_pub = self.create_publisher(
            MarkerArray,
            "/aruco/markers_viz",
            10
        )

        self.timer = self.create_timer(
            0.2,
            self._detection_loop
        )
        self.get_logger().info("ArucoMonitor started")

    def _publish_marker_viz(
        self,
        marker_id: int,
        wx: float,
        wy: float,
    ) -> None:
        """Publish a sphere + text label for a detected ArUco marker."""

        now = self.get_clock().now().to_msg()
        array = MarkerArray()

        # --- Sphere at world position ---
        sphere = Marker()
        sphere.header.frame_id = "map"
        sphere.header.stamp = now
        sphere.ns = "aruco_spheres"
        sphere.id = marker_id
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = wx
        sphere.pose.position.y = wy
        sphere.pose.position.z = 0.0
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = 0.15
        sphere.scale.y = 0.15
        sphere.scale.z = 0.15
        sphere.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
        sphere.lifetime = Duration(sec=0)  # 0 = persist forever
        array.markers.append(sphere)

        # --- Text label above sphere ---
        label = Marker()
        label.header.frame_id = "map"
        label.header.stamp = now
        label.ns = "aruco_labels"
        label.id = marker_id
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = wx
        label.pose.position.y = wy
        label.pose.position.z = 0.3
        label.pose.orientation.w = 1.0
        label.scale.z = 0.15
        label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        label.text = f"ID {marker_id}"
        label.lifetime = Duration(sec=0)
        array.markers.append(label)

        self._marker_pub.publish(array)

    def _detection_loop(self):
        frame_data = self._camera.get_frame()
        if frame_data is None:
            return

        detections = self._aruco.detect(
            frame_data["frame"]
        )
        if not detections:
            return

        for det in detections:
            marker_id = det["marker_id"]
            tvec = det["tvec"]
            distance = float(np.linalg.norm(tvec))

            if marker_id not in self._seen_markers:
                self._seen_markers.add(marker_id)
                self.get_logger().info(
                    f"NEW MARKER "
                    f"ID={marker_id} "
                    f"dist={distance:.2f}m"
                )

            world = self.localizer.localize(
                marker_id,
                tvec,
                self.slam.pose
            )
            wx = world["world_x"]
            wy = world["world_y"]

            # --- Publish to RViz ---
            self._publish_marker_viz(marker_id, wx, wy)

            self.doors.register_marker_position(
                marker_id,
                wx,
                wy
            )

            #
            # Door complete?
            #
            for door_id, data in self.doors.door_world_positions.items():
                if data.get("_blocked", False):
                    continue
                self.explorer.block_door_in_costmap(
                    data["left"],
                    data["right"]
                )
                data["_blocked"] = True
                self.get_logger().info(
                    f"Door {door_id} blocked"
                )

            #
            # Key found?
            #
            if marker_id in self.doors.registry.key_to_door_map:
                self.doors.register_key(marker_id)
                door_id = (
                    self.doors.registry
                    .key_to_door_map[marker_id]
                )
                d = self.doors.get_door(door_id)
                if d is not None:
                    self.explorer.unblock_door(
                        d["left"],
                        d["right"]
                    )
                    self.get_logger().info(
                        f"Door {door_id} unlocked"
                    )

    def seen_markers(self) -> Set[int]:
        return self._seen_markers.copy()

    def print_summary(self):
        if not self._seen_markers:
            self.get_logger().info("No markers detected")
            return
        self.get_logger().info(
            f"\nMarkers found: {sorted(self._seen_markers)}"
        )