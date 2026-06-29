#!/usr/bin/env python3
import json
from collections import defaultdict
from typing import Dict, Set, Tuple
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
        explorer,
        config_path=None,
    ):
        super().__init__("aruco_monitor")
        self._camera = camera
        self._aruco = aruco
        self.slam = slam
        self.explorer = explorer
        self._seen_markers: Set[int] = set()
        self._confirmed_markers: Set[int] = set()
        self._detection_counts: Dict[int, int] = defaultdict(int)
        self._last_frame_timestamp = None
        self._door_goals_requested: Set[int] = set()
        self.localizer = DoorLocalizer()

        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config", "key_doors.json",
            )
        door_marker_pairs, key_to_door, self.confirmation_frames = (
            self._load_config(config_path)
        )
        self.doors = DoorRegistry(
            door_marker_pairs=door_marker_pairs,
            key_to_door=key_to_door,
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
        self.get_logger().info(
            f"ArucoMonitor started with {len(door_marker_pairs)} door(s), "
            f"requiring {self.confirmation_frames} frame(s) per marker"
        )

    @staticmethod
    def _load_config(path: str) -> Tuple[Dict[Tuple[int, int], int], Dict[int, int], int]:
        with open(path) as stream:
            raw = json.load(stream)

        pairs = {}
        door_ids = set()
        marker_ids = set()
        for item in raw.get("door_marker_pairs", []):
            markers = item.get("markers", [])
            if len(markers) != 2 or int(markers[0]) == int(markers[1]):
                raise ValueError("Each door must contain two different ArUco marker IDs")
            pair = tuple(sorted((int(markers[0]), int(markers[1]))))
            door_id = int(item["door_id"])
            if door_id in door_ids or marker_ids.intersection(pair):
                raise ValueError("Door IDs and door-marker IDs must be unique")
            pairs[pair] = door_id
            door_ids.add(door_id)
            marker_ids.update(pair)

        key_to_door = {
            int(key_id): int(door_id)
            for key_id, door_id in raw.get("key_to_door", {}).items()
        }
        unknown_doors = set(key_to_door.values()) - door_ids
        if unknown_doors:
            raise ValueError(f"Keys reference undefined doors: {sorted(unknown_doors)}")
        if set(key_to_door).intersection(marker_ids):
            raise ValueError("A key ID cannot also be a door-marker ID")
        confirmation_frames = max(1, int(raw.get("confirmation_frames", 3)))
        return pairs, key_to_door, confirmation_frames

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

        # The timer is often faster than the camera. Never count the same image
        # repeatedly toward detection confirmation.
        if frame_data["timestamp"] == self._last_frame_timestamp:
            return
        self._last_frame_timestamp = frame_data["timestamp"]

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

            self._detection_counts[marker_id] += 1
            if self._detection_counts[marker_id] < self.confirmation_frames:
                continue
            if marker_id not in self._confirmed_markers:
                self._confirmed_markers.add(marker_id)
                self.get_logger().info(f"CONFIRMED MARKER ID={marker_id}")

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
                if self.doors.is_door_unlocked(door_id):
                    continue
                if data.get("_blocked", False):
                    continue
                self.explorer.block_door_in_costmap(
                    data["left"],
                    data["right"],
                    door_id=door_id,
                )
                data["_blocked"] = True
                self.get_logger().info(
                    f"Door {door_id} blocked"
                )

            #
            # Key found?
            #
            if marker_id in self.doors.registry.key_to_door_map:
                is_new_key = self.doors.register_key(
                    marker_id,
                    position=(wx, wy),
                )
                if not is_new_key:
                    continue
                door_id = (
                    self.doors.registry
                    .key_to_door_map[marker_id]
                )
                self.get_logger().info(
                    f"KEY {marker_id} COLLECTED: unlocks door {door_id}"
                )
                d = self.doors.get_door(door_id)
                if d is not None:
                    self.explorer.unblock_door(
                        d["left"],
                        d["right"],
                        door_id=door_id,
                    )
                    d["_blocked"] = False
                    self.explorer.navigate_to_door(
                        door_id,
                        d["center"],
                    )
                    self._door_goals_requested.add(door_id)
                    self.get_logger().info(
                        f"Door {door_id} unlocked; exploration pre-empted"
                    )
                else:
                    self.get_logger().warn(
                        f"Key {marker_id} stored, but door {door_id} has not been mapped yet"
                    )

        # Handles the uncommon but valid ordering where a key is seen before
        # both markers belonging to its door.
        for door_id, data in self.doors.door_world_positions.items():
            if (self.doors.is_door_unlocked(door_id)
                    and door_id not in self._door_goals_requested):
                self.explorer.unblock_door(
                    data["left"], data["right"], door_id=door_id
                )
                data["_blocked"] = False
                self.explorer.navigate_to_door(door_id, data["center"])
                self._door_goals_requested.add(door_id)

    def seen_markers(self) -> Set[int]:
        return self._seen_markers.copy()

    def print_summary(self):
        if not self._seen_markers:
            self.get_logger().info("No markers detected")
            return
        self.get_logger().info(
            f"\nMarkers found: {sorted(self._seen_markers)}"
        )
