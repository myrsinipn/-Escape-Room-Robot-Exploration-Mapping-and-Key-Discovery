#!/usr/bin/env python3
"""
DoorLocalizer: converts a detected ArUco marker's camera-frame translation
vector into world-frame (x, y) coordinates using the current SLAM pose.
"""

import math
import numpy as np


class DoorLocalizer:
    # Camera mounting offset from the robot's centre of rotation (metres).
    # Positive X is forward, positive Y is left.
    CAMERA_X = 0.13   # camera is 13 cm ahead of the rotation centre
    CAMERA_Y = 0.0    # camera is centred laterally
    CAMERA_Z = 0.131  # camera height (not used in 2-D localisation)

    def localize(
        self,
        marker_id: int,
        tvec,
        slam_pose,
    ):
        """Convert a marker translation vector to a world-frame position.

        OpenCV camera frame convention:
          x → right, y → down, z → forward

        The translation vector is reordered to the robot's 2-D plane
        before the camera offset and SLAM pose are applied.

        Returns a dict with keys: marker_id, world_x, world_y.
        """
        rx, ry, rtheta = slam_pose

        # Remap OpenCV camera axes to robot-frame 2-D coordinates:
        #   robot forward = camera z
        #   robot left    = -(camera x)
        cx = float(tvec[2])   # forward distance to marker
        cy = -float(tvec[0])  # lateral distance (sign-flip: OpenCV x is right)

        # Shift from camera to robot rotation centre.
        robot_x = cx + self.CAMERA_X
        robot_y = cy + self.CAMERA_Y

        # Rotate from robot frame into world frame using the SLAM heading.
        wx = rx + robot_x * math.cos(rtheta) - robot_y * math.sin(rtheta)
        wy = ry + robot_x * math.sin(rtheta) + robot_y * math.cos(rtheta)

        return {
            "marker_id": marker_id,
            "world_x": wx,
            "world_y": wy,
        }