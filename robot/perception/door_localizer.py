#!/usr/bin/env python3

import math
import numpy as np


class DoorLocalizer:

    CAMERA_X = 0.13
    CAMERA_Y = 0.0
    CAMERA_Z = 0.131


    def localize(
        self,
        marker_id:int,
        tvec,
        slam_pose
    ):

        rx,ry,rtheta = slam_pose

        #
        # OpenCV camera frame:
        #
        # x → right
        # y → down
        # z → forward
        #

        cx = float(tvec[2])
        cy = -float(tvec[0])

        robot_x = cx + self.CAMERA_X
        robot_y = cy + self.CAMERA_Y

        wx = (
            rx
            + robot_x*np.cos(rtheta)
            - robot_y*np.sin(rtheta)
        )

        wy = (
            ry
            + robot_x*np.sin(rtheta)
            + robot_y*np.cos(rtheta)
        )

        return {

            "marker_id":marker_id,
            "world_x":wx,
            "world_y":wy
        }