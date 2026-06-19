#!/usr/bin/env python3
"""
Main entry point — B07 Escape Room Robot.

Node graph
----------
  lidar   (LidarSensor)       ─┐
  camera  (CameraSensor)       │  shared sensors
  odom    (OdometrySensor)    ─┤
                               │
  motion  (SafeLidarMotion)  ──┤  uses lidar + preprocessor
  monitor (ArucoMonitor)     ──┤  uses camera + aruco
  slam    (EKFLidarSLAM)     ──┘  uses lidar + odom + preprocessor + motion_model

All nodes run in one MultiThreadedExecutor.
"""

import json
import os
import sys

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mapping.rrt_exploration import RRTExplorer
# ── Sensors ───────────────────────────────────────────────────────────────────
from sensors.lidar    import LidarSensor
from sensors.camera   import CameraSensor
from sensors.odometry import OdometrySensor

# ── Perception ────────────────────────────────────────────────────────────────
from perception.scan_preprocessor import ScanPreprocessor
from perception.aruco_detector     import ArucoDetector
from perception.motion_model       import OmniMotionModel

# ── Control ───────────────────────────────────────────────────────────────────
from control.safe_lidar_motion import SafeLidarMotion
from control.aruco_monitor     import ArucoMonitor
from control.path_follower import PathFollower
# ── State estimation ──────────────────────────────────────────────────────────
from state_estimation.ekf_slam import EKFLidarSLAM


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_calibration(path: str, camera_id: str = "11"):
    with open(path) as f:
        data = json.load(f)
    cam = data[camera_id]
    return (
        np.array(cam["camera_matrix"], dtype=np.float64),
        np.array(cam["dist_coeffs"],   dtype=np.float64),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rclpy.init()

    # ── Sensors ───────────────────────────────────────────────────────────
    lidar  = LidarSensor(topic_name="/scan",       min_range=0.10, max_range=8.0)
    camera = CameraSensor(topic_name="/image_raw")
    odom   = OdometrySensor(topic_name="/odom",    qos_profile=30)

    # ── Shared perception helpers (not nodes) ─────────────────────────────
    preprocessor = ScanPreprocessor(
        min_range=0.10,
        max_range=8.0,
        apply_smoothing=True,
        smoothing_kernel_size=5,
    )

    motion_model = OmniMotionModel()

    calib_path = os.path.join(os.path.dirname(__file__), "config", "camera_calibration.json")
    camera_matrix, distortion_coeffs = load_calibration(calib_path)

    aruco = ArucoDetector(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        marker_size=0.05,
    )

    # ── Control nodes ─────────────────────────────────────────────────────
    #motion  = SafeLidarMotion(lidar, preprocessor)
    monitor = ArucoMonitor(camera, aruco)
   
    # ── SLAM node  (inject sensors + helpers — no internal executor) ──────
    slam = EKFLidarSLAM(
        lidar=lidar,
        odom=odom,
        scan_preprocessor=preprocessor,
        motion_model=motion_model,
    )
    path_follower = PathFollower(slam)
    rrt = RRTExplorer(
        slam=slam,
        update_period=2.0,
    )

    # ── Executor ──────────────────────────────────────────────────────────
    executor = MultiThreadedExecutor()

    executor.add_node(lidar)
    executor.add_node(camera)
    executor.add_node(odom)

    # executor.add_node(motion)
    executor.add_node(monitor)
    executor.add_node(path_follower)

    executor.add_node(slam)
    executor.add_node(rrt)

    try:
        
        
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.print_summary()
        slam.print_debug_summary()
        stop = Twist()
        path_follower.cmd_pub.publish(stop)

        for node in [rrt, slam,  path_follower, monitor, odom, camera, lidar]:
            node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()